"""Orquestador del pipeline de features: ensambla la matriz de modelado.

Orden de las capas, que no es libre y es lo que este módulo fija:

1. **Capa 1 sobre la tabla principal.** `limpiar_application()` corre **antes del split**, sobre
   la tabla entera. Es determinista, no estima ningún parámetro, no mira al TARGET y no cruza
   clientes, así que ejecutarla fuera de la partición no filtra nada. Es la misma razón por la
   que las agregaciones de las tres auxiliares también van fuera. Los descartes de columna que
   el EDA decidió contra la tasa de default **no** se aplican aquí: sobreviven declarados en
   `COLUMNAS_PROVISIONALES` y los juzga la capa 2b sobre el split.
2. **Capa 0, el split**, sobre el resultado ya limpio. Se hace en este orden y no al revés
   porque la partición tiene que cubrir exactamente la población de modelado: si se partiera
   la tabla cruda, el fichero de split declararía 19 clientes que después no existen en la
   matriz, y sus conteos de estratificación no serían los de la población real.
3. **Capas 2a y 2b**, que sí estiman parámetros y por eso se ajustan solo sobre la parte de
   entrenamiento. De la 2a está el winsorizador, en `ajustar_capa2a()`; el resto y la 2b
   entran con el `Pipeline`.

Las 19 filas que la limpieza quita son todas TARGET a 0, así que los 24.825 positivos no se
mueven; lo que baja es el denominador, y la tasa pasa de 8,0729% a 8,0734%.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline

from src.config import cargar_config, ruta
from src.data.loader import load_table
from src.features.agg_bureau import agregar_bureau, unir_bureau, vencimiento_a_termino
from src.features.agg_bureau_balance import (
    agregar_bureau_balance,
    puente_credito_cliente,
    unir_bureau_balance,
)
from src.features.agg_previous import (
    CAPTACION_CALLE,
    agregar_previous,
    cociente_de_concesion,
    combinacion_definida,
    fin_de_ventana,
    finalidad_declarada,
    solicitudes_recientes,
    unir_previous,
)
from src.features.application import construir_features_capa1, verificar_contrato_capa1
from src.features.cleaning import (
    filas_a_eliminar,
    limpiar_application,
    limpiar_application_entrenamiento,
    limpiar_bureau,
    limpiar_previous,
)
from src.features.params import fijar_operativo, parametro, valor
from src.features.pipeline import construir_pipeline
from src.features.split import cargar_split, construir_split, solo_train
from src.features.transformers import Winsorizador, registrar_limites

logger = logging.getLogger(__name__)

# reduce_mem_usage no se aplica a la tabla principal: baja los float a float32 y el EDA ya
# documentó que eso convierte en desigualdad estricta 453 comparaciones de fechas que en
# realidad son iguales. Son 307.511 filas, la memoria no es el problema aquí.
REDUCIR_MEMORIA = False

# La rejilla con la que el EDA vio la U de DAYS_CREDIT_ENDDATE (notebook 02, celda 58), en años con
# signo. El tramo sale de ella y no de una más fina, y las dos lecturas que lo comprueban acaban en
# el mismo sitio: con tramos de un año el pico se va a 5 a 6 (11,17% sobre 1.307 filas, con 4 a 5
# detrás en 10,93% sobre 29.974, y dejando fuera el extremo de una sola fila), y un barrido por
# delta a nivel cliente sobre tramos enteros elige ese mismo 5 a 6 (+3,45pp sobre 1.228 clientes,
# frente a los +2,49pp sobre 71.580 del 2 a 5). Las dos se quedan con el tramo más estrecho que la
# rejilla permita, así que el valor lo decidiría la rejilla y no el dato.
REJILLA_VENCIMIENTO_ANIOS = (-np.inf, -5, -2, 0, 2, 5, 10, np.inf)

# La rejilla con la que el EDA barrió la sobreconcesión de previous_application (notebook 04, celda
# 160), en cociente concedido entre solicitado. El 1,3 heredado del binning no salió de un barrido
# y perdía dos tercios de la señal de la proporción.
REJILLA_SOBRECONCESION = (1.05, 1.1, 1.2, 1.3, 1.4, 1.5)

# Los cortes medidos que se consumen fuera del Pipeline, en la agregación, y que por eso se
# persisten en cortes.json. Los del winsorizador no: viajan dentro del Pipeline ajustado.
NOMBRE_FICHERO_CORTES = "cortes.json"
CORTES_AUXILIARES = (
    "bureau_enddate_tramo_min_anios",
    "bureau_enddate_tramo_max_anios",
    "bureau_count_cola",
    "bb_many_credits_corte",
    "prev_count_cola",
    "prev_actividad_12m_cola",
    "prev_sobreconcesion_corte",
    "prev_finalidades_urgentes",
)

# Los mínimos de denominador que barrió el EDA (notebook 04, celda 155): con uno no hay mínimo.
MINIMOS_DENOMINADOR = (1, 2, 3, 5)

# Los estratos de recencia de la última solicitud con los que el EDA controló la captación de
# previous_application (5C.11), en días con signo y con el borde superior dentro. Son el control
# obligatorio del 4.10: si una lectura relativa solo gana porque codifica "tu última solicitud es
# reciente", dentro de estos estratos su ventaja desaparece.
ESTRATOS_RECENCIA_PREVIOUS: dict[str, tuple[float, float]] = {
    "hasta 6m": (-180, 0),
    "6 a 12m": (-365, -180),
    "12 a 24m": (-730, -365),
    "mas de 24m": (-np.inf, -730),
}

# Las lecturas de recencia relativa del 4.10 con las contrapartes contra las que se comparan, todas
# escritas antes de medir. La recencia del último rechazo y las tres de captación se miden contra su
# propia versión absoluta; la salida en rechazo, contra las dos banderas de rechazo que ya existen;
# y el conteo de la ventana propia, además de contra el absoluto, contra el ritmo, que es la vía por
# la que puede ser la longitud de la relación con otro nombre.
LECTURAS_RECENCIA_RELATIVA: dict[str, tuple[str, ...]] = {
    "PREV_EXIT_REFUSED_FLAG": ("PREV_REFUSED_RATIO > 0", "PREV_REFUSED_SCOFR_FLAG"),
    "PREV_DAYS_SINCE_REFUSED_REL": ("PREV_DAYS_SINCE_REFUSED",),
    "PREV_COUNT_12M_REL": ("PREV_COUNT_12M", "PREV_APPLICATIONS_PER_YEAR"),
    "PREV_STREET_RATIO_REL": ("PREV_STREET_RATIO",),
    "PREV_EARLY_HOUR_RATIO_REL": ("PREV_EARLY_HOUR_RATIO",),
    "PREV_NO_SUITE_RATIO_REL": ("PREV_NO_SUITE_RATIO",),
}


def cargar_y_limpiar(nombre: str = "application_train") -> pd.DataFrame:
    """Carga la tabla principal y le aplica la limpieza determinista.

    Vale igual para `application_test`, y ahí está la diferencia que importa: la tabla que
    trae etiquetas es la de entrenamiento y es la única a la que se le pueden quitar filas.
    Sobre test se limpia sin borrar a nadie, que si no se perderían los 24 clientes sin cuota
    declarada.
    """
    app = load_table(nombre, reduce_memory=REDUCIR_MEMORIA)
    limpio = (
        limpiar_application_entrenamiento(app)
        if "TARGET" in app.columns
        else limpiar_application(app)
    )
    logger.info(
        "%s: %s filas x %s columnas -> %s x %s",
        nombre,
        f"{len(app):,}",
        app.shape[1],
        f"{len(limpio):,}",
        limpio.shape[1],
    )
    return limpio


def preparar_application(nombre: str = "application_train") -> pd.DataFrame:
    """La capa 1 entera sobre la tabla principal: limpieza más features sin parámetro.

    Las dos mitades se dejan invocables por separado porque cada una tiene su propia puerta
    de salida: la limpieza se mide en filas y columnas, y las features en la tripartita del
    bloque edificio.

    Aquí es donde se exige el contrato de esquema, y no dentro de las features: sus piezas son
    permisivas a propósito, porque a la API puede llegar un frame parcial, pero la matriz que
    ve el modelo tiene que salir siempre de las mismas columnas de origen.
    """
    limpio = cargar_y_limpiar(nombre)
    verificar_contrato_capa1(limpio)
    return construir_features_capa1(limpio)


def construir_base(sobrescribir: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La tabla principal con la capa 1 aplicada y su partición, en ese orden.

    Devuelve `(base, split)`. El split se persiste; si ya existe hay que pedir
    `sobrescribir=True`, porque rehacerlo invalida todo lo que se haya ajustado sobre él.
    """
    base = preparar_application()
    split = construir_split(app=base, sobrescribir=sobrescribir)
    return base, split


def _verificar_union(unido: pd.DataFrame, base: pd.DataFrame, etiqueta: str) -> None:
    """El `join` no puede cambiar la población: mismas filas que `base`, índice único."""
    if len(unido) != len(base):
        raise ValueError(
            f"unir_{etiqueta}() cambió el número de filas: {len(base):,} -> {len(unido):,}. "
            "Un left join no infla filas salvo que la lista de clientes traiga un duplicado."
        )
    if not unido.index.is_unique:
        raise ValueError(f"el índice deja de ser único tras unir_{etiqueta}()")


def _fijar_tipos(unido: pd.DataFrame, agregado: pd.DataFrame) -> pd.DataFrame:
    """Castea a float64 las columnas que trajo el agregado, salvo `BB_TRAYECTORY`.

    Es la frontera que construye la matriz la que exige el contrato de esquema, no cada pieza:
    `agregar_bureau()`, `agregar_bureau_balance()` y `agregar_previous()` siguen siendo
    permisivas, porque a la API puede llegar un frame parcial. Sin este casteo el esquema
    depende del lote: un cliente sin historial deja un conteo en float64 y uno con todos con
    historial lo deja en int64 o int8, el patrón 12 sobre la matriz de capa 1. Las `HAS_*` las
    crea cada `unir_*` en int8 y no se tocan aquí.
    """
    columnas = [c for c in agregado.columns if c != "BB_TRAJECTORY"]
    return unido.astype(dict.fromkeys(columnas, "float64"))


def ensamblar_auxiliares(
    base: pd.DataFrame, bureau: pd.DataFrame, bb: pd.DataFrame, prev: pd.DataFrame
) -> pd.DataFrame:
    """Agrega y une las tres auxiliares a la base de capa 1, sin tocar el split.

    Capa 1 pura: no estima nada, no mira al TARGET y el agregado de un cliente solo depende de
    sus propias filas, así que corre fuera del split y el mismo camino sirve para
    `application_test` y para un cliente suelto de la API. Las tres tablas van crudas, cada
    agregación limpia lo suyo y es idempotente.

    Ninguna agregación recibe `cortes`: los ocho de `CORTES_AUXILIARES` se leen de `params.py`,
    y pasarlos a mano es para los contrastes, nunca para construir la matriz. Si alguno sigue sin
    fijar, revienta antes de agregar nada, nombrando cómo se arregla.

    El orden de las uniones no es preferencia: `unir_bureau_balance()` exige que `clientes` ya
    traiga `BUREAU_OVERDUE_UNION`, porque `BB_OVERDUE_UNION` la lee para no perder la mora del
    cliente con historial de bureau y sin panel mensual.
    """
    faltan = [c for c in CORTES_AUXILIARES if parametro(c).valor_operativo is None]
    if faltan:
        raise ValueError(
            f"cortes de las auxiliares sin fijar: {faltan}. Refíjalos con "
            "refijar_cortes_auxiliares() y guardar_cortes(), o cárgalos con cargar_cortes() "
            "si ya están persistidos en cortes.json."
        )
    puente = puente_credito_cliente(bureau)

    ag_bureau = agregar_bureau(bureau)
    unido = unir_bureau(base, ag_bureau)
    _verificar_union(unido, base, "bureau")
    unido = _fijar_tipos(unido, ag_bureau)

    ag_bb = agregar_bureau_balance(bb, puente)
    unido = unir_bureau_balance(unido, ag_bb)
    _verificar_union(unido, base, "bureau_balance")
    unido = _fijar_tipos(unido, ag_bb)

    ag_prev = agregar_previous(prev)
    unido = unir_previous(unido, ag_prev)
    _verificar_union(unido, base, "previous")
    unido = _fijar_tipos(unido, ag_prev)

    return unido


def matriz_de_features(base: pd.DataFrame) -> pd.DataFrame:
    """La base sin la etiqueta ni el identificador, que es lo que consumen las capas 2.

    Aquí cuelga el contrato de nombres del pipeline. Ajustar sobre `base` entera dejaba
    `TARGET` y `SK_ID_CURR` dentro de `feature_names_in_`, y con eso `get_feature_names_out()`
    prometía 101 columnas mientras `transform` sobre la X de verdad devolvía 100: una promesa
    que no se cumple y que en la capa 1.4 lee el `ColumnTransformer`.

    Es permisiva porque `application_test` no trae objetivo.
    """
    cfg = cargar_config()["dataset"]
    return base.drop(columns=[cfg["id_col"], cfg["target_col"]], errors="ignore")


def ajustar_capa2a(
    base: pd.DataFrame, split: pd.DataFrame | None = None, sobrescribir: bool = False
) -> tuple[Winsorizador, pd.DataFrame]:
    """Ajusta el winsorizador sobre el 80% de entrenamiento y registra lo reestimado.

    Sigue siendo el único consumidor del registro de `params.py`: los tres cortes que estrenó el
    1.4 (`suavizado_woe`, `n_min_categoria` y `app_umbral_varianza`) son de dominio y no pasan
    por ahí. La lista de límites vive en el transformer, que es lo que se serializa y lo que
    entra en el CV; el registro solo guarda el rastro de con qué cifra y con cuántas filas se
    ajustó.

    El orden de las dos operaciones no es libre: primero se filtra la partición, que necesita el
    identificador, y después se quita. Al revés no hay por dónde filtrar.

    Devuelve `(winsorizador, informe)`, y el informe es la puerta del punto: los diez cortes
    reestimados frente a la referencia del EDA, con su desviación.

    Reajustar sobre otra población exige `sobrescribir=True`: el registro ya trae lo de la
    primera vez y pisarlo en silencio dejaría el valor de un ajuste con el rastro del otro.
    """
    entrenamiento = matriz_de_features(solo_train(base, split))
    winsorizador = Winsorizador().fit(entrenamiento)
    logger.info("capa 2a ajustada sobre %s filas de entrenamiento", f"{len(entrenamiento):,}")
    return winsorizador, registrar_limites(winsorizador, sobrescribir)


def _bureau_de_train(
    bureau: pd.DataFrame, base: pd.DataFrame, split: pd.DataFrame | None
) -> pd.DataFrame:
    """Las filas limpias de bureau de los clientes de train, con `SK_ID_CURR` y `TARGET`.

    Es la población de los refijados de bureau: el `n_train` que declaran son sus clientes, los de
    train con historial.
    """
    entrenamiento = solo_train(base, split)[["SK_ID_CURR", "TARGET"]]
    return limpiar_bureau(bureau).merge(entrenamiento, on="SK_ID_CURR")


def ajustar_tramo_bureau(
    bureau: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el tramo de `BUREAU_ENDDATE_2_5Y_COUNT` con el criterio del EDA.

    El criterio es el pico de la tasa de default fila a fila en `REJILLA_VENCIMIENTO_ANIOS`,
    medido sobre las filas a las que se aplica el corte: créditos a término, limpios y de clientes
    de train. Los extremos del tramo pico pasan a `fijar_operativo()`.

    Los refijados de bureau van fuera del `Pipeline`, igual que `registrar_limites()`, porque la
    agregación que consume sus cortes también va fuera. La consecuencia, declarada: en el CV de la
    Fase 4 el corte elegido sobre todo el 80% se usa en cada fold.

    Devuelve el informe, la tasa y la n por tramo con el pico marcado. Revienta si el pico cae en
    un vencimiento pasado, porque la feature cuenta vencimientos futuros.
    """
    filas = _bureau_de_train(bureau, base, split)
    anios = vencimiento_a_termino(filas) / valor("dias_por_anio")
    informe = (
        filas["TARGET"]
        .groupby(pd.cut(anios, REJILLA_VENCIMIENTO_ANIOS), observed=False)
        .agg(n="size", tasa="mean")
    )
    pico = informe["tasa"].idxmax()
    if pico.left < 0:
        raise ValueError(f"el pico de la tasa cae en {pico}, un vencimiento ya pasado")
    n_train = filas["SK_ID_CURR"].nunique()
    fijar_operativo("bureau_enddate_tramo_min_anios", pico.left, n_train, sobrescribir)
    fijar_operativo("bureau_enddate_tramo_max_anios", pico.right, n_train, sobrescribir)
    logger.info("tramo de vencimiento refijado en %s sobre %s clientes", pico, f"{n_train:,}")
    return informe.assign(pico=informe.index == pico)


def ajustar_cola_bureau(
    bureau: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el corte de `BUREAU_COUNT_COLA`, el primero cuyo delta cruza el umbral.

    Es el criterio de `prev_count_cola`: al subir el corte se gana delta y se pierde cobertura, así
    que se queda el primero que llega a `umbral_flags_pp` y no el de más delta, que acabaría en un
    puñado de clientes. El delta es el de la bandera sobre los clientes de train con historial.

    Devuelve el barrido, marcados y delta por corte con el elegido marcado. Revienta si ningún corte
    cruza el umbral.
    """
    filas = _bureau_de_train(bureau, base, split)
    clientes = filas.groupby("SK_ID_CURR")["TARGET"].agg(n="size", target="first")
    return _primer_corte_que_cruza(
        clientes["n"], clientes["target"], "bureau_count_cola", sobrescribir
    )


def ajustar_cola_bb(
    bb: pd.DataFrame,
    puente: pd.Series,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el corte de `BB_MANY_CREDITS_FLAG` con el criterio de `bureau_count_cola`.

    El conteo es el de `BB_N_CREDITS_WBAL`: créditos distintos del panel con padre en `puente`,
    sobre los clientes de train con histórico. El 22 del EDA era el p99 más uno, y queda marcado en
    el informe como contraste, medido sobre la misma población.

    Va fuera del `Pipeline` por lo mismo que los de bureau, así que en el CV de la Fase 4 cada fold
    usa el corte elegido sobre todo el 80%. Es más inestable que la cola de bureau: en 15 folds sale
    de 15 a 22, con 18 en seis y 17 en otros seis.
    """
    creditos = puente[puente.index.isin(bb["SK_ID_BUREAU"].unique())]
    entrenamiento = solo_train(base, split).set_index("SK_ID_CURR")["TARGET"]
    clientes = entrenamiento.to_frame().join(creditos.value_counts().rename("n"), how="inner")
    informe = _primer_corte_que_cruza(
        clientes["n"], clientes["TARGET"], "bb_many_credits_corte", sobrescribir
    )
    # el primer entero por encima del p99, que es como el EDA llegó al 22 desde un p99 de 21
    return informe.assign(p99_mas_uno=informe.index == int(clientes["n"].quantile(0.99)) + 1)


def _previous_de_train(
    prev: pd.DataFrame, base: pd.DataFrame, split: pd.DataFrame | None
) -> pd.DataFrame:
    """Las filas limpias de previous_application de los clientes de train, con `SK_ID_CURR` y
    `TARGET`.

    Es la población de los refijados de la tabla: el `n_train` que declaran son sus clientes, los
    de train con solicitudes previas.
    """
    entrenamiento = solo_train(base, split)[["SK_ID_CURR", "TARGET"]]
    return limpiar_previous(prev).merge(entrenamiento, on="SK_ID_CURR")


def ajustar_cola_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el corte de `PREV_COUNT_COLA`, el primero cuyo delta cruza el umbral.

    Es el criterio con el que el propio EDA eligió las 15 solicitudes, barriendo 8, 11, 15 y 20.
    Sobre train el barrido entero baja el corte a 11, que en la rejilla del EDA se quedaba a
    +1,91pp y aquí llega a +2,04pp.

    Va fuera del `Pipeline` por lo mismo que los de bureau, así que en el CV de la Fase 4 cada fold
    usa el corte elegido sobre todo el 80%. En 15 folds sale 11 en 10 de 15, con 10, 12 y 13 en los
    otros cinco; midiendo sobre la parte de validación, que son cinco veces menos clientes, se abre
    de 9 a 25.
    """
    filas = _previous_de_train(prev, base, split)
    clientes = filas.groupby("SK_ID_CURR")["TARGET"].agg(n="size", target="first")
    return _primer_corte_que_cruza(
        clientes["n"], clientes["target"], "prev_count_cola", sobrescribir
    )


def ajustar_actividad_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el corte de `PREV_ACTIVIDAD_12M_COLA` con el mismo criterio de la cola.

    El EDA registró el 4 de su rejilla descriptiva sin declarar criterio, y su propio barrido ya
    cruzaba en 3 (+2,01pp). Aquí se unifica con las otras tres colas del proyecto y el corte se
    sostiene: sobre train el 3 se queda en +1,99pp y el primero que cruza vuelve a ser el 4.

    **El barrido empieza en 1 y no en 2** porque aquí el cero es un nivel real, el 42,08% de los
    clientes de train con previas. Marcar a quien pide alguna vez en el año no cruza (+1,31pp),
    pero eso hay que medirlo, no suponerlo.

    **Es la más frágil de las cuatro colas del proyecto**, y por eso va escrito: el 3 se queda a
    0,014pp del umbral, así que en 15 folds el corte sale 4 en 10 y 3 en 5, y midiendo sobre la
    parte de validación el 3 gana en 9 de 15. Lo que el corte separa no se mueve; lo que se mueve
    es de qué lado del umbral cae el 3.
    """
    filas = _previous_de_train(prev, base, split)
    reciente = solicitudes_recientes(filas, valor("prev_ventana_reciente_dias"))
    clientes = (
        filas.assign(_reciente=reciente)
        .groupby("SK_ID_CURR")
        .agg(n=("_reciente", "sum"), target=("TARGET", "first"))
    )
    return _primer_corte_que_cruza(
        clientes["n"], clientes["target"], "prev_actividad_12m_cola", sobrescribir, desde=1
    )


def ajustar_sobreconcesion_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train el corte de `PREV_OVERGRANTED_RATIO`, el de mayor r_rb de la proporción.

    Se barre `REJILLA_SOBRECONCESION` sobre la proporción por cliente de solicitudes que superan el
    corte, que es la codificación que entra en la matriz. Sobre la bandera el delta crece con el
    corte (+1,65pp a +9,28pp en train) porque marca a cada vez menos gente, y elegiría el 1,5. Y no
    es un primer cruce como en las colas: aquí el efecto tiene pico y se elige el máximo.

    La población es la de la feature: los clientes de train con alguna solicitud con las dos cifras
    positivas, y ese es el `n_train` que se declara. Sale 1,1 con r_rb 0,1208, y el borde queda
    fuera como en la agregación. Con una rejilla más fina el pico es plano entre 1,1 y 1,125 (0,1208
    y 0,1200), así que la rejilla del EDA decide el valor, como en el tramo de bureau.

    Es el más estable de los seis: sale 1,1 en los 15 folds, tanto midiendo sobre la parte de
    entrenamiento como sobre la de validación, y en el CV de la Fase 4 cada fold usa el elegido
    sobre todo el 80%, como los de bureau.

    Devuelve el barrido, n, r_rb y delta de la bandera por corte con el elegido marcado.
    """
    filas = _previous_de_train(prev, base, split)
    concesion = cociente_de_concesion(filas)
    objetivo = filas.groupby("SK_ID_CURR")["TARGET"].first()
    barrido = []
    for corte in REJILLA_SOBRECONCESION:
        proporcion = (
            concesion.gt(corte)
            .astype(float)
            .where(concesion.notna())
            .groupby(filas["SK_ID_CURR"])
            .mean()
            .dropna()
        )
        target = objetivo.loc[proporcion.index]
        r_rb, _ = _r_rb(pd.DataFrame({"valor": proporcion, "TARGET": target}))
        marcados = proporcion > 0
        delta = (target[marcados].mean() - target[~marcados].mean()) * 100
        barrido.append((corte, len(proporcion), r_rb, delta))
    informe = pd.DataFrame(barrido, columns=["corte", "n", "r_rb", "delta_bandera_pp"])
    informe = informe.set_index("corte")
    elegido = informe["r_rb"].idxmax()
    fijar_operativo("prev_sobreconcesion_corte", elegido, int(informe["n"].iloc[0]), sobrescribir)
    logger.info(
        "sobreconcesion refijada en %s sobre %s clientes", elegido, f"{informe['n'].iloc[0]:,}"
    )
    return informe.assign(elegido=informe.index == elegido)


def ajustar_finalidades_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Refija sobre train la lista de `PREV_URGENT_PURPOSE_RATIO`: las finalidades que destacan.

    Entra la finalidad declarada con al menos `n_min_categoria` solicitudes de train y una tasa de
    default que supera en `umbral_flags_pp` la global de las declaradas. Reutiliza los dos cortes
    ya declarados y no fija ningún k: el EDA tomó las cinco primeras por tasa, y esa lista es la
    misma con la que midió el efecto, así que no es evidencia independiente. La tasa es por
    solicitud, como la celda 92, y la global es la de todas las declaradas.

    Sobre train salen tres, Gasification, Car repairs y Payments on other loans. **`Urgent needs`
    queda fuera**, con +1,85pp en train y +1,93pp en el crudo, y con ella `Building a house or an
    annex`: la lista del EDA tenía cinco. Decidido con el usuario al ver la cifra. La regla es
    más estricta que la del EDA a propósito, y el 4.11 remide el efecto con la lista refijada.

    **Es la lista menos estable de las seis.** En 15 folds sobre la parte de entrenamiento, Car
    repairs entra en 15, Gasification y Payments en 14 y Urgent needs en 6; la lista exacta se
    repite en 5 de 15. Midiendo sobre la parte de validación, con cinco veces menos clientes, no se
    repite ninguna. Va fuera del `Pipeline`, así que en el CV cada fold usa la elegida sobre todo
    el 80%.

    El `n_train` son los clientes de train con alguna solicitud con finalidad declarada. La tupla
    sale ordenada, para que no dependa del orden de las filas ni de un empate en la tasa.

    Devuelve el barrido: n, tasa y delta sobre la global por finalidad, con las elegidas marcadas.
    Revienta si ninguna cruza.
    """
    filas = _previous_de_train(prev, base, split)
    declaradas = filas[finalidad_declarada(filas)]
    global_ = declaradas["TARGET"].mean()
    por_finalidad = declaradas.groupby("NAME_CASH_LOAN_PURPOSE", observed=True)["TARGET"]
    informe = por_finalidad.agg(n="size", tasa="mean")
    informe["delta_pp"] = (informe["tasa"] - global_) * 100
    informe["tasa"] *= 100
    informe["elegida"] = informe["n"].ge(valor("n_min_categoria")) & informe["delta_pp"].ge(
        valor("umbral_flags_pp")
    )
    if not informe["elegida"].any():
        raise ValueError("ninguna finalidad cruza el umbral de banderas con la n mínima")
    elegidas = tuple(sorted(informe.index[informe["elegida"]]))
    n_train = declaradas["SK_ID_CURR"].nunique()
    fijar_operativo("prev_finalidades_urgentes", elegidas, n_train, sobrescribir)
    logger.info("finalidades urgentes refijadas en %s sobre %s clientes", elegidas, f"{n_train:,}")
    return informe.sort_values("tasa", ascending=False)


def refijar_cortes_auxiliares(
    bureau: pd.DataFrame,
    bb: pd.DataFrame,
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> dict[str, pd.DataFrame]:
    """Corre sobre train los siete refijados de las auxiliares y devuelve sus informes por nombre.

    Las tres tablas van crudas, cada refijado limpia lo suyo, y el puente sale del mismo `bureau`.
    Fija los ocho de `CORTES_AUXILIARES`, que `guardar_cortes()` persiste para la API.
    """
    puente = puente_credito_cliente(bureau)
    return {
        "tramo_bureau": ajustar_tramo_bureau(bureau, base, split, sobrescribir),
        "cola_bureau": ajustar_cola_bureau(bureau, base, split, sobrescribir),
        "cola_bb": ajustar_cola_bb(bb, puente, base, split, sobrescribir),
        "cola_previous": ajustar_cola_previous(prev, base, split, sobrescribir),
        "actividad_previous": ajustar_actividad_previous(prev, base, split, sobrescribir),
        "sobreconcesion_previous": ajustar_sobreconcesion_previous(prev, base, split, sobrescribir),
        "finalidades_previous": ajustar_finalidades_previous(prev, base, split, sobrescribir),
    }


def huella_split(split: pd.DataFrame) -> str:
    """Sha256 del contenido de la partición: `SK_ID_CURR` y `split`, ordenados por cliente.

    Del contenido y no de los bytes del parquet, que cambian con la versión de pyarrow sin que
    cambie la partición. Sirve para que `cargar_cortes()` detecte un split distinto del que midió
    los cortes, no para reproducir el fichero.
    """
    contenido = split[["SK_ID_CURR", "split"]].sort_values("SK_ID_CURR").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(contenido, index=False).values).hexdigest()


def _destino_cortes() -> Path:
    """Ruta del fichero de cortes, como `split._destino()` con la suya."""
    return ruta("processed_data") / NOMBRE_FICHERO_CORTES


def guardar_cortes(
    split: pd.DataFrame | None = None, destino: Path | None = None, sobrescribir: bool = False
) -> Path:
    """Persiste los ocho cortes de `CORTES_AUXILIARES` en `cortes.json`, atados a la huella del
    split.

    Revienta si alguno todavía no está fijado, antes de escribir nada: no deja un fichero a medias.
    Como `construir_split()`, no pisa un fichero existente sin `sobrescribir=True`.
    """
    ruta_destino = destino or _destino_cortes()
    if ruta_destino.exists() and not sobrescribir:
        raise FileExistsError(
            f"ya hay cortes guardados en {ruta_destino}. Sobrescribirlos deja inválido en "
            "silencio todo lo agregado con los anteriores. Pasa sobrescribir=True si de verdad "
            "quieres reemplazarlos."
        )
    faltan = [n for n in CORTES_AUXILIARES if parametro(n).valor_operativo is None]
    if faltan:
        raise ValueError(
            f"cortes sin fijar, no se guarda nada: {faltan}. Córrelos con "
            "refijar_cortes_auxiliares() antes de guardar."
        )
    cortes = {}
    for nombre in CORTES_AUXILIARES:
        p = parametro(nombre)
        if isinstance(p.valor_operativo, tuple):
            valor_json = list(p.valor_operativo)
        elif isinstance(p.valor_operativo, (int, np.integer)):
            # entero de verdad, no un float que resulte entero (como el tramo o la sobreconcesión):
            # se guarda sin ".0" para que valor() devuelva el mismo tipo tras cargar_cortes()
            valor_json = int(p.valor_operativo)
        else:
            valor_json = float(p.valor_operativo)
        cortes[nombre] = {"valor": valor_json, "n_train": int(p.n_train_operativo)}
    split_efectivo = split if split is not None else cargar_split()
    contenido = {"huella_split": huella_split(split_efectivo), "cortes": cortes}
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    ruta_destino.write_text(json.dumps(contenido, indent=2, ensure_ascii=False, sort_keys=True))
    logger.info("cortes guardados en %s", ruta_destino)
    return ruta_destino


def cargar_cortes(
    split: pd.DataFrame | None = None, origen: Path | None = None, sobrescribir: bool = False
) -> None:
    """Lee `cortes.json` y fija los ocho cortes con `fijar_operativo()`.

    Revienta si la huella del split no casa con la que guardó el fichero: es lo que impide que la
    API, o un proceso nuevo, aplique cortes medidos sobre una partición distinta. Revienta también
    si el fichero no trae exactamente los ocho de `CORTES_AUXILIARES`, ni de más ni de menos.
    """
    ruta_origen = origen or _destino_cortes()
    if not ruta_origen.exists():
        raise FileNotFoundError(
            f"no hay cortes en {ruta_origen}. Créalos con refijar_cortes_auxiliares() y "
            "guardar_cortes() sobre el split de entrenamiento."
        )
    contenido = json.loads(ruta_origen.read_text())
    split_efectivo = split if split is not None else cargar_split()
    huella_actual = huella_split(split_efectivo)
    if contenido["huella_split"] != huella_actual:
        raise ValueError(
            f"la huella de {ruta_origen} no casa con la del split actual: los cortes se "
            "midieron sobre otra partición y aplicarlos aquí sería una fuga."
        )
    presentes = set(contenido["cortes"])
    esperados = set(CORTES_AUXILIARES)
    if presentes != esperados:
        raise ValueError(
            f"cortes.json no trae exactamente los ocho esperados. "
            f"faltan: {sorted(esperados - presentes)}, de más: {sorted(presentes - esperados)}"
        )
    for nombre in CORTES_AUXILIARES:
        entrada = contenido["cortes"][nombre]
        valor_nuevo = tuple(entrada["valor"]) if isinstance(entrada["valor"], list) else entrada[
            "valor"
        ]
        fijar_operativo(nombre, valor_nuevo, entrada["n_train"], sobrescribir)
    logger.info("cortes cargados desde %s", ruta_origen)


def informe_denominador_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    cortes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Barre el mínimo de denominador de las seis proporciones de conteos de previous_application.

    Cada proporción se cuenta sobre su propio denominador y no sobre el de solicitudes, y lo que
    decide es la señal **dentro del grupo que el mínimo deja fuera**: exigir denominador sube
    siempre el efecto, porque selecciona población. Solo informa, no fija nada: sobre train ninguna
    de las seis lleva mínimo, y las tres medias de importes quedan fuera por el criterio del 2.3.

    Una fila por proporción y mínimo, sobre los clientes de train con la proporción construida:
    n y r_rb de los que quedan, n de los que se quedan fuera y su r_rb y su p. Cuando el grupo
    excluido solo vale 0 o 1, que con mínimo 2 es siempre, añade su delta de bandera en pp. Los
    `cortes` son los de la agregación, que la sobreconcesión y las finalidades urgentes necesitan.
    """
    filas = _previous_de_train(prev, base, split)
    todas = pd.Series(True, index=filas.index)
    denominador = pd.DataFrame(
        {
            feature: mascara.groupby(filas["SK_ID_CURR"]).sum()
            for feature, mascara in {
                "PREV_REFUSED_RATIO": todas,
                "PREV_OVERGRANTED_RATIO": cociente_de_concesion(filas).notna(),
                "PREV_STREET_RATIO": combinacion_definida(filas),
                "PREV_NO_SUITE_RATIO": todas,
                "PREV_EARLY_HOUR_RATIO": todas,
                "PREV_URGENT_PURPOSE_RATIO": finalidad_declarada(filas),
            }.items()
        }
    )
    proporciones = agregar_previous(prev, cortes)[list(denominador.columns)]
    target = filas.groupby("SK_ID_CURR")["TARGET"].first()
    informe = []
    for feature in denominador:
        clientes = target.to_frame().join(proporciones[feature].rename("valor"), how="inner")
        clientes = clientes.join(denominador[feature].rename("den"))[lambda d: d["valor"].notna()]
        for minimo in MINIMOS_DENOMINADOR:
            dentro, fuera = clientes[clientes["den"] >= minimo], clientes[clientes["den"] < minimo]
            r_dentro, _ = _r_rb(dentro)
            r_fuera, p_fuera = _r_rb(fuera)
            informe.append(
                {
                    "feature": feature,
                    "minimo": minimo,
                    "n": len(dentro),
                    "r_rb": r_dentro,
                    "n_fuera": len(fuera),
                    "r_rb_fuera": r_fuera,
                    "p_fuera": p_fuera,
                    "delta_fuera_pp": _delta_bandera(fuera),
                }
            )
    return pd.DataFrame(informe).set_index(["feature", "minimo"])


def lecturas_relativas_previous(
    filas: pd.DataFrame, reciente: pd.Series, hora_max: float
) -> pd.DataFrame:
    """Las seis lecturas de recencia relativa del 4.10, una columna por cliente.

    `reciente` es la máscara de la ventana. El informe se la pasa relativa al fin de ventana de
    cada cliente; con la absoluta, el conteo y las tres de captación tienen que dar exactamente
    las columnas que construye `agregar_previous()`, y eso lo fija un test, que es lo que impide
    que la relativa y la absoluta se separen por una diferencia de escritura.

    Ninguna se guarda en la matriz: el 4.10 es exploratorio y solo construye lo que gane.
    """
    cliente = filas["SK_ID_CURR"]
    fin = fin_de_ventana(filas)
    rechazada = filas["NAME_CONTRACT_STATUS"].eq("Refused")
    # alguna de las del día más reciente, no la primera fila de ese día: el orden no decide, como
    # en PREV_HISTORIAL_RECORTADO en el otro extremo del historial
    ultima = filas["DAYS_DECISION"].eq(fin)
    calle = (
        filas["PRODUCT_COMBINATION"]
        .str.contains(CAPTACION_CALLE, na=False)
        .astype(float)
        .where(combinacion_definida(filas))
    )
    temprana = filas["HOUR_APPR_PROCESS_START"].le(hora_max).astype(float)
    sin_acompanante = filas["NAME_TYPE_SUITE"].isna().astype(float)
    return pd.DataFrame(
        {
            "PREV_EXIT_REFUSED_FLAG": (rechazada & ultima).groupby(cliente).max().astype(float),
            # el último rechazo contra el fin de ventana del cliente: 0 es salir en rechazo
            "PREV_DAYS_SINCE_REFUSED_REL": (filas["DAYS_DECISION"] - fin)
            .where(rechazada)
            .groupby(cliente)
            .max(),
            "PREV_COUNT_12M_REL": reciente.groupby(cliente).sum().astype(float),
            "PREV_STREET_RATIO_REL": calle.where(reciente).groupby(cliente).mean(),
            "PREV_EARLY_HOUR_RATIO_REL": temprana.where(reciente).groupby(cliente).mean(),
            "PREV_NO_SUITE_RATIO_REL": sin_acompanante.where(reciente).groupby(cliente).mean(),
        }
    )


def informe_recencia_relativa_previous(
    prev: pd.DataFrame,
    base: pd.DataFrame,
    split: pd.DataFrame | None = None,
    cortes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Mide las seis lecturas de recencia relativa del 4.10 contra sus contrapartes absolutas.

    El pendiente 6 de la auditoría transversal: en `bureau_balance` la recencia contra el fin de
    ventana propio ganó a la absoluta, y aquí el fin de ventana de cada cliente es su última
    solicitud. Solo informa, no fija nada y no construye ninguna columna de la matriz.

    Una fila por lectura, contraparte y estrato de recencia, con el estrato `todos` como fila
    global. En las banderas el efecto es el delta en pp y en las continuas el rank-biserial en
    valor absoluto, los dos sobre la misma población: los clientes de train donde la lectura
    relativa está definida. `n_solo_abs` son los que pierde frente a su contraparte, que es la
    cobertura que cuesta relativizar. La `p` sale de Mann-Whitney también en las banderas, como en
    el grupo excluido de `informe_denominador_previous()`.

    La redundancia va solo en la fila global, porque por estrato es ruido: Pearson entre las dos
    continuas, y entre dos banderas el mismo número es la V de Cramér de su tabla 2x2.

    La familia de Bonferroni son los contrastes que emite, dos por fila.
    """
    filas = _previous_de_train(prev, base, split)
    hora_max = valor("prev_hora_temprana_max")
    relativa = lecturas_relativas_previous(
        filas,
        solicitudes_recientes(filas, valor("prev_ventana_reciente_dias"), fin_de_ventana(filas)),
        hora_max,
    )
    rechazada = filas["NAME_CONTRACT_STATUS"].eq("Refused")
    agregado = agregar_previous(prev, cortes).reindex(relativa.index)
    absoluta = agregado.assign(
        **{
            "PREV_REFUSED_RATIO > 0": agregado["PREV_REFUSED_RATIO"].gt(0).astype(float),
            "PREV_DAYS_SINCE_REFUSED": filas["DAYS_DECISION"]
            .where(rechazada)
            .groupby(filas["SK_ID_CURR"])
            .max(),
        }
    )
    target = filas.groupby("SK_ID_CURR")["TARGET"].first().reindex(relativa.index)
    recencia = agregado["PREV_DAYS_DECISION_MAX"]
    estratos = {"todos": (-np.inf, np.inf), **ESTRATOS_RECENCIA_PREVIOUS}
    informe = []
    for lectura, contrapartes in LECTURAS_RECENCIA_RELATIVA.items():
        flag = lectura.endswith("_FLAG")
        for contraparte in contrapartes:
            datos = pd.DataFrame(
                {
                    "valor": relativa[lectura],
                    "abs": absoluta[contraparte],
                    "TARGET": target,
                    "recencia": recencia,
                }
            )
            for estrato, (desde, hasta) in estratos.items():
                dentro = datos[datos["recencia"].gt(desde) & datos["recencia"].le(hasta)]
                d = dentro[dentro["valor"].notna()]
                contra = d.assign(valor=d["abs"]).dropna(subset=["valor"])
                efecto, p = _r_rb(d)
                efecto_abs, p_abs = _r_rb(contra)
                informe.append(
                    {
                        "lectura": lectura,
                        "contraparte": contraparte,
                        "estrato": estrato,
                        "tipo": "flag" if flag else "continua",
                        "n": len(d),
                        "n_marcados": int(d["valor"].sum()) if flag else np.nan,
                        "n_solo_abs": int(
                            (dentro["valor"].isna() & dentro["abs"].notna()).sum()
                        ),
                        "efecto": _delta_bandera(d) if flag else efecto,
                        "p": p,
                        "efecto_abs": _delta_bandera(contra) if flag else efecto_abs,
                        "p_abs": p_abs,
                        "redundancia": _redundancia(d) if estrato == "todos" else np.nan,
                    }
                )
    return pd.DataFrame(informe).set_index(["lectura", "contraparte", "estrato"])


def _redundancia(d: pd.DataFrame) -> float:
    """Pearson en valor absoluto entre la lectura y su contraparte; NaN si alguna es constante."""
    if d["valor"].nunique() < 2 or d["abs"].nunique() < 2:
        return np.nan
    return abs(d["valor"].corr(d["abs"]))


def _r_rb(clientes: pd.DataFrame) -> tuple[float, float]:
    """Rank-biserial en valor absoluto y p de una proporción contra el TARGET; NaN si no separa."""
    sanos = clientes.loc[clientes["TARGET"] == 0, "valor"]
    morosos = clientes.loc[clientes["TARGET"] == 1, "valor"]
    if sanos.empty or morosos.empty or clientes["valor"].nunique() < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(sanos, morosos)
    return abs(2 * u / (len(sanos) * len(morosos)) - 1), p


def _delta_bandera(clientes: pd.DataFrame) -> float:
    """Tasa de default con la proporción a 1 menos con ella a 0, en pp; NaN si no es solo 0 o 1."""
    tasa = clientes.groupby("valor")["TARGET"].mean()
    return (tasa[1] - tasa[0]) * 100 if set(tasa.index) == {0, 1} else np.nan


def _primer_corte_que_cruza(
    conteo: pd.Series, target: pd.Series, nombre: str, sobrescribir: bool, desde: int = 2
) -> pd.DataFrame:
    """Barre los cortes de un conteo por cliente y fija el primero cuyo delta cruza el umbral.

    `desde` es 2 salvo en un conteo donde el cero es un nivel real: con el mínimo del conteo el
    grupo sin marcar queda vacío y el delta no existe.
    """
    barrido = []
    for corte in range(desde, conteo.max() + 1):
        cola = conteo.ge(corte)
        delta = target[cola].mean() - target[~cola].mean()
        barrido.append((corte, int(cola.sum()), delta * 100))
    informe = pd.DataFrame(barrido, columns=["corte", "marcados", "delta_pp"]).set_index("corte")
    cruzan = informe.index[informe["delta_pp"] >= valor("umbral_flags_pp")]
    if cruzan.empty:
        raise ValueError("ningún corte del conteo cruza el umbral de banderas")
    fijar_operativo(nombre, int(cruzan[0]), len(conteo), sobrescribir)
    logger.info("%s refijado en %s sobre %s clientes", nombre, cruzan[0], f"{len(conteo):,}")
    return informe.assign(elegido=informe.index == cruzan[0])


def ajustar_pipeline(
    base: pd.DataFrame, split: pd.DataFrame | None = None
) -> tuple[Pipeline, pd.DataFrame]:
    """Ajusta el pipeline entero de las capas 2 sobre el 80% de entrenamiento.

    Es **la única forma sancionada de ajustarlo**, y existe por eso. `construir_pipeline()`
    devuelve el objeto sin ajustar, así que cualquiera podía hacerle `fit` con la tabla entera y
    nada lo distinguía: sobre el dato real los diez cortes de la capa 2a salen idénticos con
    partición y sin ella, o sea que la cifra no delata la fuga. Teniendo un solo camino, la
    guarda se pone una vez y se prueba una vez.

    El orden de las dos operaciones es el mismo que en `ajustar_capa2a()`: primero se filtra la
    partición, que necesita el identificador, y después se quita.

    Devuelve `(pipeline, matriz)`, igual que `ajustar_capa2a()` devuelve `(winsorizador,
    informe)`, y **la matriz sale de `fit_transform` a propósito**. Es la que hay que darle al
    modelo en la Fase 4: transformar el entrenamiento con el pipeline ya ajustado le da a cada
    fila la media de su propia categoría calculada **con ella dentro**, que es fuga de etiqueta
    en la matriz con la que se entrena. Con la codificación cruzada son 95 valores distintos
    fuera de fold frente a los 19 niveles de dentro, medido sobre train. Devolver las dos cosas
    es lo que hace que nadie tenga que acordarse: sobre validación sí se usa `transform`, que
    ahí es lo correcto.

    **Eso vale para la ocupación y no para la organización.** El `WoEEncoder` no cruza folds, así
    que su columna de esta misma matriz sigue siendo de dentro de fold. Está declarado en su
    docstring con la magnitud medida, y decidirlo es de la Fase 4.
    """
    entrenamiento = solo_train(base, split)
    cfg = cargar_config()["dataset"]
    pipeline = construir_pipeline()
    matriz = pipeline.fit_transform(
        matriz_de_features(entrenamiento), entrenamiento[cfg["target_col"]]
    )
    logger.info("capas 2 ajustadas sobre %s filas de entrenamiento", f"{len(entrenamiento):,}")
    return pipeline, matriz


def informe_base(
    app_cruda: pd.DataFrame, limpio: pd.DataFrame, con_features: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Puerta de salida de la capa 1: qué entra, qué sale y cuánto se mueve la tasa.

    Con `con_features` añade el recuento de columnas de la segunda mitad, que si no hay que
    contar a mano y acaba citado de memoria: las cifras del cierre de este punto se escribieron
    una vez desde un commit anterior, cuando dos descartes pasaron a provisionales y la limpieza
    dejó de llevárselos por delante.
    """
    fuera = filas_a_eliminar(app_cruda)
    columnas = {"medida": "columnas", "cruda": app_cruda.shape[1], "limpia": limpio.shape[1]}
    filas_extra = []
    if con_features is not None:
        columnas["con features"] = con_features.shape[1]
        # sin cruda ni limpia: en esas dos columnas la medida no existe, y un 0 se leería
        # como que se midió y salió cero
        filas_extra.append(
            {
                "medida": "features de capa 1",
                "con features": con_features.shape[1] - limpio.shape[1],
            }
        )
    return pd.DataFrame(
        [
            {"medida": "filas", "cruda": len(app_cruda), "limpia": len(limpio)},
            columnas,
            *filas_extra,
            {
                "medida": "filas eliminadas (netas)",
                "cruda": int(fuera.sum()),
                "limpia": len(app_cruda) - len(limpio),
            },
            {
                "medida": "positivos",
                "cruda": int(app_cruda["TARGET"].sum()),
                "limpia": int(limpio["TARGET"].sum()),
            },
            {
                "medida": "% default",
                "cruda": round(app_cruda["TARGET"].mean() * 100, 4),
                "limpia": round(limpio["TARGET"].mean() * 100, 4),
            },
            {
                "medida": "centinela marcado",
                "cruda": int(app_cruda["DAYS_EMPLOYED"].eq(365243).sum()),
                "limpia": int(limpio["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()),
            },
        ]
    )
