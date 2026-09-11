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

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import cargar_config
from src.data.loader import load_table
from src.features.agg_bureau import vencimiento_a_termino
from src.features.application import construir_features_capa1, verificar_contrato_capa1
from src.features.cleaning import (
    filas_a_eliminar,
    limpiar_application,
    limpiar_application_entrenamiento,
    limpiar_bureau,
)
from src.features.params import fijar_operativo, valor
from src.features.pipeline import construir_pipeline
from src.features.split import construir_split, solo_train
from src.features.transformers import Winsorizador, registrar_limites

logger = logging.getLogger(__name__)

# reduce_mem_usage no se aplica a la tabla principal: baja los float a float32 y el EDA ya
# documentó que eso convierte en desigualdad estricta 453 comparaciones de fechas que en
# realidad son iguales. Son 307.511 filas, la memoria no es el problema aquí.
REDUCIR_MEMORIA = False

# La rejilla con la que el EDA vio la U de DAYS_CREDIT_ENDDATE (notebook 02, celda 58), en años con
# signo. El tramo sale de ella y no de una más fina: sobre train, la fina sube el pico hasta 4 a 7
# años y un barrido por delta a nivel cliente lo lleva a 3 a 5, así que el valor lo decidiría la
# rejilla y no el dato.
REJILLA_VENCIMIENTO_ANIOS = (-np.inf, -5, -2, 0, 2, 5, 10, np.inf)


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
    barrido = []
    for corte in range(2, clientes["n"].max() + 1):
        cola = clientes["n"].ge(corte)
        delta = clientes["target"][cola].mean() - clientes["target"][~cola].mean()
        barrido.append((corte, int(cola.sum()), delta * 100))
    informe = pd.DataFrame(barrido, columns=["corte", "marcados", "delta_pp"]).set_index("corte")
    cruzan = informe.index[informe["delta_pp"] >= valor("umbral_flags_pp")]
    if cruzan.empty:
        raise ValueError("ningún corte del conteo cruza el umbral de banderas")
    fijar_operativo("bureau_count_cola", int(cruzan[0]), len(clientes), sobrescribir)
    logger.info("cola del conteo refijada en %s sobre %s clientes", cruzan[0], f"{len(clientes):,}")
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
