"""El `Pipeline` de las capas 2 montado, con su `ColumnTransformer` dentro.

Convierte 101 columnas heterogéneas en una matriz numérica de 139 con nombres. Se ajusta **solo
sobre el 80% de entrenamiento**: dos de sus pasos estiman parámetros sobre covariables y tres los
estiman con el TARGET.

**Ojo con ese 101, que es otro.** La capa 1 deja 101 columnas contando `TARGET` y `SK_ID_CURR`, o
sea 99 features, y lo que ve el `ColumnTransformer` son esas 99 más los dos ratios que añade el
winsorizador. Los dos números coinciden porque se quitan dos y se añaden dos, y no porque sean el
mismo conjunto.

El orden de los pasos no es libre y lo fija `sklearn.md`:

1. `winsor`, que recorta los extremos.
2. `derivadas`, los dos ratios que llevan una columna winsorizada en el denominador. Antes del
   winsorizador dejarían el error de captura de 117M de ingreso dentro del divisor de la carga.
3. `dominio`, las recodificaciones deterministas de dominio.
4. `contrato`, la guarda de que los buckets cubren el frame exactamente. Va aquí porque es el
   único punto donde existe lo que el `ColumnTransformer` va a ver, con los dos ratios ya
   construidos y la hora ya convertida en franja.
5. `columnas`, el `ColumnTransformer` que reparte cada bucket a su codificación.
6. `varianza`, la red contra la columna que se quede constante.

El `SelectorIV` que el boceto de `sklearn.md` dibuja al final **no está**: `iv.py` es del bloque 5
y este pipeline se cierra en el `VarianceThreshold`.

**Pendiente declarado, para que no viva solo en el docstring que lo comete:** `aplicar_dominio()`
es capa 1 por la regla de la fase, porque no estima nada y no cruza filas, y sin embargo vive
aquí. Está así para no reabrir el contrato de esquema del punto 1.2 ni mover sus cifras de puerta
ya cerradas. Cuando se toque ese contrato, su sitio es `application.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder

from src.features.application import COLUMNAS_EDIFICIO
from src.features.params import valor
from src.features.transformers import (
    AgrupadorDeRaras,
    RatiosPosteriores,
    Winsorizador,
    WoEEncoder,
)

# Las 48 continuas. `HOUR_APPR_PROCESS_START` no está: el paso de dominio la convierte en franja
# y se va al bucket de OHE, así que entra en la matriz como categoría y no como número.
NUMERICAS: tuple[str, ...] = (
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "BUILDING_INFO_COUNT",
    "AGE_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
    "LTV",
    "ANNUITY_TO_INCOME_RATIO",
    "CHILDREN_TO_FAM_RATIO",
)

# Las 36 que ya son 0/1 y pasan tal cual. Ninguna trae nulos, así que no necesitan imputación:
# las de documento y las geográficas vienen así del origen, y las cinco de capa 1 se construyen
# con `notna()`, que no puede producir uno.
BINARIAS: tuple[str, ...] = (
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "FLAG_DAYS_EMPLOYED_ANOMALY",
    "HAS_BUILDING_INFO",
    "HAS_BUREAU_INFO",
    "HAS_SOCIAL_INFO",
    "FLAG_EXT_SOURCE_1_NULL",
    "FLAG_EXT_SOURCE_3_NULL",
)

# De dónde se recupera la ausencia de cada numérica que la imputación rellena. El plan de la
# fase solo sancionaba cuatro columnas (los dos scores externos, el precio del bien y la
# antigüedad laboral) y aquí se imputan las 48, con 32 que traen algún nulo y 14 de ellas por
# encima del 50%. Ampliarlo es inevitable, porque la matriz no puede salir con nulos, pero deja
# de ser gratis: rellenar sin rastro borra la diferencia entre "vale la mediana" y "no se sabe".
# Por eso se declara de dónde se recupera cada una, y un test lo comprueba contra la tabla real.
#
# Las trece cuya bandera reproduce el patrón de nulos exacto, o sea sin pérdida ninguna.
PRESENCIA_POR_BANDERA: dict[str, str] = {
    "DAYS_EMPLOYED": "FLAG_DAYS_EMPLOYED_ANOMALY",
    "EMPLOYED_TO_AGE_RATIO": "FLAG_DAYS_EMPLOYED_ANOMALY",
    "EXT_SOURCE_1": "FLAG_EXT_SOURCE_1_NULL",
    "EXT_SOURCE_3": "FLAG_EXT_SOURCE_3_NULL",
    "OBS_30_CNT_SOCIAL_CIRCLE": "HAS_SOCIAL_INFO",
    "DEF_30_CNT_SOCIAL_CIRCLE": "HAS_SOCIAL_INFO",
    "DEF_60_CNT_SOCIAL_CIRCLE": "HAS_SOCIAL_INFO",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "HAS_BUREAU_INFO",
    "AMT_REQ_CREDIT_BUREAU_DAY": "HAS_BUREAU_INFO",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "HAS_BUREAU_INFO",
    "AMT_REQ_CREDIT_BUREAU_MON": "HAS_BUREAU_INFO",
    "AMT_REQ_CREDIT_BUREAU_QRT": "HAS_BUREAU_INFO",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "HAS_BUREAU_INFO",
}

# La antigüedad del coche va aparte porque su bandera **no** es exacta: 4 clientes de los 245.993
# de entrenamiento declaran coche y no declaran su antigüedad, que es el mismo residuo diminuto
# que el EDA ya había visto al clasificarla. A esos cuatro la mediana les borra el dato y nada lo
# señala; al resto los recupera `FLAG_OWN_CAR`, que va por el bucket de OHE y no por el binario.
PRESENCIA_CASI_EXACTA: dict[str, str] = {"OWN_CAR_AGE": "FLAG_OWN_CAR"}
N_COCHE_SIN_EDAD = 4

# Las quince del bloque edificio se recuperan **solo en agregado**: `HAS_BUILDING_INFO` reproduce
# exacto el grupo sin ni un dato y `BUILDING_INFO_COUNT` cuenta cuántos hay, pero ninguna de las
# dos dice cuáles. Es la pérdida consciente de este bucket, y son las que más nulos traen (de
# 48,17% en `TOTALAREA_MODE` a 69,80% en `COMMONAREA_AVG`). La alternativa, una bandera por
# columna, son quince columnas más que el IV del bloque 5 tendría que juzgar.
PRESENCIA_POR_BLOQUE: tuple[str, ...] = COLUMNAS_EDIFICIO

# Y las tres que se imputan sin ningún rastro, aceptado por volumen: 232, 232 y 529 clientes,
# o sea el 0,09%, el 0,09% y el 0,22% del entrenamiento.
IMPUTACION_SIN_RASTRO: tuple[str, ...] = ("AMT_GOODS_PRICE", "LTV", "EXT_SOURCE_2")

# Las dos banderas de documento que el EDA decidió conservar pase lo que pase, por su correlación
# con el objetivo sobre la tabla completa (+0,0443 la 3 y -0,0286 la 6; sobre los 245.993 de
# entrenamiento son +0,0446 y -0,0284). El plan de la fase pide dejarlas **fuera** del
# filtro de varianza, y aquí no se implementa esa exclusión: con el suelo a cero solo caen las
# constantes, y ninguna de las dos puede serlo (70,99% y 8,79% de unos sobre train), así que
# montar un desvío para las dos columnas sería maquinaria para un caso que no ocurre.
#
# Lo que sí hay es la alarma: un test comprueba que las dos llegan a la matriz. Si alguien sube
# `app_umbral_varianza`, ese test se pone rojo y obliga a decidir de verdad, en vez de que las dos
# desaparezcan en silencio. La protección de verdad, frente a una selección por señal, es del
# bloque 5 y su `control: true`, que es el mecanismo que las recetas ya tienen para esto.
COLUMNAS_PROTEGIDAS_DE_VARIANZA: tuple[str, ...] = ("FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6")

COL_HORA = "HOUR_APPR_PROCESS_START"
COL_FRANJA = "HORA_FRANJA"
COL_DIA = "WEEKDAY_APPR_PROCESS_START"
COL_EDUCACION = "NAME_EDUCATION_TYPE"
COL_ORGANIZACION = "ORGANIZATION_TYPE"
COL_OCUPACION = "OCCUPATION_TYPE"

# Las 14 de baja cardinalidad, con la franja horaria que fabrica el paso de dominio. Las cuatro
# binarias por categoría (contrato, sexo, coche y vivienda) van aquí con `drop="if_binary"` en vez
# de por cuatro mapeos a mano: sale una columna por cada una igual y no hay diccionarios que
# mantener. Las cuatro del bloque edificio entran con el nulo como nivel más, sin decidir nada:
# el EDA las dejó en "binarizar o eliminar, pendiente de IV" y el IV es del bloque 5.
CATEGORICAS_OHE: tuple[str, ...] = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    COL_DIA,
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    COL_FRANJA,
)

# La jerarquía es de dominio y el EDA solo confirmó que la monotonía la acompaña. Remedida sobre
# los 245.993 de entrenamiento, de menor a mayor nivel: 10,944%, 8,929%, 8,569%, 5,379% y 1,504%.
JERARQUIA_EDUCACION: tuple[str, ...] = (
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
)

# El nivel que se le pone al nulo de la ocupación. El plan de la fase lo llama `Retired/Inactive`
# por el cruce determinista con `NAME_INCOME_TYPE`, pero la propia skill del EDA matiza que ese
# determinismo solo se cumple en Unemployed (100,00%) y Pensioner (99,99%), y que en Working el
# nulo es del 15,70% y se comporta como ruido de captura. La codificación es idéntica con
# cualquiera de los dos nombres, porque es un nivel único para todos los nulos, así que se usa
# uno neutro y la discrepancia queda declarada en vez de resuelta callando.
#
# Va sin guarda de colisión, y es la única de las tres del módulo que no la lleva: el agrupador
# revienta si una categoría real se llama como su residual y `_clave` si se llama como la clave
# del nulo. Aquí no hace falta porque el nivel es una etiqueta en castellano y el vocabulario de
# la columna son 18 oficios en inglés, así que la colisión no puede darse sin que alguien cambie
# antes este nombre. Si se cambia, hay que mirar que el nuevo no exista ya en la columna.
NIVEL_SIN_OCUPACION = "Sin declarar"

# Los nombres de las tres franjas de la hora de solicitud. Las fronteras no están aquí: son
# cortes y viven en `params.py`, que es donde declaran su procedencia.
FRANJA_MANANA, FRANJA_TARDE, FRANJA_FUERA = "manana", "tarde", "fuera de horario"

# El código que el `OrdinalEncoder` le pone al nivel de educación que no vio al ajustar. Queda
# fuera de la escala por abajo, que es la señal correcta, y no mete un nulo en una matriz que ya
# no vuelve a imputarse.
CODIGO_EDUCACION_DESCONOCIDA = -1

# Qué columnas lleva cada bucket. El `ColumnTransformer` las reparte, `columnas_declaradas()`
# las exige y `informe_buckets()` las cuenta, y los tres tienen que decir lo mismo: un bucket
# nuevo declarado en dos de los tres sitios cambia la matriz sin que falle nada. Los nombres son
# los del `ColumnTransformer` y por ahí los cruza `test_el_column_transformer_reparte_lo_declarado`.
REPARTO: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("num", NUMERICAS),
    ("ohe", CATEGORICAS_OHE),
    ("ord", (COL_EDUCACION,)),
    ("woe", (COL_ORGANIZACION,)),
    ("tgt", (COL_OCUPACION,)),
    ("bin", BINARIAS),
)

FIN_DE_SEMANA = ("SATURDAY", "SUNDAY")
DIAS_LABORABLES = ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY")
DIA_LABORABLE, DIA_FIN_DE_SEMANA = "entresemana", "fin de semana"

# El mapa va con los siete días escritos y no con un `else` que mande todo lo demás a
# entresemana. Con el `else`, aplicar la recodificación dos veces convertía "fin de semana" en
# "entresemana", porque en la segunda pasada ya no es SATURDAY: la no idempotencia que borra, que
# es el mismo fallo de la bandera del centinela. Y de paso, un valor que no sea un día no se
# etiqueta como laborable por descarte.
MAPA_DIA = {
    **{dia: DIA_LABORABLE for dia in DIAS_LABORABLES},
    **{dia: DIA_FIN_DE_SEMANA for dia in FIN_DE_SEMANA},
}


def franja_horaria(horas: pd.Series) -> pd.Series:
    """La hora de solicitud reducida a mañana, tarde y fuera de horario.

    Las tres fronteras se leen de `params.py` en cada llamada y no al importar el módulo: así
    quien las cambie ve el efecto sin reimportar, y no queda una copia congelada del valor.
    """
    inicio_manana = valor("app_hora_inicio_manana")
    inicio_tarde = valor("app_hora_inicio_tarde")
    fin_tarde = valor("app_hora_fin_tarde")
    franja = pd.Series(FRANJA_FUERA, index=horas.index, dtype=object)
    franja[(horas >= inicio_manana) & (horas < inicio_tarde)] = FRANJA_MANANA
    franja[(horas >= inicio_tarde) & (horas < fin_tarde)] = FRANJA_TARDE
    return franja.where(horas.notna())


def aplicar_dominio(datos: pd.DataFrame) -> pd.DataFrame:
    """Las recodificaciones deterministas que no estiman nada y no cruzan filas.

    Por la regla de la propia fase son capa 1, y viven aquí y no en `application.py` para no
    reabrir el contrato de esquema del punto 1.2 ni mover sus cifras de puerta ya cerradas (92
    columnas limpias y 101 con features, ese 101 con `TARGET` y `SK_ID_CURR` dentro). Es un sitio
    provisional declarado, no una reclasificación de capa.

    **Sustituye y no añade**, que es lo que deja el conteo cuadrado en 101: la hora sale del
    bucket numérico y entra como franja en el de OHE, y el día de la semana se reescribe en su
    sitio y sigue siendo categórico. Añadir la franja al lado dejaría dos columnas de hora
    correlacionadas casi al máximo, que el control de redundancia del bloque 5 tendría que
    deshacer.

    Permisivo con la columna ausente, igual que la capa 1: quien exige el contrato es
    `verificar_contrato_columnas()`.
    """
    datos = datos.copy()
    if COL_HORA in datos.columns:
        datos[COL_HORA] = franja_horaria(datos[COL_HORA])
        datos = datos.rename(columns={COL_HORA: COL_FRANJA})
    if COL_DIA in datos.columns:
        # `fillna(dias)` devuelve lo que el mapa no reconoce, que es lo que hace la
        # recodificación idempotente: en la segunda pasada "fin de semana" ya no es SATURDAY y
        # sin esto caería a "entresemana"
        datos[COL_DIA] = datos[COL_DIA].map(MAPA_DIA).fillna(datos[COL_DIA])
    if COL_OCUPACION in datos.columns:
        datos[COL_OCUPACION] = datos[COL_OCUPACION].fillna(NIVEL_SIN_OCUPACION)
    return datos


def _nombres_dominio(_: object, input_features: list[str]) -> np.ndarray:
    """Los nombres que salen del paso de dominio: los mismos con la hora renombrada."""
    return np.asarray([COL_FRANJA if c == COL_HORA else c for c in input_features], dtype=object)


def columnas_declaradas() -> tuple[str, ...]:
    """Todas las que el `ColumnTransformer` reparte, o sea el contrato de entrada de la matriz."""
    return tuple(columna for _, columnas in REPARTO for columna in columnas)


def verificar_contrato_columnas(datos: pd.DataFrame) -> None:
    """Revienta si el frame y los buckets no se cubren exactamente, en las dos direcciones.

    Sin esto, `remainder="drop"` se lleva en silencio cualquier columna que no esté declarada:
    la matriz saldría más pobre sin un solo error y sin que cambie ningún nombre. Se llama sobre
    el frame **ya pasado por el paso de dominio**, que es el que ve el `ColumnTransformer`.

    Las dos direcciones no se rompen igual y por eso hay que mirar las dos: la columna que falta
    revienta sola, porque el `ColumnTransformer` la pide por nombre, y la que sobra desaparece
    callando. La que sobra es la que van a traer los bloques 2 a 4, que dejan sus agregaciones
    en el mismo frame de capa 1.
    """
    declaradas = set(columnas_declaradas())
    presentes = set(datos.columns)
    sobran = sorted(presentes - declaradas)
    faltan = sorted(declaradas - presentes)
    if sobran or faltan:
        raise ValueError(
            f"el reparto de buckets no cubre el frame: sobran {sobran}, faltan {faltan}. "
            "Con remainder='drop' las que sobran se caerían de la matriz sin avisar"
        )


def _exigir_contrato(datos: pd.DataFrame) -> pd.DataFrame:
    """Paso de guarda: exige el contrato de buckets y devuelve el frame intacto.

    Va como paso del `Pipeline` y no como llamada dentro de `ajustar_pipeline()`, que era lo
    natural, porque ahí los dos ratios posteriores todavía no existen: los fabrica `derivadas`,
    así que el contrato los daría por ausentes. Este es el único sitio donde el frame que ve el
    `ColumnTransformer` existe de verdad, y de paso cubre también el `transform`.
    """
    verificar_contrato_columnas(datos)
    return datos


def construir_pipeline() -> Pipeline:
    """El pipeline de las capas 2, sin ajustar. Se ajusta con `fit(X, y)` sobre `solo_train()`.

    `set_output(transform="pandas")` no es opcional: por defecto el `ColumnTransformer` devuelve
    un array de numpy y se pierden los nombres de columna, que hacen falta enteros para SHAP en
    la Fase 5, para el scorecard y para el contrato de la API.
    """
    ohe = Pipeline(
        [
            ("agrupa", AgrupadorDeRaras()),
            # `handle_unknown="ignore"` porque la API va a recibir categorías que no estaban en
            # el entrenamiento y no puede reventar por eso. `min_frequency` no sirve aquí: agrupa
            # por frecuencia y quien decide la rareza es el paso de delante, con su umbral.
            (
                "codifica",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    columnas = ColumnTransformer(
        [
            # La mediana sobre las 48, de las que 32 traen algún nulo. De dónde se recupera
            # la ausencia de cada una está declarado arriba, en los cuatro grupos, y un test lo
            # comprueba contra la tabla real. Contrastar la mediana contra dejar el NaN, que es
            # el proviso que el EDA dejó abierto para `DAYS_EMPLOYED`, necesita un modelo y por
            # eso es de la Fase 4.
            ("num", SimpleImputer(strategy="median"), list(NUMERICAS)),
            ("ohe", ohe, list(CATEGORICAS_OHE)),
            (
                "ord",
                OrdinalEncoder(
                    categories=[list(JERARQUIA_EDUCACION)],
                    handle_unknown="use_encoded_value",
                    unknown_value=CODIGO_EDUCACION_DESCONOCIDA,
                ),
                [COL_EDUCACION],
            ),
            ("woe", WoEEncoder(), [COL_ORGANIZACION]),
            # `TargetEncoder` hace codificación cruzada interna, o sea que resuelve por diseño el
            # sobreajuste del target encoding sin escribirlo a mano.
            ("tgt", TargetEncoder(), [COL_OCUPACION]),
            ("bin", "passthrough", list(BINARIAS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("winsor", Winsorizador()),
            ("derivadas", RatiosPosteriores()),
            ("dominio", FunctionTransformer(aplicar_dominio, feature_names_out=_nombres_dominio)),
            ("contrato", FunctionTransformer(_exigir_contrato, feature_names_out="one-to-one")),
            ("columnas", columnas),
            # El suelo va declarado en `params.py`. A cero no elimina ninguna hoy: medido sobre
            # train, la más pobre de las 20 `FLAG_DOCUMENT_*` es la 12, con una sola observación
            # a 1. Está por el fold del CV de la Fase 4 donde esa bandera sí se quede constante y
            # el número de columnas cambie entre folds.
            ("varianza", VarianceThreshold(threshold=valor("app_umbral_varianza"))),
        ]
    ).set_output(transform="pandas")


def informe_buckets(datos: pd.DataFrame, pipeline: Pipeline | None = None) -> pd.DataFrame:
    """Puerta del punto: qué columnas entrega cada bucket y cuántas saca cada uno.

    Con `pipeline` ya ajustado añade el `salen`, que es donde está la única cifra que no se puede
    contar a mano: el OHE entrega 14 columnas y saca 52, y esa expansión es la diferencia entre
    las 101 de entrada y las 139 de la matriz. Sin él, `salen` va a nulo en vez de desaparecer,
    para que el informe no cambie de forma según con qué se le llame.

    `salen` sale de `output_indices_` del `ColumnTransformer` y no de recontar niveles, que sería
    reimplementar lo que el codificador ya sabe. Cuenta lo que emite el reparto de buckets: el
    filtro de varianza va después y podría quitar alguna, aunque con el suelo a cero no quita
    ninguna.
    """
    tras_dominio = aplicar_dominio(datos)
    emitidas = {} if pipeline is None else pipeline.named_steps["columnas"].output_indices_
    return pd.DataFrame(
        [
            {
                "bucket": nombre,
                "entran": len(columnas),
                "presentes": len(set(columnas) & set(tras_dominio.columns)),
                "salen": (
                    emitidas[nombre].stop - emitidas[nombre].start if nombre in emitidas else pd.NA
                ),
            }
            for nombre, columnas in REPARTO
        ]
    )
