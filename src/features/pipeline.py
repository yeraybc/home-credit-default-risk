"""El `Pipeline` de las capas 2 montado, con su `ColumnTransformer` dentro.

Convierte las 101 columnas heterogéneas que deja la capa 1 (más los dos ratios del winsorizador)
en una matriz numérica con nombres. Se ajusta **solo sobre el 80% de entrenamiento**: dos de sus
pasos estiman parámetros sobre covariables y tres los estiman con el TARGET.

El orden de los pasos no es libre y lo fija `sklearn.md`:

1. `winsor`, que recorta los extremos.
2. `derivadas`, los dos ratios que llevan una columna winsorizada en el denominador. Antes del
   winsorizador dejarían el error de captura de 117M de ingreso dentro del divisor de la carga.
3. `dominio`, las recodificaciones deterministas de dominio.
4. `columnas`, el `ColumnTransformer` que reparte cada bucket a su codificación.
5. `varianza`, la red contra la columna que se quede constante.

El `SelectorIV` que el boceto de `sklearn.md` dibuja al final **no está**: `iv.py` es del bloque 5
y este pipeline se cierra en el `VarianceThreshold`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder

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
NIVEL_SIN_OCUPACION = "Sin declarar"

# Franjas de la hora de solicitud, criterio de dominio del EDA: pedir fuera de horario laboral
# puede señalar un perfil distinto. Reduce 24 valores a 3. El efecto medido es flojo (8,490%,
# 7,720% y 8,026% sobre train, o sea 0,77pp de recorrido frente a los 2pp del umbral), así que
# quien la mata es el IV del bloque 5 con su medida delante, no esto.
FRANJAS_HORA: tuple[tuple[str, int, int], ...] = (("manana", 6, 12), ("tarde", 12, 18))
FRANJA_FUERA = "fuera de horario"

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
    """La hora de solicitud reducida a mañana, tarde y fuera de horario."""
    franja = pd.Series(FRANJA_FUERA, index=horas.index, dtype=object)
    for nombre, desde, hasta in FRANJAS_HORA:
        franja[(horas >= desde) & (horas < hasta)] = nombre
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
    return (
        *NUMERICAS,
        *CATEGORICAS_OHE,
        COL_EDUCACION,
        COL_ORGANIZACION,
        COL_OCUPACION,
        *BINARIAS,
    )


def verificar_contrato_columnas(datos: pd.DataFrame) -> None:
    """Revienta si el frame y los buckets no se cubren exactamente, en las dos direcciones.

    Sin esto, `remainder="drop"` se lleva en silencio cualquier columna que no esté declarada:
    la matriz saldría más pobre sin un solo error y sin que cambie ningún nombre. Se llama sobre
    el frame **ya pasado por el paso de dominio**, que es el que ve el `ColumnTransformer`.
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
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
            ),
        ]
    )
    columnas = ColumnTransformer(
        [
            # La mediana con la bandera al lado, que ya está en la matriz desde la capa 1: el
            # modelo puede recuperar el grupo imputado. Los dos casos que el EDA dejó con proviso
            # son `DAYS_EMPLOYED`, con sus 55.374 del centinela y `FLAG_DAYS_EMPLOYED_ANOMALY`
            # detrás, y `OWN_CAR_AGE`, con sus nulos estructurales y `FLAG_OWN_CAR`. Contrastar
            # la mediana contra dejar el NaN necesita un modelo, así que es de la Fase 4.
            ("num", SimpleImputer(strategy="median"), list(NUMERICAS)),
            ("ohe", ohe, list(CATEGORICAS_OHE)),
            # `-1` y no NaN para lo no visto: queda fuera de la escala por abajo, que es la señal
            # correcta, y no mete un nulo en una matriz que ya no vuelve a imputarse.
            (
                "ord",
                OrdinalEncoder(
                    categories=[list(JERARQUIA_EDUCACION)],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
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
            ("columnas", columnas),
            # Umbral 0, o sea solo constantes. Hoy no elimina ninguna: medido sobre train, la más
            # pobre de las 20 `FLAG_DOCUMENT_*` es la 12, con una sola observación a 1. Está por
            # el fold del CV de la Fase 4 donde esa bandera sí se quede constante y el número de
            # columnas cambie entre folds. Con umbral 0 la exclusión declarada de la 3 y la 6 es
            # inocua, y solo pasaría a importar si alguien sube el umbral.
            ("varianza", VarianceThreshold()),
        ]
    ).set_output(transform="pandas")


def informe_buckets(datos: pd.DataFrame) -> pd.DataFrame:
    """Puerta del punto: cuántas columnas entrega cada bucket y cuántas salen de la matriz."""
    reparto = {
        "num": NUMERICAS,
        "ohe": CATEGORICAS_OHE,
        "ord": (COL_EDUCACION,),
        "woe": (COL_ORGANIZACION,),
        "tgt": (COL_OCUPACION,),
        "bin": BINARIAS,
    }
    tras_dominio = aplicar_dominio(datos)
    return pd.DataFrame(
        [
            {
                "bucket": nombre,
                "entran": len(columnas),
                "presentes": len(set(columnas) & set(tras_dominio.columns)),
            }
            for nombre, columnas in reparto.items()
        ]
    )
