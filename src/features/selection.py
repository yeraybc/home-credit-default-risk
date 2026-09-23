"""
src.features.selection: Métodos de selección de variables y cálculo de métricas estadísticas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import agg_bureau, agg_bureau_balance, agg_previous, pipeline
from src.features.iv import BANDERAS_RARAS, calcular_iv, iv_condicionado
from src.features.params import valor
from src.features.recipes import cargar_receta


def obtener_categorias(df: pd.DataFrame) -> dict[str, list]:
    """Identifica las columnas categóricas en el DataFrame y devuelve sus valores únicos ordenados.
    filtro e itero de forma simple sobre tipos categóricos.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols}


def recomendar_codificacion(df: pd.DataFrame) -> pd.DataFrame:
    """Recomienda la codificación de cada variable categórica o conceptualmente categórica.

    Aplica las conclusiones de negocio alcanzadas en el EDA para variables conocidas,
    y cae a un enfoque basado en cardinalidad (Binary, OHE, Target/WoE) para el resto.

    Aplico un diccionario estático para conclusiones específicas y heurística básica para el resto.

    **Esto recoge lo que concluyó el EDA, y quien codifica de verdad es `pipeline.py`.** Las dos
    cosas coinciden salvo en seis entradas, donde el punto 1.4 midió sobre el split y llegó a
    otra conclusión: los agrupamientos por tasa de `NAME_TYPE_SUITE`, `NAME_INCOME_TYPE` y
    `NAME_HOUSING_TYPE`, las bandas de riesgo previas al WoE de `ORGANIZATION_TYPE` y al target
    encoding de `OCCUPATION_TYPE`, y la exclusión del filtro de varianza que el bloque de
    `FLAG_DOCUMENT_*` daba por hecha. Las seis lo dicen en su detalle. No se borran porque la
    conclusión del EDA es la que era y el rastro vale, pero la fuente ejecutable es la otra.
    """
    especificas = {
        "NAME_CONTRACT_TYPE": (
            "Binary Mapping",
            "Mapear directamente: 'Cash loans' -> 1, 'Revolving loans' -> 0.",
        ),
        "CODE_GENDER": (
            "Binary Mapping / Clean",
            "Eliminar registros con 'XNA' (solo 4 registros) y binarizar ('M'/'F').",
        ),
        "NAME_TYPE_SUITE": (
            "One-Hot Encoding (Consolidado)",
            "Crear categoría 'Unknown' para los 1.292 nulos (0,42%) en vez de imputar por moda, "
            "para no inventar dato sobre 1.292 obs y no por señal diferencial: el análisis de "
            "missings reclasifica la variable a MCAR. Agrupar 'Other_A' (8,78%), 'Other_B' "
            "(9,83%) y 'Group of people' (8,49%) en 'Other_High' (2.907 obs, 9,39%), y fusionar "
            "'Children' (7,38%) con 'Family' (7,49%). 'Unaccompanied' y 'Spouse, partner' se "
            "mantienen solas, luego aplicar OHE. El pipeline no agrupa: remedidos sobre el 80% de "
            "entrenamiento, sus siete niveles con dato caben dentro de 2,2pp, así que cualquier "
            "criterio de tasas equivalentes los funde en uno solo. Van a OHE tal cual.",
        ),
        "NAME_INCOME_TYPE": (
            "One-Hot / Target Encoding",
            "Agrupar las cuatro categorías de 5 a 22 registros: 'Maternity leave' (40,00%) y "
            "'Unemployed' (36,36%) en 'High Risk Other' (27 obs, 37,04%), 'Businessman' y "
            "'Student' en 'Low Risk Other' (28 obs, 0,00%), luego codificar. El pipeline no parte "
            "en dos: 'Low Risk Other' existía solo por no tener ni un positivo, y sobre el split "
            "son 20 observaciones, donde ver cero positivos a la tasa base tiene probabilidad "
            "0,186. Las cuatro van al mismo residual por rareza, sin dirección propia.",
        ),
        "NAME_EDUCATION_TYPE": (
            "Ordinal Encoding",
            "Jerarquía: Lower secondary (1) → Academic degree (5)",
        ),
        "NAME_FAMILY_STATUS": (
            "One-Hot Encoding / Clean",
            "Eliminar registros 'Unknown' (2 registros) y aplicar OHE sobre el resto (baja "
            "cardinalidad).",
        ),
        "NAME_HOUSING_TYPE": (
            "One-Hot Encoding (Consolidado)",
            "Fusionar 'Co-op apartment' (1.122 obs, 7,93%) con 'House / apartment' (7,80%) por "
            "tasa equivalente. 'With parents', 'Rented apartment', 'Municipal apartment' y "
            "'Office apartment' se mantienen separadas, luego aplicar OHE. El pipeline no fusiona: "
            "aquí un criterio de tasas sí separaría en dos grupos (hay 3,00pp entre 'Municipal "
            "apartment' y 'With parents' sobre train), pero la puerta del pipeline es de rareza y "
            "ninguno de los seis baja del mínimo. Lo decide el IV del bloque 5.",
        ),
        "WEEKDAY_APPR_PROCESS_START": (
            "Binary (0/1)",
            "Mapear: días entresemana → 1, fin de semana → 0",
        ),
        "ORGANIZATION_TYPE": (
            "Weight of Evidence (WoE)",
            "Alta cardinalidad (58 categorías). Agrupar por sectores o tasa de default similar "
            "(no por frecuencia) en entrenamiento, luego aplicar WoE en Fase 3. El pipeline "
            "aplica el WoE nivel a nivel, sin agrupar: quien absorbe el nivel con poca evidencia "
            "propia es el suavizado del prior, que lo acerca al comportamiento medio.",
        ),
        "OCCUPATION_TYPE": (
            "Target / WoE Encoding",
            "Agrupar por tasa de default en entrenamiento. Tratar nulos (31%) como "
            "'Retired/Inactive' tras cruzar con NAME_INCOME_TYPE, luego aplicar WoE/Target. El "
            "pipeline no agrupa en bandas: el TargetEncoder de la librería ya le da a cada nivel "
            "su propia media, con codificación cruzada interna. Y al nulo lo llama 'Sin declarar', "
            "porque el cruce solo es determinista en Unemployed y Pensioner: en Working el nulo "
            "es del 15,70% y se comporta como ruido de captura.",
        ),
        "FONDKAPREMONT_MODE": (
            "Binarize / Drop",
            "68% de nulos. Evaluar si el nulo tiene tasa de default diferencial. Binarizar (Nulo "
            "vs No Nulo) o eliminar.",
        ),
        "HOUSETYPE_MODE": (
            "Binarize / Drop",
            "50% de nulos. Evaluar si el nulo tiene tasa de default diferencial. Binarizar (Nulo "
            "vs No Nulo) o eliminar.",
        ),
        "WALLSMATERIAL_MODE": (
            "OHE / Binarize / Drop",
            "50% de nulos. Consolidar categorías minoritarias. Evaluar si el nulo tiene tasa "
            "diferencial antes de decidir entre binarizar, OHE o eliminar.",
        ),
        "EMERGENCYSTATE_MODE": (
            "Binary Mapping / Drop",
            "47% de nulos. Evaluar si el nulo equivale a 'No' o si tiene tasa de default "
            "diferencial, luego binarizar (Nulo vs No Nulo).",
        ),
        "FLAG_OWN_CAR": (
            "Conservar como binaria",
            "Mapear directamente a boolean/entero. Aporta información patrimonial directa.",
        ),
        "FLAG_OWN_REALTY": (
            "Conservar como binaria",
            "Mapear directamente a boolean/entero. Aporta información patrimonial directa.",
        ),
        "FLAG_MOBIL": (
            "Eliminar / Varianza nula",
            "Media 0,999997 y varianza 0,0000033: un único registro a 0 en 307.511. No "
            "discrimina entre clientes.",
        ),
        "FLAG_CONT_MOBILE": (
            "Eliminar / Sin señal",
            "Correlación de 0,0004 con TARGET, la más baja del bloque de contacto. Tiene "
            "varianza (574 registros a 0) pero no aporta señal.",
        ),
        "FLAG_EMP_PHONE": (
            "Eliminar / Redundante",
            "Redundante con DAYS_EMPLOYED (r = -0,9998): solo 12 registros de 307.511 discrepan "
            "del código de inactivo 365243. Se conserva DAYS_EMPLOYED por ser continua y más "
            "granular.",
        ),
        "REG_CITY_NOT_WORK_CITY": (
            "Conservar como binaria",
            "La más predictiva del bloque geográfico (r = +0,0510 con TARGET). Se conserva "
            "frente a LIVE_CITY_NOT_WORK_CITY (+0,0325), con la que correlaciona 0,83.",
        ),
        "REG_CITY_NOT_LIVE_CITY": (
            "Conservar como binaria",
            "Segunda del bloque (r = +0,0444) y señal casi independiente: correlaciona solo 0,44 "
            "con REG_CITY_NOT_WORK_CITY, así que no es redundante con ella.",
        ),
        "LIVE_CITY_NOT_WORK_CITY": (
            "Pendiente de IV / Candidata a eliminar",
            "r = +0,0325, por debajo de REG_CITY_NOT_WORK_CITY (+0,0510) y redundante con ella "
            "(r = 0,83). La decisión final se toma con el IV en Fase 3.",
        ),
        "REG_REGION_NOT_WORK_REGION": (
            "Pendiente de IV / Candidata a eliminar",
            "r = +0,0069 con TARGET. Las tres variantes de región quedan un orden de magnitud "
            "por debajo de las de ciudad y correlacionan 0,86 entre sí. Se decide con el IV en "
            "Fase 3.",
        ),
        "REG_REGION_NOT_LIVE_REGION": (
            "Pendiente de IV / Candidata a eliminar",
            "r = +0,0056 con TARGET, señal despreciable. Se decide con el IV en Fase 3.",
        ),
        "LIVE_REGION_NOT_WORK_REGION": (
            "Pendiente de IV / Candidata a eliminar",
            "r = +0,0028 con TARGET, la más débil del bloque, y redundante con "
            "REG_REGION_NOT_WORK_REGION (r = 0,86). Se decide con el IV en Fase 3.",
        ),
        "FLAG_DOCUMENT_3": (
            "Conservar como binaria",
            "Conservar para evaluación. Muestra correlación positiva significativa con TARGET "
            "(+0,0443).",
        ),
        "FLAG_DOCUMENT_6": (
            "Conservar como binaria",
            "Conservar para evaluación. Muestra correlación negativa significativa con TARGET "
            "(-0,0286).",
        ),
        # Las tres del bloque de contacto que caían en la rama por defecto y salían con el
        # detalle genérico, sin decisión propia. Ninguna se decide aquí: las tres son binarias
        # ya codificadas y van al passthrough, y quien las juzgue es el IV del bloque 5. Lo que
        # faltaba era su motivo, que es lo que la rama por defecto no puede dar.
        "FLAG_PHONE": (
            "Conservar como binaria",
            "r = -0,0238 con TARGET, la más fuerte de las tres del bloque de contacto y con "
            "signo negativo: tener teléfono fijo declarado acompaña a menos default, y el EDA lo "
            "lee como estabilidad residencial residual. Se decide con el IV.",
        ),
        "FLAG_WORK_PHONE": (
            "Conservar como binaria",
            "r = +0,0285 con TARGET, y es la única de las tres con signo positivo. Declarar "
            "teléfono del trabajo va con más default, no con menos, así que no es una segunda "
            "copia de FLAG_PHONE. Se decide con el IV.",
        ),
        "FLAG_EMAIL": (
            "Conservar como binaria",
            "r = -0,0018 con TARGET, señal despreciable, y solo el 5,67% de los clientes lo "
            "declara. Es la candidata más clara del bloque a caerse, pero cae por IV y no por "
            "correlación.",
        ),
    }

    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # Encuentro las variables binarias 0/1
    binary_num_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col.upper() != "TARGET"
        and len(u := df[col].dropna().unique()) > 0
        and set(u).issubset({0, 1})
    ]

    target_cols = sorted(list(set(cat_cols + binary_num_cols)))

    recoms = []
    for col in target_cols:
        cats = sorted(df[col].dropna().unique())
        card = len(cats)

        # las dos que tienen entrada propia se salen de esta rama por `especificas` y no por una
        # lista con sus nombres, que era el mismo par escrito otra vez al lado del de
        # `pipeline.COLUMNAS_PROTEGIDAS_DE_VARIANZA`
        if col.startswith("FLAG_DOCUMENT_") and col not in especificas:
            strategy, detail = (
                "Filtrar con VarianceThreshold",
                "Baja varianza (medias de 0,000007 a 0,015) y correlación insignificante con "
                "TARGET. El conjunto exacto lo fija el VarianceThreshold en Fase 3 sobre el "
                "split de entrenamiento. El pipeline no exceptúa a FLAG_DOCUMENT_3 ni a "
                "FLAG_DOCUMENT_6 del filtro: con el suelo a cero solo caen las constantes y "
                "ninguna de las dos puede serlo, así que no hay nada de lo que salvarlas. Quien "
                "las protege de verdad es el control por IV del bloque 5.",
            )
        elif col in binary_num_cols and col not in especificas:
            strategy, detail = "Conservar como binaria", "Ya es una variable numérica binaria 0/1."
        else:
            strategy, detail = especificas.get(
                col,
                (
                    ("Binary (0/1)", f"Mapear: {cats[0]} -> 0, {cats[1]} -> 1")
                    if card == 2
                    else (
                        ("One-Hot Encoding", f"Crear {card} columnas dummy")
                        if card <= 10
                        else (
                            "Target / Frequency Encoding",
                            f"Evitar OHE por alta cardinalidad ({card} categorías)",
                        )
                    )
                ),
            )

        recoms.append(
            {
                "Variable": col,
                "Categorías": cats,
                "Cardinalidad": card,
                "Estrategia Recomendada": strategy,
                "Detalle": detail,
            }
        )
    return pd.DataFrame(recoms)


# --- el registro de selección del 5.6, que el 5.8 consume ---------------------------------------


@dataclass(frozen=True)
class DecisionIV:
    """Una de las cinco decisiones del 5.6, con su criterio y su motivo.

    `decision` es del vocabulario cerrado de las recetas (conservar | iv | degradada | descartar |
    control | referencia). La cifra que sostiene la decisión va en `motivo`, nunca consumida.
    """

    decision: str
    criterio: str
    motivo: str

    def __post_init__(self) -> None:
        if not self.decision.strip() or not self.criterio.strip() or not self.motivo.strip():
            raise ValueError("una decisión del 5.6 declara decision, criterio y motivo")


# Las cinco decisiones que el EDA dejó pendientes de IV, cerradas en el 5.6 sobre los 245.993
# clientes de train. `BUREAU_HAS_CURRENT_OVERDUE` es la única que ya estaba en CANDIDATAS_IV;
# las otras cuatro no compiten contra min_iv, así que no van ahí.
DECISIONES_IV: dict[str, DecisionIV] = {
    "BUREAU_HAS_CURRENT_OVERDUE": DecisionIV(
        "conservar",
        "mayor IV sin presencia; empate en la cuarta cifra decae a la construcción",
        "empata con BUREAU_CURRENT_OVERDUE_SUM > 0 en 0,0099 (2.688 vs 2.687 marcados); gana la "
        "foto porque no pierde al cliente cuya única mora activa está en otra moneda",
    ),
    "tripartita_mora": DecisionIV(
        "descartar",
        "IV incremental sobre BUREAU_OVERDUE_UNION >= min_iv (0,02), dentro de con historial",
        "el incremental es 0,0014, muy por debajo de 0,02: separar reportada a cero (6,79%, "
        "n=90.085) de sin reportar (7,38%, n=62.863) no añade nada sobre la unión (9,55% vs "
        "7,03%); no se construye columna y sigue BUREAU_OVERDUE_UNION",
    ),
    "BUILDING_INFO_COUNT": DecisionIV(
        "descartar",
        "IV incremental sobre HAS_BUILDING_INFO >= min_iv (0,02), dentro de quien tiene dato",
        "el incremental es 0,00015, muy por debajo de 0,02, aunque el marginal (0,0212) llega: "
        "cuántos campos del edificio tiene un cliente no separa más que tener alguno o no; sale "
        "y se queda HAS_BUILDING_INFO",
    ),
    "PREV_FUTURE_DUE_VIVAS": DecisionIV(
        "descartar",
        "IV incremental sobre PREV_FUTURE_DUE_MAX >= min_iv (0,02), dentro de con previas",
        "el incremental es 0,0067; sobre train la actual tiene más IV sin presencia (0,0116 "
        "frente a 0,0082 de solo vivas), al revés de lo medido sobre la tabla cruda sin split "
        "(0,0842 frente a 0,0856); no se construye columna y sigue PREV_FUTURE_DUE_MAX",
    ),
    # el incremental sobre EXT_SOURCE_3 (pendiente del 4.10 y del 2.3 cerrado en el 2.3): 13
    # columnas de bureau y bureau_balance con |Pearson| >= solape_ext3_min (0,20) sobre train,
    # cinco pasan min_iv (0,02) dentro de con historial y ocho no; degradada y no descartar,
    # porque el 5.8 da el corte final con la redundancia y la banda de revisión delante
    "BUREAU_CREDITS_PER_YEAR": DecisionIV(
        "conservar", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,4214, incremental 0,0240"
    ),
    "BUREAU_DAYS_CREDIT_MAX": DecisionIV(
        "conservar", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,3896, incremental 0,0311"
    ),
    "BUREAU_ACTIVE_COUNT": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,3888, incremental 0,0075"
    ),
    "BUREAU_DAYS_CREDIT_UPDATE_FLAG": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,3481, incremental 0,0136"
    ),
    "BB_MONTHS_SINCE_LAST_DPD_REL": DecisionIV(
        "conservar", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,3149, incremental 0,0300"
    ),
    "BB_RECENT_DPD_FLAG_REL": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2992, incremental 0,0084"
    ),
    "BB_OVERDUE_UNION": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2681, incremental 0,0118"
    ),
    "BB_ANY_DPD_FLAG": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2370, incremental 0,0115"
    ),
    "BUREAU_DAYS_CREDIT_MIN": DecisionIV(
        "conservar", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2263, incremental 0,0281"
    ),
    "BB_CREDITS_WITH_DPD_COUNT": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2261, incremental 0,0056"
    ),
    "BUREAU_OVERDUE_UNION": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2186, incremental 0,0059"
    ),
    "BUREAU_DEBT_CREDIT_RATIO": DecisionIV(
        "conservar", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2034, incremental 0,0373"
    ),
    "BUREAU_HAS_ANY_OVERDUE": DecisionIV(
        "degradada", "IV incremental sobre EXT_SOURCE_3 >= 0,02", "r=-0,2014, incremental 0,0055"
    ),
}


# --- la redundancia entre tablas del 5.7 ---------------------------------------------------------

TABLA_PRINCIPAL = "application_train"

# cada auxiliar con el módulo que declara lo que añade sin receta
AUXILIARES = {
    "bureau": agg_bureau,
    "bureau_balance": agg_bureau_balance,
    "previous_application": agg_previous,
}


def _nombres_por_tabla() -> dict[str, set[str]]:
    return {
        tabla: {f["nombre"] for f in cargar_receta(tabla)["features"]}
        | set(modulo.COLUMNAS_SIN_RECETA)
        for tabla, modulo in AUXILIARES.items()
    }


def tabla_de(columna: str, nombres: dict[str, set[str]] | None = None) -> str:
    """La tabla que produce la columna: la auxiliar de su receta o de su `COLUMNAS_SIN_RECETA`, y
    si no está en ninguna, la principal. `BUREAU_OVERDUE_UNION` sale de bureau aunque la receta de
    bureau_balance la cite como referencia, porque bureau se mira antes.

    `nombres` es para no releer las tres recetas por columna; quien llame desde fuera no lo pasa.
    """
    for tabla, de_tabla in (nombres or _nombres_por_tabla()).items():
        if columna in de_tabla:
            return tabla
    return TABLA_PRINCIPAL


def columnas_protegidas() -> frozenset[str]:
    """Las que no pasan por el umbral del 5.8, adelantadas aquí porque el empate del 5.7 las mira:
    las `control: true` de las recetas, las tres `HAS_*` (`pipeline.PRESENCIA_AUX`), los dos
    documentos protegidos del filtro de varianza (`pipeline.COLUMNAS_PROTEGIDAS_DE_VARIANZA`),
    las banderas raras y `PREV_ACTIVIDAD_12M_COLA` como término de `PREV_RELACION_CORTA_ACTIVA`.

    Las dos listas de `pipeline.py` se importan y no se copian, que copiarlas es la misma trampa
    de las dos fuentes de verdad que ya ha mordido tres veces en este proyecto: un tercer
    documento protegido en `pipeline.py` no se enteraría aquí.

    Desde el 5.8, además, toda fuente de presencia (`fuentes_de_presencia()`) mientras le quede
    alguna columna recuperada fuera de los descartes: sin ella, la mediana deja de distinguir
    "vale la mediana" de "no se sabe". Es lo que reabrió dos pares del 5.7.
    """
    recetas = [cargar_receta(tabla)["features"] for tabla in AUXILIARES]
    controles = {f["nombre"] for receta in recetas for f in receta if f.get("control")}
    brutos = set(_descartes_brutos())
    de_presencia = {f for f, recuperadas in fuentes_de_presencia().items() if recuperadas - brutos}
    return frozenset(
        controles
        | set(pipeline.PRESENCIA_AUX)
        | set(pipeline.COLUMNAS_PROTEGIDAS_DE_VARIANZA)
        | {"PREV_ACTIVIDAD_12M_COLA"}
        | set(BANDERAS_RARAS)
        | de_presencia
    )


def _descartadas() -> set[str]:
    """Lo que ya sale por su receta o por el 5.6, y por eso no entra al detector."""
    de_receta = {
        f["nombre"]
        for tabla in AUXILIARES
        for f in cargar_receta(tabla)["features"]
        if f["decision"] == "descartar"
    }
    return de_receta | {c for c, d in DECISIONES_IV.items() if d.decision == "descartar"}


# Los dos de dentro de bureau que el 2.3 le dejó al bloque 5. El plan citaba el primero como
# BUREAU_COUNT_COLA frente a BUREAU_ACTIVE_COUNT (0,692), pero ese par da 0,3167 sobre train: el
# 0,6919 es el conteo del que se corta la cola. El segundo cruza con ACTIVE_COUNT > 0 (0,7106) y no
# con el conteo (0,4355), y se mide contra el conteo porque su tramo 0 es el sin ningún activo.
PARES_DECLARADOS: dict[tuple[str, str], str] = {
    ("BUREAU_ACTIVE_COUNT", "BUREAU_LOAN_COUNT"): (
        "la cola del conteo sale de BUREAU_LOAN_COUNT, que correlaciona 0,692 con el activo (2.3)"
    ),
    ("BUREAU_ACTIVE_COUNT", "BUREAU_DAYS_CREDIT_UPDATE_FLAG"): (
        "la señal de la bandera vive en los clientes con algún crédito activo (2.3)"
    ),
}


def _es_bandera01(serie: pd.Series) -> bool:
    return bool(serie.dropna().isin((0, 1)).all())


def pares_redundantes(train: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Los pares de columnas de tablas distintas que cruzan el umbral de redundancia o caen en la
    banda de revisión, más los `PARES_DECLARADOS`. No mira el TARGET.

    Entran las numéricas vivas: fuera lo que su receta o el 5.6 ya descartan, porque su salida no
    depende de este informe (si el 5.8 recupera alguna, se vuelve a pasar con `columnas`). Las
    categóricas se saltan: `BB_TRAJECTORY`, la única de una auxiliar, se queda en 0,365 de V de
    Cramér contra las 55 banderas vivas de otras tablas (auditoría del 5.7), y en 0,1548 contra
    las magnitudes con el mismo binning de `tramos()` que usa el IV; una V de Cramér directa
    contra una magnitud sin binar es engañosa (sale por encima de 0,9 contra `EXT_SOURCE_1`), que
    es el motivo de no comparar la categórica contra la magnitud en bruto.

    El Pearson va sobre los clientes con las dos columnas, y el umbral según el tipo:

    - dos magnitudes, o bandera y magnitud: por encima de `redundancia_pearson`, y en la banda
      desde `banda_revision_pearson`;
    - dos banderas: por encima de `redundancia_cramer`, porque en una 2x2 el |r| es la V;
    - bandera y magnitud sin negativos, además: la bandera contra `magnitud > 0` por encima de
      `redundancia_cramer`, que es la redundancia del evento y no de la escala
      (`metodologia-estadistica` 8.4).

    Cada par sale con sus dos nombres en orden alfabético, que es la clave del registro.
    """
    if columnas is None:
        fuera = _descartadas() | {"SK_ID_CURR", "TARGET"}
        columnas = [c for c in train.columns if c not in fuera]
    columnas = [c for c in columnas if pd.api.types.is_numeric_dtype(train[c])]
    datos = train[columnas].astype(float)
    bandera = {c: _es_bandera01(datos[c]) for c in columnas}
    binarizable = [c for c in columnas if not bandera[c] and datos[c].min() >= 0]
    evento = datos[binarizable].gt(0).where(datos[binarizable].notna()).add_suffix(" > 0")
    r = pd.concat([datos, evento], axis=1).corr()
    nombres = _nombres_por_tabla()
    tabla = {c: tabla_de(c, nombres) for c in columnas}
    umbral_mm = valor("redundancia_pearson")
    umbral_ff = valor("redundancia_cramer")
    banda = valor("banda_revision_pearson")

    filas = []
    ordenadas = sorted(columnas)
    for i, a in enumerate(ordenadas):
        for b in ordenadas[i + 1 :]:
            declarado = (a, b) in PARES_DECLARADOS
            if tabla[a] == tabla[b] and not declarado:
                continue
            tipo = {2: "FF", 1: "FM", 0: "MM"}[bandera[a] + bandera[b]]
            r_ab = r.loc[a, b]
            r_bin = np.nan
            if tipo == "FM":
                f, m = (a, b) if bandera[a] else (b, a)
                if m in binarizable:
                    r_bin = r.loc[f, f"{m} > 0"]
            umbral = umbral_ff if tipo == "FF" else umbral_mm
            if abs(r_ab) > umbral or abs(r_bin) > umbral_ff:
                zona = "por encima"
            elif tipo != "FF" and abs(r_ab) >= banda:
                zona = "banda"
            elif declarado:
                zona = "declarado"
            else:
                continue
            filas.append(
                {
                    "a": a,
                    "b": b,
                    "tabla_a": tabla[a],
                    "tabla_b": tabla[b],
                    "tipo": tipo,
                    "r": r_ab,
                    "r_bin": r_bin,
                    "n": int((datos[a].notna() & datos[b].notna()).sum()),
                    "zona": zona,
                    "declarado": declarado,
                }
            )
    return pd.DataFrame(filas).set_index(["a", "b"])


def informe_redundancia(train: pd.DataFrame, pares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resuelve cada par de `pares_redundantes()` con estratos cruzados en las dos direcciones.

    **Criterio, escrito antes de medir:** sobre la población donde las dos columnas tienen dato,
    `inc_a` es el IV de `a` dentro de los tramos de `b` (`iv_condicionado()`) y `inc_b` al revés.
    - Si una de las dos es protegida (`columnas_protegidas()`), se queda; la otra se queda también
      si su incremental sobre la protegida llega a `min_iv`, y si no, sale.
    - Si ninguna es protegida y las dos llegan a `min_iv`, las dos aportan señal propia y se quedan.
    - Si llega solo una, esa es la que se queda.
    - Si no llega ninguna, es la misma información contada dos veces: se queda la de más IV
      marginal dentro de la población del par.

    `queda` es la tupla de columnas del par que sobreviven, `iv_a` e `iv_b` el IV marginal de cada
    una sobre esa misma población y `motivo` la frase que sostiene la decisión, con las cifras.
    """
    if pares is None:
        pares = pares_redundantes(train)
    protegidas = columnas_protegidas()
    min_iv = valor("min_iv")
    filas = []
    for a, b in pares.index:
        dentro = train[train[a].notna() & train[b].notna()]
        y = dentro["TARGET"]
        iv_a = calcular_iv(dentro[a], y)
        iv_b = calcular_iv(dentro[b], y)
        inc_a = iv_condicionado(dentro[a], dentro[b], y)
        inc_b = iv_condicionado(dentro[b], dentro[a], y)
        prot_a, prot_b = a in protegidas, b in protegidas

        if prot_a and prot_b:
            queda, motivo = (a, b), "las dos protegidas, ninguna sale por redundancia"
        elif prot_a or prot_b:
            protegida, otra = (a, b) if prot_a else (b, a)
            inc_otra = inc_b if prot_a else inc_a
            llega = "llega" if inc_otra >= min_iv else "no llega"
            motivo = (
                f"{protegida} protegida; {otra} {llega} a min_iv sobre ella "
                f"(incremental {inc_otra:.4f})"
            )
            queda = (a, b) if inc_otra >= min_iv else (protegida,)
        elif inc_a >= min_iv and inc_b >= min_iv:
            queda = (a, b)
            motivo = f"las dos llegan a min_iv la una sobre la otra ({inc_a:.4f} y {inc_b:.4f})"
        elif inc_a >= min_iv:
            queda = (a,)
            motivo = f"{a} llega a min_iv sobre {b} (incremental {inc_a:.4f}); {b} no ({inc_b:.4f})"
        elif inc_b >= min_iv:
            queda = (b,)
            motivo = f"{b} llega a min_iv sobre {a} (incremental {inc_b:.4f}); {a} no ({inc_a:.4f})"
        else:
            ganadora = a if iv_a >= iv_b else b
            queda = (ganadora,)
            motivo = (
                f"ninguna aporta sobre la otra (incremental {inc_a:.4f} y {inc_b:.4f}); "
                f"queda {ganadora} por mayor IV marginal ({iv_a:.4f} frente a {iv_b:.4f})"
            )

        filas.append(
            {
                "a": a,
                "b": b,
                "n": len(dentro),
                "iv_a": iv_a,
                "iv_b": iv_b,
                "inc_a": inc_a,
                "inc_b": inc_b,
                "queda": queda,
                "motivo": motivo,
            }
        )
    return pd.DataFrame(filas).set_index(["a", "b"])


@dataclass(frozen=True)
class DecisionRedundancia:
    """La resolución de un par de `informe_redundancia()`, con la cifra que la sostiene en
    `motivo`, nunca consumida."""

    queda: tuple[str, ...]
    motivo: str

    def __post_init__(self) -> None:
        if not self.queda or not self.motivo.strip():
            raise ValueError("una decisión de redundancia declara queda y motivo")


# Los 15 pares del 5.7 sobre los 245.993 de train: los 13 que cruzan el umbral o la banda entre
# tablas más los 2 declarados dentro de bureau (PARES_DECLARADOS). Congelado con
# `informe_redundancia()`, no se recalcula al importar: el 5.8 lee esto para saber qué columna
# sale por redundancia sin volver a ensamblar nada.
DECISIONES_REDUNDANCIA: dict[tuple[str, str], DecisionRedundancia] = {
    ("BB_MANY_CREDITS_FLAG", "BUREAU_COUNT_COLA"): DecisionRedundancia(
        ("BUREAU_COUNT_COLA",),
        "r=0,9981, la misma cola del conteo refijada en 18 por las dos tablas (2.3 y 3.9); "
        "incremental 0,0000 en las dos direcciones, empatan en IV (0,0022) y queda "
        "BUREAU_COUNT_COLA, de la tabla que hizo el refijado original",
    ),
    ("BB_MONTHS_REPORTED", "BUREAU_CLOSED_COUNT"): DecisionRedundancia(
        ("BB_MONTHS_REPORTED", "BUREAU_CLOSED_COUNT"),
        "r=0,6227, en la banda de revisión; cada una aporta sobre la otra (incremental 0,0219 "
        "y 0,0371, las dos por encima de min_iv), así que se quedan las dos",
    ),
    ("BB_MONTHS_REPORTED", "BUREAU_LOAN_COUNT"): DecisionRedundancia(
        ("BB_MONTHS_REPORTED", "BUREAU_LOAN_COUNT"),
        "r=0,6932, en la banda de revisión; cada una aporta sobre la otra (incremental 0,0233 "
        "y 0,0220), así que se quedan las dos",
    ),
    ("BB_MONTHS_TOTAL", "BUREAU_CLOSED_COUNT"): DecisionRedundancia(
        ("BB_MONTHS_TOTAL",),
        "r=0,8037, por encima del umbral; BB_MONTHS_TOTAL llega a min_iv sobre "
        "BUREAU_CLOSED_COUNT (incremental 0,0293) y BUREAU_CLOSED_COUNT no llega sobre ella "
        "(0,0123), así que solo se queda BB_MONTHS_TOTAL",
    ),
    ("BB_MONTHS_TOTAL", "BUREAU_DAYS_CREDIT_MIN"): DecisionRedundancia(
        ("BB_MONTHS_TOTAL", "BUREAU_DAYS_CREDIT_MIN"),
        "r=-0,6047, en la banda de revisión; cada una aporta sobre la otra (incremental 0,0213 "
        "y 0,0447), así que se quedan las dos",
    ),
    ("BB_MONTHS_TOTAL", "BUREAU_LOAN_COUNT"): DecisionRedundancia(
        ("BB_MONTHS_TOTAL", "BUREAU_LOAN_COUNT"),
        "r=0,7989, por encima del umbral pero cada una aporta sobre la otra (incremental "
        "0,1077 y 0,0745), así que se quedan las dos",
    ),
    ("BB_OVERDUE_UNION", "BUREAU_HAS_ANY_OVERDUE"): DecisionRedundancia(
        ("BB_OVERDUE_UNION",),
        "r=0,8502; ninguna aporta sobre la otra (incremental 0,0109 y 0,0001, las dos por "
        "debajo de min_iv), queda BB_OVERDUE_UNION por mayor IV marginal (0,0300 frente a "
        "0,0196)",
    ),
    ("BB_OVERDUE_UNION", "BUREAU_MAX_OVERDUE_EVER"): DecisionRedundancia(
        ("BB_OVERDUE_UNION",),
        "cruza binarizado (r_bin=0,8936, r en bruto solo 0,0945 porque la magnitud no escala "
        "con el evento); ninguna aporta sobre la otra (incremental 0,0065 y 0,0029), queda "
        "BB_OVERDUE_UNION por mayor IV marginal (0,0323 frente a 0,0293)",
    ),
    ("BB_OVERDUE_UNION", "BUREAU_OVERDUE_UNION"): DecisionRedundancia(
        ("BB_OVERDUE_UNION",),
        "r=0,8667; ninguna aporta sobre la otra (incremental 0,0070 y 0,0000), queda "
        "BB_OVERDUE_UNION por mayor IV marginal (0,0300 frente a 0,0234)",
    ),
    ("BUREAU_ACTIVE_COUNT", "BUREAU_DAYS_CREDIT_UPDATE_FLAG"): DecisionRedundancia(
        ("BUREAU_ACTIVE_COUNT",),
        "declarado en el 2.3: cruza binarizado (r_bin=0,7106 contra ACTIVE_COUNT > 0, r en "
        "bruto 0,4355); BUREAU_ACTIVE_COUNT llega a min_iv sobre la bandera (incremental "
        "0,0319) y la bandera no llega sobre el conteo (0,0063), así que solo se queda "
        "BUREAU_ACTIVE_COUNT",
    ),
    ("BUREAU_ACTIVE_COUNT", "BUREAU_LOAN_COUNT"): DecisionRedundancia(
        ("BUREAU_ACTIVE_COUNT", "BUREAU_LOAN_COUNT"),
        "declarado en el 2.3: r=0,6919, en la banda; cada una aporta sobre la otra "
        "(incremental 0,0995 y 0,0583), así que se quedan las dos",
    ),
    ("BUREAU_ANNUITY_ACTIVE_RATIO", "HAS_BUREAU_BALANCE"): DecisionRedundancia(
        ("HAS_BUREAU_BALANCE",),
        "cruza binarizado (r_bin=0,7420, r en bruto 0,6152 porque el ratio no escala con el "
        "evento); HAS_BUREAU_BALANCE es protegida y BUREAU_ANNUITY_ACTIVE_RATIO no llega a "
        "min_iv sobre ella (incremental 0,0078), así que solo se queda la protegida",
    ),
    ("BUREAU_CREDITS_WITH_ANNUITY_COUNT", "HAS_BUREAU_BALANCE"): DecisionRedundancia(
        ("HAS_BUREAU_BALANCE",),
        "cruza binarizado (r_bin=0,7420, r en bruto 0,5373); HAS_BUREAU_BALANCE es protegida y "
        "BUREAU_CREDITS_WITH_ANNUITY_COUNT no llega a min_iv sobre ella (incremental 0,0048), "
        "así que solo se queda la protegida",
    ),
    # los dos últimos se reabrieron en el 5.8: sus perdedoras son fuentes de presencia, que desde
    # entonces van protegidas mientras recuperen alguna columna que siga en la matriz
    ("FLAG_EXT_SOURCE_3_NULL", "HAS_BUREAU_HISTORY"): DecisionRedundancia(
        ("FLAG_EXT_SOURCE_3_NULL", "HAS_BUREAU_HISTORY"),
        "r=-0,7903, que el EDA no había anticipado; las dos protegidas desde el 5.8, porque "
        "FLAG_EXT_SOURCE_3_NULL es la presencia de EXT_SOURCE_3. En el 5.7 salía por no llegar a "
        "min_iv sobre HAS_BUREAU_HISTORY (incremental 0,0010)",
    ),
    ("HAS_BUREAU_HISTORY", "HAS_BUREAU_INFO"): DecisionRedundancia(
        ("HAS_BUREAU_HISTORY", "HAS_BUREAU_INFO"),
        "r=0,9667; las dos protegidas desde el 5.8, porque HAS_BUREAU_INFO es la presencia de "
        "las seis AMT_REQ_CREDIT_BUREAU_*. En el 5.7 salía por no llegar a min_iv sobre "
        "HAS_BUREAU_HISTORY (incremental 0,0015)",
    ),
}


# --- la selección final del 5.8 ------------------------------------------------------------------


class _Lector(dict):
    """Un frame de mentira que anota qué columnas pide una función de `PRESENCIA_POR_COLUMNA`."""

    def __missing__(self, columna: str) -> pd.Series:
        self[columna] = pd.Series([0.0])
        return self[columna]


def fuentes_de_presencia() -> dict[str, set[str]]:
    """Cada columna de la que el pipeline recupera una ausencia, con las que recupera.

    Sale de los cuatro grupos de `pipeline.py` y no de una lista escrita aquí: las funciones de
    `PRESENCIA_POR_COLUMNA` se ejecutan sobre un `_Lector`, que anota qué columnas leen.
    """
    fuentes: dict[str, set[str]] = {}
    for recuperada, fuente in {
        **pipeline.PRESENCIA_POR_BANDERA,
        **pipeline.PRESENCIA_CASI_EXACTA,
    }.items():
        fuentes.setdefault(fuente, set()).add(recuperada)
    for recuperada, condicion in pipeline.PRESENCIA_POR_COLUMNA.items():
        lector = _Lector()
        condicion(lector)
        for fuente in lector:
            fuentes.setdefault(fuente, set()).add(recuperada)
    fuentes.setdefault(pipeline.BANDERA_BLOQUE, set()).update(pipeline.PRESENCIA_POR_BLOQUE)
    return fuentes


def _descartes_brutos() -> dict[str, str]:
    """Lo que sale por una decisión ya congelada, antes de mirar las protegidas: los `descartar`
    provisionales de receta que llegan a la matriz, los `descartar` y `degradada` del 5.6 que son
    columna y lo que pierde su par en el 5.7. Cada columna con su motivo, o sus motivos."""
    declaradas = set(pipeline.columnas_declaradas())
    motivos: dict[str, list[str]] = {}
    for tabla in AUXILIARES:
        for f in cargar_receta(tabla)["features"]:
            # BB_STATUS_WORST sale dos veces en su receta, una por población
            texto = f"receta de {tabla}: {f['estado']}"
            if f["decision"] == "descartar" and f["nombre"] in declaradas:
                if texto not in motivos.setdefault(f["nombre"], []):
                    motivos[f["nombre"]].append(texto)
    for columna, d in DECISIONES_IV.items():
        if d.decision in ("descartar", "degradada") and columna in declaradas:
            motivos.setdefault(columna, []).append(f"{d.decision} en el 5.6 ({d.motivo})")
    for (a, b), d in DECISIONES_REDUNDANCIA.items():
        for columna in {a, b} - set(d.queda):
            motivos.setdefault(columna, []).append(
                f"redundante con {' y '.join(d.queda)} en el 5.7"
            )
    return {columna: "; ".join(textos) for columna, textos in motivos.items()}


def descartes_fijos() -> dict[str, str]:
    """Lo que el `SelectorIV` saca sin mirar el IV del fold, con su motivo: `_descartes_brutos()`
    menos las protegidas. No se reajusta por fold, igual que los cortes `medido` del 5.1."""
    protegidas = columnas_protegidas()
    return {c: m for c, m in _descartes_brutos().items() if c not in protegidas}
