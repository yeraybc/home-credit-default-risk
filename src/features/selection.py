"""
src.features.selection: Métodos de selección de variables y cálculo de métricas estadísticas.
"""

from __future__ import annotations

import pandas as pd


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
            "mantienen solas, luego aplicar OHE.",
        ),
        "NAME_INCOME_TYPE": (
            "One-Hot / Target Encoding",
            "Agrupar las cuatro categorías de 5 a 22 registros: 'Maternity leave' (40,00%) y "
            "'Unemployed' (36,36%) en 'High Risk Other' (27 obs, 37,04%), 'Businessman' y "
            "'Student' en 'Low Risk Other' (28 obs, 0,00%), luego codificar.",
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
            "'Office apartment' se mantienen separadas, luego aplicar OHE.",
        ),
        "WEEKDAY_APPR_PROCESS_START": (
            "Binary (0/1)",
            "Mapear: días entresemana → 1, fin de semana → 0",
        ),
        "ORGANIZATION_TYPE": (
            "Weight of Evidence (WoE)",
            "Alta cardinalidad (58 categorías). Agrupar por sectores o tasa de default similar "
            "(no por frecuencia) en entrenamiento, luego aplicar WoE en Fase 3.",
        ),
        "OCCUPATION_TYPE": (
            "Target / WoE Encoding",
            "Agrupar por tasa de default en entrenamiento. Tratar nulos (31%) como "
            "'Retired/Inactive' tras cruzar con NAME_INCOME_TYPE, luego aplicar WoE/Target.",
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
            "(+0.044).",
        ),
        "FLAG_DOCUMENT_6": (
            "Conservar como binaria",
            "Conservar para evaluación. Muestra correlación negativa significativa con TARGET "
            "(-0.029).",
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

        if col.startswith("FLAG_DOCUMENT_") and col not in ["FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6"]:
            strategy, detail = (
                "Filtrar con VarianceThreshold",
                "Baja varianza (medias de 0,000007 a 0,015) y correlación insignificante con "
                "TARGET. El conjunto exacto lo fija el VarianceThreshold en Fase 3 sobre el "
                "split de entrenamiento; FLAG_DOCUMENT_3 y FLAG_DOCUMENT_6 quedan fuera del "
                "filtro para evaluarlas con IV.",
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
