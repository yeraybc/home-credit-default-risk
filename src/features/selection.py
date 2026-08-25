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
            "Imputar nulos (0.42%) como Moda o 'Unknown'. Agrupar categorías minoritarias (ej. "
            "'Other_A', 'Other_B', 'Children', 'Family') según tasa de default similar antes de "
            "aplicar OHE.",
        ),
        "NAME_INCOME_TYPE": (
            "One-Hot / Target Encoding",
            "Agrupar categorías con < 22 registros ('Maternity', 'Unemployed', 'Businessman', "
            "'Student') en 'High Risk Other' / 'Low Risk Other' según tasa de default, luego "
            "codificar.",
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
            "Agrupar las 3 categorías poco representadas ('Co-op apartment', 'House/apartment' "
            "fusionadas por tasas similares), luego aplicar OHE.",
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
                "Eliminar / Baja Varianza",
                "Candidata a eliminación por baja varianza y correlación insignificante con "
                "TARGET.",
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
