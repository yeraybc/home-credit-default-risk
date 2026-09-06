"""Limpieza determinista de nivel fila, capa 1 del pipeline de features.

Solo entra aquí lo que no estima ningún parámetro a partir de datos: centinelas, filas y
columnas que se eliminan, y caps con una constante fija de dominio. Todo lo que salga de un
percentil o de una mediana vive en la capa 2a, aunque parezca la misma clase de operación.

Las decisiones son las del EDA y no se reabren aquí; lo que sí se declara es el motivo de
cada una, para que el informe pueda explicar el recuento línea a línea.
"""

from __future__ import annotations

import pandas as pd

from src.features.params import valor

# --- filas -----------------------------------------------------------------------------------
# Las cinco son residuales y suman 19 filas, con 2 de solape. Las tres de nulos son el
# resultado matemático que el EDA marcó como MCAR: con 0% de default, su delta es exactamente
# la media global invertida, así que no hay nada que imputar ni señal que preservar.
FILAS_POR_CATEGORIA = {
    "CODE_GENDER": "los 4 registros XNA, que impiden binarizar el sexo",
    "NAME_FAMILY_STATUS": "los 2 registros Unknown",
}
FILAS_POR_NULO = {
    "CNT_FAM_MEMBERS": "2 nulos residuales",
    "DAYS_LAST_PHONE_CHANGE": "1 nulo residual",
    "AMT_ANNUITY": "12 nulos residuales",
}

# --- columnas --------------------------------------------------------------------------------
COLUMNAS_SUELTAS = {
    "FLAG_MOBIL": "varianza nula: un único registro a 0 en 307.511",
    "FLAG_CONT_MOBILE": "sin señal, r = 0,0004 con TARGET, pese a tener 574 registros a 0",
    "FLAG_EMP_PHONE": (
        "redundante con DAYS_EMPLOYED, r = -0,9998 y solo 12 discrepancias con el centinela; "
        "se conserva la continua, que es más granular"
    ),
    "OBS_60_CNT_SOCIAL_CIRCLE": "redundante con OBS_30, r = 0,9985, y r con TARGET algo menor",
    # La skill daba 1,00 de correlación y son 0,8605, así que el motivo no es redundancia
    # numérica sino señal prestada, que es el caso de flags anidadas de la metodología: con
    # DEF_30 a cero no hay ni un caso de DEF_60 por encima de cero, dentro de cada estrato de
    # DEF_30 la de 60 días no aporta nada significativo (+0,94pp con p = 0,016, +1,85pp con
    # p = 0,13 y +3,86pp con p = 0,20) y al revés DEF_30 sí aporta (+1,76pp con p = 4,4e-10).
    "DEF_60_CNT_SOCIAL_CIRCLE": (
        "señal prestada de DEF_30, que la contiene; no aporta efecto propio dentro de sus "
        "estratos y r con TARGET es menor (0,0313 frente a 0,0322)"
    ),
}
# La limpieza de bureau (importes por encima de 50M, cuota por encima de 10M, deuda capada al
# crédito, moneda) entra en el bloque 2, con sus propias constantes.

# Del bloque edificio se conserva solo la versión AVG de cada concepto numérico: MODE y MEDI
# son estadísticamente intercambiables. La lista se deriva del propio frame en vez de
# escribirse a mano, porque hay tres columnas que terminan en _MODE y no pertenecen al grupo:
# las cuatro categóricas del bloque, que no tienen AVG ni MEDI, y TOTALAREA_MODE, que es el
# concepto 15 y no tiene pareja.
SUFIJOS_EDIFICIO = ("_MODE", "_MEDI")


def columnas_edificio_redundantes(app: pd.DataFrame) -> list[str]:
    """Las versiones MODE y MEDI de los conceptos del bloque edificio que sí tienen AVG."""
    con_avg = {c[: -len("_AVG")] for c in app.columns if c.endswith("_AVG")}
    return sorted(
        c
        for c in app.columns
        for suf in SUFIJOS_EDIFICIO
        if c.endswith(suf) and c[: -len(suf)] in con_avg
    )


def columnas_a_eliminar(app: pd.DataFrame) -> dict[str, str]:
    """Columna y motivo, incluido el bloque edificio derivado del propio frame."""
    motivos = {c: m for c, m in COLUMNAS_SUELTAS.items() if c in app.columns}
    for c in columnas_edificio_redundantes(app):
        motivos[c] = "bloque edificio: se conserva solo la versión AVG del concepto"
    return motivos


def filas_a_eliminar(app: pd.DataFrame) -> pd.Series:
    """Máscara de las filas residuales que se eliminan, con los solapes ya resueltos."""
    fuera = pd.Series(False, index=app.index)
    for col in FILAS_POR_CATEGORIA:
        valores = {"CODE_GENDER": "XNA", "NAME_FAMILY_STATUS": "Unknown"}[col]
        if col in app.columns:
            fuera |= app[col].eq(valores)
    for col in FILAS_POR_NULO:
        if col in app.columns:
            fuera |= app[col].isna()
    return fuera


COL_ANOMALIA = "FLAG_DAYS_EMPLOYED_ANOMALY"


def aplicar_centinela(app: pd.DataFrame) -> pd.DataFrame:
    """DAYS_EMPLOYED a NaN donde lleva el código de inactivo, con su bandera al lado.

    La bandera se calcula **antes** de sustituir: después el dato que la define ya no existe.
    Y no se sobrescribe si ya está, sino que se acumula con un o lógico: en una segunda pasada
    el centinela ya es NaN, así que recalcularla la pondría a cero y borraría en silencio los
    55.374 marcados de la primera.
    """
    centinela = valor("centinela_365243")
    anomalo = app["DAYS_EMPLOYED"].eq(centinela)
    app = app.copy()
    previa = app[COL_ANOMALIA].astype("int8") if COL_ANOMALIA in app.columns else 0
    app[COL_ANOMALIA] = (anomalo.astype("int8") | previa).astype("int8")
    app.loc[anomalo, "DAYS_EMPLOYED"] = pd.NA
    return app


def aplicar_caps_de_dominio(app: pd.DataFrame) -> pd.DataFrame:
    """Caps con constante fija. Los que salen de un percentil van en la capa 2a, no aquí."""
    app = app.copy()
    col = "AMT_REQ_CREDIT_BUREAU_DAY"
    if col in app.columns:
        app[col] = app[col].clip(upper=valor("app_amt_req_bureau_day_max"))
    return app


def limpiar_application(app: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza determinista completa a application_train o application_test."""
    fuera = filas_a_eliminar(app)
    limpio = app.loc[~fuera].copy()
    limpio = limpio.drop(columns=list(columnas_a_eliminar(limpio)))
    limpio = aplicar_centinela(limpio)
    limpio = aplicar_caps_de_dominio(limpio)
    return limpio.reset_index(drop=True)


def informe_limpieza(app: pd.DataFrame) -> pd.DataFrame:
    """Recuento línea a línea de lo que la limpieza quita y por qué."""
    filas = []
    for col, motivo in FILAS_POR_CATEGORIA.items():
        v = {"CODE_GENDER": "XNA", "NAME_FAMILY_STATUS": "Unknown"}[col]
        filas.append(
            {"tipo": "fila", "objeto": col, "n": int(app[col].eq(v).sum()), "motivo": motivo}
        )
    for col, motivo in FILAS_POR_NULO.items():
        filas.append(
            {"tipo": "fila", "objeto": col, "n": int(app[col].isna().sum()), "motivo": motivo}
        )
    filas.append(
        {
            "tipo": "fila",
            "objeto": "unión, con solapes resueltos",
            "n": int(filas_a_eliminar(app).sum()),
            "motivo": "es lo que de verdad se elimina",
        }
    )
    for col, motivo in columnas_a_eliminar(app).items():
        filas.append({"tipo": "columna", "objeto": col, "n": 1, "motivo": motivo})
    return pd.DataFrame(filas)
