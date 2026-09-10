"""Agregación de bureau por cliente, capa 1 del pipeline de features.

Traslada el `groupby("SK_ID_CURR").agg(...)` del notebook 02 (la celda 88 cuando se escribió).
Es capa 1 porque no estima nada, no mira al TARGET y **el agregado de un cliente solo depende de
sus propias filas**: se calcula una vez sobre bureau entero y se reindexa con `unir_bureau()` a la
lista de clientes que toque, sea train, valid, `application_test` o un cliente suelto en la API.
Por eso es función pura y no transformer, y queda fuera del `Pipeline`.

Tres diferencias con el notebook, las tres de la regla de NaN de la capa 1:

1. **La presencia y el signo de los importes se leen de la foto `*_SIGNO`**, nunca del importe
   limpio. La limpieza anula importes en moneda extranjera y por encima del cap, y leídas de ahí
   las banderas cambian de valor sin que falle nada: 60 clientes pierden la mora reportada y 27
   cambian de nivel en la tripartita de la cuota, sobre bureau completo.
2. **Las sumas llevan `min_count=1`.** `sum()` da 0 cuando todo es NaN, que es rellenar: en el
   EDA, 7.254 de los 76.030 clientes con ratio de deuda 0 no tenían ni una deuda reportada y se
   leían como deuda cero. Un NaN parcial sí suma como 0, que es la magnitud de lo reportado.
3. **El tope de 20 años del vencimiento no se aplica aquí**, lo aplica `acotar_ventanas_bureau()`
   a nivel fila. Repetirlo sería el mismo corte en dos sitios.

Los dos descartes con `firmeza: firme` de la receta (`BUREAU_MAX_DAYS_OVERDUE` y
`BUREAU_DAYS_ENDDATE_FACT_MIN`) no se construyen, porque son redundancia estructural y valen en
cualquier submuestra. Los provisionales sí, que la capa 2b tiene que poder remedirlos sobre el
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.cleaning import (
    COL_MONEDA,
    COL_MONEDA_EXTRANJERA,
    IMPORTES_BUREAU,
    SUFIJO_SIGNO,
    limpiar_bureau,
)
from src.features.params import CORTES_POR_FEATURE, valor

# Las columnas de bureau de las que sale alguna feature. La agregación es la frontera que produce
# las features de la tabla, así que exige su esquema: una columna ausente en silencio saldría
# como "sin dato" en todas las features que la leen.
CATEGORICAS_ORIGEN: tuple[str, ...] = ("CREDIT_ACTIVE", "CREDIT_TYPE", COL_MONEDA)
NUMERICAS_ORIGEN: tuple[str, ...] = (
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
    "CNT_CREDIT_PROLONG",
    *IMPORTES_BUREAU,
)
COLUMNAS_ORIGEN: tuple[str, ...] = ("SK_ID_CURR", *CATEGORICAS_ORIGEN, *NUMERICAS_ORIGEN)

# Los cortes que lleva dentro alguna feature de la tabla, del mismo registro que usa `params.py`.
CORTES = tuple(sorted({c for cortes in CORTES_POR_FEATURE["bureau"].values() for c in cortes}))

# Lo que la agregación añade y la receta no tiene, con su motivo. Es la lista que el bloque 5
# tiene que repartir en buckets, y ninguna de las dos tiene aún efecto medido.
COLUMNAS_SIN_RECETA: dict[str, str] = {
    "BUREAU_HAS_FOREIGN_CURRENCY": (
        "max de la bandera de fila de la limpieza: algún importe del cliente no se sumó por estar "
        "en otra moneda"
    ),
    "BUREAU_HAS_CURRENT_OVERDUE": (
        "el > 0 con el que la receta usa BUREAU_CURRENT_OVERDUE_SUM, leído de la foto: desde la "
        "magnitud limpia pierde al cliente cuya única mora activa está en otra moneda"
    ),
}


def agregar_bureau(bureau: pd.DataFrame, cortes: dict[str, float] | None = None) -> pd.DataFrame:
    """Una fila por cliente con historial, indexada por `SK_ID_CURR`.

    Llama a `limpiar_bureau()` antes de agregar: es idempotente, así que un frame ya limpio no
    cambia, y uno crudo no puede saltarse el orden de dominio antes de agregar ni leer una foto que
    no existe.

    `cortes` sustituye a los de `params.py`, y los tres `medido` revientan en `valor()` hasta que
    alguien los refija sobre `solo_train()`. Pasarlos a mano es lo que hace el barrido que los
    refija, y el contraste del suelo de medio año, que es de dominio y aun así tiene uno pendiente.
    """
    faltan = [c for c in COLUMNAS_ORIGEN if c not in bureau.columns]
    if faltan:
        raise ValueError(f"a bureau le faltan columnas para agregar: {faltan}")
    # a float en la frontera: un frame vacío construido sin tipos llega en object y el esquema de
    # la salida dependería del de la entrada. De paso, un importe que llegue como texto revienta
    # aquí y no más adelante
    bureau = bureau.astype(dict.fromkeys(NUMERICAS_ORIGEN, float))
    cortes = cortes or {}
    # sin esta guarda, un nombre mal escrito dejaría el contraste del suelo con el valor de
    # params.py sin avisar
    desconocidos = set(cortes) - set(CORTES)
    if desconocidos:
        raise KeyError(f"cortes que ninguna feature de bureau usa: {sorted(desconocidos)}")
    c = {n: cortes[n] if n in cortes else valor(n) for n in CORTES}
    dias = valor("dias_por_anio")
    b = limpiar_bureau(bureau)

    def signo(col: str) -> pd.Series:
        return b[col + SUFIJO_SIGNO]

    activo = b["CREDIT_ACTIVE"].eq("Active")
    cerrado = b["CREDIT_ACTIVE"].eq("Closed")
    tarjeta = b["CREDIT_TYPE"].eq("Credit card")
    # vencimiento solo de créditos a término: sin revolving, y los placeholders ya los anuló la
    # limpieza
    a_termino = b["DAYS_CREDIT_ENDDATE"].where(~tarjeta)
    tramo = (a_termino > c["bureau_enddate_tramo_min_anios"] * dias) & (
        a_termino <= c["bureau_enddate_tramo_max_anios"] * dias
    )
    mora_maxima = signo("AMT_CREDIT_MAX_OVERDUE") > 0
    mora_activa = signo("AMT_CREDIT_SUM_OVERDUE") > 0
    # las auxiliares van en booleano y no en int8: la suma de un int8 vuelve a int8 si el
    # resultado cabe y se queda en int64 si no, así que un cliente de 128 créditos cambiaría el
    # tipo de la columna de todo su lote. La de un booleano sale siempre en int64
    f = b.assign(
        _is_active=activo,
        _is_closed=cerrado,
        _is_bad_debt=b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]),
        _active_card=activo & tarjeta,
        _active_consumer=activo & b["CREDIT_TYPE"].eq("Consumer credit"),
        _closed_late=cerrado & (b["DAYS_ENDDATE_FACT"] > b["DAYS_CREDIT_ENDDATE"]),
        _term_enddate=a_termino,
        _enddate_2_5y=tramo,
        # presencia y signo, de la foto
        _neg_limit=signo("AMT_CREDIT_SUM_LIMIT") < 0,
        _has_annuity=signo("AMT_ANNUITY") > 0,
        _annuity_reported=signo("AMT_ANNUITY").notna(),
        _has_overdue=mora_maxima,
        _has_overdue_hist=signo("AMT_CREDIT_MAX_OVERDUE").notna(),
        _has_fin_detail=signo("AMT_CREDIT_SUM_LIMIT").notna()
        | signo("AMT_CREDIT_SUM_DEBT").notna(),
        _current_overdue=mora_activa,
        _overdue_union=mora_maxima | mora_activa | (b["CREDIT_DAY_OVERDUE"] > 0),
    )
    g = f.groupby("SK_ID_CURR")
    agregado = g.agg(
        BUREAU_LOAN_COUNT=("SK_ID_CURR", "size"),
        BUREAU_ACTIVE_COUNT=("_is_active", "sum"),
        BUREAU_CLOSED_COUNT=("_is_closed", "sum"),
        BUREAU_BAD_DEBT_COUNT=("_is_bad_debt", "sum"),
        BUREAU_CREDIT_TYPE_NUNIQUE=("CREDIT_TYPE", "nunique"),
        BUREAU_MAX_OVERDUE_EVER=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_HAS_ANY_OVERDUE=("_has_overdue", "max"),
        BUREAU_HAS_CURRENT_OVERDUE=("_current_overdue", "max"),
        BUREAU_NEGATIVE_LIMIT_FLAG=("_neg_limit", "max"),
        BUREAU_CREDITS_WITH_ANNUITY_COUNT=("_has_annuity", "sum"),
        BUREAU_ANNUITY_ACTIVE_RATIO=("_has_annuity", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_CLOSED_AFTER_ENDDATE=("_closed_late", "sum"),
        HAS_BUREAU_FINANCIAL_DETAIL=("_has_fin_detail", "max"),
        HAS_BUREAU_OVERDUE_HISTORY=("_has_overdue_hist", "max"),
        HAS_BUREAU_ANNUITY=("_annuity_reported", "max"),
        BUREAU_ACTIVE_CARD_COUNT=("_active_card", "sum"),
        BUREAU_ACTIVE_CONSUMER_COUNT=("_active_consumer", "sum"),
        BUREAU_ENDDATE_2_5Y_COUNT=("_enddate_2_5y", "sum"),
        BUREAU_DAYS_CREDIT_ENDDATE_MAX=("_term_enddate", "max"),
        BUREAU_OVERDUE_UNION=("_overdue_union", "max"),
        BUREAU_HAS_FOREIGN_CURRENCY=(COL_MONEDA_EXTRANJERA, "max"),
        _days_update_max=("DAYS_CREDIT_UPDATE", "max"),
        _prolong_max=("CNT_CREDIT_PROLONG", "max"),
    )
    # las sumas van aparte porque la agregación nombrada no acepta min_count sin un lambda, que
    # sobre 305.811 grupos es una llamada de Python por cliente
    sumas = g[["AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"]].sum(min_count=1)
    agregado["BUREAU_CURRENT_OVERDUE_SUM"] = sumas["AMT_CREDIT_SUM_OVERDUE"]
    agregado["BUREAU_DEBT_CREDIT_RATIO"] = sumas["AMT_CREDIT_SUM_DEBT"] / sumas[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    # el conteo está censurado por la ventana de ocho años del buró: se normaliza por los años
    # observados, con suelo para no dividir por casi cero
    anios = (-agregado["BUREAU_DAYS_CREDIT_MIN"] / dias).clip(lower=c["suelo_anios_denominador"])
    agregado["BUREAU_CREDITS_PER_YEAR"] = agregado["BUREAU_LOAN_COUNT"] / anios
    agregado["HAS_BEEN_PROLONGED"] = agregado.pop("_prolong_max") > 0
    banderas = agregado.select_dtypes("bool").columns
    agregado[banderas] = agregado[banderas].astype("int8")
    # NaN y no 0 cuando no queda ninguna actualización válida, que no es un registro dormido. En
    # float siempre: `where` solo cambia el tipo si hay algún NaN en el lote, y el esquema de un
    # cliente no puede depender de quién le acompañe
    actualizacion = agregado.pop("_days_update_max")
    reciente = actualizacion > -c["bureau_update_reciente_dias"]
    agregado["BUREAU_DAYS_CREDIT_UPDATE_FLAG"] = reciente.astype(float).where(actualizacion.notna())
    return agregado


def unir_bureau(clientes: pd.DataFrame, agregado: pd.DataFrame) -> pd.DataFrame:
    """Left join del agregado a una lista de clientes, con `HAS_BUREAU_HISTORY`.

    No rellena: el cliente sin historial queda con NaN en todo, conteos incluidos, porque su grupo
    es de mayor riesgo (10,12% frente a 7,73%) y un 0 lo mezclaría con quien tiene historial y
    ningún crédito activo. Qué se hace con ese NaN es de la capa 2.

    El `validate` revienta con un agregado de clientes repetidos, que duplicaría filas en silencio.
    """
    unido = clientes.join(agregado, on="SK_ID_CURR", validate="m:1")
    unido["HAS_BUREAU_HISTORY"] = unido["BUREAU_LOAN_COUNT"].notna().astype("int8")
    return unido
