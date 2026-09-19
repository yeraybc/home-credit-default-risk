"""Agregación de previous_application por cliente, capa 1 del pipeline de features.

Traslada la construcción del notebook 04. Es capa 1 por lo mismo que `agg_bureau.py`: no estima
nada, no mira al TARGET y el agregado de un cliente solo depende de sus propias solicitudes, así
que se calcula una vez sobre la tabla entera y se reindexa con `unir_previous()` a la lista de
clientes que toque.

Una diferencia con el notebook: **el recorte del historial no depende del orden de las filas.**
El notebook lee el tipo de cliente de la solicitud más antigua con un `idxmin`, que ante dos
solicitudes el mismo día se queda con la primera fila. Aquí basta con que alguna de las del día
más antiguo sea de recurrente. Sobre la tabla entera empatan 21.951 clientes y solo en uno
discrepan las solicitudes del día, que no está en train.
"""

from __future__ import annotations

import pandas as pd

from src.features.cleaning import limpiar_previous
from src.features.params import valor

# Las columnas de previous_application de las que sale alguna feature. Como en bureau, la
# agregación exige su esquema: una columna ausente en silencio saldría como "sin dato".
NUMERICAS_ORIGEN: tuple[str, ...] = ("DAYS_DECISION",)
COLUMNAS_ORIGEN: tuple[str, ...] = ("SK_ID_CURR", "NAME_CLIENT_TYPE", *NUMERICAS_ORIGEN)

# Los cortes que lee la agregación, y solo esos: uno del registro que aún no se consume pasaría la
# guarda de `cortes` y se ignoraría en silencio. Crece con cada punto que añade una feature con
# corte, y un test lo cruza con `CORTES_POR_FEATURE`.
CORTES: tuple[str, ...] = ("prev_ventana_reciente_dias", "suelo_anios_denominador")

# El tipo de cliente que delata solicitudes anteriores a la ventana de la tabla.
TIPOS_RECURRENTES: tuple[str, ...] = ("Repeater", "Refreshed")


def agregar_previous(prev: pd.DataFrame, cortes: dict[str, float] | None = None) -> pd.DataFrame:
    """Una fila por cliente con solicitudes previas, indexada por `SK_ID_CURR`.

    Llama a `limpiar_previous()` antes de agregar, que es idempotente. `cortes` sustituye a los
    de `params.py` para los contrastes. Los dos que lee hoy son de dominio, así que no hace falta
    refijar nada para construir.
    """
    faltan = [c for c in COLUMNAS_ORIGEN if c not in prev.columns]
    if faltan:
        raise ValueError(f"a previous_application le faltan columnas para agregar: {faltan}")
    # a float en la frontera: las fechas llegan en int16 o float32 según el lote, y un frame
    # vacío sin tipos en object
    prev = prev.astype(dict.fromkeys(NUMERICAS_ORIGEN, float))
    cortes = cortes or {}
    desconocidos = set(cortes) - set(CORTES)
    if desconocidos:
        raise KeyError(f"cortes que ninguna feature de previous usa: {sorted(desconocidos)}")

    c = {n: cortes[n] if n in cortes else valor(n) for n in CORTES}
    p = limpiar_previous(prev)
    decision = p["DAYS_DECISION"]
    primera = decision.eq(decision.groupby(p["SK_ID_CURR"]).transform("min"))
    # booleanas y no int8 por lo mismo que en bureau: su suma sale siempre en int64
    f = p.assign(
        _reciente=decision > -c["prev_ventana_reciente_dias"],
        _recurrente_en_la_primera=primera & p["NAME_CLIENT_TYPE"].isin(TIPOS_RECURRENTES),
    )
    agregado = f.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_CURR", "size"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_COUNT_12M=("_reciente", "sum"),
        PREV_HISTORIAL_RECORTADO=("_recurrente_en_la_primera", "max"),
    )
    # el conteo está censurado por la ventana de ocho años de la tabla: se normaliza por los años
    # de relación, con suelo para no dividir por casi cero
    anios = (-agregado["PREV_DAYS_DECISION_MIN"] / valor("dias_por_anio")).clip(
        lower=c["suelo_anios_denominador"]
    )
    agregado["PREV_APPLICATIONS_PER_YEAR"] = agregado["PREV_APPLICATION_COUNT"] / anios
    # el esquema no puede depender del lote: un frame vacío deja los conteos y la bandera en
    # object o bool
    return agregado.astype(
        {
            "PREV_APPLICATION_COUNT": "int64",
            "PREV_COUNT_12M": "int64",
            "PREV_HISTORIAL_RECORTADO": "int8",
        }
    )


def unir_previous(clientes: pd.DataFrame, agregado: pd.DataFrame) -> pd.DataFrame:
    """Left join del agregado a una lista de clientes, con `HAS_PREV_APPLICATION`.

    No rellena: el cliente sin solicitudes queda con NaN en todo, conteos incluidos. Su grupo es
    de menor riesgo (5,96% frente a 8,19%), el signo contrario al de bureau, y qué se hace con
    ese NaN es de la capa 2.

    El `validate` revienta con un agregado de clientes repetidos, que duplicaría filas en silencio.
    """
    unido = clientes.join(agregado, on="SK_ID_CURR", validate="m:1")
    unido["HAS_PREV_APPLICATION"] = unido["PREV_APPLICATION_COUNT"].notna().astype("int8")
    return unido
