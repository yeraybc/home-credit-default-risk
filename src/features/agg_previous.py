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
NUMERICAS_ORIGEN: tuple[str, ...] = (
    "DAYS_DECISION",
    "CNT_PAYMENT",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "RATE_DOWN_PAYMENT",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "HOUR_APPR_PROCESS_START",
)
COLUMNAS_ORIGEN: tuple[str, ...] = (
    "SK_ID_CURR",
    "NAME_CLIENT_TYPE",
    "NAME_CONTRACT_STATUS",
    "NAME_CONTRACT_TYPE",
    "CODE_REJECT_REASON",
    "PRODUCT_COMBINATION",
    "NAME_TYPE_SUITE",
    "NAME_CASH_LOAN_PURPOSE",
    *NUMERICAS_ORIGEN,
)

# Los cortes que lee la agregación, y solo esos: uno del registro que aún no se consume pasaría la
# guarda de `cortes` y se ignoraría en silencio. Crece con cada punto que añade una feature con
# corte, y un test lo cruza con `CORTES_POR_FEATURE`.
CORTES: tuple[str, ...] = (
    "prev_ventana_reciente_dias",
    "suelo_anios_denominador",
    "prev_plazo_largo_cuotas",
    "prev_sobreconcesion_corte",
    "prev_adelanto_liquidacion_dias",
    "prev_hora_temprana_max",
    "prev_finalidades_urgentes",
    "prev_count_cola",
    "prev_actividad_12m_cola",
    "prev_relacion_larga_anios",
)

# El tipo de cliente que delata solicitudes anteriores a la ventana de la tabla.
TIPOS_RECURRENTES: tuple[str, ...] = ("Repeater", "Refreshed")

# El canal de captación que la combinación de producto lleva escrito, y las dos etiquetas con las
# que la tabla dice que la finalidad no se declaró. Son etiquetas del dato, sin nada que refijar.
CAPTACION_CALLE = "Street"
FINALIDAD_NO_DECLARADA: tuple[str, ...] = ("XNA", "XAP")

# Las features que no se agregan sobre todas las solicitudes del cliente sino sobre las que la
# variable tiene sentido; fuera de ese denominador la solicitud no cuenta, y sin ninguna dentro
# la feature es NaN aunque el cliente tenga previas. Es el `DENOMINADOR` del notebook 04 más el
# coste implícito, que allí no figuraba y también se mide sobre un subconjunto.
DENOMINADOR: dict[str, str] = {
    "PREV_CREDIT_APPLICATION_RATIO": "las dos cifras positivas",
    "PREV_OVERGRANTED_RATIO": "las dos cifras positivas",
    "PREV_CNT_PAYMENT_MEAN": "plazo positivo",
    "PREV_DOWN_PAYMENT_RATE_MEAN": "solo consumo",
    "PREV_IMPLIED_COST_MEAN": "aprobadas con cuota, plazo y crédito positivos",
    "PREV_EARLY_SETTLED_FLAG": "con operación terminada, las dos fechas de fin",
    "PREV_FUTURE_DUE_MAX": "fin previsto por vencer",
    "PREV_STREET_RATIO": "combinación de producto definida",
    "PREV_URGENT_PURPOSE_RATIO": "finalidad declarada",
}

# Las poblaciones sobre las que la receta midió cada feature, como máscara del frame que devuelve
# `unir_previous()`, como en `agg_bureau.py`. None es la población del propio evaluador. Las dos
# últimas salen del NaN de su feature, que es su denominador: la finalidad es la declarada y no
# la de la lista de urgentes.
POBLACIONES = {
    "global": None,
    "no nulos (auto-cond.)": None,
    "con previas": lambda d: d["HAS_PREV_APPLICATION"].eq(1),
    "con algún rechazo": lambda d: d["PREV_REFUSED_COUNT"].gt(0),
    "con operación terminada": lambda d: d["PREV_EARLY_SETTLED_FLAG"].notna(),
    "con finalidad informada": lambda d: d["PREV_URGENT_PURPOSE_RATIO"].notna(),
}

# Lo que la agregación añade y la receta no tiene, con su motivo. Es la lista que el bloque 5
# tiene que repartir en buckets.
COLUMNAS_SIN_RECETA: dict[str, str] = {
    "PREV_RELACION_CORTA_ACTIVA": (
        "el término de la interacción de la relación corta con la actividad alta, el pendiente 7 "
        "de la auditoría transversal: la receta conserva PREV_ACTIVIDAD_12M_COLA solo para él"
    ),
}


def fin_de_ventana(p: pd.DataFrame) -> pd.Series:
    """La fecha de la última solicitud del cliente, repetida en cada una de sus filas.

    Es el fin de ventana propio de cada cliente, el equivalente aquí del último mes reportado de
    cada crédito en `bureau_balance`. Lo usa la recencia relativa del 4.10.
    """
    return p.groupby("SK_ID_CURR")["DAYS_DECISION"].transform("max")


def solicitudes_recientes(
    p: pd.DataFrame, ventana: float, origen: pd.Series | None = None
) -> pd.Series:
    """Máscara de las solicitudes decididas dentro de la ventana, con el borde fuera.

    Vive aquí y no en el refijado del 4.8 porque la cola de actividad se barre sobre la misma
    población que después cuenta la feature: escrita dos veces, un `>=` en una de las dos elige
    el corte sobre un conteo distinto del que entra en la matriz.

    Sin `origen` la ventana se cuenta desde hoy, que es lo que entra en la matriz. Con él se
    cuenta hacia atrás desde esa fecha por fila, que es la ventana relativa del 4.10: el mismo
    borde y la misma comparación, y por eso no se escribe dos veces.
    """
    desde = 0 if origen is None else origen
    return p["DAYS_DECISION"] - desde > -ventana


def cociente_de_concesion(p: pd.DataFrame) -> pd.Series:
    """Lo concedido entre lo solicitado, solo donde las dos cifras son positivas.

    Con la solicitud a 0 y el crédito positivo el cociente sería infinito, y con el crédito a 0
    daría 0: ninguno es un dato. Pasa las dos cifras a `float` para no depender del tipo del lote,
    que con `float32` mueve de lado las filas que caen justo en un corte. Lo usan la agregación y
    el refijado de `prev_sobreconcesion_corte`, que tienen que barrer la misma población.
    """
    solicitado, concedido = p["AMT_APPLICATION"].astype(float), p["AMT_CREDIT"].astype(float)
    return (concedido / solicitado).where(solicitado.gt(0) & concedido.gt(0))


def finalidad_declarada(p: pd.DataFrame) -> pd.Series:
    """Máscara de las solicitudes con finalidad declarada, que es el denominador de la urgente.

    La finalidad solo existe donde el cliente la declara: ni el NaN ni las dos etiquetas con las
    que la tabla dice que no se declaró cuentan. La comparten la agregación y el refijado de
    `prev_finalidades_urgentes`, que tienen que medir sobre la misma población.
    """
    finalidad = p["NAME_CASH_LOAN_PURPOSE"]
    return finalidad.notna() & ~finalidad.isin(FINALIDAD_NO_DECLARADA)


def lectura_solo_vivas(prev: pd.DataFrame) -> pd.Series:
    """El fin previsto por cliente, solo de las operaciones sin fin efectivo (vivas).

    Es la lectura que el 5.6 compara contra `PREV_FUTURE_DUE_MAX`, que cuenta también las ya
    liquidadas. Llama a `limpiar_previous()`, así que es idempotente sobre un frame ya limpio y no
    depende de que quien la llame ya lo haya limpiado. No crea columna en la agregación, igual que
    `lecturas_relativas_previous()` del 4.10: solo informa.
    """
    p = limpiar_previous(prev)
    previsto = p["DAYS_LAST_DUE_1ST_VERSION"]
    viva = previsto.where(previsto.gt(0) & p["DAYS_LAST_DUE"].isna())
    return viva.groupby(p["SK_ID_CURR"]).max()


def combinacion_definida(p: pd.DataFrame) -> pd.Series:
    """Máscara de las solicitudes con combinación de producto, que es el denominador de la calle.

    Sin combinación es el registro fantasma y va aparte de la calle. La comparten la agregación y
    el informe del denominador, que tienen que contar la misma población.
    """
    return p["PRODUCT_COMBINATION"].notna()


def agregar_previous(prev: pd.DataFrame, cortes: dict[str, float] | None = None) -> pd.DataFrame:
    """Una fila por cliente con solicitudes previas, indexada por `SK_ID_CURR`.

    Llama a `limpiar_previous()` antes de agregar, que es idempotente. `cortes` sustituye a los
    de `params.py` para los contrastes; sin él, las dos colas, la sobreconcesión y la lista de
    finalidades urgentes, que son `medido`, revientan hasta que se refijan sobre train.

    `PREV_REFUSED_LONG_TERM_FLAG` marca el plazo largo en una solicitud **no aprobada**, Canceled
    incluida, como la midió el notebook: con solo las Refused los 78 marcados de train son 59.

    `PREV_EARLY_SETTLED_FLAG` es NaN sin ninguna operación terminada, y no 0 como en el notebook:
    se midió sobre los que tienen alguna. `PREV_FUTURE_DUE_MAX` lee el fin **previsto**, como el
    notebook, así que cuenta también las operaciones ya liquidadas antes de ese fin: es el plan
    pactado y no la deuda viva (84.673 de las 224.392 solicitudes por vencer ya se cerraron).

    `PREV_URGENT_PURPOSE_RATIO` es una proporción y nunca un conteo.
    `PREV_PHANTOM_FLAG` vale 0 en quien tiene previas y ninguna sin combinación de producto.

    `PREV_ACTIVIDAD_12M_COLA` se conserva solo como término de `PREV_RELACION_CORTA_ACTIVA`:
    fijado el ritmo anual, su efecto propio desaparece.
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
    rechazada = p["NAME_CONTRACT_STATUS"].eq("Refused")
    concesion = cociente_de_concesion(p)
    # lo que se devuelve frente a lo que se recibe, solo donde hay algo que devolver
    valida = (
        p["NAME_CONTRACT_STATUS"].eq("Approved")
        & p["AMT_ANNUITY"].gt(0)
        & p["CNT_PAYMENT"].gt(0)
        & p["AMT_CREDIT"].gt(0)
    )
    coste = (p["AMT_ANNUITY"] * p["CNT_PAYMENT"] / p["AMT_CREDIT"]).where(valida)
    definida = combinacion_definida(p)
    declarada = finalidad_declarada(p)
    # el fin previsto frente al efectivo; sin el segundo la operación no ha terminado
    previsto = p["DAYS_LAST_DUE_1ST_VERSION"]
    adelanto = previsto - p["DAYS_LAST_DUE"]
    # booleanas y no int8 por lo mismo que en bureau: su suma sale siempre en int64
    f = p.assign(
        _concesion=concesion,
        # proporción y no bandera: el corte se barrió sobre la proporción; el borde queda fuera
        _sobreconcedida=concesion.gt(c["prev_sobreconcesion_corte"])
        .astype(float)
        .where(concesion.notna()),
        _coste=coste,
        _entrada=p["RATE_DOWN_PAYMENT"].where(p["NAME_CONTRACT_TYPE"].eq("Consumer loans")),
        _plazo=p["CNT_PAYMENT"].where(p["CNT_PAYMENT"].gt(0)),
        _reciente=solicitudes_recientes(p, c["prev_ventana_reciente_dias"]),
        _recurrente_en_la_primera=primera & p["NAME_CLIENT_TYPE"].isin(TIPOS_RECURRENTES),
        _rechazada=rechazada,
        _scofr=rechazada & p["CODE_REJECT_REASON"].eq("SCOFR"),
        _plazo_largo=p["CNT_PAYMENT"].gt(c["prev_plazo_largo_cuotas"])
        & p["NAME_CONTRACT_STATUS"].ne("Approved"),
        _liquidada=adelanto.gt(c["prev_adelanto_liquidacion_dias"])
        .astype(float)
        .where(adelanto.notna()),
        _por_vencer=previsto.where(previsto.gt(0)),
        _calle=p["PRODUCT_COMBINATION"]
        .str.contains(CAPTACION_CALLE, na=False)
        .astype(float)
        .where(definida),
        _temprana=p["HOUR_APPR_PROCESS_START"].le(c["prev_hora_temprana_max"]),
        _sin_acompanante=p["NAME_TYPE_SUITE"].isna(),
        _urgente=p["NAME_CASH_LOAN_PURPOSE"]
        .isin(c["prev_finalidades_urgentes"])
        .astype(float)
        .where(declarada),
        _fantasma=~definida,
    )
    # el motivo va como bandera y no como conteo: está a cero en casi todos
    agregado = f.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_CURR", "size"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_COUNT_12M=("_reciente", "sum"),
        PREV_HISTORIAL_RECORTADO=("_recurrente_en_la_primera", "max"),
        PREV_REFUSED_COUNT=("_rechazada", "sum"),
        PREV_REFUSED_RATIO=("_rechazada", "mean"),
        PREV_REFUSED_SCOFR_FLAG=("_scofr", "max"),
        PREV_REFUSED_LONG_TERM_FLAG=("_plazo_largo", "max"),
        PREV_CREDIT_APPLICATION_RATIO=("_concesion", "mean"),
        PREV_OVERGRANTED_RATIO=("_sobreconcedida", "mean"),
        PREV_CNT_PAYMENT_MEAN=("_plazo", "mean"),
        PREV_IMPLIED_COST_MEAN=("_coste", "mean"),
        PREV_DOWN_PAYMENT_RATE_MEAN=("_entrada", "mean"),
        PREV_EARLY_SETTLED_FLAG=("_liquidada", "max"),
        PREV_FUTURE_DUE_MAX=("_por_vencer", "max"),
        PREV_STREET_RATIO=("_calle", "mean"),
        PREV_EARLY_HOUR_RATIO=("_temprana", "mean"),
        PREV_NO_SUITE_RATIO=("_sin_acompanante", "mean"),
        PREV_URGENT_PURPOSE_RATIO=("_urgente", "mean"),
        PREV_PHANTOM_FLAG=("_fantasma", "max"),
    )
    # el conteo está censurado por la ventana de ocho años de la tabla: se normaliza por los años
    # de relación, con suelo para no dividir por casi cero
    anios = -agregado["PREV_DAYS_DECISION_MIN"] / valor("dias_por_anio")
    agregado["PREV_APPLICATIONS_PER_YEAR"] = agregado["PREV_APPLICATION_COUNT"] / anios.clip(
        lower=c["suelo_anios_denominador"]
    )
    agregado["PREV_COUNT_COLA"] = agregado["PREV_APPLICATION_COUNT"] >= c["prev_count_cola"]
    activa = agregado["PREV_COUNT_12M"] >= c["prev_actividad_12m_cola"]
    agregado["PREV_ACTIVIDAD_12M_COLA"] = activa
    # la relación corta y la actividad alta se refuerzan más de lo que suman: el término explícito
    agregado["PREV_RELACION_CORTA_ACTIVA"] = activa & (anios < c["prev_relacion_larga_anios"])
    # las banderas a int8 como en las otras dos agregaciones: el esquema no puede depender del
    # lote, y un bool queda en object en cuanto el lote trae algún NaN
    banderas = agregado.select_dtypes("bool").columns
    agregado[banderas] = agregado[banderas].astype("int8")
    return agregado


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
