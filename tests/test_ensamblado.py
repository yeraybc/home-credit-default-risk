"""Tests del ensamblado de capa 1 de las tres auxiliares (punto 5.2).

Todos sobre un fixture sintético de cuatro clientes, así que corren en CI sin los CSV, salvo la
puerta contra el dato real del final, que se salta sin las tres tablas, `application_train.csv`
y el split, y solo corre en local.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_bureau_balance import unir_bureau_balance
from src.features.build_features import (
    CORTES_AUXILIARES,
    cargar_cortes,
    ensamblar_auxiliares,
    preparar_application,
)
from src.features.cleaning import COL_MONEDA
from src.features.params import fijar_operativo, parametro
from src.features.split import NOMBRE_FICHERO, cargar_split

# Valores cualesquiera, solo para que los ocho dejen de estar sin fijar: el 5.1 ya prueba que el
# refijado y la persistencia dan los correctos, aquí solo hace falta que existan.
CORTES_SINTETICOS = {
    "bureau_enddate_tramo_min_anios": 2.0,
    "bureau_enddate_tramo_max_anios": 5.0,
    "bureau_count_cola": 2,
    "bb_many_credits_corte": 2,
    "prev_count_cola": 2,
    "prev_actividad_12m_cola": 2,
    "prev_sobreconcesion_corte": 1.1,
    "prev_finalidades_urgentes": ("Car repairs",),
}


@pytest.fixture(autouse=True)
def _cortes_fijados():
    """Deja los ocho fijados con un valor cualquiera, salvo que `dato_real` (module-scoped, y por
    tanto ya instanciado para esta llamada) los haya fijado ya con los de verdad: ahí no se pisan.
    """
    for nombre, v in CORTES_SINTETICOS.items():
        if parametro(nombre).valor_operativo is None:
            fijar_operativo(nombre, v, n_train=100)


# un crédito nacional, activo, de consumo, sin cuota ni mora, con las columnas de origen mínimas
BASE_CREDITO = {
    "CREDIT_ACTIVE": "Active",
    "CREDIT_TYPE": "Consumer credit",
    COL_MONEDA: "currency 1",
    "DAYS_CREDIT": -1000,
    "CREDIT_DAY_OVERDUE": 0,
    "DAYS_CREDIT_ENDDATE": 100.0,
    "DAYS_ENDDATE_FACT": np.nan,
    "DAYS_CREDIT_UPDATE": -30.0,
    "CNT_CREDIT_PROLONG": 0,
    "AMT_CREDIT_SUM": 100_000.0,
    "AMT_CREDIT_SUM_DEBT": 50_000.0,
    "AMT_CREDIT_SUM_LIMIT": np.nan,
    "AMT_CREDIT_SUM_OVERDUE": 0.0,
    "AMT_CREDIT_MAX_OVERDUE": np.nan,
    "AMT_ANNUITY": np.nan,
}


def credito(cliente, id_bureau, **campos):
    return {"SK_ID_CURR": cliente, "SK_ID_BUREAU": id_bureau, **BASE_CREDITO, **campos}


# una solicitud aprobada, mediodía, con acompañante y sin finalidad declarada
BASE_SOLICITUD = {
    "NAME_CLIENT_TYPE": "New",
    "NAME_CONTRACT_STATUS": "Approved",
    "CODE_REJECT_REASON": "XAP",
    "CNT_PAYMENT": 12.0,
    "AMT_APPLICATION": 100.0,
    "AMT_CREDIT": 100.0,
    "AMT_ANNUITY": 10.0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "RATE_DOWN_PAYMENT": np.nan,
    "DAYS_LAST_DUE_1ST_VERSION": np.nan,
    "DAYS_LAST_DUE": np.nan,
    "PRODUCT_COMBINATION": "Cash X-Sell: low",
    "HOUR_APPR_PROCESS_START": 12.0,
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_CASH_LOAN_PURPOSE": "XAP",
    "DAYS_DECISION": -100,
}


def solicitud(cliente, **campos):
    return {"SK_ID_CURR": cliente, **BASE_SOLICITUD, **campos}


@pytest.fixture
def base():
    """Cuatro clientes: con las tres fuentes, con solo bureau, con solo previas, y sin nada."""
    return pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4]})


@pytest.fixture
def bureau():
    """Clientes 1 y 2 con un crédito cada uno; 3 y 4 no aparecen."""
    return pd.DataFrame([credito(1, 1001), credito(2, 1002)])


@pytest.fixture
def bb():
    """Panel mensual solo del crédito 1001, del cliente 1. El 1002 del cliente 2 queda huérfano
    de panel, que es la rama de bureau sin histórico mensual."""
    return pd.DataFrame(
        [
            {"SK_ID_BUREAU": 1001, "MONTHS_BALANCE": -2, "STATUS": "0"},
            {"SK_ID_BUREAU": 1001, "MONTHS_BALANCE": -1, "STATUS": "1"},
            {"SK_ID_BUREAU": 1001, "MONTHS_BALANCE": 0, "STATUS": "0"},
        ]
    )


@pytest.fixture
def prev():
    """Clientes 1 y 3 con una solicitud cada uno; 2 y 4 no aparecen."""
    return pd.DataFrame([solicitud(1), solicitud(3)])


@pytest.fixture
def ensamblado(base, bureau, bb, prev):
    return ensamblar_auxiliares(base, bureau, bb, prev)


def test_el_fixture_ejercita_cada_rama(ensamblado):
    """Guardián: si el fixture pierde un cliente, los tests de abajo dejan de probar la rama que
    dicen probar."""
    por_cliente = ensamblado.set_index("SK_ID_CURR")
    assert por_cliente.loc[1, "HAS_BUREAU_HISTORY"] == 1, "falta el cliente con las tres fuentes"
    assert por_cliente.loc[1, "HAS_BUREAU_BALANCE"] == 1, "falta el cliente con panel mensual"
    assert por_cliente.loc[1, "HAS_PREV_APPLICATION"] == 1, "falta el cliente con previas"
    assert (
        por_cliente.loc[2, "HAS_BUREAU_HISTORY"] == 1
        and por_cliente.loc[2, "HAS_BUREAU_BALANCE"] == 0
    ), "falta el cliente con bureau y sin panel mensual"
    assert (
        por_cliente.loc[3, "HAS_BUREAU_HISTORY"] == 0
        and por_cliente.loc[3, "HAS_PREV_APPLICATION"] == 1
    ), "falta el cliente con solo previas"
    assert (
        por_cliente.loc[4, ["HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE", "HAS_PREV_APPLICATION"]]
        .eq(0)
        .all()
    ), "falta el cliente sin ninguna fuente"


# --- forma y población ---------------------------------------------------------------------


def test_la_forma_es_la_base_mas_las_79_de_las_auxiliares(base, ensamblado):
    assert ensamblado.shape == (len(base), base.shape[1] + 79)
    assert list(ensamblado["SK_ID_CURR"]) == list(base["SK_ID_CURR"])


def test_las_79_del_ensamblado_son_exactamente_las_del_reparto_de_buckets(base, ensamblado):
    """El guardián del 5.3: una columna nueva de una agregación que nadie reparta se caería de
    la matriz en silencio, porque `verificar_contrato_columnas()` la rechaza y `remainder="drop"`
    se la lleva. Corre en CI, sin CSV, contra el fixture sintético del ensamblado; el fixture no
    trae las columnas de `application_train`, así que aquí se cruzan solo las 79 y no el
    contrato completo, que exige también esas."""
    from src.features.pipeline import BANDERAS_AUX, COL_TRAYECTORIA, NUMERICAS_AUX, PRESENCIA_AUX

    nuevas = set(ensamblado.columns) - set(base.columns)
    declaradas = set(NUMERICAS_AUX) | set(BANDERAS_AUX) | set(PRESENCIA_AUX) | {COL_TRAYECTORIA}

    assert nuevas == declaradas, (
        f"sin repartir: {sorted(nuevas - declaradas)}; declaradas sin producir: "
        f"{sorted(declaradas - nuevas)}"
    )


def test_el_indice_es_unico(ensamblado):
    assert ensamblado.index.is_unique


def test_ningun_nan_de_capa1_sale_rellenado(ensamblado):
    sin_historial = ensamblado.loc[ensamblado["HAS_BUREAU_HISTORY"] == 0]
    columnas_bureau = [c for c in ensamblado.columns if c.startswith("BUREAU_")]
    assert sin_historial[columnas_bureau].isna().all().all()


# --- el orden de las uniones ------------------------------------------------------------------


def test_el_orden_de_las_uniones_es_bureau_bb_previous(base, bureau, bb, prev, monkeypatch):
    """bureau_balance necesita BUREAU_OVERDUE_UNION, que solo trae unir_bureau(): invertir el
    orden revienta en vez de dejar la mora sin marcar."""
    from src.features import build_features as mod

    orden = []
    for nombre in ("unir_bureau", "unir_bureau_balance", "unir_previous"):
        original = getattr(mod, nombre)

        def espia(*args, _orig=original, _nombre=nombre, **kwargs):
            orden.append(_nombre)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(mod, nombre, espia)

    mod.ensamblar_auxiliares(base, bureau, bb, prev)
    assert orden == ["unir_bureau", "unir_bureau_balance", "unir_previous"]


def test_unir_bureau_balance_sin_bureau_antes_revienta(base, bb):
    from src.features.agg_bureau_balance import agregar_bureau_balance

    puente = pd.Series(
        [1], index=pd.Index([1001], name="SK_ID_BUREAU", dtype="int64"), name="SK_ID_CURR"
    )
    agregado = agregar_bureau_balance(bb, puente)
    with pytest.raises(ValueError, match="BUREAU_OVERDUE_UNION"):
        unir_bureau_balance(base, agregado)


def test_una_fila_de_mas_en_una_union_revienta(base, bureau, bb, prev, monkeypatch):
    """El `_verificar_union` caza lo que el `validate='m:1'` de cada `unir_*` no ve: una lista de
    clientes de entrada con un duplicado."""
    from src.features import build_features as mod

    original = mod.unir_previous

    def infla(clientes, agregado):
        unido = original(clientes, agregado)
        return pd.concat([unido, unido.iloc[[0]]])

    monkeypatch.setattr(mod, "unir_previous", infla)
    with pytest.raises(ValueError, match="cambió el número de filas"):
        mod.ensamblar_auxiliares(base, bureau, bb, prev)


def test_aplicarlo_dos_veces_revienta_por_columnas_solapadas(base, bureau, bb, prev):
    unido = ensamblar_auxiliares(base, bureau, bb, prev)
    with pytest.raises(ValueError, match="overlap"):
        ensamblar_auxiliares(unido, bureau, bb, prev)


# --- el patrón 12, el esquema fijo en absoluto -------------------------------------------------

BANDERAS_HAS = {"HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE", "HAS_PREV_APPLICATION"}
CATEGORICAS = {"BB_TRAJECTORY"}


def test_el_esquema_de_las_79_es_fijo_en_absoluto(base, ensamblado):
    nuevas = [c for c in ensamblado.columns if c not in base.columns]
    assert len(nuevas) == 79
    for c in nuevas:
        if c in BANDERAS_HAS:
            assert ensamblado[c].dtype == "int8", c
        elif c in CATEGORICAS:
            assert isinstance(ensamblado[c].dtype, pd.CategoricalDtype), c
        else:
            assert ensamblado[c].dtype == "float64", c


def test_el_esquema_es_el_mismo_con_ausentes_que_sin_ninguno(base, bureau, bb, prev, ensamblado):
    """Comparar dos lotes y no solo declarar el esquema a mano: es lo que dejó pasar que
    `bureau` perdiera el int8 de sus banderas en los dos lotes a la vez."""
    base_completa = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    completo = ensamblar_auxiliares(base_completa, bureau, bb, prev)
    comunes = [c for c in ensamblado.columns if c in completo.columns and c != "SK_ID_CURR"]
    pd.testing.assert_series_equal(
        ensamblado[comunes].dtypes.sort_index(), completo[comunes].dtypes.sort_index()
    )


@pytest.mark.parametrize("dtype", ["uint32", "int64"])
def test_sk_id_curr_da_lo_mismo_venga_como_venga_de_los_cargadores(base, bureau, bb, prev, dtype):
    convertido = ensamblar_auxiliares(
        base.astype({"SK_ID_CURR": dtype}),
        bureau.astype({"SK_ID_CURR": dtype}),
        bb,
        prev.astype({"SK_ID_CURR": dtype}),
    )
    referencia = ensamblar_auxiliares(base, bureau, bb, prev)
    pd.testing.assert_frame_equal(
        convertido.reset_index(drop=True), referencia.reset_index(drop=True), check_dtype=False
    )


# --- la guarda de cortes sin fijar --------------------------------------------------------------


def test_con_los_ocho_cortes_fijados_no_revienta(base, bureau, bb, prev):
    ensamblar_auxiliares(base, bureau, bb, prev)  # no revienta


def test_con_un_corte_sin_fijar_revienta_nombrando_como_arreglarlo(base, bureau, bb, prev):
    from dataclasses import replace

    from src.features import params as params_mod

    nombre = "bureau_count_cola"
    p = parametro(nombre)
    params_mod.PARAMS[nombre] = replace(p, valor_operativo=None, n_train_operativo=None)
    with pytest.raises(ValueError, match="refijar_cortes_auxiliares"):
        ensamblar_auxiliares(base, bureau, bb, prev)


def test_los_ocho_de_cortes_auxiliares_son_los_que_guarda_la_funcion():
    """Guardián: si `CORTES_AUXILIARES` cambia, el guarda de arriba deja de cubrir uno de verdad."""
    assert set(CORTES_AUXILIARES) == set(CORTES_SINTETICOS)


# --- la puerta contra el dato real --------------------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not all(
        (ruta("raw_data") / TABLE_FILES[t]).exists()
        for t in ("bureau", "bureau_balance", "previous_application", "application_train")
    )
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)

# Sobre la población de modelado (307.492, la del split), no la cruda del EDA. Coberturas ya
# fijadas en las puertas de los bloques 2 a 4; aquí se comprueba que el ensamblado las reproduce.
PUERTA = {
    "filas": 307_492,
    "columnas": 180,
    "con bureau": 263_475,
    "sin bureau": 44_017,
    "con historico mensual": 92_220,
    "con previas": 291_041,
    "sin previas": 16_451,
}


@pytest.fixture(scope="module")
def tablas_reales():
    """Lo caro, cacheado por módulo: no toca `PARAMS`, así que no hay nada que `restaurar_params`
    tenga que deshacer entre tests. `cargar_cortes()` sí muta `PARAMS`, y una fijación hecha en el
    setup de un fixture de módulo se ejecuta antes que el snapshot de ese fixture (function-scoped),
    así que quedaría fuera de lo que se restaura y se filtraría a `test_params.py`: por eso va en
    un fixture aparte, de function scope, más abajo."""
    split = cargar_split()
    base = preparar_application()
    bureau = load_table("bureau", reduce_memory=False)
    bb = load_table("bureau_balance")
    prev = load_table("previous_application", reduce_memory=False)
    return base, split, bureau, bb, prev


@pytest.fixture
def dato_real(tablas_reales):
    base, split, bureau, bb, prev = tablas_reales
    cargar_cortes(split, sobrescribir=True)
    return base, ensamblar_auxiliares(base, bureau, bb, prev), bureau, bb, prev


@sin_dato_real
def test_la_puerta_del_5_2_sobre_el_dato_real(dato_real):
    base, matriz, *_ = dato_real
    medido = {
        "filas": len(matriz),
        "columnas": matriz.shape[1],
        "con bureau": int(matriz["HAS_BUREAU_HISTORY"].sum()),
        "sin bureau": int((matriz["HAS_BUREAU_HISTORY"] == 0).sum()),
        "con historico mensual": int(matriz["HAS_BUREAU_BALANCE"].sum()),
        "con previas": int(matriz["HAS_PREV_APPLICATION"].sum()),
        "sin previas": int((matriz["HAS_PREV_APPLICATION"] == 0).sum()),
    }
    assert medido == PUERTA


@sin_dato_real
def test_application_test_conserva_sus_48_744_clientes(dato_real):
    _, _, bureau, bb, prev = dato_real
    test = preparar_application("application_test")
    matriz_test = ensamblar_auxiliares(test, bureau, bb, prev)
    assert len(matriz_test) == 48_744
    assert "TARGET" not in matriz_test.columns
    assert matriz_test.shape[1] == test.shape[1] + 79
