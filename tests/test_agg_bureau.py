"""Tests de la agregación de bureau por cliente (capa 1, punto 2.2).

Todos sobre un frame sintético, así que corren en CI sin los CSV. La puerta contra el dato real
(263.491 y 44.020 sobre la tabla cruda, 263.475 y 44.017 sobre la de modelado, y las presencias
leídas de la foto idénticas a la receta) se reprodujo fuera de la suite al cerrar el punto.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import RAIZ
from src.features.agg_bureau import (
    COLUMNAS_ORIGEN,
    COLUMNAS_SIN_RECETA,
    CORTES,
    agregar_bureau,
    unir_bureau,
)
from src.features.cleaning import COL_MONEDA, limpiar_bureau
from src.features.params import fijar_operativo, parametro, valor

# Los cortes salen del registro y no de literales, por lo mismo que en test_cleaning: dos copias
# del mismo corte dejan el fixture adaptándose en silencio al que cambie. Los `medido` se pasan
# con su referencia porque `valor()` los bloquea hasta el 2.3, y aquí solo se prueba el cómputo.
DIAS = valor("dias_por_anio")
SUELO_FECHA = -valor("bureau_cierre_max_anios") * DIAS
CUOTA_MAX = valor("bureau_cuota_max")
SUELO_ANIOS = valor("suelo_anios_denominador")
REFERENCIA = {
    n: parametro(n).valor_referencia for n in CORTES if parametro(n).procedencia == "medido"
}
UPDATE = REFERENCIA["bureau_update_reciente_dias"]
TRAMO_MIN = REFERENCIA["bureau_enddate_tramo_min_anios"] * DIAS
TRAMO_MAX = REFERENCIA["bureau_enddate_tramo_max_anios"] * DIAS

# un crédito nacional, activo, de consumo, sin cuota ni mora reportadas y fuera del tramo
BASE = {
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


def credito(cliente, **campos):
    return {"SK_ID_CURR": cliente, **BASE, **campos}


def agregar(bureau):
    return agregar_bureau(bureau, REFERENCIA)


@pytest.fixture
def bureau():
    """Un cliente por caso de borde, cada uno con el agregado que se espera de él a mano."""
    filas = [
        # 1: mezcla de monedas; la cuota, la mora activa y el sobregiro de la extranjera cuentan
        # por la foto
        credito(1, AMT_ANNUITY=1_000.0, AMT_CREDIT_SUM_DEBT=40_000.0, AMT_CREDIT_MAX_OVERDUE=0.0),
        credito(
            1,
            **{COL_MONEDA: "currency 2"},
            AMT_ANNUITY=500.0,
            AMT_CREDIT_SUM_OVERDUE=300.0,
            AMT_CREDIT_MAX_OVERDUE=300.0,
            AMT_CREDIT_SUM_LIMIT=-100.0,
            AMT_CREDIT_SUM=9_000.0,
            AMT_CREDIT_SUM_DEBT=9_000.0,
        ),
        # 2: solo extranjera, con la cuota reportada a cero: todos sus importes se anulan
        credito(2, **{COL_MONEDA: "currency 3"}, AMT_ANNUITY=0.0, AMT_CREDIT_MAX_OVERDUE=0.0),
        # 3: cuota rota por encima del cap, ninguna deuda reportada, y un activo con cierre real
        # posterior al vencimiento, que no es cierre tardío porque no está cerrado (435 filas así
        # en la tabla real)
        credito(3, AMT_ANNUITY=2 * CUOTA_MAX, AMT_CREDIT_SUM_DEBT=np.nan),
        credito(
            3,
            CREDIT_TYPE="Car loan",
            AMT_CREDIT_SUM_DEBT=np.nan,
            DAYS_CREDIT_ENDDATE=-100.0,
            DAYS_ENDDATE_FACT=-50.0,
        ),
        # 4: denominador cero con deuda positiva, que sin anular el cero daría infinito
        credito(4, AMT_CREDIT_SUM=0.0, AMT_CREDIT_SUM_DEBT=5_000.0),
        # 5: un solo crédito, historial más corto que el suelo, impagado, días de mora sin
        # importe, y la actualización justo en el corte, que no cuenta como reciente
        credito(
            5,
            CREDIT_ACTIVE="Bad debt",
            DAYS_CREDIT=-100,
            CREDIT_DAY_OVERDUE=10,
            DAYS_CREDIT_UPDATE=-float(UPDATE),
        ),
        # 6: historial largo con Sold, cierre tardío, sobregiro, prórroga, mora activa nacional,
        # y la actualización un día dentro del corte
        credito(6, CREDIT_ACTIVE="Sold", DAYS_CREDIT=-2000, DAYS_CREDIT_UPDATE=-500.0),
        credito(
            6,
            CREDIT_ACTIVE="Closed",
            DAYS_CREDIT_ENDDATE=-200.0,
            DAYS_ENDDATE_FACT=-100.0,
            CNT_CREDIT_PROLONG=1,
            DAYS_CREDIT_UPDATE=-500.0,
        ),
        credito(
            6,
            CREDIT_TYPE="Credit card",
            DAYS_CREDIT_ENDDATE=np.nan,
            AMT_CREDIT_SUM_LIMIT=-5_000.0,
            AMT_CREDIT_SUM_OVERDUE=1_000.0,
            AMT_CREDIT_MAX_OVERDUE=2_000.0,
            DAYS_CREDIT_UPDATE=-float(UPDATE - 1),
        ),
        # 7: los bordes del tramo de vencimiento, y dos tarjetas que no son a término
        credito(7, DAYS_CREDIT_ENDDATE=TRAMO_MIN),
        credito(7, DAYS_CREDIT_ENDDATE=TRAMO_MIN + 1),
        credito(7, DAYS_CREDIT_ENDDATE=TRAMO_MAX),
        credito(7, CREDIT_TYPE="Credit card", DAYS_CREDIT_ENDDATE=TRAMO_MIN + 1),
        credito(7, CREDIT_TYPE="Credit card", DAYS_CREDIT_ENDDATE=6 * DIAS),
        # 8: su única actualización está por debajo del suelo de fechas y la limpieza la anula
        credito(
            8,
            CREDIT_ACTIVE="Closed",
            DAYS_ENDDATE_FACT=-825.0,
            DAYS_CREDIT_UPDATE=SUELO_FECHA - 1,
        ),
        # 10: huérfano, en bureau y en ninguna lista de clientes
        credito(10),
    ]
    return pd.DataFrame(filas)


def test_el_fixture_ejercita_cada_rama(bureau):
    """Guardián: si el fixture pierde un caso, los tests de abajo dejan de probar lo que dicen."""
    extranjera = bureau[COL_MONEDA].ne("currency 1")
    por_cliente = extranjera.groupby(bureau.SK_ID_CURR).agg(["any", "all"])
    assert por_cliente["all"].sum() == 1, "falta el cliente solo en moneda extranjera"
    assert (por_cliente["any"] & ~por_cliente["all"]).sum() == 1, "falta el de mezcla"
    assert (extranjera & (bureau.AMT_ANNUITY > 0) & (bureau.AMT_CREDIT_SUM_OVERDUE > 0)).any()
    assert (extranjera & (bureau.AMT_ANNUITY == 0)).any(), "falta la cuota extranjera a cero"
    assert (bureau.AMT_ANNUITY > CUOTA_MAX).any(), "falta la cuota rota"
    sin_deuda = bureau.AMT_CREDIT_SUM_DEBT.isna().groupby(bureau.SK_ID_CURR).all()
    assert sin_deuda.any(), "falta el cliente sin ninguna deuda reportada"
    cero = (bureau.AMT_CREDIT_SUM == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0)
    assert cero.any(), "falta el denominador cero con deuda"
    assert (extranjera & (bureau.AMT_CREDIT_SUM_LIMIT < 0)).any(), "falta el sobregiro extranjero"
    activo_con_cierre = bureau.CREDIT_ACTIVE.eq("Active") & (
        bureau.DAYS_ENDDATE_FACT > bureau.DAYS_CREDIT_ENDDATE
    )
    assert activo_con_cierre.any(), "falta el activo con cierre posterior al vencimiento"
    assert (bureau.DAYS_CREDIT > -SUELO_ANIOS * DIAS).any(), "falta el historial bajo el suelo"
    assert set(bureau.CREDIT_ACTIVE) == {"Active", "Closed", "Sold", "Bad debt"}
    ultima = bureau.groupby("SK_ID_CURR").DAYS_CREDIT_UPDATE.max()
    assert (ultima == -UPDATE).any() and (ultima == -(UPDATE - 1)).any(), "faltan los bordes"
    assert (bureau.DAYS_CREDIT_UPDATE < SUELO_FECHA).any(), "falta la actualización rota"
    fin = bureau.DAYS_CREDIT_ENDDATE
    assert (fin == TRAMO_MIN).any() and (fin == TRAMO_MAX).any(), "faltan los bordes del tramo"
    tarjeta = bureau.CREDIT_TYPE.eq("Credit card")
    assert (tarjeta & (fin > TRAMO_MIN) & (fin <= TRAMO_MAX)).any(), "falta la tarjeta en tramo"
    assert ((bureau.CREDIT_DAY_OVERDUE > 0) & (bureau.AMT_CREDIT_SUM_OVERDUE == 0)).any()
    assert (bureau.AMT_CREDIT_SUM_LIMIT < 0).any() and (bureau.CNT_CREDIT_PROLONG > 0).any()


def test_cada_columna_de_la_salida_varia_entre_clientes(bureau):
    """Una columna constante en el fixture es una columna que ningún test está midiendo."""
    agregado = agregar(bureau)
    constantes = [c for c in agregado.columns if agregado[c].nunique(dropna=False) < 2]
    assert not constantes, f"el fixture no hace variar: {constantes}"


# El agregado que se espera de cada cliente, calculado a mano desde el fixture. Es la única fuente
# de los valores esperados: los comentarios dicen qué propiedad fija cada uno.
ESPERADO = {
    1: {
        "BUREAU_LOAN_COUNT": 2,
        # la presencia y el signo se leen de la foto: la cuota, la mora y el sobregiro de la fila
        # extranjera cuentan aunque la limpieza anule sus importes. Leídas del importe limpio, 27
        # clientes cambiaban de nivel en la tripartita y 60 perdían la mora sobre la tabla real
        "BUREAU_CREDITS_WITH_ANNUITY_COUNT": 2,
        "BUREAU_ANNUITY_ACTIVE_RATIO": 1.0,
        "BUREAU_HAS_ANY_OVERDUE": 1,
        "BUREAU_NEGATIVE_LIMIT_FLAG": 1,
        # el > 0 de la receta leído de la magnitud perdería a este cliente: su suma nacional es 0
        "BUREAU_HAS_CURRENT_OVERDUE": 1,
        "BUREAU_CURRENT_OVERDUE_SUM": 0.0,
        "BUREAU_MAX_OVERDUE_EVER": 0.0,
        "BUREAU_OVERDUE_UNION": 1,
        "BUREAU_HAS_FOREIGN_CURRENCY": 1,
        # un NaN parcial suma como 0: queda la deuda nacional sobre el crédito nacional
        "BUREAU_DEBT_CREDIT_RATIO": 0.4,
    },
    2: {
        # reportada a cero, el nivel protector de la tripartita
        "HAS_BUREAU_ANNUITY": 1,
        "BUREAU_CREDITS_WITH_ANNUITY_COUNT": 0,
        "HAS_BUREAU_FINANCIAL_DETAIL": 1,
        "HAS_BUREAU_OVERDUE_HISTORY": 1,
        "BUREAU_HAS_ANY_OVERDUE": 0,
        # sum() rellena con 0 lo que es todo NaN, que en el EDA leía como deuda cero a 7.254
        "BUREAU_CURRENT_OVERDUE_SUM": np.nan,
        "BUREAU_MAX_OVERDUE_EVER": np.nan,
        "BUREAU_DEBT_CREDIT_RATIO": np.nan,
    },
    3: {
        # la cuota rota por encima del cap sigue contando: el cap dice que no se sabe cuánto vale,
        # no que no se reportara
        "BUREAU_CREDITS_WITH_ANNUITY_COUNT": 1,
        "BUREAU_ANNUITY_ACTIVE_RATIO": 0.5,
        "BUREAU_CREDIT_TYPE_NUNIQUE": 2,
        "HAS_BUREAU_FINANCIAL_DETAIL": 0,
        "HAS_BUREAU_OVERDUE_HISTORY": 0,
        "BUREAU_CLOSED_AFTER_ENDDATE": 0,
        "BUREAU_DEBT_CREDIT_RATIO": np.nan,
    },
    4: {"BUREAU_DEBT_CREDIT_RATIO": np.nan, "HAS_BUREAU_ANNUITY": 0},
    5: {
        "BUREAU_BAD_DEBT_COUNT": 1,
        "BUREAU_ACTIVE_COUNT": 0,
        "BUREAU_CREDITS_PER_YEAR": 1 / SUELO_ANIOS,
        "BUREAU_DAYS_CREDIT_UPDATE_FLAG": 0,
        "BUREAU_OVERDUE_UNION": 1,
        "BUREAU_HAS_ANY_OVERDUE": 0,
    },
    6: {
        "BUREAU_LOAN_COUNT": 3,
        "BUREAU_ACTIVE_COUNT": 1,
        "BUREAU_CLOSED_COUNT": 1,
        "BUREAU_BAD_DEBT_COUNT": 1,
        "BUREAU_ACTIVE_CARD_COUNT": 1,
        "BUREAU_ACTIVE_CONSUMER_COUNT": 0,
        "BUREAU_CLOSED_AFTER_ENDDATE": 1,
        "BUREAU_NEGATIVE_LIMIT_FLAG": 1,
        "HAS_BEEN_PROLONGED": 1,
        "BUREAU_HAS_CURRENT_OVERDUE": 1,
        "BUREAU_CURRENT_OVERDUE_SUM": 1_000.0,
        "BUREAU_MAX_OVERDUE_EVER": 2_000.0,
        "BUREAU_DAYS_CREDIT_UPDATE_FLAG": 1,
        "BUREAU_DAYS_CREDIT_MIN": -2000,
        "BUREAU_DAYS_CREDIT_MAX": -1000,
        "BUREAU_CREDITS_PER_YEAR": 3 / (2000 / DIAS),
        "BUREAU_DEBT_CREDIT_RATIO": 0.5,
    },
    7: {
        "BUREAU_ENDDATE_2_5Y_COUNT": 2,
        "BUREAU_DAYS_CREDIT_ENDDATE_MAX": TRAMO_MAX,
        "BUREAU_ACTIVE_CARD_COUNT": 2,
    },
    8: {"BUREAU_DAYS_CREDIT_UPDATE_FLAG": np.nan, "BUREAU_CLOSED_COUNT": 1},
}


@pytest.mark.parametrize("cliente", sorted(ESPERADO))
def test_el_agregado_de_cada_cliente_es_el_calculado_a_mano(bureau, cliente):
    fila = agregar(bureau).loc[cliente]
    for col, esperado in ESPERADO[cliente].items():
        assert fila[col] == pytest.approx(esperado, nan_ok=True), f"{col}: {fila[col]}"


def test_cada_cliente_agregado_solo_da_lo_mismo_que_acompanado(bureau):
    """La premisa que deja a la agregación correr fuera del split: no cruza clientes.

    Comprobado metiendo un estadístico de la tabla entera dentro de la agregación: lo caza.
    """
    juntos = agregar(bureau)
    for cliente in bureau.SK_ID_CURR.unique():
        solo = agregar(bureau[bureau.SK_ID_CURR == cliente])
        pd.testing.assert_frame_equal(solo, juntos.loc[[cliente]])


def test_la_tripartita_de_la_cuota_tiene_sus_tres_niveles(bureau):
    """Sin cuota reportada, reportada a cero y con valor: el nulo y el cero no son lo mismo."""
    agregado = agregar(bureau)
    nivel = np.select(
        [
            agregado.HAS_BUREAU_ANNUITY == 0,
            agregado.BUREAU_CREDITS_WITH_ANNUITY_COUNT == 0,
        ],
        ["sin", "cero"],
        "valor",
    )
    niveles = pd.Series(nivel, index=agregado.index)
    assert niveles.loc[[2, 4, 1]].tolist() == ["cero", "sin", "valor"]


def test_la_particion_de_conteos_por_estado_cuadra_con_el_total(bureau):
    agregado = agregar(bureau)
    suma = (
        agregado.BUREAU_ACTIVE_COUNT + agregado.BUREAU_CLOSED_COUNT + agregado.BUREAU_BAD_DEBT_COUNT
    )
    pd.testing.assert_series_equal(suma, agregado.BUREAU_LOAN_COUNT, check_names=False)


def test_el_tipo_de_cada_columna_no_depende_del_volumen_del_cliente():
    """La suma de un int8 vuelve a int8 si cabe y se queda en int64 si no.

    Con las auxiliares en int8, un cliente de 128 créditos cambiaría el tipo de la columna de
    todo su lote, y el esquema no puede depender de quién venga en él.
    """
    uno = agregar(pd.DataFrame([credito(1)]))
    muchos = agregar(pd.DataFrame([credito(1)] * 130))
    assert muchos.loc[1, "BUREAU_ACTIVE_COUNT"] == 130
    pd.testing.assert_series_equal(uno.dtypes, muchos.dtypes)


def test_la_salida_cumple_el_contrato_con_la_receta(bureau):
    """Las columnas son las de la receta menos los descartes firmes, más las declaradas sin receta.

    `HAS_BUREAU_HISTORY` no sale de aquí sino de `unir_bureau()`, que es donde existe el cliente
    sin historial. Al exigir el conjunto exacto deja fuera la moneda, su bandera de fila, las fotos
    y las auxiliares.
    """
    receta = yaml.safe_load((RAIZ / "config" / "bureau_features.yaml").read_text())["features"]
    nombres = {f["nombre"] for f in receta}
    firmes = {f["nombre"] for f in receta if f["firmeza"] == "firme"}
    assert firmes == {"BUREAU_MAX_DAYS_OVERDUE", "BUREAU_DAYS_ENDDATE_FACT_MIN"}
    esperadas = (nombres - firmes - {"HAS_BUREAU_HISTORY"}) | set(COLUMNAS_SIN_RECETA)
    agregado = agregar(bureau)
    assert set(agregado.columns) == esperadas
    assert agregado.index.name == "SK_ID_CURR" and agregado.index.is_unique
    for motivo in COLUMNAS_SIN_RECETA.values():
        assert motivo.strip()


def test_sin_cortes_revienta_mientras_los_medidos_esten_sin_fijar(bureau):
    with pytest.raises(ValueError, match="sin fijar"):
        agregar_bureau(bureau)


def test_con_los_cortes_fijados_sobre_train_no_hace_falta_pasarlos(bureau):
    for nombre, referencia in REFERENCIA.items():
        fijar_operativo(nombre, referencia, n_train=1)
    pd.testing.assert_frame_equal(agregar_bureau(bureau), agregar(bureau))


def test_un_corte_que_ninguna_feature_usa_revienta(bureau):
    with pytest.raises(KeyError, match="bureau_update_reciente"):
        agregar_bureau(bureau, {**REFERENCIA, "bureau_update_reciente": 90})


def test_agregar_el_crudo_y_el_limpio_da_lo_mismo(bureau):
    """Limpia dentro y la limpieza es idempotente, así que el orden de dominio no se salta."""
    pd.testing.assert_frame_equal(agregar(bureau), agregar(limpiar_bureau(bureau)))


def test_no_muta_el_frame_de_entrada(bureau):
    antes = bureau.copy()
    agregar(bureau)
    pd.testing.assert_frame_equal(bureau, antes)


@pytest.mark.parametrize("sin_tipos", [False, True], ids=["recorte", "sin_tipos"])
def test_un_frame_vacio_da_un_agregado_vacio_con_el_mismo_esquema(bureau, sin_tipos):
    """Es lo que se agrega para un cliente sin historial: ni revienta ni cambia el esquema.

    Sobre un frame vacío pandas decide los tipos por su cuenta (`patrones-de-fallo` §9). El
    construido sin tipos, que es lo probable desde la API, trae todas las columnas en object, y
    sin pasar las numéricas a float siete columnas de la salida salían en object.
    """
    lleno = agregar(bureau)
    vacio = agregar(pd.DataFrame([], columns=bureau.columns) if sin_tipos else bureau.iloc[:0])
    assert vacio.empty
    pd.testing.assert_series_equal(vacio.dtypes, lleno.dtypes)
    unido = unir_bureau(pd.DataFrame({"SK_ID_CURR": [99]}), vacio)
    assert unido.HAS_BUREAU_HISTORY.tolist() == [0]
    assert unido[lleno.columns].isna().all().all()


def test_un_importe_que_no_es_numero_revienta_en_la_frontera(bureau):
    roto = bureau.astype({"AMT_ANNUITY": object})
    roto.loc[0, "AMT_ANNUITY"] = "mil"
    with pytest.raises(ValueError, match="mil"):
        agregar(roto)


@pytest.mark.parametrize("columna", COLUMNAS_ORIGEN)
def test_una_columna_de_origen_ausente_revienta_con_su_nombre(bureau, columna):
    with pytest.raises(ValueError, match=columna):
        agregar(bureau.drop(columns=columna))


# --- unir_bureau -----------------------------------------------------------------------------


def test_unir_conserva_filas_orden_e_indice_y_no_rellena(bureau):
    """El 99 no tiene historial: NaN en todo, conteos incluidos, y la bandera a 0.

    El índice se conserva para que la predicción se pueda alinear de vuelta con la petición, y el
    huérfano del agregado no añade ninguna fila.
    """
    agregado = agregar(bureau)
    clientes = pd.DataFrame({"SK_ID_CURR": [99, 5, 1]}, index=[7, 8, 9])
    unido = unir_bureau(clientes, agregado)
    assert unido.index.tolist() == [7, 8, 9]
    assert unido.SK_ID_CURR.tolist() == [99, 5, 1]
    assert unido.HAS_BUREAU_HISTORY.tolist() == [0, 1, 1]
    assert unido.loc[7, agregado.columns].isna().all()
    assert unido.loc[[8, 9], "BUREAU_LOAN_COUNT"].tolist() == [1, 2]


def test_unir_revienta_si_el_agregado_trae_un_cliente_repetido(bureau):
    agregado = agregar(bureau)
    duplicado = pd.concat([agregado, agregado.loc[[5]]])
    with pytest.raises(ValueError, match="repetidos"):
        unir_bureau(pd.DataFrame({"SK_ID_CURR": [5, 1]}), duplicado)
