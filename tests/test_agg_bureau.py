"""Tests de la agregación de bureau por cliente (punto 2.2) y del refijado de su tramo (2.3).

Todos sobre un frame sintético, así que corren en CI sin los CSV, salvo la puerta contra el dato
real del final, que se salta sin `bureau.csv` y el split y solo corre en local.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import RAIZ, ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_bureau import (
    COLUMNAS_ORIGEN,
    COLUMNAS_SIN_RECETA,
    CORTES,
    POBLACIONES,
    agregar_bureau,
    unir_bureau,
)
from src.features.build_features import ajustar_tramo_bureau
from src.features.cleaning import COL_MONEDA, limpiar_bureau
from src.features.eval import remedir_receta
from src.features.params import fijar_operativo, parametro, valor
from src.features.recipes import cargar_receta
from src.features.split import NOMBRE_FICHERO, cargar_split

# Los cortes salen del registro y no de literales, por lo mismo que en test_cleaning: dos copias
# del mismo corte dejan el fixture adaptándose en silencio al que cambie. Los `medido` se pasan
# con su referencia porque `valor()` los bloquea hasta que se refijan sobre train, y aquí solo se
# prueba el cómputo.
DIAS = valor("dias_por_anio")
SUELO_FECHA = -valor("bureau_cierre_max_anios") * DIAS
CUOTA_MAX = valor("bureau_cuota_max")
SUELO_ANIOS = valor("suelo_anios_denominador")
REFERENCIA = {
    n: parametro(n).valor_referencia for n in CORTES if parametro(n).procedencia == "medido"
}
UPDATE = valor("bureau_update_reciente_dias")
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


def nivel_de_cuota(agregado):
    """La tripartita: sin cuota reportada, reportada a cero y con valor."""
    nivel = np.select(
        [agregado.HAS_BUREAU_ANNUITY == 0, agregado.BUREAU_CREDITS_WITH_ANNUITY_COUNT == 0],
        ["sin", "cero"],
        "valor",
    )
    return pd.Series(nivel, index=agregado.index)


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
        # 2: solo extranjera, con la cuota reportada a cero y mora activa sin mora máxima: todos
        # sus importes se anulan, y la unión solo la ve por la foto de la mora activa
        credito(
            2,
            **{COL_MONEDA: "currency 3"},
            AMT_ANNUITY=0.0,
            AMT_CREDIT_MAX_OVERDUE=0.0,
            AMT_CREDIT_SUM_OVERDUE=200.0,
        ),
        credito(2, **{COL_MONEDA: "currency 4"}),
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
        # 9: dos filas que encienden las mismas banderas, que siguen a 1 y no cuentan filas; la
        # segunda es una tarjeta cerrada, que no es tarjeta activa
        credito(
            9,
            AMT_CREDIT_MAX_OVERDUE=100.0,
            AMT_CREDIT_SUM_OVERDUE=50.0,
            AMT_CREDIT_SUM_LIMIT=-10.0,
        ),
        credito(
            9,
            CREDIT_ACTIVE="Closed",
            CREDIT_TYPE="Credit card",
            AMT_CREDIT_MAX_OVERDUE=100.0,
            AMT_CREDIT_SUM_OVERDUE=50.0,
            AMT_CREDIT_SUM_LIMIT=-10.0,
        ),
        # 10: huérfano, en bureau y en ninguna lista de clientes
        credito(10),
        # 11: solo una tarjeta, así que ningún vencimiento a término, con el límite reportado y
        # sin deuda: tiene detalle financiero solo por el límite
        credito(
            11, CREDIT_TYPE="Credit card", AMT_CREDIT_SUM_LIMIT=0.0, AMT_CREDIT_SUM_DEBT=np.nan
        ),
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
    # sin estos, un sum en vez de un max, una tarjeta cerrada contada como activa o un relleno
    # del vencimiento a término pasan en verde
    detalle = bureau.AMT_CREDIT_SUM_LIMIT.notna() | bureau.AMT_CREDIT_SUM_DEBT.notna()
    filas_que_encienden = {
        "moneda extranjera": extranjera,
        "mora máxima": bureau.AMT_CREDIT_MAX_OVERDUE > 0,
        "mora activa": bureau.AMT_CREDIT_SUM_OVERDUE > 0,
        "sobregiro": bureau.AMT_CREDIT_SUM_LIMIT < 0,
        "detalle financiero": detalle,
        "historial de mora": bureau.AMT_CREDIT_MAX_OVERDUE.notna(),
        "cuota reportada": bureau.AMT_ANNUITY.notna(),
    }
    for nombre, filas in filas_que_encienden.items():
        assert filas.groupby(bureau.SK_ID_CURR).sum().max() >= 2, f"falta el doble de {nombre}"
    assert (tarjeta & bureau.CREDIT_ACTIVE.ne("Active")).any(), "falta la tarjeta no activa"
    solo_tarjetas = tarjeta.groupby(bureau.SK_ID_CURR).all()
    assert solo_tarjetas.any(), "falta el cliente sin ningún vencimiento a término"
    limite_sin_deuda = bureau.AMT_CREDIT_SUM_LIMIT.notna() & bureau.AMT_CREDIT_SUM_DEBT.isna()
    assert limite_sin_deuda.any(), "falta el detalle financiero solo por el límite"
    activa_sin_maxima = (bureau.AMT_CREDIT_SUM_OVERDUE > 0) & ~(bureau.AMT_CREDIT_MAX_OVERDUE > 0)
    assert (extranjera & activa_sin_maxima).any(), "falta la mora activa extranjera sin máxima"


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
        # la proporción es de créditos con cuota positiva, no de cuotas reportadas
        "BUREAU_ANNUITY_ACTIVE_RATIO": 0.0,
        # su única mora es activa y extranjera: leída del importe limpio, la unión la perdería
        "BUREAU_HAS_CURRENT_OVERDUE": 1,
        "BUREAU_OVERDUE_UNION": 1,
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
        # cinco créditos de dos tipos: cuenta tipos, no filas
        "BUREAU_CREDIT_TYPE_NUNIQUE": 2,
    },
    8: {"BUREAU_DAYS_CREDIT_UPDATE_FLAG": np.nan, "BUREAU_CLOSED_COUNT": 1},
    9: {
        "BUREAU_ACTIVE_CARD_COUNT": 0,
        "BUREAU_CLOSED_COUNT": 1,
        # la magnitud suma las dos filas; las banderas no, y eso lo fija el test de binarias
        "BUREAU_CURRENT_OVERDUE_SUM": 100.0,
        "BUREAU_MAX_OVERDUE_EVER": 100.0,
    },
    11: {
        # sin ningún crédito a término no hay vencimiento: NaN, ni 0 ni una mediana de la tabla
        "BUREAU_DAYS_CREDIT_ENDDATE_MAX": np.nan,
        "BUREAU_ENDDATE_2_5Y_COUNT": 0,
        "HAS_BUREAU_FINANCIAL_DETAIL": 1,
        "BUREAU_NEGATIVE_LIMIT_FLAG": 0,
        "BUREAU_DEBT_CREDIT_RATIO": np.nan,
    },
}


@pytest.mark.parametrize("cliente", sorted(ESPERADO))
def test_el_agregado_de_cada_cliente_es_el_calculado_a_mano(bureau, cliente):
    fila = agregar(bureau).loc[cliente]
    for col, esperado in ESPERADO[cliente].items():
        assert fila[col] == pytest.approx(esperado, nan_ok=True), f"{col}: {fila[col]}"


def test_cada_cliente_agregado_solo_da_lo_mismo_que_acompanado(bureau):
    """La premisa que deja a la agregación correr fuera del split: no cruza clientes.

    Caza un estadístico de la tabla entera solo si algún cliente lo deja ver: una normalización
    por el máximo del lote la ve cualquiera, y una mediana que rellena el vencimiento a término
    solo el cliente 11, que no tiene ninguno. Sin él pasaba en verde.
    """
    juntos = agregar(bureau)
    for cliente in bureau.SK_ID_CURR.unique():
        solo = agregar(bureau[bureau.SK_ID_CURR == cliente])
        pd.testing.assert_frame_equal(solo, juntos.loc[[cliente]])


def test_la_tripartita_de_la_cuota_tiene_sus_tres_niveles(bureau):
    """Sin cuota reportada, reportada a cero y con valor: el nulo y el cero no son lo mismo."""
    niveles = nivel_de_cuota(agregar(bureau))
    assert niveles.loc[[2, 4, 1]].tolist() == ["cero", "sin", "valor"]


def test_la_particion_de_conteos_por_estado_cuadra_con_el_total(bureau):
    agregado = agregar(bureau)
    suma = (
        agregado.BUREAU_ACTIVE_COUNT + agregado.BUREAU_CLOSED_COUNT + agregado.BUREAU_BAD_DEBT_COUNT
    )
    pd.testing.assert_series_equal(suma, agregado.BUREAU_LOAN_COUNT, check_names=False)


# El tipo de cada columna, que es el contrato que el bloque 5 reparte en buckets; el resto sale en
# float64. Fijado en absoluto y no solo entre dos lotes: una bandera que pierde el int8 lo pierde
# en todos los lotes a la vez, y compararlos entre sí no lo ve.
BANDERAS = {
    "BUREAU_HAS_ANY_OVERDUE",
    "BUREAU_HAS_CURRENT_OVERDUE",
    "BUREAU_HAS_FOREIGN_CURRENCY",
    "BUREAU_NEGATIVE_LIMIT_FLAG",
    "BUREAU_OVERDUE_UNION",
    "HAS_BEEN_PROLONGED",
    "HAS_BUREAU_ANNUITY",
    "HAS_BUREAU_FINANCIAL_DETAIL",
    "HAS_BUREAU_OVERDUE_HISTORY",
}
CONTEOS = {
    "BUREAU_LOAN_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_BAD_DEBT_COUNT",
    "BUREAU_ACTIVE_CARD_COUNT",
    "BUREAU_ACTIVE_CONSUMER_COUNT",
    "BUREAU_CLOSED_AFTER_ENDDATE",
    "BUREAU_CREDITS_WITH_ANNUITY_COUNT",
    "BUREAU_CREDIT_TYPE_NUNIQUE",
    "BUREAU_ENDDATE_2_5Y_COUNT",
}


def test_la_salida_tiene_el_esquema_declarado(bureau):
    agregado = agregar(bureau)
    esperado = {
        c: "int8" if c in BANDERAS else "int64" if c in CONTEOS else "float64"
        for c in agregado.columns
    }
    assert agregado.dtypes.astype(str).to_dict() == esperado


def test_las_banderas_valen_cero_o_uno(bureau):
    """Una bandera es el max de su auxiliar: con dos filas que la encienden sigue a 1.

    Recorre las declaradas y no las `int8` de la salida, que serían ninguna si las banderas
    perdiesen el tipo.
    """
    agregado = agregar(bureau)
    for col in BANDERAS | {"BUREAU_DAYS_CREDIT_UPDATE_FLAG"}:
        assert set(agregado[col].dropna()) <= {0, 1}, f"{col}: {sorted(agregado[col].unique())}"


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
    with pytest.raises(ValueError, match="not unique"):
        unir_bureau(pd.DataFrame({"SK_ID_CURR": [5, 1]}), duplicado)


def test_las_poblaciones_son_exactamente_las_de_la_receta(bureau):
    """Ninguna etiqueta de lo provisional sin máscara, ninguna máscara que no use nadie, y ninguna
    población condicionada deja entrar a un cliente sin historial."""
    receta = cargar_receta("bureau")["features"]
    assert {f["poblacion_medicion"] for f in receta if f["firmeza"] == "provisional"} == set(
        POBLACIONES
    )
    clientes = pd.DataFrame({"SK_ID_CURR": [99, *bureau.SK_ID_CURR.unique()]})
    unido = unir_bureau(clientes, agregar(bureau))
    for etiqueta, mascara in POBLACIONES.items():
        if mascara is not None:
            dentro = mascara(unido)
            assert dentro.dtype == bool and dentro.any(), etiqueta
            assert not dentro.iloc[0], f"{etiqueta} incluye al cliente sin historial"


# --- el refijado del tramo sobre train -------------------------------------------------------

TRAMO = ("bureau_enddate_tramo_min_anios", "bureau_enddate_tramo_max_anios")


def escenario(*grupos):
    """Cada grupo es (años de vencimiento, targets, parte, tipo[, créditos por cliente])."""
    filas, clientes = [], []
    for anios, targets, parte, tipo, *creditos in grupos:
        for target in targets:
            cliente = len(clientes) + 1
            clientes.append({"SK_ID_CURR": cliente, "TARGET": target, "split": parte})
            fila = credito(cliente, DAYS_CREDIT_ENDDATE=anios * DIAS, CREDIT_TYPE=tipo)
            filas += [fila] * (creditos[0] if creditos else 1)
    return pd.DataFrame(filas), pd.DataFrame(clientes)


def refijar(*grupos, sobrescribir=False):
    """El tramo que sale de refijar sobre el escenario, y el informe."""
    bureau, clientes = escenario(*grupos)
    informe = ajustar_tramo_bureau(bureau, clientes, clientes, sobrescribir)
    return tuple(valor(n) for n in TRAMO), informe


# el pico en 5 a 10 años (67%), por encima de 0 a 2 (25%) y 2 a 5 (33%), y dos grupos que lo
# llevarían a 2 a 5 si contaran
PICO = [
    (-1, [0, 0], "train", "Consumer credit"),
    (1, [0, 0, 0, 1], "train", "Consumer credit"),
    (3, [0, 0, 1], "train", "Consumer credit"),
    (7, [1, 1, 0], "train", "Consumer credit"),
]
VALID = (3, [1] * 10, "valid", "Consumer credit")
TARJETAS = (3, [1] * 10, "train", "Credit card")


def test_el_refijado_toma_los_extremos_del_tramo_pico():
    tramo, informe = refijar(*PICO)
    assert tramo == (5, 10)
    assert informe.pico.sum() == 1
    assert informe.index[informe.pico][0].left == 5


@pytest.mark.parametrize("fuera", [VALID, TARJETAS], ids=["valid", "tarjetas"])
def test_ni_valid_ni_las_tarjetas_mueven_el_pico(fuera):
    """Las dos direcciones: el grupo no cuenta, y el mismo grupo contado sí movería el pico."""
    assert refijar(*PICO, fuera)[0] == (5, 10)
    contado = (fuera[0], fuera[1], "train", "Consumer credit")
    assert refijar(*PICO, contado, sobrescribir=True)[0] == (2, 5)


def test_n_train_son_los_clientes_de_train_con_historial():
    """Clientes y no créditos, las tarjetas cuentan aunque no tengan vencimiento a término, y ni
    valid ni el cliente de train sin historial suman."""
    bureau, clientes = escenario(*PICO, TARJETAS, VALID)
    segundo_credito = bureau.iloc[[0]]
    sin_historial = pd.DataFrame([{"SK_ID_CURR": 999, "TARGET": 1, "split": "train"}])
    clientes = pd.concat([clientes, sin_historial], ignore_index=True)
    ajustar_tramo_bureau(pd.concat([bureau, segundo_credito]), clientes, clientes)
    assert {parametro(n).n_train_operativo for n in TRAMO} == {12 + 10}


def test_refijar_otra_vez_exige_sobrescribir():
    refijar(*PICO)
    with pytest.raises(ValueError, match="sobrescribir"):
        refijar(*PICO)
    assert refijar(*PICO, sobrescribir=True)[0] == (5, 10)


def test_un_pico_de_0_a_2_anios_es_futuro_y_se_acepta():
    """El borde del 0 no revienta: vencer dentro de dos años es vencimiento futuro."""
    assert refijar(*PICO, (1, [1] * 10, "train", "Consumer credit"))[0] == (0, 2)


def test_un_pico_en_un_vencimiento_pasado_revienta():
    with pytest.raises(ValueError, match="ya pasado"):
        refijar(*PICO, (-1, [1] * 10, "train", "Consumer credit"))


# --- la puerta contra el dato real -----------------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["bureau"]).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)

# La puerta del bloque 2 sobre sus dos poblaciones: la tabla cruda, que es la del EDA, y la de
# modelado, que es la del split. Las presencias son las de la receta, idénticas porque se leen de
# la foto; el ratio pasó de 262.408 a 255.109 por la limpieza y el min_count.
PUERTA = {
    "crudo": {
        "con historial": 263_491,
        "sin historial": 44_020,
        "ratio de deuda no nulo": 255_109,
        "BUREAU_NEGATIVE_LIMIT_FLAG": 323,
        "BUREAU_HAS_ANY_OVERDUE": 70_440,
        "BUREAU_OVERDUE_UNION": 72_419,
        "BUREAU_HAS_CURRENT_OVERDUE": 3_334,
        "HAS_BUREAU_FINANCIAL_DETAIL": 256_134,
        "HAS_BUREAU_OVERDUE_HISTORY": 183_886,
        "HAS_BUREAU_ANNUITY": 80_009,
        "cuota sin": 183_482,
        "cuota cero": 19_737,
        "cuota valor": 60_272,
    },
    "modelado": {
        "con historial": 263_475,
        "sin historial": 44_017,
        "ratio de deuda no nulo": 255_094,
        "BUREAU_NEGATIVE_LIMIT_FLAG": 323,
        "BUREAU_HAS_ANY_OVERDUE": 70_435,
        "BUREAU_OVERDUE_UNION": 72_413,
        "BUREAU_HAS_CURRENT_OVERDUE": 3_332,
        "HAS_BUREAU_FINANCIAL_DETAIL": 256_119,
        "HAS_BUREAU_OVERDUE_HISTORY": 183_877,
        "HAS_BUREAU_ANNUITY": 79_998,
        "cuota sin": 183_477,
        "cuota cero": 19_736,
        "cuota valor": 60_262,
    },
}


@pytest.fixture(scope="module")
def dato_real():
    bureau = load_table("bureau")
    poblaciones = {
        "crudo": load_table("application_train", usecols=["SK_ID_CURR"]).SK_ID_CURR,
        "modelado": cargar_split().SK_ID_CURR,
    }
    return bureau, agregar(bureau), poblaciones


@sin_dato_real
@pytest.mark.parametrize("poblacion", sorted(PUERTA))
def test_la_puerta_del_bloque_sobre_el_dato_real(dato_real, poblacion):
    _, agregado, poblaciones = dato_real
    unido = unir_bureau(pd.DataFrame({"SK_ID_CURR": poblaciones[poblacion]}), agregado)
    con = unido[unido.HAS_BUREAU_HISTORY == 1]
    presencias = [c for c in PUERTA[poblacion] if c.isupper()]
    medido = {
        "con historial": len(con),
        "sin historial": len(unido) - len(con),
        "ratio de deuda no nulo": int(con.BUREAU_DEBT_CREDIT_RATIO.notna().sum()),
        **{c: int(con[c].sum()) for c in presencias},
        **{f"cuota {k}": int(n) for k, n in nivel_de_cuota(con).value_counts().items()},
    }
    assert medido == PUERTA[poblacion]


@sin_dato_real
def test_sobre_el_dato_real_la_particion_cuadra_en_todos_los_clientes(dato_real):
    _, agregado, _ = dato_real
    estados = ["BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT", "BUREAU_BAD_DEBT_COUNT"]
    assert len(agregado) == 305_811
    assert (agregado[estados].sum(axis=1) == agregado.BUREAU_LOAN_COUNT).all()


@sin_dato_real
def test_sobre_el_dato_real_cada_cliente_solo_da_lo_mismo_que_acompanado(dato_real):
    """El fila a fila, sobre 300 clientes al azar y el único sin ninguna actualización válida."""
    bureau, agregado, _ = dato_real
    muestra = [*np.random.default_rng(0).choice(agregado.index, 300, replace=False), 170_304]
    trozo = bureau[bureau.SK_ID_CURR.isin(muestra)]
    for cliente, filas in trozo.groupby("SK_ID_CURR"):
        pd.testing.assert_frame_equal(agregar(filas), agregado.loc[[cliente]])


def train_real(dato_real):
    """Los clientes de train con su TARGET, y las filas de bureau que son suyas."""
    split = cargar_split()
    train = split.loc[split.split == "train", ["SK_ID_CURR", "TARGET"]]
    bureau = dato_real[0]
    return train, bureau[bureau.SK_ID_CURR.isin(train.SK_ID_CURR)]


@sin_dato_real
def test_sobre_el_split_el_tramo_refijado_sigue_en_2_a_5_anios(dato_real):
    """El pico sigue en 2 a 5 años, con 10,09% frente a 8,37% y 8,31% a los lados."""
    split = cargar_split()
    informe = ajustar_tramo_bureau(dato_real[0], split, split)
    assert tuple(valor(n) for n in TRAMO) == (2, 5)
    assert parametro(TRAMO[0]).n_train_operativo == 210_875
    assert (informe.tasa * 100).round(2).tolist() == [5.75, 6.48, 7.69, 8.37, 10.09, 8.31, 5.29]


@sin_dato_real
def test_contraste_de_la_ventana_de_actualizacion(dato_real):
    """La señal no depende de los 180 días, que por eso son dominio y no un corte medido.

    Fila a fila la tasa baja sin saltos del 9,35% al 5,77%, y la bandera separa con cualquier
    ventana de 90 a 730 días, de +2,62pp a +1,91pp.
    """
    train, bureau = train_real(dato_real)
    filas = limpiar_bureau(bureau).merge(train, on="SK_ID_CURR")
    antiguedad = pd.cut(-filas.DAYS_CREDIT_UPDATE / DIAS, [-np.inf, 0.5, 1, 2, 3, 5, np.inf])
    tasas = filas.TARGET.groupby(antiguedad, observed=False).mean()
    assert tasas.notna().sum() == 6, "sin los seis tramos, el all() de abajo sería vacuo"
    assert tasas.diff().dropna().lt(0).all()
    for dias in (90, 180, 365, 730):
        cortes = {**REFERENCIA, "bureau_update_reciente_dias": dias}
        unido = unir_bureau(train, agregar_bureau(bureau, cortes))
        bandera = unido.BUREAU_DAYS_CREDIT_UPDATE_FLAG
        assert unido.TARGET[bandera == 1].mean() > unido.TARGET[bandera == 0].mean(), dias


@sin_dato_real
def test_contraste_del_suelo_de_medio_anio(dato_real):
    """El gradiente del ritmo anual es monótono con suelos de 0,25 a 1 año.

    Con el de medio año va del 6,16% al 15,93%. Con 2 años se rompe en el primer tramo, porque el
    suelo mete los historiales cortos, que son los de más riesgo, entre los de ritmo bajo.
    """
    train, bureau = train_real(dato_real)
    for suelo in (0.25, 0.5, 1.0):
        cortes = {**REFERENCIA, "suelo_anios_denominador": suelo}
        unido = unir_bureau(train, agregar_bureau(bureau, cortes))
        con = unido[unido.HAS_BUREAU_HISTORY == 1]
        ritmo = pd.cut(con.BUREAU_CREDITS_PER_YEAR, [0, 0.5, 1, 2, 3, np.inf], include_lowest=True)
        tasas = con.TARGET.groupby(ritmo, observed=False).mean()
        assert tasas.notna().sum() == 5, suelo
        assert tasas.diff().dropna().gt(0).all(), suelo


@sin_dato_real
def test_sobre_el_split_lo_provisional_sigue_en_el_mismo_orden(dato_real):
    """Las 28 provisionales, con cocientes frente a la receta entre 0,85 y 1,31."""
    split = cargar_split()
    ajustar_tramo_bureau(dato_real[0], split, split)
    train, _ = train_real(dato_real)
    unido = unir_bureau(train, agregar_bureau(dato_real[0]))
    tabla = remedir_receta(unido, unido.TARGET, cargar_receta("bureau"), POBLACIONES)
    assert len(tabla) == 28
    # medida sobre la población de la receta: su 80%, con el margen de lo que mueve el split en una
    # bandera rara (0,746 en la de límite negativo). El cociente del efecto no basta, que una
    # máscara que mida el ratio de cuota sobre todos los clientes con historial lo deja en 0,58
    proporcion = tabla.n / tabla.n_receta
    assert proporcion.between(0.7, 0.9).all(), tabla.feature[~proporcion.between(0.7, 0.9)].tolist()
    # eq(True) y no all(): sobre object, un vacío cuenta como verdadero
    assert tabla.mismo_orden.eq(True).all(), tabla.feature[~tabla.mismo_orden.eq(True)].tolist()
    assert tabla.set_index("feature").loc["HAS_BUREAU_HISTORY", "n"] == 210_875
