"""Tests de la agregación de bureau_balance a nivel crédito y cliente (puntos 3.2 a 3.7).

Todos sobre un panel sintético, así que corren en CI sin los CSV, salvo la puerta contra el dato
real del final, que se salta sin `bureau_balance.csv`, `bureau.csv` y `application_train.csv` y
solo corre en local.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats.contingency import association

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_bureau import CORTES as CORTES_BUREAU
from src.features.agg_bureau import agregar_bureau, unir_bureau
from src.features.agg_bureau_balance import (
    BANDERAS_TRAYECTORIA,
    COLUMNAS_ORIGEN,
    COLUMNAS_SIN_RECETA,
    CORTES_CLIENTE,
    TRAYECTORIAS,
    agregar_bureau_balance,
    agregar_por_credito,
    puente_credito_cliente,
    unir_bureau_balance,
)
from src.features.cleaning import (
    DTYPE_STATUS,
    SUFIJO_SIGNO,
    limpiar_bureau,
    limpiar_bureau_balance,
)
from src.features.params import parametro
from src.features.recipes import cargar_receta
from src.features.split import NOMBRE_FICHERO, cargar_split

# Un crédito por caso de borde, con los meses en orden de más antiguo a más reciente. El agregado
# que se espera de cada uno está calculado a mano en `ESPERADO`.
PANEL = {
    # 1: sin ningún estado numérico mezclando cierre y ausencia, que es donde el sin dato se leía
    # como sin mora
    1: (-2, ["C", "X", "C"]),
    # 2: el 100% en X y además censurado, el perfil de los 85.569 de la tabla entera
    2: (-6, ["X", "X"]),
    # 3: un solo mes reportado, el que rompe todo denominador, y encima sale en mora
    3: (0, ["1"]),
    # 4: mora en dos meses no consecutivos, para que la recencia sea el último impago y no el
    # primero, y para que la intensidad sea una suma y no un máximo
    4: (-4, ["1", "0", "2", "0", "C"]),
    # 5: censurado que se despide en mora, que es lo que separa la censura del estado final
    5: (-3, ["0", "5"]),
    # 6: con meses reportados y ninguna mora, la referencia que exige el 0 donde el ciego lleva NaN
    6: (-2, ["X", "0", "0"]),
    # Del 8 al 15, la trayectoria: todos con la ventana mínima. El 4 hace de borde que no
    # entra, con 5 meses y las dos mitades reportadas, y el 7 lo usa un test de la guarda.
    # 8: sin mora con la ventana justa de 6, el borde que sí entra
    8: (-5, ["0", "0", "X", "0", "0", "C"]),
    # 9: la única mora en el mes central de una ventana impar, que va a la mitad antigua; si fuera
    # a la reciente saldría empeora
    9: (-6, ["0", "0", "0", "1", "0", "0", "0"]),
    # 10: mejora por severidad sin salir de mora, censurado y en el borde antiguo del panel, donde
    # `ini + fin` ya no cabe en el int8 del mes y un punto medio sumado se desbordaría. El 4 es
    # además el único estado grave que no es el fallido, que es lo que separa `== 5` de `>= 4`
    10: (-95, ["4", "4", "4", "1", "1", "1"]),
    # 11: empeora
    11: (-5, ["0", "0", "0", "0", "1", "2"]),
    # 12: estable, con el mismo peor estado en las dos mitades
    12: (-7, ["1", "0", "0", "0", "0", "0", "1", "X"]),
    # 13: la reciente toda C, que el notebook leía como 0 y clasificaba de mejora
    13: (-5, ["1", "0", "0", "C", "C", "C"]),
    # 14: la antigua toda X, que el notebook clasificaba de empeora
    14: (-5, ["X", "X", "X", "0", "1", "0"]),
    # 15: sin mora, censurado y cerrado; en el cliente 500 deja la peor trayectoria en medio y no al
    # final, que es lo que separa el max de un last
    15: (-11, ["0", "0", "0", "0", "0", "C"]),
    # 16: fallido que después se cierra, lo que separa el fallido por severidad del último estado
    16: (-3, ["5", "C", "C"]),
}

# Los seis primeros y el 16 no llegan a la ventana mínima, así que su trayectoria es NaN aunque sus
# mitades tengan valor
ESPERADO = {
    1: {
        "BB_MONTHS_OBS": 3, "BB_MONTHS_REPORTED": 0, "BB_WORST": np.nan, "BB_PCT_X": 1 / 3,
        "BB_WINDOW_INI": -2, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": np.nan, "BB_PCT_DPD": np.nan,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": np.nan, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": np.nan, "BB_WORST_RECENT_HALF": np.nan, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    2: {
        "BB_MONTHS_OBS": 2, "BB_MONTHS_REPORTED": 0, "BB_WORST": np.nan, "BB_PCT_X": 1.0,
        "BB_WINDOW_INI": -6, "BB_WINDOW_END": -5, "BB_DPD_MONTHS": np.nan, "BB_PCT_DPD": np.nan,
        "BB_LAST_STATUS": "X", "BB_LAST_DPD_MONTH": np.nan, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": np.nan, "BB_WORST_RECENT_HALF": np.nan, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    3: {
        "BB_MONTHS_OBS": 1, "BB_MONTHS_REPORTED": 1, "BB_WORST": 1.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": 0, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 1.0,
        "BB_LAST_STATUS": "1", "BB_LAST_DPD_MONTH": 0.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 1.0, "BB_WORST_RECENT_HALF": np.nan, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    4: {
        "BB_MONTHS_OBS": 5, "BB_MONTHS_REPORTED": 4, "BB_WORST": 2.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -4, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 2.0, "BB_PCT_DPD": 0.5,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": -2.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 2.0, "BB_WORST_RECENT_HALF": 0.0, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    5: {
        "BB_MONTHS_OBS": 2, "BB_MONTHS_REPORTED": 2, "BB_WORST": 5.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -3, "BB_WINDOW_END": -2, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 0.5,
        "BB_LAST_STATUS": "5", "BB_LAST_DPD_MONTH": -2.0, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": 0.0, "BB_WORST_RECENT_HALF": 5.0, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    6: {
        "BB_MONTHS_OBS": 3, "BB_MONTHS_REPORTED": 2, "BB_WORST": 0.0, "BB_PCT_X": 1 / 3,
        "BB_WINDOW_INI": -2, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 0.0, "BB_PCT_DPD": 0.0,
        "BB_LAST_STATUS": "0", "BB_LAST_DPD_MONTH": np.nan, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 0.0, "BB_WORST_RECENT_HALF": 0.0, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    8: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 4, "BB_WORST": 0.0, "BB_PCT_X": 1 / 6,
        "BB_WINDOW_INI": -5, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 0.0, "BB_PCT_DPD": 0.0,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": np.nan, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 0.0, "BB_WORST_RECENT_HALF": 0.0, "BB_CREDIT_TRAJECTORY": "sin mora",
    },
    9: {
        "BB_MONTHS_OBS": 7, "BB_MONTHS_REPORTED": 7, "BB_WORST": 1.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -6, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 1 / 7,
        "BB_LAST_STATUS": "0", "BB_LAST_DPD_MONTH": -3.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 1.0, "BB_WORST_RECENT_HALF": 0.0, "BB_CREDIT_TRAJECTORY": "mejora",
    },
    10: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 6, "BB_WORST": 4.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -95, "BB_WINDOW_END": -90, "BB_DPD_MONTHS": 6.0, "BB_PCT_DPD": 1.0,
        "BB_LAST_STATUS": "1", "BB_LAST_DPD_MONTH": -90.0, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": 4.0, "BB_WORST_RECENT_HALF": 1.0, "BB_CREDIT_TRAJECTORY": "mejora",
    },
    11: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 6, "BB_WORST": 2.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -5, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 2.0, "BB_PCT_DPD": 1 / 3,
        "BB_LAST_STATUS": "2", "BB_LAST_DPD_MONTH": 0.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 0.0, "BB_WORST_RECENT_HALF": 2.0, "BB_CREDIT_TRAJECTORY": "empeora",
    },
    12: {
        "BB_MONTHS_OBS": 8, "BB_MONTHS_REPORTED": 7, "BB_WORST": 1.0, "BB_PCT_X": 1 / 8,
        "BB_WINDOW_INI": -7, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 2.0, "BB_PCT_DPD": 2 / 7,
        "BB_LAST_STATUS": "X", "BB_LAST_DPD_MONTH": -1.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 1.0, "BB_WORST_RECENT_HALF": 1.0, "BB_CREDIT_TRAJECTORY": "estable",
    },
    13: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 3, "BB_WORST": 1.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -5, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 1 / 3,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": -5.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": 1.0, "BB_WORST_RECENT_HALF": np.nan, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    14: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 3, "BB_WORST": 1.0, "BB_PCT_X": 0.5,
        "BB_WINDOW_INI": -5, "BB_WINDOW_END": 0, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 1 / 3,
        "BB_LAST_STATUS": "0", "BB_LAST_DPD_MONTH": -1.0, "BB_CENSORED": 0,
        "BB_WORST_OLD_HALF": np.nan, "BB_WORST_RECENT_HALF": 1.0, "BB_CREDIT_TRAJECTORY": np.nan,
    },
    15: {
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 5, "BB_WORST": 0.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -11, "BB_WINDOW_END": -6, "BB_DPD_MONTHS": 0.0, "BB_PCT_DPD": 0.0,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": np.nan, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": 0.0, "BB_WORST_RECENT_HALF": 0.0, "BB_CREDIT_TRAJECTORY": "sin mora",
    },
    16: {
        "BB_MONTHS_OBS": 3, "BB_MONTHS_REPORTED": 1, "BB_WORST": 5.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -3, "BB_WINDOW_END": -1, "BB_DPD_MONTHS": 1.0, "BB_PCT_DPD": 1.0,
        "BB_LAST_STATUS": "C", "BB_LAST_DPD_MONTH": -3.0, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": 5.0, "BB_WORST_RECENT_HALF": np.nan, "BB_CREDIT_TRAJECTORY": np.nan,
    },
}


def panel(trozos=PANEL):
    """El panel crudo, tal y como llega la tabla: tres columnas y una fila por crédito-mes."""
    filas = [
        {"SK_ID_BUREAU": credito, "MONTHS_BALANCE": ini + i, "STATUS": estado}
        for credito, (ini, estados) in trozos.items()
        for i, estado in enumerate(estados)
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def bb():
    return panel()


@pytest.fixture
def agregado(bb):
    return agregar_por_credito(bb)


# --- el agregado, crédito a crédito -----------------------------------------------------------


@pytest.mark.parametrize("credito", sorted(ESPERADO))
def test_el_agregado_de_cada_credito_es_el_calculado_a_mano(agregado, credito):
    fila = agregado.loc[credito]
    for columna, esperado in ESPERADO[credito].items():
        obtenido = fila[columna]
        if isinstance(esperado, str):
            assert obtenido == esperado, columna
        elif np.isnan(esperado):
            assert pd.isna(obtenido), columna
        else:
            assert obtenido == pytest.approx(esperado), columna


def test_el_fixture_ejercita_cada_rama(agregado):
    """Guardián: si el fixture pierde un caso de borde, los tests de arriba dejan de cazarlo."""
    ciego = agregado["BB_MONTHS_REPORTED"].eq(0)
    assert ciego.any(), "falta el crédito sin ningún estado numérico"
    assert (ciego & agregado["BB_PCT_X"].eq(1.0)).any(), "falta el crédito que es todo X"
    assert (ciego & agregado["BB_PCT_X"].lt(1.0)).any(), "falta el ciego que mezcla C y X"
    assert agregado["BB_MONTHS_REPORTED"].eq(1).any(), "falta el de un solo mes reportado"
    assert agregado["BB_DPD_MONTHS"].gt(1).any(), "falta el de dos meses en mora, que un max tapa"
    assert (
        agregado["BB_DPD_MONTHS"].eq(0) & agregado["BB_MONTHS_REPORTED"].gt(0)
    ).any(), "falta el reportado sin mora, que exige 0 donde el ciego lleva NaN"
    assert agregado["BB_CENSORED"].eq(1).any() and agregado["BB_CENSORED"].eq(0).any()
    sale_en_mora = agregado["BB_LAST_STATUS"].isin(["1", "5"]).any()
    assert sale_en_mora, "falta el que sale del histórico en mora"
    assert agregado["BB_LAST_STATUS"].eq("C").any(), "falta el que sale cerrado"
    # la recencia tiene que poder distinguirse del primer impago en algún crédito
    ultimo_no_es_primero = agregado.loc[4, "BB_LAST_DPD_MONTH"] > ESPERADO[4]["BB_WINDOW_INI"]
    assert ultimo_no_es_primero, "falta el crédito con mora en meses no consecutivos"
    # la trayectoria
    trayectoria = agregado["BB_CREDIT_TRAJECTORY"]
    assert set(trayectoria.dropna()) == set(TRAYECTORIAS.categories), "falta alguna clase"
    ant, rec = agregado["BB_WORST_OLD_HALF"], agregado["BB_WORST_RECENT_HALF"]
    larga = agregado["BB_MONTHS_OBS"].ge(6)
    assert (larga & ant.isna() & rec.notna()).any(), "falta el de la mitad antigua ciega"
    assert (larga & rec.isna() & ant.notna()).any(), "falta el de la mitad reciente ciega"
    corta_evaluable = agregado["BB_MONTHS_OBS"].eq(5) & ant.notna() & rec.notna()
    assert corta_evaluable.any(), "falta la ventana de 5 con las dos mitades, el borde que no entra"
    justa = agregado["BB_MONTHS_OBS"].eq(6) & trayectoria.notna()
    assert justa.any(), "falta la ventana de 6 clasificada, el borde que sí entra"
    assert (trayectoria.eq("mejora") & rec.gt(0)).any(), "falta la mejora que sigue en mora"
    # la ventana sale en int8, así que la suma de sus extremos va en un tipo ancho
    extremos = agregado["BB_WINDOW_INI"].astype(int) + agregado["BB_WINDOW_END"].astype(int)
    impar = agregado["BB_MONTHS_OBS"].mod(2).eq(1) & larga
    medio = extremos // 2
    mora_en_el_medio = impar & agregado["BB_LAST_DPD_MONTH"].eq(medio) & agregado["BB_WORST"].gt(0)
    assert mora_en_el_medio.any(), "falta la ventana impar con la mora en el mes central"
    desborda = extremos.lt(-128) & trayectoria.notna()
    assert desborda.any(), "falta el clasificado con ini + fin fuera del int8, el que desborda"


def test_cada_columna_de_la_salida_varia_entre_creditos(agregado):
    """Una columna constante por construcción pasa cualquier test sin medir nada."""
    constantes = [c for c in agregado.columns if agregado[c].nunique(dropna=False) < 2]
    assert not constantes


# --- la premisa de la capa 1 ------------------------------------------------------------------


def test_cada_credito_agregado_solo_da_lo_mismo_que_acompanado(bb, agregado):
    """Lo que autoriza a agregar fuera del split: ningún estadístico cruza créditos.

    Con `assert_frame_equal` y tipos incluidos, que es lo que caza la columna cuyo dtype depende
    de los vecinos.
    """
    for credito in ESPERADO:
        solo = agregar_por_credito(bb[bb["SK_ID_BUREAU"].eq(credito)])
        assert_frame_equal(solo, agregado.loc[[credito]])


def test_el_orden_de_las_filas_no_cambia_el_agregado(bb, agregado):
    revuelto = bb.sample(frac=1, random_state=0)
    assert_frame_equal(agregar_por_credito(revuelto), agregado)


def test_el_estado_final_es_el_del_mes_maximo_y_no_el_de_la_ultima_fila(bb):
    """Sin esto, el estado final se leería del orden de llegada y no del calendario."""
    al_reves = bb.iloc[::-1]
    esperados = {c: ESPERADO[c]["BB_LAST_STATUS"] for c in ESPERADO}
    assert agregar_por_credito(al_reves)["BB_LAST_STATUS"].astype(str).to_dict() == esperados


def test_no_muta_el_frame_de_entrada(bb):
    copia = bb.copy()
    agregar_por_credito(bb)
    assert_frame_equal(bb, copia)


def test_agregar_el_crudo_y_el_limpio_da_lo_mismo(bb, agregado):
    """La limpieza va por dentro y es idempotente, así que da igual quién la haya pasado ya."""
    assert_frame_equal(agregar_por_credito(limpiar_bureau_balance(bb)), agregado)


# --- la regla de NaN --------------------------------------------------------------------------


def test_sin_mes_reportado_la_severidad_y_la_intensidad_son_nan_y_no_cero(agregado):
    """Sin dato no es sin mora: con 0, el cliente de créditos ciegos saldría limpio del 3.5."""
    ciego = agregado["BB_MONTHS_REPORTED"].eq(0)
    assert agregado["BB_WORST"].isna().equals(ciego)
    assert agregado["BB_DPD_MONTHS"].isna().equals(ciego)
    assert agregado["BB_PCT_DPD"].isna().equals(ciego)
    # y el reportado sin mora sigue siendo 0, que es lo que el NaN no puede comerse
    assert agregado.loc[6, "BB_DPD_MONTHS"] == 0


def test_la_mitad_ciega_no_se_lee_como_sin_mora_y_deja_la_trayectoria_en_nan(agregado):
    """La ceguera se concentra en sin mora y mejora, así que con un 0 esas dos se inflan."""
    ciega = agregado["BB_WORST_OLD_HALF"].isna() | agregado["BB_WORST_RECENT_HALF"].isna()
    corta = agregado["BB_MONTHS_OBS"].lt(6)
    assert agregado["BB_CREDIT_TRAJECTORY"].isna().equals(ciega | corta)


def test_la_particion_de_estados_cuadra_con_la_ventana(bb, agregado):
    """Los meses cerrados, los ausentes y los reportados suman la ventana entera."""
    limpio = limpiar_bureau_balance(bb)
    g = limpio.groupby("SK_ID_BUREAU")
    cerrados = g["STATUS"].agg(lambda s: s.eq("C").sum())
    ausentes = g["BB_IS_X"].sum()
    assert (cerrados + ausentes + agregado["BB_MONTHS_REPORTED"] == agregado["BB_MONTHS_OBS"]).all()


def test_la_recencia_es_el_ultimo_impago_y_no_el_primero(agregado):
    assert agregado.loc[4, "BB_LAST_DPD_MONTH"] == -2
    assert agregado["BB_LAST_DPD_MONTH"].isna().equals(agregado["BB_DPD_MONTHS"].fillna(0).eq(0))


# --- el esquema, que no puede depender del lote -----------------------------------------------


def test_un_frame_vacio_da_un_agregado_vacio_con_el_mismo_esquema(agregado):
    """El cliente sin histórico y el frame sin tipos de la API, que es el patrón 12."""
    for vacio in (
        panel().iloc[:0],
        pd.DataFrame(columns=list(COLUMNAS_ORIGEN)),
    ):
        salida = agregar_por_credito(vacio)
        assert salida.empty
        assert list(salida.columns) == list(agregado.columns)
        assert salida.dtypes.to_dict() == agregado.dtypes.to_dict()


@pytest.mark.parametrize("dtype", ["uint32", "int64", "float64"])
def test_el_indice_sale_en_int64_venga_como_venga_la_clave(bb, dtype):
    """`load_table` da uint32 y un `read_csv` a pelo int64: el índice no puede depender de eso."""
    salida = agregar_por_credito(bb.astype({"SK_ID_BUREAU": dtype}))
    assert salida.index.dtype == "int64"
    assert salida.index.tolist() == sorted(ESPERADO)


def test_el_estado_final_conserva_los_ocho_niveles_aunque_el_lote_traiga_tres(agregado):
    assert list(agregado["BB_LAST_STATUS"].cat.categories) == list(DTYPE_STATUS.categories)


def test_la_trayectoria_conserva_sus_cuatro_niveles_ordenados_aunque_el_lote_traiga_uno(bb):
    """El orden es la prioridad del peor recorrido, que el nivel cliente lee con un max."""
    solo = agregar_por_credito(bb[bb["SK_ID_BUREAU"].eq(11)])["BB_CREDIT_TRAJECTORY"]
    assert solo.dtype == TRAYECTORIAS
    assert solo.cat.ordered
    assert list(solo.cat.categories) == ["sin mora", "mejora", "empeora", "estable"]


def test_la_salida_tiene_el_esquema_declarado(agregado):
    """En absoluto y no comparando dos lotes entre sí.

    Comparar el frame vacío con el poblado no ve la columna que pierde su tipo en los dos a la
    vez, que es lo que dejó pasar quitarle el `int8` a las banderas de `bureau`.
    """
    assert agregado.dtypes.astype(str).to_dict() == {
        "BB_MONTHS_OBS": "int64",
        "BB_MONTHS_REPORTED": "int64",
        "BB_WORST": "float32",
        "BB_PCT_X": "float64",
        "BB_WINDOW_INI": "int8",
        "BB_WINDOW_END": "int8",
        "BB_DPD_MONTHS": "float64",
        "BB_PCT_DPD": "float64",
        "BB_LAST_STATUS": "category",
        "BB_LAST_DPD_MONTH": "float64",
        "BB_CENSORED": "int8",
        "BB_WORST_OLD_HALF": "float32",
        "BB_WORST_RECENT_HALF": "float32",
        "BB_CREDIT_TRAJECTORY": "category",
    }


# --- las guardas, por los dos lados -----------------------------------------------------------


@pytest.mark.parametrize("columna", COLUMNAS_ORIGEN)
def test_una_columna_de_origen_ausente_revienta_con_su_nombre(bb, columna):
    with pytest.raises(ValueError, match=columna):
        agregar_por_credito(bb.drop(columns=[columna]))


def test_una_clave_nula_revienta_en_vez_de_perder_sus_meses(bb):
    """El `groupby` tira la fila sin clave. La tabla no trae ninguna: solo puede traerla la API."""
    sin_clave = bb.astype({"SK_ID_BUREAU": float})
    sin_clave.loc[0, "SK_ID_BUREAU"] = np.nan
    with pytest.raises(ValueError, match="sin SK_ID_BUREAU"):
        agregar_por_credito(sin_clave)


def test_un_hueco_en_la_ventana_revienta_nombrando_el_credito(bb):
    con_hueco = bb[~(bb["SK_ID_BUREAU"].eq(4) & bb["MONTHS_BALANCE"].eq(-2))]
    with pytest.raises(ValueError, match="ventana mensual rota"):
        agregar_por_credito(con_hueco)


def test_un_par_credito_mes_duplicado_revienta_por_la_misma_guarda(bb):
    """La limpieza no los busca a propósito, y aquí inflarían la ventana sin que nada avise."""
    with pytest.raises(ValueError, match="ventana mensual rota"):
        agregar_por_credito(pd.concat([bb, bb.iloc[[0]]], ignore_index=True))


def test_un_duplicado_que_tapa_un_hueco_revienta(bb):
    """El caso que comparar la ventana con las filas deja pasar: el largo sale justo.

    La tabla no trae ninguno, así que solo existe aquí, y por la API es un `concat` mal hecho.
    """
    credito = bb["SK_ID_BUREAU"].eq(4)
    tapado = pd.concat(
        [bb[~(credito & bb["MONTHS_BALANCE"].eq(-2))], bb[credito & bb["MONTHS_BALANCE"].eq(-3)]],
        ignore_index=True,
    )
    meses = tapado.loc[tapado["SK_ID_BUREAU"].eq(4), "MONTHS_BALANCE"]
    assert meses.max() - meses.min() + 1 == len(meses), "el largo tiene que salir justo"
    with pytest.raises(ValueError, match="ventana mensual rota"):
        agregar_por_credito(tapado)


def test_la_ventana_contigua_de_un_solo_mes_no_revienta():
    """El otro lado de la guarda, en el borde de la ventana que el fixture no tiene."""
    assert agregar_por_credito(panel({7: (-96, ["0"])})).loc[7, "BB_MONTHS_OBS"] == 1


# --- 3.4, el puente credito a cliente -----------------------------------------------------------


@pytest.fixture
def bureau():
    """Un bureau sintético mínimo: solo las dos columnas que el puente necesita."""
    return pd.DataFrame({"SK_ID_BUREAU": [1, 2, 3, 4], "SK_ID_CURR": [100, 100, 200, 300]})


def test_el_puente_es_el_esperado_a_mano(bureau):
    puente = puente_credito_cliente(bureau)
    assert puente.to_dict() == {1: 100, 2: 100, 3: 200, 4: 300}
    assert puente.index.dtype == "int64"
    assert puente.index.name == "SK_ID_BUREAU"
    assert puente.name == "SK_ID_CURR"


def test_el_tipo_del_indice_coincide_con_el_del_paso_credito(bureau, agregado):
    """Es lo que permite `cred.join(puente)` sin cruzar dos tipos en el 3.5."""
    assert puente_credito_cliente(bureau).index.dtype == agregado.index.dtype


@pytest.mark.parametrize("dtype", ["uint32", "int64", "float64"])
def test_el_indice_del_puente_sale_en_int64_venga_como_venga_la_clave(bureau, dtype):
    puente = puente_credito_cliente(bureau.astype({"SK_ID_BUREAU": dtype}))
    assert puente.index.dtype == "int64"
    assert puente.to_dict() == {1: 100, 2: 100, 3: 200, 4: 300}


def test_un_bureau_vacio_da_un_puente_vacio_en_int64():
    vacio = pd.DataFrame(columns=["SK_ID_BUREAU", "SK_ID_CURR"])
    puente = puente_credito_cliente(vacio)
    assert puente.empty
    assert puente.index.dtype == "int64"


def test_el_puente_no_muta_el_frame_de_entrada(bureau):
    copia = bureau.copy()
    puente_credito_cliente(bureau)
    assert_frame_equal(bureau, copia)


def test_una_sk_id_bureau_con_decimales_revienta():
    """Un 1.5 truncado por el cast cruzaría con el crédito 1, como el mes de bureau_balance."""
    con_decimales = pd.DataFrame({"SK_ID_BUREAU": [1.0, 1.5, 2.0], "SK_ID_CURR": [100, 200, 300]})
    with pytest.raises(ValueError, match="decimales"):
        puente_credito_cliente(con_decimales)


@pytest.mark.parametrize("columna", ["SK_ID_BUREAU", "SK_ID_CURR"])
def test_una_columna_ausente_revienta_con_su_nombre(bureau, columna):
    with pytest.raises(ValueError, match=columna):
        puente_credito_cliente(bureau.drop(columns=[columna]))


def test_una_sk_id_bureau_nula_revienta(bureau):
    con_nula = bureau.astype({"SK_ID_BUREAU": float})
    con_nula.loc[0, "SK_ID_BUREAU"] = np.nan
    with pytest.raises(ValueError, match="SK_ID_BUREAU"):
        puente_credito_cliente(con_nula)


def test_una_sk_id_curr_nula_revienta(bureau):
    con_nula = bureau.astype({"SK_ID_CURR": float})
    con_nula.loc[0, "SK_ID_CURR"] = np.nan
    with pytest.raises(ValueError, match="SK_ID_CURR"):
        puente_credito_cliente(con_nula)


def test_un_sk_id_bureau_repetido_revienta_aunque_sea_el_mismo_cliente(bureau):
    repetido = pd.concat([bureau, bureau.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        puente_credito_cliente(repetido)


def test_un_sk_id_bureau_repetido_con_otro_cliente_revienta(bureau):
    otro_cliente = pd.DataFrame({"SK_ID_BUREAU": [1], "SK_ID_CURR": [999]})
    repetido = pd.concat([bureau, otro_cliente], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        puente_credito_cliente(repetido)


# --- 3.5, la agregación a nivel cliente -------------------------------------------------------

# Los cortes salen del registro y no de literales, como en `test_agg_bureau`: los tres son `medido`
# y `valor()` los bloquea hasta que el 3.8 y el 3.9 los refijan sobre train, así que aquí se pasan
# con su referencia del EDA, que es lo que la puerta reproduce.
REFERENCIA = {n: parametro(n).valor_referencia for n in CORTES_CLIENTE}
# el fixture no llega a los 22 créditos de la cola, así que la bajo a 3: cae justo en el 500, que
# es el borde que separa `>=` de `>`
CORTES_FIXTURE = {**REFERENCIA, "bb_many_credits_corte": 3}

# El reparto de los créditos del panel entre clientes, uno por caso de borde del nivel cliente.
# Los que no aparecen son los huérfanos, que están en el panel y no en bureau; el 7 es lo
# contrario, un crédito de bureau sin histórico mensual.
CLIENTES = {
    100: [1],  # todos sus créditos ciegos: severidad y conteos a NaN, banderas a 0
    200: [2, 3],  # uno ciego y otro reportado: el que separa min_count=1 de "ningún mes reportado"
    300: [4, 5],  # dos créditos con mora, que es lo que un max taparía, más el fallido
    400: [6],  # reportado sin mora: conteos a 0, que es lo que el NaN no puede comerse
    # tres créditos, y uno reportado sin mora junto a dos con ella; su trayectoria es el max de
    # mejora, empeora y sin mora, con la peor en medio, y es el borde de la cola
    500: [9, 11, 15],
    600: [10],  # mora antigua: la genérica marca y la reciente no, pero la relativa sí
    700: [12],  # estable, la única persistente
    800: [7],  # en bureau y sin histórico mensual: no sale del agregado
    900: [8],  # sin mora y todo cerrado, con las tres banderas de trayectoria a 0 y no a NaN
    1000: [16],  # fallido que no sale en 5: con el último estado en vez de la severidad no marca
}
HUERFANOS = (13, 14)

ESPERADO_CLIENTE = {
    100: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 3, "BB_MONTHS_REPORTED": 0,
        "BB_STATUS_WORST": np.nan, "BB_ANY_DPD_FLAG": 0, "BB_RECENT_DPD_FLAG": 0,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": np.nan, "BB_CENSORED_RATIO": 0.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": np.nan, "BB_EXITS_IN_DPD_FLAG": 0,
        "BB_ALL_CLOSED_FLAG": 1, "BB_TRAJECTORY": np.nan,
        "BB_CREDITS_WITH_DPD_COUNT": np.nan, "BB_DPD_MONTHS_COUNT": np.nan,
        "BB_PCT_MONTHS_DPD": np.nan, "BB_RECENT_DPD_FLAG_REL": 0, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": np.nan, "BB_WORSENING_DPD_FLAG": np.nan,
        "BB_RECOVERED_DPD_FLAG": np.nan,
    },
    200: {
        "BB_N_CREDITS_WBAL": 2, "BB_MONTHS_TOTAL": 3, "BB_MONTHS_REPORTED": 1,
        "BB_STATUS_WORST": 1.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 1,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": 0.0, "BB_CENSORED_RATIO": 0.5,
        "BB_MONTHS_SINCE_LAST_DPD_REL": 0.0, "BB_EXITS_IN_DPD_FLAG": 1,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": np.nan,
        "BB_CREDITS_WITH_DPD_COUNT": 1.0, "BB_DPD_MONTHS_COUNT": 1.0,
        "BB_PCT_MONTHS_DPD": np.nan, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": np.nan, "BB_WORSENING_DPD_FLAG": np.nan,
        "BB_RECOVERED_DPD_FLAG": np.nan,
    },
    300: {
        "BB_N_CREDITS_WBAL": 2, "BB_MONTHS_TOTAL": 7, "BB_MONTHS_REPORTED": 6,
        "BB_STATUS_WORST": 5.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 1,
        "BB_WRITEOFF_FLAG": 1, "BB_MONTHS_SINCE_LAST_DPD": -2.0, "BB_CENSORED_RATIO": 0.5,
        # el 4 da -2 y el censurado 5 da 0: la relativa es el max, y no coincide con la absoluta
        "BB_MONTHS_SINCE_LAST_DPD_REL": 0.0, "BB_EXITS_IN_DPD_FLAG": 1,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": np.nan,
        "BB_CREDITS_WITH_DPD_COUNT": 2.0, "BB_DPD_MONTHS_COUNT": 3.0,
        "BB_PCT_MONTHS_DPD": 0.5, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": np.nan, "BB_WORSENING_DPD_FLAG": np.nan,
        "BB_RECOVERED_DPD_FLAG": np.nan,
    },
    400: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 3, "BB_MONTHS_REPORTED": 2,
        "BB_STATUS_WORST": 0.0, "BB_ANY_DPD_FLAG": 0, "BB_RECENT_DPD_FLAG": 0,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": np.nan, "BB_CENSORED_RATIO": 0.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": np.nan, "BB_EXITS_IN_DPD_FLAG": 0,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": np.nan,
        "BB_CREDITS_WITH_DPD_COUNT": 0.0, "BB_DPD_MONTHS_COUNT": 0.0,
        "BB_PCT_MONTHS_DPD": np.nan, "BB_RECENT_DPD_FLAG_REL": 0, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": np.nan, "BB_WORSENING_DPD_FLAG": np.nan,
        "BB_RECOVERED_DPD_FLAG": np.nan,
    },
    500: {
        "BB_N_CREDITS_WBAL": 3, "BB_MONTHS_TOTAL": 19, "BB_MONTHS_REPORTED": 18,
        "BB_STATUS_WORST": 2.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 1,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": 0.0, "BB_CENSORED_RATIO": 1 / 3,
        "BB_MONTHS_SINCE_LAST_DPD_REL": 0.0, "BB_EXITS_IN_DPD_FLAG": 1,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": "empeora",
        "BB_CREDITS_WITH_DPD_COUNT": 2.0, "BB_DPD_MONTHS_COUNT": 3.0,
        "BB_PCT_MONTHS_DPD": 3 / 18, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 1,
        "BB_PERSISTENT_DPD_FLAG": 0.0, "BB_WORSENING_DPD_FLAG": 1.0,
        "BB_RECOVERED_DPD_FLAG": 0.0,
    },
    600: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 6, "BB_MONTHS_REPORTED": 6,
        "BB_STATUS_WORST": 4.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 0,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": -90.0, "BB_CENSORED_RATIO": 1.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": 0.0, "BB_EXITS_IN_DPD_FLAG": 1,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": "mejora",
        "BB_CREDITS_WITH_DPD_COUNT": 1.0, "BB_DPD_MONTHS_COUNT": 6.0,
        "BB_PCT_MONTHS_DPD": 1.0, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": 0.0, "BB_WORSENING_DPD_FLAG": 0.0,
        "BB_RECOVERED_DPD_FLAG": 1.0,
    },
    700: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 8, "BB_MONTHS_REPORTED": 7,
        "BB_STATUS_WORST": 1.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 1,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": -1.0, "BB_CENSORED_RATIO": 0.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": -1.0, "BB_EXITS_IN_DPD_FLAG": 0,
        "BB_ALL_CLOSED_FLAG": 0, "BB_TRAJECTORY": "estable",
        "BB_CREDITS_WITH_DPD_COUNT": 1.0, "BB_DPD_MONTHS_COUNT": 2.0,
        "BB_PCT_MONTHS_DPD": 2 / 7, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": 1.0, "BB_WORSENING_DPD_FLAG": 0.0,
        "BB_RECOVERED_DPD_FLAG": 0.0,
    },
    900: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 6, "BB_MONTHS_REPORTED": 4,
        "BB_STATUS_WORST": 0.0, "BB_ANY_DPD_FLAG": 0, "BB_RECENT_DPD_FLAG": 0,
        "BB_WRITEOFF_FLAG": 0, "BB_MONTHS_SINCE_LAST_DPD": np.nan, "BB_CENSORED_RATIO": 0.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": np.nan, "BB_EXITS_IN_DPD_FLAG": 0,
        "BB_ALL_CLOSED_FLAG": 1, "BB_TRAJECTORY": "sin mora",
        "BB_CREDITS_WITH_DPD_COUNT": 0.0, "BB_DPD_MONTHS_COUNT": 0.0,
        "BB_PCT_MONTHS_DPD": np.nan, "BB_RECENT_DPD_FLAG_REL": 0, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": 0.0, "BB_WORSENING_DPD_FLAG": 0.0,
        "BB_RECOVERED_DPD_FLAG": 0.0,
    },
    1000: {
        "BB_N_CREDITS_WBAL": 1, "BB_MONTHS_TOTAL": 3, "BB_MONTHS_REPORTED": 1,
        "BB_STATUS_WORST": 5.0, "BB_ANY_DPD_FLAG": 1, "BB_RECENT_DPD_FLAG": 1,
        "BB_WRITEOFF_FLAG": 1, "BB_MONTHS_SINCE_LAST_DPD": -3.0, "BB_CENSORED_RATIO": 1.0,
        "BB_MONTHS_SINCE_LAST_DPD_REL": -2.0, "BB_EXITS_IN_DPD_FLAG": 0,
        "BB_ALL_CLOSED_FLAG": 1, "BB_TRAJECTORY": np.nan,
        "BB_CREDITS_WITH_DPD_COUNT": 1.0, "BB_DPD_MONTHS_COUNT": 1.0,
        "BB_PCT_MONTHS_DPD": np.nan, "BB_RECENT_DPD_FLAG_REL": 1, "BB_MANY_CREDITS_FLAG": 0,
        "BB_PERSISTENT_DPD_FLAG": np.nan, "BB_WORSENING_DPD_FLAG": np.nan,
        "BB_RECOVERED_DPD_FLAG": np.nan,
    },
}

@pytest.fixture
def bureau_cliente():
    """El `bureau` que reparte los créditos del panel entre clientes, sin el huérfano."""
    return pd.DataFrame(
        [
            {"SK_ID_BUREAU": credito, "SK_ID_CURR": cliente}
            for cliente, creditos in CLIENTES.items()
            for credito in creditos
        ]
    )


@pytest.fixture
def puente(bureau_cliente):
    return puente_credito_cliente(bureau_cliente)


@pytest.fixture
def por_cliente(bb, puente):
    return agregar_bureau_balance(bb, puente, CORTES_FIXTURE)


@pytest.mark.parametrize("cliente", sorted(ESPERADO_CLIENTE))
def test_el_agregado_de_cada_cliente_es_el_calculado_a_mano(por_cliente, cliente):
    fila = por_cliente.loc[cliente]
    for columna, esperado in ESPERADO_CLIENTE[cliente].items():
        obtenido = fila[columna]
        if pd.isna(esperado):
            assert pd.isna(obtenido), columna
        else:
            assert obtenido == pytest.approx(esperado), columna


def test_el_fixture_de_clientes_ejercita_cada_rama(por_cliente, agregado, bb, puente):
    """Guardián: si el reparto pierde un caso, los tests de arriba dejan de cazarlo.

    Los tres primeros miran el nivel crédito porque el agregado de cliente no deja ver de cuántos
    créditos sale: un cliente de un crédito reportado y otro que mezcla uno ciego con uno
    reportado son indistinguibles desde sus columnas.
    """
    minimo = CORTES_FIXTURE["bb_min_meses_reportados"]
    credito = agregado.loc[agregado.index.isin(puente.index)]
    de_cada = credito["BB_MONTHS_REPORTED"].groupby(credito.index.map(puente))
    ciegos, total = de_cada.agg(lambda s: s.eq(0).sum()), de_cada.size()
    assert ciegos.eq(total).any(), "falta el cliente con todos sus créditos ciegos"
    assert (ciegos.gt(0) & ciegos.lt(total)).any(), (
        "falta el cliente que mezcla un crédito ciego con uno reportado, que es el que separa "
        "min_count=1 de contar los meses reportados"
    )
    suma, maximo = de_cada.sum(), de_cada.max()
    assert (suma.ge(minimo) & maximo.lt(minimo)).any(), (
        "falta el cliente cuyo denominador solo llega al mínimo sumando sus créditos"
    )
    assert (
        por_cliente["BB_DPD_MONTHS_COUNT"].eq(0) & por_cliente["BB_MONTHS_REPORTED"].gt(0)
    ).any(), "falta el reportado sin mora, que exige 0 donde el ciego lleva NaN"
    assert por_cliente["BB_CREDITS_WITH_DPD_COUNT"].gt(1).any(), (
        "falta el cliente de dos créditos con mora, sin el cual el sum pasa por el max"
    )
    antigua = por_cliente["BB_ANY_DPD_FLAG"].eq(1) & por_cliente["BB_RECENT_DPD_FLAG"].eq(0)
    assert antigua.any(), "falta el cliente de mora antigua, que separa las dos banderas"
    assert por_cliente["BB_WRITEOFF_FLAG"].eq(1).any(), "falta el fallido"
    grave = por_cliente["BB_STATUS_WORST"].between(3, 4) & por_cliente["BB_WRITEOFF_FLAG"].eq(0)
    assert grave.any(), "falta el grave que no es fallido, sin el cual `== 5` y `>= 4` dan lo mismo"
    assert (credito["BB_WORST"].eq(5) & credito["BB_LAST_STATUS"].ne("5")).any(), (
        "falta el fallido que no acaba en 5, sin el cual leerlo del último estado da lo mismo"
    )
    assert total.max() >= 3, "falta el cliente de tres créditos, donde las sumas dejan de ser pares"
    con_mora = por_cliente["BB_CREDITS_WITH_DPD_COUNT"]
    assert (con_mora.gt(0) & con_mora.lt(por_cliente["BB_N_CREDITS_WBAL"]) & ~ciegos.gt(0)).any(), (
        "falta el cliente que mezcla créditos con mora y sin ella, los dos reportados"
    )
    ultimos = credito["BB_LAST_DPD_MONTH"].groupby(credito.index.map(puente))
    assert (ultimos.max() != ultimos.min()).any(), (
        "falta el cliente con impagos en meses distintos, sin el cual la recencia da igual con min"
    )
    for ratio in ("BB_CENSORED_RATIO", "BB_PCT_MONTHS_DPD"):
        valores = por_cliente[ratio].dropna()
        assert valores.between(0, 1).all() and valores.gt(0).any(), ratio
    justo = por_cliente["BB_MONTHS_REPORTED"].eq(minimo)
    assert (justo & por_cliente["BB_PCT_MONTHS_DPD"].notna()).any(), "falta el denominador justo"
    assert por_cliente["BB_PCT_MONTHS_DPD"].isna().any(), "falta el denominador corto"
    # las derivadas del 3.6
    trayectoria = por_cliente["BB_TRAJECTORY"]
    assert set(trayectoria.dropna()) == set(TRAYECTORIAS.categories), "falta alguna clase"
    clases = credito["BB_CREDIT_TRAJECTORY"].groupby(credito.index.map(puente)).nunique()
    assert clases.gt(1).any(), "falta el cliente de clases distintas, sin el cual el max no se ve"
    assert (trayectoria.isna() & por_cliente["BB_MONTHS_REPORTED"].gt(0)).any(), (
        "falta el reportado sin trayectoria, sin el cual el NaN de las banderas solo es el ciego"
    )
    rel, absoluta = (
        por_cliente["BB_MONTHS_SINCE_LAST_DPD_REL"],
        por_cliente["BB_MONTHS_SINCE_LAST_DPD"],
    )
    assert (rel.notna() & rel.ne(absoluta)).any(), "falta la relativa distinta de la absoluta"
    assert por_cliente["BB_RECENT_DPD_FLAG_REL"].ne(por_cliente["BB_RECENT_DPD_FLAG"]).any(), (
        "falta el cliente que solo marca la mora reciente relativa"
    )
    cerrado = credito["BB_LAST_STATUS"].eq("C").groupby(credito.index.map(puente))
    assert (cerrado.any() & ~cerrado.all()).any(), "falta el que cierra unos créditos y otros no"
    todo_cerrado = por_cliente["BB_ALL_CLOSED_FLAG"].eq(1)
    assert (todo_cerrado & por_cliente["BB_MONTHS_REPORTED"].gt(0)).any(), (
        "falta el todo cerrado con meses reportados, que no es solo el ciego"
    )
    en_mora = (
        credito["BB_LAST_STATUS"].isin(["1", "2", "3", "4", "5"]).groupby(credito.index.map(puente))
    )
    assert (en_mora.any() & ~en_mora.all()).any(), "falta el que sale en mora solo en un crédito"
    borde = por_cliente["BB_N_CREDITS_WBAL"].eq(CORTES_FIXTURE["bb_many_credits_corte"])
    assert borde.any(), "falta el cliente justo en el corte de la cola"
    assert len(HUERFANOS) > 1, "con un solo huérfano, tirar solo el primero pasaría en verde"
    assert set(HUERFANOS) <= set(bb["SK_ID_BUREAU"]) - set(puente.index), (
        "los huérfanos tienen que estar en el panel y no en bureau"
    )


def test_cada_columna_de_la_salida_varia_entre_clientes(por_cliente):
    """Una columna constante por construcción pasa cualquier test sin medir nada."""
    constantes = [c for c in por_cliente.columns if por_cliente[c].nunique(dropna=False) < 2]
    assert not constantes


def test_la_salida_de_cliente_tiene_el_esquema_declarado(por_cliente):
    assert list(por_cliente.columns) == list(ESPERADO_CLIENTE[100])
    assert por_cliente.index.name == "SK_ID_CURR"
    conteos = ["BB_N_CREDITS_WBAL", "BB_MONTHS_TOTAL", "BB_MONTHS_REPORTED"]
    assert (por_cliente[conteos].dtypes == "int64").all()
    banderas = [
        "BB_ANY_DPD_FLAG", "BB_RECENT_DPD_FLAG", "BB_WRITEOFF_FLAG", "BB_EXITS_IN_DPD_FLAG",
        "BB_ALL_CLOSED_FLAG", "BB_RECENT_DPD_FLAG_REL", "BB_MANY_CREDITS_FLAG",
    ]
    assert (por_cliente[banderas].dtypes == "int8").all()
    assert (por_cliente[list(BANDERAS_TRAYECTORIA)].dtypes == "float64").all()
    assert por_cliente["BB_TRAJECTORY"].dtype == TRAYECTORIAS


def test_la_salida_cumple_el_contrato_con_la_receta(por_cliente, clientes):
    """Lo que añade la unión es la receta menos el descarte firme, más las declaradas sin receta.

    `BUREAU_OVERDUE_UNION` es la referencia de `bureau` y llega ya en la lista de clientes.
    """
    receta = cargar_receta("bureau_balance")["features"]
    nombres = {f["nombre"] for f in receta}
    firmes = {f["nombre"] for f in receta if f["firmeza"] == "firme"}
    assert firmes == {"BB_MONTHS_MAX"}
    referencia = {f["nombre"] for f in receta if f["decision"] == "referencia"}
    assert referencia == {"BUREAU_OVERDUE_UNION"} and referencia <= set(clientes.columns)
    anadidas = set(unir_bureau_balance(clientes, por_cliente).columns) - set(clientes.columns)
    assert anadidas == (nombres - firmes - referencia) | set(COLUMNAS_SIN_RECETA)
    assert not set(COLUMNAS_SIN_RECETA) & nombres, "una columna sin receta que sí está en ella"
    for motivo in COLUMNAS_SIN_RECETA.values():
        assert motivo.strip()


# --- la premisa de la capa 1, un nivel más arriba ---------------------------------------------


def test_cada_cliente_agregado_solo_da_lo_mismo_que_acompanado(bb, puente, por_cliente):
    """Lo que autoriza a agregar fuera del split: ningún estadístico cruza clientes."""
    for cliente in ESPERADO_CLIENTE:
        creditos = puente.index[puente.eq(cliente)]
        solo = bb[bb["SK_ID_BUREAU"].isin(creditos)]
        assert_frame_equal(
            agregar_bureau_balance(solo, puente[puente.eq(cliente)], CORTES_FIXTURE),
            por_cliente.loc[[cliente]],
        )


def test_el_orden_de_las_filas_no_cambia_el_agregado_de_cliente(bb, puente, por_cliente):
    revuelto = bb.sample(frac=1, random_state=0)
    assert_frame_equal(agregar_bureau_balance(revuelto, puente, CORTES_FIXTURE), por_cliente)


def test_el_nivel_cliente_no_muta_el_frame_de_entrada(bb, puente):
    copia, copia_puente = bb.copy(), puente.copy()
    agregar_bureau_balance(bb, puente, CORTES_FIXTURE)
    assert_frame_equal(bb, copia)
    pd.testing.assert_series_equal(puente, copia_puente)


# --- la regla de NaN a nivel cliente ----------------------------------------------------------


def test_el_cliente_de_creditos_todos_ciegos_va_a_nan_y_no_a_cero(por_cliente):
    """Sin dato no es sin mora: con 0, el cliente sin ningún estado reportado saldría limpio."""
    ciego = por_cliente["BB_MONTHS_REPORTED"].eq(0)
    for columna in ("BB_STATUS_WORST", "BB_DPD_MONTHS_COUNT", "BB_CREDITS_WITH_DPD_COUNT"):
        assert por_cliente[columna].isna().equals(ciego), columna


def test_las_banderas_del_cliente_ciego_se_quedan_en_cero(por_cliente):
    """La única excepción declarada a la regla, y la que el EDA midió: +2,74 frente a +2,76pp."""
    ciego = por_cliente["BB_MONTHS_REPORTED"].eq(0)
    assert ciego.any(), "sin ningún cliente ciego el `.all()` de abajo es verdadero por vacuidad"
    banderas = ["BB_ANY_DPD_FLAG", "BB_RECENT_DPD_FLAG", "BB_WRITEOFF_FLAG"]
    assert (por_cliente.loc[ciego, banderas] == 0).all().all()


def test_la_proporcion_sin_denominador_suficiente_es_nan(bb, puente):
    """El extremo del ratio es ruido de reporte: con un mes reportado vale 1 sin decir nada."""
    for minimo in (1, 6, 12):
        cortes = {**CORTES_FIXTURE, "bb_min_meses_reportados": minimo}
        agregado = agregar_bureau_balance(bb, puente, cortes)
        corto = agregado["BB_MONTHS_REPORTED"].lt(minimo)
        assert agregado["BB_PCT_MONTHS_DPD"].isna().equals(corto), minimo


def test_la_mora_reciente_se_corta_sobre_el_ultimo_impago(bb, puente):
    """Y el cliente sin ningún impago no la enciende, que es lo que el NaN podría colar."""
    for meses in (3, 6, 12, 96):
        cortes = {**CORTES_FIXTURE, "bb_mora_reciente_meses": meses}
        agregado = agregar_bureau_balance(bb, puente, cortes)
        esperado = agregado["BB_MONTHS_SINCE_LAST_DPD"].ge(-meses).astype("int8")
        assert agregado["BB_RECENT_DPD_FLAG"].equals(esperado), meses
        # con la ventana entera, la reciente y la genérica dicen lo mismo
        if meses == 96:
            assert agregado["BB_RECENT_DPD_FLAG"].equals(agregado["BB_ANY_DPD_FLAG"])


# --- 3.6, las derivadas de recencia, trayectoria y cola ---------------------------------------


def test_la_mora_reciente_relativa_se_corta_sobre_la_recencia_relativa(bb, puente):
    """Con el mismo corte que la absoluta, y sin encenderse en el cliente sin impago."""
    for meses in (0, 1, 6, 96):
        cortes = {**CORTES_FIXTURE, "bb_mora_reciente_meses": meses}
        agregado = agregar_bureau_balance(bb, puente, cortes)
        esperado = agregado["BB_MONTHS_SINCE_LAST_DPD_REL"].ge(-meses).astype("int8")
        assert agregado["BB_RECENT_DPD_FLAG_REL"].equals(esperado), meses


def test_la_cola_del_conteo_se_corta_sobre_los_creditos_con_historico(bb, puente):
    for corte in (1, 2, 3, 4):
        agregado = agregar_bureau_balance(
            bb, puente, {**CORTES_FIXTURE, "bb_many_credits_corte": corte}
        )
        esperado = agregado["BB_N_CREDITS_WBAL"].ge(corte).astype("int8")
        assert agregado["BB_MANY_CREDITS_FLAG"].equals(esperado), corte


# --- el puente, desde el nivel cliente --------------------------------------------------------


def test_el_credito_huerfano_no_llega_a_ningun_cliente(por_cliente, bb, puente):
    """Los créditos sin padre en bureau los tira el join interno, no el paso crédito."""
    con_padre = bb["SK_ID_BUREAU"].isin(puente.index).groupby(bb["SK_ID_BUREAU"]).first()
    esperados = len(PANEL) - len(HUERFANOS)
    assert por_cliente["BB_N_CREDITS_WBAL"].sum() == int(con_padre.sum()) == esperados


def test_el_cliente_sin_historico_mensual_no_sale_del_agregado(por_cliente):
    assert 800 not in por_cliente.index


def test_la_suma_de_meses_iguala_las_filas_enlazadas(por_cliente, bb, puente):
    """El assert del multi-tabla: la doble agregación no pierde ni duplica un mes.

    Quien impide que dos defectos de signo contrario lo dejen cuadrando no es este conteo sino
    el `validate` del join, que prueba el test de abajo: un crédito perdido y otro duplicado a la
    vez revientan antes de que la suma llegue a verlos.
    """
    enlazadas = bb["SK_ID_BUREAU"].isin(puente.index).sum()
    assert por_cliente["BB_MONTHS_TOTAL"].sum() == enlazadas


def test_un_credito_perdido_y_otro_duplicado_no_se_compensan(bb, puente):
    """La guarda por conteo que dos defectos dejan cuadrando, el pariente del patrón 8."""
    compensado = pd.concat([puente.drop(puente.index[0]), puente.iloc[[1]]])
    compensado.index.name = puente.index.name
    with pytest.raises(pd.errors.MergeError):
        agregar_bureau_balance(bb, compensado, CORTES_FIXTURE)


def test_un_puente_con_el_credito_repetido_revienta_en_vez_de_duplicarlo(bb, puente):
    """`puente_credito_cliente()` ya lo impide, pero el join no puede fiarse de quién le llama."""
    repetido = pd.concat([puente, puente.iloc[[0]]])
    with pytest.raises(pd.errors.MergeError):
        agregar_bureau_balance(bb, repetido, CORTES_FIXTURE)


# --- los cortes, por los dos lados ------------------------------------------------------------


def test_sin_cortes_revienta_mientras_los_medidos_esten_sin_fijar(bb, puente):
    with pytest.raises(ValueError, match="sin fijar"):
        agregar_bureau_balance(bb, puente)


def test_un_corte_desconocido_revienta_con_su_nombre(bb, puente):
    with pytest.raises(KeyError, match="bb_min_meses_reportado"):
        agregar_bureau_balance(bb, puente, {**CORTES_FIXTURE, "bb_min_meses_reportado": 6})


def test_la_ventana_de_la_trayectoria_se_puede_variar_por_el_argumento(bb):
    """Lo que el 3.9 necesita para su contraste, y que `valor()` por dentro no dejaba hacer."""
    corto = agregar_por_credito(bb, {"bb_min_meses_trayectoria": 2})
    largo = agregar_por_credito(bb, {"bb_min_meses_trayectoria": 90})
    assert corto["BB_CREDIT_TRAJECTORY"].notna().sum() > largo["BB_CREDIT_TRAJECTORY"].notna().sum()
    assert largo["BB_CREDIT_TRAJECTORY"].isna().all()


def test_el_nivel_cliente_reenvia_los_cortes_al_de_credito(bb, puente, por_cliente):
    """El punto de entrada del contraste del 3.9, visto en la trayectoria que sube a cliente."""
    largo = agregar_bureau_balance(bb, puente, {**CORTES_FIXTURE, "bb_min_meses_trayectoria": 90})
    assert por_cliente["BB_TRAJECTORY"].notna().any()
    assert largo["BB_TRAJECTORY"].isna().all()


def test_el_nivel_credito_corre_sin_haber_fijado_los_cortes_del_de_cliente(bb):
    """Solo resuelve el suyo, de dominio; si resolviera los cuatro reventaría en `valor()`."""
    assert len(agregar_por_credito(bb)) == len(PANEL)


# --- la unión a la lista de clientes ----------------------------------------------------------


# La lista de clientes tal como sale de `unir_bureau()`, con la mora de bureau de cada uno y lo que
# la unión con la mensual tiene que dar. El 850 es un cliente de bureau cuyos créditos no tienen
# panel, como el 800 pero sin mora, para que "toma el valor de bureau" no pase por un 1 constante.
UNION_MORA = {
    100: (1, 1.0),  # todo ciego, con su BB_ANY_DPD_FLAG a 0: la mora de bureau no se apaga
    200: (1, 1.0),  # ambas fuentes
    300: (0, 1.0),  # solo la mensual
    400: (0, 0.0),  # con panel y ninguna
    # solo la mensual otra vez, con banderas de mora apagadas que el 300 lleva encendidas: sin
    # ellos, leer la reciente, el fallido o la salida en mora en vez de la genérica pasaba en CI
    600: (0, 1.0),  # sin mora reciente ni fallido
    700: (0, 1.0),  # sin salir en mora
    800: (1, 1.0),  # sin panel, solo bureau
    850: (0, 0.0),  # sin panel y sin mora: 0 y no NaN
    999: (np.nan, np.nan),  # sin historial de bureau: NaN, la capa 1 no rellena
}


@pytest.fixture
def clientes():
    return pd.DataFrame(
        {
            "SK_ID_CURR": list(UNION_MORA),
            "BUREAU_OVERDUE_UNION": [b for b, _ in UNION_MORA.values()],
            "OTRA": range(len(UNION_MORA)),
        }
    )


def test_la_union_no_rellena_y_marca_la_presencia(clientes, por_cliente):
    unido = unir_bureau_balance(clientes, por_cliente)
    assert unido["HAS_BUREAU_BALANCE"].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0]
    sin_historico = unido[unido["HAS_BUREAU_BALANCE"].eq(0)]
    assert sin_historico[por_cliente.columns].isna().all().all()
    # y el cliente con histórico y sin mora conserva su 0, que el relleno no puede inventar
    assert unido.loc[unido["SK_ID_CURR"].eq(300), "BB_CREDITS_WITH_DPD_COUNT"].iloc[0] == 2


def test_la_union_no_altera_filas_ni_orden(clientes, por_cliente):
    unido = unir_bureau_balance(clientes, por_cliente)
    assert len(unido) == len(clientes)
    assert unido["SK_ID_CURR"].tolist() == clientes["SK_ID_CURR"].tolist()
    assert unido["OTRA"].tolist() == clientes["OTRA"].tolist()


def test_la_union_revienta_con_un_agregado_de_clientes_repetidos(clientes, por_cliente):
    """El left join contra una clave duplicada infla la matriz sin dar error."""
    repetido = pd.concat([por_cliente, por_cliente.loc[[100]]])
    with pytest.raises(pd.errors.MergeError):
        unir_bureau_balance(clientes, repetido)


# --- 3.7, la mora de las dos fuentes -----------------------------------------------------------


@pytest.mark.parametrize("cliente", sorted(UNION_MORA))
def test_la_union_de_mora_es_la_calculada_a_mano(clientes, por_cliente, cliente):
    unido = unir_bureau_balance(clientes, por_cliente).set_index("SK_ID_CURR")
    obtenido, esperado = unido.loc[cliente, "BB_OVERDUE_UNION"], UNION_MORA[cliente][1]
    assert (pd.isna(obtenido) and pd.isna(esperado)) or obtenido == esperado


def test_el_fixture_de_la_union_ejercita_cada_rama(clientes, por_cliente):
    """Guardián: cada combinación de fuentes que la unión distingue tiene su cliente.

    Y un cliente de solo mensual con cada bandera de mora vecina apagada, que es lo que separa la
    genérica de ellas. La relativa no tiene ninguno, porque en el panel toda mora cae dentro de los
    seis meses del fin de ventana: esa sustitución solo la caza la puerta sobre el dato real.
    """
    unido = unir_bureau_balance(clientes, por_cliente)
    bureau, mensual = unido["BUREAU_OVERDUE_UNION"], unido["BB_ANY_DPD_FLAG"]
    con_panel = unido["HAS_BUREAU_BALANCE"].eq(1)
    ciego = unido["BB_MONTHS_REPORTED"].eq(0)
    ramas = {
        "ciego con mora de bureau": ciego & bureau.eq(1),
        "ambas": bureau.eq(1) & mensual.eq(1),
        **{
            f"solo mensual sin {vecina}": bureau.eq(0) & mensual.eq(1) & unido[vecina].eq(0)
            for vecina in ("BB_RECENT_DPD_FLAG", "BB_WRITEOFF_FLAG", "BB_EXITS_IN_DPD_FLAG")
        },
        "ninguna con panel": con_panel & bureau.eq(0) & mensual.eq(0),
        "solo bureau sin panel": ~con_panel & bureau.eq(1),
        "sin mora y sin panel": ~con_panel & bureau.eq(0),
        "sin historial de bureau": bureau.isna(),
    }
    assert {rama: bool(m.any()) for rama, m in ramas.items()} == dict.fromkeys(ramas, True)


def test_la_union_sin_la_mora_de_bureau_revienta_con_su_nombre(clientes, por_cliente):
    """Sin ella el cliente con mora solo en bureau saldría sin marcar y nada avisaría."""
    with pytest.raises(ValueError, match="BUREAU_OVERDUE_UNION"):
        unir_bureau_balance(clientes.drop(columns="BUREAU_OVERDUE_UNION"), por_cliente)


def test_la_union_de_mora_va_en_float_aunque_el_lote_no_traiga_nan(clientes, por_cliente):
    con_bureau = clientes[clientes["BUREAU_OVERDUE_UNION"].notna()]
    assert unir_bureau_balance(con_bureau, por_cliente)["BB_OVERDUE_UNION"].dtype == "float64"


def test_un_panel_vacio_da_un_agregado_vacio_con_el_mismo_esquema(por_cliente, puente):
    """El frame sin tipos de la API, que es el patrón 12 un nivel más arriba."""
    for vacio in (panel().iloc[:0], pd.DataFrame(columns=list(COLUMNAS_ORIGEN))):
        agregado = agregar_bureau_balance(vacio, puente.iloc[:0], CORTES_FIXTURE)
        assert agregado.empty
        assert_frame_equal(agregado.dtypes.to_frame(), por_cliente.dtypes.to_frame())


# --- la puerta del 3.2 contra el dato real ----------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not all(
        (ruta("raw_data") / TABLE_FILES[t]).exists()
        for t in ("bureau_balance", "bureau", "application_train")
    ),
    reason="data/raw no viaja con el repo",
)

# Las dos poblaciones, como en el bloque 2. La del EDA son los 523.515 créditos enlazables a los
# 307.511 clientes de la tabla cruda, que es lo que él podía medir; la del pipeline son los
# 817.395 de la tabla entera, porque la agregación es capa 1 y no ve el TARGET. Las dos están
# bien y ninguna corrige a la otra, igual que `bureau_t` frente a `bureau`.
PUERTA = {
    "enlazables a train (EDA)": {
        "créditos": 523_515,
        "huecos en la ventana": 0,
        "censurados": 177_845,
        "% censurado": 33.97,
        "con algún DPD": 67_962,
        "sin estado numérico": 67_116,
        "% sin estado numérico": 12.82,
        "de ellos 100% X": 47_488,
        "un mes reportado o menos": 93_033,
        "% un mes reportado o menos": 17.77,
    },
    "tabla entera (pipeline)": {
        "créditos": 817_395,
        "huecos en la ventana": 0,
        "censurados": 206_430,
        "% censurado": 25.25,
        "con algún DPD": 103_264,
        "sin estado numérico": 130_368,
        "% sin estado numérico": 15.95,
        "de ellos 100% X": 85_569,
        "un mes reportado o menos": 169_718,
        "% un mes reportado o menos": 20.76,
    },
}

# El estado del último mes observado, que es el mecanismo detrás de la recencia relativa. Los
# cuatro que el EDA publica, más los tramos que completan la partición.
ULTIMO_ESTADO = {
    "enlazables a train (EDA)": {"C": 278_937, "X": 127_153, "0": 111_872, "5": 1_043},
    "tabla entera (pipeline)": {"C": 449_603, "X": 203_003, "0": 157_328, "5": 1_258},
}

# La puerta del 3.3. La trayectoria solo clasifica con las dos mitades reportadas, así que el
# contraste sin restringir es el del notebook, que leía la mitad ciega como 0; se recalcula aquí
# para que la diferencia quede medida y no supuesta.
PUERTA_TRAYECTORIA = {
    "enlazables a train (EDA)": {
        "ventana de 6 meses o más": 470_762,
        "con alguna mitad ciega": 274_388,
        "clases": {"sin mora": 157_225, "mejora": 13_664, "empeora": 16_380, "estable": 9_105},
        "sin restringir": {
            "sin mora": 404_631, "mejora": 39_931, "empeora": 17_095, "estable": 9_105
        },
    },
    "tabla entera (pipeline)": {
        "ventana de 6 meses o más": 751_038,
        "con alguna mitad ciega": 462_491,
        "clases": {"sin mora": 230_623, "mejora": 21_465, "empeora": 23_217, "estable": 13_242},
        "sin restringir": {
            "sin mora": 649_963, "mejora": 63_528, "empeora": 24_305, "estable": 13_242
        },
    },
}

# el orden del EDA, con las tasas de default de cada clase sobre los enlazables
TASA_TRAYECTORIA_EDA = {"sin mora": 8.14, "mejora": 9.57, "empeora": 11.98, "estable": 12.40}


@pytest.fixture(scope="module")
def dato_real():
    bb = load_table("bureau_balance")
    cred = agregar_por_credito(bb)
    bureau = load_table("bureau", usecols=["SK_ID_BUREAU", "SK_ID_CURR"])
    train = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    puente = puente_credito_cliente(bureau)
    # el TARGET solo existe para los créditos cuyo cliente está en train, con panel o sin él
    target = puente.map(train.set_index("SK_ID_CURR")["TARGET"]).dropna()
    # 300 créditos al azar más los casos que el azar puede no traer, que es lo que decide si el
    # fila a fila caza algo: el ciego, el que es todo X, el de más meses en mora, el de la ventana
    # más larga y el estable más largo, que es la clase más rara de la trayectoria
    estables = cred["BB_CREDIT_TRAJECTORY"].eq("estable")
    muestra = {
        *np.random.default_rng(0).choice(cred.index, 300, replace=False).tolist(),
        int(cred.index[cred["BB_MONTHS_REPORTED"].eq(0)][0]),
        int(cred.index[cred["BB_PCT_X"].eq(1.0)][0]),
        int(cred["BB_DPD_MONTHS"].idxmax()),
        int(cred["BB_MONTHS_OBS"].idxmax()),
        int(cred.loc[estables, "BB_MONTHS_OBS"].idxmax()),
    }
    return {
        "tabla entera (pipeline)": cred,
        "enlazables a train (EDA)": cred[cred.index.isin(target.index)],
        "target": target,
        "muestra": bb[bb.SK_ID_BUREAU.isin(muestra)],
        "bb": bb,
        "puente": puente,
        "train_curr": train["SK_ID_CURR"],
        "target_cliente": train.set_index("SK_ID_CURR")["TARGET"],
    }


@sin_dato_real
@pytest.mark.parametrize("poblacion", sorted(PUERTA))
def test_la_puerta_del_3_2_sobre_el_dato_real(dato_real, poblacion):
    c = dato_real[poblacion]
    ciego = c["BB_MONTHS_REPORTED"].eq(0)
    un_mes = c["BB_MONTHS_REPORTED"].le(1)
    medido = {
        "créditos": len(c),
        # la guarda ya habría reventado; se mide para que la puerta lo declare y no lo suponga
        "huecos en la ventana": int(
            (c.BB_WINDOW_END - c.BB_WINDOW_INI + 1 != c.BB_MONTHS_OBS).sum()
        ),
        "censurados": int(c["BB_CENSORED"].sum()),
        "% censurado": round(c["BB_CENSORED"].mean() * 100, 2),
        "con algún DPD": int(c["BB_LAST_DPD_MONTH"].notna().sum()),
        "sin estado numérico": int(ciego.sum()),
        "% sin estado numérico": round(ciego.mean() * 100, 2),
        "de ellos 100% X": int((ciego & c["BB_PCT_X"].eq(1.0)).sum()),
        "un mes reportado o menos": int(un_mes.sum()),
        "% un mes reportado o menos": round(un_mes.mean() * 100, 2),
    }
    assert medido == PUERTA[poblacion]


@sin_dato_real
@pytest.mark.parametrize("poblacion", sorted(ULTIMO_ESTADO))
def test_el_estado_final_sobre_el_dato_real(dato_real, poblacion):
    conteo = dato_real[poblacion]["BB_LAST_STATUS"].value_counts()
    assert {k: int(conteo[k]) for k in ULTIMO_ESTADO[poblacion]} == ULTIMO_ESTADO[poblacion]
    assert int(conteo.sum()) == PUERTA[poblacion]["créditos"]


@sin_dato_real
def test_sobre_el_dato_real_cada_credito_solo_da_lo_mismo_que_acompanado(dato_real):
    """La premisa de la capa 1 sobre el dato, que es donde el fixture sintético no llega.

    Es lo que `agg_bureau` ya hace sobre 301 clientes reales: el fila a fila del fixture solo caza
    el estadístico que algún crédito suyo deja ver, y la tabla trae perfiles que él no tiene.
    """
    agregado = dato_real["tabla entera (pipeline)"]
    for credito, filas in dato_real["muestra"].groupby("SK_ID_BUREAU"):
        assert_frame_equal(agregar_por_credito(filas), agregado.loc[[int(credito)]])


@sin_dato_real
def test_la_severidad_y_la_intensidad_son_nan_en_los_130368_ciegos(dato_real):
    """La regla de NaN sobre el dato real, que es donde el 0 se habría colado sin ruido."""
    c = dato_real["tabla entera (pipeline)"]
    ciego = c["BB_MONTHS_REPORTED"].eq(0)
    assert int(ciego.sum()) == 130_368
    assert c["BB_WORST"].isna().equals(ciego)
    assert c["BB_DPD_MONTHS"].isna().equals(ciego)


# --- la puerta del 3.3 contra el dato real ----------------------------------------------------


@sin_dato_real
@pytest.mark.parametrize("poblacion", sorted(PUERTA_TRAYECTORIA))
def test_la_puerta_del_3_3_sobre_el_dato_real(dato_real, poblacion):
    c = dato_real[poblacion]
    larga = c["BB_MONTHS_OBS"].ge(6)
    trayectoria = c["BB_CREDIT_TRAJECTORY"]
    # el notebook: la mitad ciega cuenta como 0 y todo crédito largo recibe clase
    ant = c.loc[larga, "BB_WORST_OLD_HALF"].fillna(0)
    rec = c.loc[larga, "BB_WORST_RECENT_HALF"].fillna(0)
    notebook = np.select(
        [(ant == 0) & (rec == 0), rec > ant, rec < ant],
        ["sin mora", "empeora", "mejora"],
        "estable",
    )
    medido = {
        "ventana de 6 meses o más": int(larga.sum()),
        "con alguna mitad ciega": int((larga & trayectoria.isna()).sum()),
        "clases": {k: int(v) for k, v in trayectoria.value_counts().items()},
        "sin restringir": {k: int(v) for k, v in pd.Series(notebook).value_counts().items()},
    }
    # las clases se cuentan sobre todos los créditos, así que un corto clasificado ya la descuadra
    assert medido == PUERTA_TRAYECTORIA[poblacion]


@sin_dato_real
def test_la_trayectoria_reproduce_las_tasas_del_eda(dato_real):
    """Con la jerarquía en el orden del EDA, y la persistencia por encima del deterioro."""
    c = dato_real["enlazables a train (EDA)"]
    target = dato_real["target"].reindex(c.index)
    tasas = target.groupby(c["BB_CREDIT_TRAJECTORY"], observed=True).mean().mul(100).round(2)
    assert tasas.to_dict() == TASA_TRAYECTORIA_EDA


# --- la puerta del 3.4 contra el dato real ------------------------------------------------------

sin_split = pytest.mark.skipif(
    not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="el split no viaja con el repo",
)

# El mismo desfase que ya declara el bloque 2 entre la tabla cruda del EDA (los 307.511 clientes
# de train) y la población de modelado que deja el split (307.492): mueve las tres últimas cifras,
# porque son las únicas que dependen de la lista de clientes.
PUERTA_PUENTE = {
    "crudo": {
        "creditos con panel": 774_354,
        "huerfanos": 43_041,
        "filas huerfanas": 3_120_184,
        "filas con padre": 24_179_741,
        "filas con cliente fuera": 9_478_129,
        "filas analizables": 14_701_612,
        "creditos enlazables": 523_515,
        "clientes con panel": 92_231,
    },
    "modelado": {
        "creditos con panel": 774_354,
        "huerfanos": 43_041,
        "filas huerfanas": 3_120_184,
        "filas con padre": 24_179_741,
        "filas con cliente fuera": 9_479_158,
        "filas analizables": 14_700_583,
        "creditos enlazables": 523_473,
        "clientes con panel": 92_220,
    },
}


@sin_dato_real
@sin_split
@pytest.mark.parametrize("poblacion", sorted(PUERTA_PUENTE))
def test_la_puerta_del_3_4_sobre_el_dato_real(dato_real, poblacion):
    cred, puente, bb = dato_real["tabla entera (pipeline)"], dato_real["puente"], dato_real["bb"]
    con_panel = puente.index.isin(cred.index)
    huerfano = ~cred.index.isin(puente.index)
    filas_huerfanas = int(bb["SK_ID_BUREAU"].astype("int64").isin(cred.index[huerfano]).sum())
    enlazado = cred.join(puente, how="inner")
    clientes = (
        dato_real["train_curr"] if poblacion == "crudo" else cargar_split()["SK_ID_CURR"]
    )
    dentro = enlazado["SK_ID_CURR"].isin(clientes)
    medido = {
        "creditos con panel": int(con_panel.sum()),
        "huerfanos": int(huerfano.sum()),
        "filas huerfanas": filas_huerfanas,
        "filas con padre": int(enlazado["BB_MONTHS_OBS"].sum()),
        "filas con cliente fuera": int(enlazado.loc[~dentro, "BB_MONTHS_OBS"].sum()),
        "filas analizables": int(enlazado.loc[dentro, "BB_MONTHS_OBS"].sum()),
        "creditos enlazables": int(dentro.sum()),
        "clientes con panel": enlazado.loc[dentro, "SK_ID_CURR"].nunique(),
    }
    assert medido == PUERTA_PUENTE[poblacion]


# El perfil de los huérfanos frente a los créditos con padre: la mediana de meses es por crédito
# (BB_MONTHS_OBS) y las tres proporciones son por fila de bb. Es lo que deja claro que lo que se
# pierde al enlazar es historia antigua, cerrada y limpia.
PERFIL_HUERFANOS = {
    "huerfanos": {"meses (mediana)": 97.0, "% C": 67.03, "% X": 22.29, "% DPD": 0.93},
    "con padre": {"meses (mediana)": 25.0, "% C": 47.79, "% X": 21.15, "% DPD": 1.30},
}


@sin_dato_real
def test_el_perfil_de_los_huerfanos_sobre_el_dato_real(dato_real):
    cred, puente = dato_real["tabla entera (pipeline)"], dato_real["puente"]
    b = limpiar_bureau_balance(dato_real["bb"])
    huerfano_credito = ~cred.index.isin(puente.index)
    fila_huerfana = b["SK_ID_BUREAU"].astype("int64").isin(cred.index[huerfano_credito])
    grupos = {
        "huerfanos": (huerfano_credito, fila_huerfana),
        "con padre": (~huerfano_credito, ~fila_huerfana),
    }
    for nombre, (m_cred, m_fila) in grupos.items():
        s = b.loc[m_fila, "STATUS"].astype(str)
        medido = {
            "meses (mediana)": cred.loc[m_cred, "BB_MONTHS_OBS"].median(),
            "% C": round(s.eq("C").mean() * 100, 2),
            "% X": round(s.eq("X").mean() * 100, 2),
            "% DPD": round(s.isin(list("12345")).mean() * 100, 2),
        }
        assert medido == PERFIL_HUERFANOS[nombre]


# --- la puerta del 3.5 contra el dato real ----------------------------------------------------

# Las tres poblaciones. La agregación se calcula sobre la tabla entera y se reindexa a la lista de
# clientes que toque, así que la cifra depende de esa lista: las del EDA están sobre los 307.511 de
# la tabla cruda y la población de modelado son los 307.492 que deja la limpieza, el mismo desfase
# que el bloque 0 declara con el 8,0729% frente al 8,0734%.
PUERTA_CLIENTE = {
    "tabla entera (pipeline)": {
        "clientes con historico": 134_542,
        "cobertura completa": 134_108,
        "cobertura parcial": 434,
        "suma de BB_MONTHS_TOTAL": 24_179_741,
        "con mora": 48_522,
        "fallidos": 2_986,
        "sin mes reportado": 3_769,
        "% sin mes reportado": 2.80,
        "con algun mes reportado": 130_773,
        "con denominador suficiente": 122_845,
    },
    "train del EDA (307.511)": {
        "clientes con historico": 92_231,
        "cobertura completa": 92_028,
        "cobertura parcial": 203,
        "suma de BB_MONTHS_TOTAL": 14_701_612,
        "con mora": 31_052,
        "fallidos": 2_206,
        "sin mes reportado": 2_375,
        "% sin mes reportado": 2.58,
        "con algun mes reportado": 89_856,
        "con denominador suficiente": 83_971,
    },
    "modelado (307.492)": {
        "clientes con historico": 92_220,
        "cobertura completa": 92_017,
        "cobertura parcial": 203,
        "suma de BB_MONTHS_TOTAL": 14_700_583,
        "con mora": 31_048,
        "fallidos": 2_206,
        "sin mes reportado": 2_375,
        "% sin mes reportado": 2.58,
        "con algun mes reportado": 89_845,
        "con denominador suficiente": 83_960,
    },
}


@pytest.fixture(scope="module")
def cliente_real(dato_real):
    """El agregado a nivel cliente sobre la tabla entera, con los cortes de referencia del EDA."""
    return agregar_bureau_balance(dato_real["bb"], dato_real["puente"], REFERENCIA)


@sin_dato_real
@sin_split
@pytest.mark.parametrize("poblacion", sorted(PUERTA_CLIENTE))
def test_la_puerta_del_3_5_sobre_el_dato_real(dato_real, cliente_real, poblacion):
    clientes = {
        "tabla entera (pipeline)": cliente_real.index,
        "train del EDA (307.511)": dato_real["train_curr"],
        "modelado (307.492)": cargar_split()["SK_ID_CURR"],
    }[poblacion]
    a = cliente_real[cliente_real.index.isin(clientes)]
    # la cobertura es del cliente y no del crédito: o el buró reporta su historial entero o ninguno
    puente = dato_real["puente"]
    con_panel = puente[puente.index.isin(dato_real["tabla entera (pipeline)"].index)].value_counts()
    completa = con_panel.reindex(a.index).eq(puente.value_counts().reindex(a.index))
    sin_rep = a["BB_MONTHS_REPORTED"].eq(0)
    medido = {
        "clientes con historico": len(a),
        "cobertura completa": int(completa.sum()),
        "cobertura parcial": int((~completa).sum()),
        "suma de BB_MONTHS_TOTAL": int(a["BB_MONTHS_TOTAL"].sum()),
        "con mora": int(a["BB_ANY_DPD_FLAG"].sum()),
        "fallidos": int(a["BB_WRITEOFF_FLAG"].sum()),
        "sin mes reportado": int(sin_rep.sum()),
        "% sin mes reportado": round(sin_rep.mean() * 100, 2),
        "con algun mes reportado": int((~sin_rep).sum()),
        "con denominador suficiente": int(a["BB_PCT_MONTHS_DPD"].notna().sum()),
    }
    assert medido == PUERTA_CLIENTE[poblacion]


@sin_dato_real
def test_la_doble_agregacion_no_pierde_ni_duplica_un_mes(dato_real, cliente_real):
    """El assert del multi-tabla sobre el dato: la suma iguala las filas con padre en bureau."""
    enlazadas = dato_real["bb"]["SK_ID_BUREAU"].isin(dato_real["puente"].index).sum()
    assert int(cliente_real["BB_MONTHS_TOTAL"].sum()) == int(enlazadas)


@sin_dato_real
def test_sobre_el_dato_real_cada_cliente_solo_da_lo_mismo_que_acompanado(dato_real, cliente_real):
    """La premisa de la capa 1 a nivel cliente, donde el fixture sintético no llega."""
    bb, puente = dato_real["bb"], dato_real["puente"]
    clientes = puente.reindex(dato_real["muestra"]["SK_ID_BUREAU"].unique()).dropna().unique()
    for cliente in clientes:
        creditos = puente.index[puente.eq(cliente)]
        solo = bb[bb["SK_ID_BUREAU"].astype("int64").isin(creditos)]
        assert_frame_equal(
            agregar_bureau_balance(solo, puente[puente.eq(cliente)], REFERENCIA),
            cliente_real.loc[[cliente]],
        )


# --- la puerta del 3.6 contra el dato real ----------------------------------------------------

# Las del EDA sobre los 307.511 de la tabla cruda, más la población de modelado y la tabla entera.
# Las banderas de trayectoria cuentan sus marcados, y sus no nulos son los evaluables.
PUERTA_DERIVADAS = {
    "tabla entera (pipeline)": {
        "recencia relativa no nula": 48_522,
        "mora reciente relativa": 20_544,
        "mora reciente absoluta": 17_854,
        "sale en mora": 5_868,
        "todo cerrado": 24_329,
        "muchos creditos": 1_215,
        "trayectoria evaluable": 100_245,
        "persistentes": 9_140,
        "empeoran": 13_986,
        "mejoran": 10_309,
    },
    "train del EDA (307.511)": {
        "recencia relativa no nula": 31_052,
        "mora reciente relativa": 14_288,
        "mora reciente absoluta": 11_602,
        "sale en mora": 4_274,
        "todo cerrado": 18_997,
        "muchos creditos": 749,
        "trayectoria evaluable": 66_663,
        "persistentes": 6_013,
        "empeoran": 9_411,
        "mejoran": 6_098,
    },
    "modelado (307.492)": {
        "recencia relativa no nula": 31_048,
        "mora reciente relativa": 14_286,
        "mora reciente absoluta": 11_600,
        "sale en mora": 4_274,
        "todo cerrado": 18_997,
        "muchos creditos": 749,
        "trayectoria evaluable": 66_653,
        "persistentes": 6_013,
        "empeoran": 9_410,
        "mejoran": 6_097,
    },
}

# la tasa de default de cada trayectoria de cliente, la del notebook sobre los 307.511
TASA_TRAYECTORIA_CLIENTE_EDA = {
    "sin mora": 7.10, "mejora": 8.72, "empeora": 11.16, "estable": 11.94
}


@sin_dato_real
@sin_split
@pytest.mark.parametrize("poblacion", sorted(PUERTA_DERIVADAS))
def test_la_puerta_del_3_6_sobre_el_dato_real(dato_real, cliente_real, poblacion):
    clientes = {
        "tabla entera (pipeline)": cliente_real.index,
        "train del EDA (307.511)": dato_real["train_curr"],
        "modelado (307.492)": cargar_split()["SK_ID_CURR"],
    }[poblacion]
    a = cliente_real[cliente_real.index.isin(clientes)]
    evaluable = a["BB_TRAJECTORY"].notna()
    for bandera in BANDERAS_TRAYECTORIA:
        assert a[bandera].notna().equals(evaluable), bandera
    medido = {
        "recencia relativa no nula": int(a["BB_MONTHS_SINCE_LAST_DPD_REL"].notna().sum()),
        "mora reciente relativa": int(a["BB_RECENT_DPD_FLAG_REL"].sum()),
        "mora reciente absoluta": int(a["BB_RECENT_DPD_FLAG"].sum()),
        "sale en mora": int(a["BB_EXITS_IN_DPD_FLAG"].sum()),
        "todo cerrado": int(a["BB_ALL_CLOSED_FLAG"].sum()),
        "muchos creditos": int(a["BB_MANY_CREDITS_FLAG"].sum()),
        "trayectoria evaluable": int(evaluable.sum()),
        "persistentes": int(a["BB_PERSISTENT_DPD_FLAG"].sum()),
        "empeoran": int(a["BB_WORSENING_DPD_FLAG"].sum()),
        "mejoran": int(a["BB_RECOVERED_DPD_FLAG"].sum()),
    }
    assert medido == PUERTA_DERIVADAS[poblacion]


@sin_dato_real
def test_la_trayectoria_de_cliente_reproduce_las_tasas_del_eda(dato_real, cliente_real):
    a = cliente_real[cliente_real.index.isin(dato_real["train_curr"])]
    a = a[a["BB_TRAJECTORY"].notna()]
    target = dato_real["target_cliente"].reindex(a.index)
    tasas = target.groupby(a["BB_TRAJECTORY"], observed=True).mean().mul(100).round(2)
    assert tasas.to_dict() == TASA_TRAYECTORIA_CLIENTE_EDA


# --- la puerta del 3.7 contra el dato real ----------------------------------------------------

# Sobre los clientes con historial de bureau, que es la población de medición de la unión, y la
# asociación entre las dos fuentes sobre los que tienen panel, como la celda 60 del notebook. La del
# EDA sale idéntica, y la de modelado pierde 7 marcados de la unión y 6 de bureau.
PUERTA_UNION = {
    "train del EDA (307.511)": {
        "con historial de bureau": 263_491,
        "sin historial de bureau": 44_020,
        "BB_OVERDUE_UNION": 88_419,
        "BUREAU_OVERDUE_UNION": 72_419,
        "V de Cramer": 0.3945,
        "Jaccard": 0.3889,
        "solo mensual": 16_000,
        "solo bureau": 7_653,
    },
    "modelado (307.492)": {
        "con historial de bureau": 263_475,
        "sin historial de bureau": 44_017,
        "BB_OVERDUE_UNION": 88_412,
        "BUREAU_OVERDUE_UNION": 72_413,
        "V de Cramer": 0.3945,
        "Jaccard": 0.3889,
        "solo mensual": 15_999,
        "solo bureau": 7_653,
    },
}

# Crédito a crédito sobre los 523.515 enlazables a train: n y tasa de default de cada grupo. Solo
# tiene la población del EDA, porque mide el dato con el TARGET y no una salida de la agregación.
CONCORDANCIA_CREDITO = {
    "creditos": 523_515,
    "% acuerdo": 85.59,
    "ninguna": (435_381, 7.86),
    "solo bureau": (20_172, 10.26),
    "solo mensual": (55_281, 10.09),
    "ambas": (12_681, 11.41),
}


@pytest.fixture(scope="module")
def bureau_real():
    """`bureau` entero y su agregado, con los cortes `medido` en su referencia del EDA.

    La cola del conteo no tiene referencia y no toca la mora, así que va con los 18 del 2.3.
    """
    bureau = load_table("bureau")
    medidos = [n for n in CORTES_BUREAU if parametro(n).procedencia == "medido"]
    cortes = {**{n: parametro(n).valor_referencia for n in medidos}, "bureau_count_cola": 18}
    return bureau, agregar_bureau(bureau, cortes)


@sin_dato_real
@sin_split
@pytest.mark.parametrize("poblacion", sorted(PUERTA_UNION))
def test_la_puerta_del_3_7_sobre_el_dato_real(dato_real, cliente_real, bureau_real, poblacion):
    ids = {
        "train del EDA (307.511)": dato_real["train_curr"],
        "modelado (307.492)": cargar_split()["SK_ID_CURR"],
    }[poblacion]
    base = unir_bureau(pd.DataFrame({"SK_ID_CURR": ids.to_numpy()}), bureau_real[1])
    unido = unir_bureau_balance(base, cliente_real)
    con = unido[unido["HAS_BUREAU_HISTORY"].eq(1)]
    assert unido["BB_OVERDUE_UNION"].isna().equals(unido["HAS_BUREAU_HISTORY"].eq(0))
    con_panel = unido[unido["HAS_BUREAU_BALANCE"].eq(1)]
    mensual, bureau = con_panel["BB_ANY_DPD_FLAG"].eq(1), con_panel["BUREAU_OVERDUE_UNION"].eq(1)
    medido = {
        "con historial de bureau": len(con),
        "sin historial de bureau": len(unido) - len(con),
        "BB_OVERDUE_UNION": int(con["BB_OVERDUE_UNION"].sum()),
        "BUREAU_OVERDUE_UNION": int(con["BUREAU_OVERDUE_UNION"].sum()),
        "V de Cramer": round(
            association(pd.crosstab(mensual, bureau), method="cramer", correction=True), 4
        ),
        "Jaccard": round(int((mensual & bureau).sum()) / int((mensual | bureau).sum()), 4),
        "solo mensual": int((mensual & ~bureau).sum()),
        "solo bureau": int((bureau & ~mensual).sum()),
    }
    assert medido == PUERTA_UNION[poblacion]


@sin_dato_real
def test_la_concordancia_credito_a_credito_reproduce_la_del_eda(dato_real, bureau_real):
    """Ninguna fuente domina, que es lo que justifica la unión.

    Repite aquí la mora de tres condiciones de `agregar_bureau()`, porque el pipeline no la deja a
    nivel crédito; la lee del signo de la foto, como ella.
    """
    b = limpiar_bureau(bureau_real[0])
    mora = (
        b["AMT_CREDIT_MAX_OVERDUE" + SUFIJO_SIGNO].gt(0)
        | b["AMT_CREDIT_SUM_OVERDUE" + SUFIJO_SIGNO].gt(0)
        | b["CREDIT_DAY_OVERDUE"].gt(0)
    ).set_axis(b["SK_ID_BUREAU"].astype("int64"))
    cred = dato_real["enlazables a train (EDA)"]
    de_bureau = mora.reindex(cred.index)
    assert de_bureau.notna().all(), "un crédito enlazable sin su fila en bureau"
    de_bureau = de_bureau.astype(bool)
    mensual = cred["BB_DPD_MONTHS"].gt(0)
    target = dato_real["target"].reindex(cred.index)
    grupos = {
        "ninguna": ~de_bureau & ~mensual,
        "solo bureau": de_bureau & ~mensual,
        "solo mensual": ~de_bureau & mensual,
        "ambas": de_bureau & mensual,
    }
    medido = {
        "creditos": len(cred),
        "% acuerdo": round(de_bureau.eq(mensual).mean() * 100, 2),
        **{k: (int(m.sum()), round(target[m].mean() * 100, 2)) for k, m in grupos.items()},
    }
    assert medido == CONCORDANCIA_CREDITO
