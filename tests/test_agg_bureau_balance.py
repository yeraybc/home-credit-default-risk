"""Tests de la agregación de bureau_balance a nivel crédito (puntos 3.2 y 3.3).

Todos sobre un panel sintético, así que corren en CI sin los CSV, salvo la puerta contra el dato
real del final, que se salta sin `bureau_balance.csv`, `bureau.csv` y `application_train.csv` y
solo corre en local.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_bureau_balance import COLUMNAS_ORIGEN, TRAYECTORIAS, agregar_por_credito
from src.features.cleaning import DTYPE_STATUS, limpiar_bureau_balance

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
    # Del 8 en adelante, la trayectoria: todos con la ventana mínima. El 4 hace de borde que no
    # entra, con 5 meses y las dos mitades reportadas, y el 7 lo usa un test de la guarda.
    # 8: sin mora con la ventana justa de 6, el borde que sí entra
    8: (-5, ["0", "0", "X", "0", "0", "C"]),
    # 9: la única mora en el mes central de una ventana impar, que va a la mitad antigua; si fuera
    # a la reciente saldría empeora
    9: (-6, ["0", "0", "0", "1", "0", "0", "0"]),
    # 10: mejora por severidad sin salir de mora, censurado y en el borde antiguo del panel, donde
    # `ini + fin` ya no cabe en el int8 del mes y un punto medio sumado se desbordaría
    10: (-95, ["3", "3", "3", "1", "1", "1"]),
    # 11: empeora
    11: (-5, ["0", "0", "0", "0", "1", "2"]),
    # 12: estable, con el mismo peor estado en las dos mitades
    12: (-7, ["1", "0", "0", "0", "0", "0", "1", "X"]),
    # 13: la reciente toda C, que el notebook leía como 0 y clasificaba de mejora
    13: (-5, ["1", "0", "0", "C", "C", "C"]),
    # 14: la antigua toda X, que el notebook clasificaba de empeora
    14: (-5, ["X", "X", "X", "0", "1", "0"]),
}

# Los seis primeros no llegan a la ventana mínima, así que su trayectoria es NaN aunque sus mitades
# tengan valor
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
        "BB_MONTHS_OBS": 6, "BB_MONTHS_REPORTED": 6, "BB_WORST": 3.0, "BB_PCT_X": 0.0,
        "BB_WINDOW_INI": -95, "BB_WINDOW_END": -90, "BB_DPD_MONTHS": 6.0, "BB_PCT_DPD": 1.0,
        "BB_LAST_STATUS": "1", "BB_LAST_DPD_MONTH": -90.0, "BB_CENSORED": 1,
        "BB_WORST_OLD_HALF": 3.0, "BB_WORST_RECENT_HALF": 1.0, "BB_CREDIT_TRAJECTORY": "mejora",
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
    impar = agregado["BB_MONTHS_OBS"].mod(2).eq(1) & larga
    medio = (agregado["BB_WINDOW_INI"] + agregado["BB_WINDOW_END"]) // 2
    mora_en_el_medio = impar & agregado["BB_LAST_DPD_MONTH"].eq(medio) & agregado["BB_WORST"].gt(0)
    assert mora_en_el_medio.any(), "falta la ventana impar con la mora en el mes central"
    extremos = agregado["BB_WINDOW_INI"].astype(int) + agregado["BB_WINDOW_END"].astype(int)
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
    # el puente de producción es el 3.4; aquí solo hace falta el TARGET de los créditos enlazables
    enlazados = bureau.merge(train, on="SK_ID_CURR")
    target = pd.Series(enlazados.TARGET.to_numpy(), index=enlazados.SK_ID_BUREAU.astype("int64"))
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
