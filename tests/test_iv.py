"""Tests de iv.py: el binning, la tabla de WoE, el IV y el IV condicionado."""

import numpy as np
import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features import agg_bureau, agg_bureau_balance, agg_previous
from src.features.build_features import (
    NOMBRE_FICHERO_CORTES,
    cargar_cortes,
    ensamblar_auxiliares,
    preparar_application,
)
from src.features.cleaning import COLUMNAS_PROVISIONALES
from src.features.iv import (
    BANDERAS_RARAS,
    CANDIDATAS_IV,
    NULO,
    Candidata,
    calcular_iv,
    informe_iv,
    iv_condicionado,
    tabla_woe,
    tramos,
)
from src.features.params import valor
from src.features.pipeline import PRESENCIA_AUX, columnas_declaradas
from src.features.recipes import cargar_receta
from src.features.selection import recomendar_codificacion
from src.features.split import NOMBRE_FICHERO, cargar_split, solo_train


def _repetir(tramos_y_recuentos):
    """Una columna categórica y su TARGET a partir de (tramo, malos, clientes)."""
    x, y = [], []
    for tramo, malos, n in tramos_y_recuentos:
        x += [tramo] * n
        y += [1] * malos + [0] * (n - malos)
    return pd.Series(x), pd.Series(y)


@pytest.fixture(scope="module")
def senal():
    """Una continua que mueve la probabilidad de default, una independiente y el TARGET."""
    rng = np.random.default_rng(7)
    n = 100_000
    x, z = rng.normal(size=n), rng.normal(size=n)
    # intercepto para una tasa en torno al 8%, como la cartera
    p = 1 / (1 + np.exp(-(-2.6 + 0.6 * x + 0.6 * z)))
    y = pd.Series((rng.random(n) < p).astype(int))
    return pd.Series(x, name="x"), pd.Series(z, name="z"), y


# --- el IV a mano -----------------------------------------------------------------------------


def test_el_iv_de_dos_tramos_sin_suavizar_es_el_de_la_formula():
    # 80 malos y 920 buenos: A tiene 40 y 660 (5,7%), B tiene 40 y 260 (13,3%)
    # (0,5 − 660/920)·ln(0,5/(660/920)) + (0,5 − 260/920)·ln(0,5/(260/920))
    x, y = _repetir([("A", 40, 700), ("B", 40, 300)])
    assert calcular_iv(x, y, alfa=0) == pytest.approx(0.20251265, abs=1e-8)


def test_el_iv_de_tres_tramos_con_el_prior_repartido_por_la_tasa_global():
    # alfa 20 con la tasa global del 8%: 1,6 de prior a los malos y 18,4 a los buenos de cada tramo
    x, y = _repetir([("A", 20, 500), ("B", 30, 300), ("C", 30, 200)])
    tabla = tabla_woe(x, y, alfa=20)
    assert tabla.loc["A", "parte_malos"] == pytest.approx((20 + 1.6) / 80)
    assert tabla.loc["A", "parte_buenos"] == pytest.approx((480 + 18.4) / 920)
    assert calcular_iv(x, y, alfa=20) == pytest.approx(0.33303399, abs=1e-8)


def test_el_woe_positivo_es_el_tramo_de_mas_riesgo():
    x, y = _repetir([("A", 40, 700), ("B", 40, 300)])
    woe = tabla_woe(x, y)["woe"]
    assert woe["B"] > 0 > woe["A"]


def test_el_prior_impide_el_iv_infinito_de_un_tramo_sin_malos():
    x, y = _repetir([("A", 0, 200), ("B", 80, 800)])
    assert np.isfinite(calcular_iv(x, y))
    # la otra dirección: sin prior el tramo sin malos es log de cero
    with np.errstate(divide="ignore"):
        assert np.isinf(calcular_iv(x, y, alfa=0))


# --- propiedades --------------------------------------------------------------------------------


def test_una_columna_permutada_tiene_un_iv_casi_nulo(senal):
    x, _, y = senal
    permutada = pd.Series(np.random.default_rng(0).permutation(x.to_numpy()))
    assert calcular_iv(x, y) > 0.1, "el fixture perdió la señal que la permutación tiene que borrar"
    assert calcular_iv(permutada, y) < 0.005


def test_el_iv_no_depende_del_orden_de_las_filas(senal):
    x, _, y = senal
    orden = np.random.default_rng(1).permutation(len(x))
    barajada = x.iloc[orden].reset_index(drop=True)
    assert calcular_iv(barajada, y.iloc[orden]) == pytest.approx(calcular_iv(x, y), abs=1e-12)


def test_el_iv_nunca_es_negativo():
    x, y = _repetir([("A", 1, 300), ("B", 3, 50), ("C", 76, 650)])
    assert (tabla_woe(x, y)["iv"] >= 0).all()


# --- el binning ---------------------------------------------------------------------------------


def test_una_continua_sale_en_n_bins_max_cuantiles(senal):
    x, _, _ = senal
    corte = tramos(x)
    assert corte.nunique() == 10
    # sin nulos, la categoría del nulo existe y queda vacía
    assert (corte.value_counts().drop(NULO) == 10_000).all()


def test_los_empates_colapsan_tramos_en_vez_de_partirlos():
    conteo = pd.Series([0] * 600 + [1] * 200 + [2] * 100 + list(range(3, 103)))
    corte = tramos(conteo)
    assert corte.nunique() < 10
    # los 600 ceros van enteros a un mismo tramo
    assert corte[conteo == 0].nunique() == 1


def test_el_nulo_es_tramo_propio_y_conserva_su_senal():
    rng = np.random.default_rng(3)
    ruido = pd.Series(np.r_[rng.normal(size=9_000), [np.nan] * 1_000])
    y = pd.Series(np.r_[(rng.random(9_000) < 0.07), (rng.random(1_000) < 0.20)].astype(int))
    assert ruido.isna().sum() == 1_000, "el fixture perdió el grupo ausente"
    tabla = tabla_woe(tramos(ruido), y)
    assert NULO in tabla.index
    assert tabla.loc[NULO, "woe"] > 0
    # la señal está en el nulo: sin él la columna es ruido
    assert calcular_iv(ruido, y) > 0.05 > calcular_iv(ruido.dropna(), y[ruido.notna()])


@pytest.mark.parametrize(
    "bandera, esperado",
    [
        (pd.Series([0, 1, 1, 0, 1]), 2),
        (pd.Series([0, 1, np.nan, 0, 1]), 3),
        (pd.Series([True, False, True, True, False]), 2),
    ],
)
def test_una_bandera_va_en_dos_tramos_mas_el_nulo(bandera, esperado):
    assert tramos(bandera).nunique() == esperado


def test_una_categorica_va_por_nivel_con_el_nulo_aparte():
    corte = tramos(pd.Series(["a", "b", None, "c", "a"]))
    assert set(corte.unique()) == {"a", "b", "c", NULO}


def test_una_categoria_literal_como_la_clave_del_nulo_revienta():
    with pytest.raises(ValueError, match="categoría literal"):
        tramos(pd.Series(["a", NULO]))


# --- el IV condicionado -------------------------------------------------------------------------

# sin la guarda del tramo que no separa, el nulo vacío de un estrato continuo divide 0 entre 0 y
# el tramo sin malos sale NaN, que la suma se salta en silencio: el resultado no cambia, el aviso sí
sin_avisos = pytest.mark.filterwarnings("error")


@sin_avisos
def test_una_copia_del_estrato_no_aporta_nada_dentro_de_sus_tramos(senal):
    x, _, y = senal
    copia = 2 * x + 1
    assert calcular_iv(copia, y) > 0.1, "la copia tiene que traer la señal del estrato"
    assert iv_condicionado(copia, x, y) < 1e-9


@sin_avisos
def test_una_senal_independiente_conserva_su_iv_dentro_del_estrato(senal):
    x, z, y = senal
    marginal = calcular_iv(z, y)
    assert marginal > 0.1
    assert iv_condicionado(z, x, y) == pytest.approx(marginal, rel=0.25)


@sin_avisos
def test_un_tramo_del_estrato_sin_malos_aporta_cero_y_cada_tramo_pesa_lo_que_mide():
    # tamaños distintos a propósito: con dos iguales, pesar por tamaño o a partes iguales coincide
    estrato = pd.Series(["a"] * 100 + ["b"] * 300)
    y = pd.Series([0] * 100 + [1] * 60 + [0] * 240)
    serie = pd.Series(np.r_[np.zeros(100), np.ones(60), np.zeros(240)])
    solo_b = tabla_woe(tramos(serie[100:]).reset_index(drop=True), y[100:].reset_index(drop=True))
    # el tramo `a` no tiene malos, así que solo cuenta `b`, con sus 300 de los 400
    assert iv_condicionado(serie, estrato, y) == pytest.approx(solo_b["iv"].sum() * 0.75)


# --- la lista cerrada del 5.5 -----------------------------------------------------------------

# cada auxiliar con su bandera de presencia, y los módulos que declaran lo que añaden sin receta
TABLAS = {
    "bureau": ("HAS_BUREAU_HISTORY", agg_bureau),
    "bureau_balance": ("HAS_BUREAU_BALANCE", agg_bureau_balance),
    "previous_application": ("HAS_PREV_APPLICATION", agg_previous),
}


def _tabla_de(columna):
    """La auxiliar que produce la columna, por su receta o por su `COLUMNAS_SIN_RECETA`."""
    for tabla, (_, modulo) in TABLAS.items():
        nombres = {f["nombre"] for f in cargar_receta(tabla)["features"]}
        if columna in nombres | set(modulo.COLUMNAS_SIN_RECETA):
            return tabla
    return None


def _pendientes_de_selection():
    """Las de la tabla principal que `recomendar_codificacion()` deja escritas para el IV."""
    tabla = recomendar_codificacion(pd.DataFrame({c: [0, 1] for c in columnas_declaradas()}))
    deja = tabla["Estrategia Recomendada"].str.contains("IV") | tabla["Detalle"].str.contains(
        "decide con el IV|decide el IV|cae por IV|se toma con el IV"
    )
    return set(tabla.loc[deja, "Variable"])


def test_la_lista_del_iv_contiene_todo_lo_que_sus_fuentes_dejan_al_iv():
    """Una `decision: iv` nueva en una receta, o una columna sin receta nueva, la hace fallar."""
    fuentes = {
        f["nombre"]
        for tabla in TABLAS
        for f in cargar_receta(tabla)["features"]
        if f["decision"] == "iv"
    }
    assert len(fuentes) == 11
    selection = _pendientes_de_selection()
    assert len(selection) == 8, selection
    fuentes |= selection
    for _, modulo in TABLAS.values():
        fuentes |= set(modulo.COLUMNAS_SIN_RECETA)
    fuentes |= set(COLUMNAS_PROVISIONALES) | set(BANDERAS_RARAS)
    assert fuentes <= set(CANDIDATAS_IV), sorted(fuentes - set(CANDIDATAS_IV))
    assert len(CANDIDATAS_IV) == 38


def test_las_de_receta_son_decision_iv_en_la_suya():
    """Al revés: lo que la lista atribuye a una receta sigue siendo `decision: iv` en ella."""
    for columna, candidata in CANDIDATAS_IV.items():
        if candidata.fuente.startswith("receta de "):
            tabla = candidata.fuente.removeprefix("receta de ")
            decision = {f["nombre"]: f["decision"] for f in cargar_receta(tabla)["features"]}
            assert decision.get(columna) == "iv", columna


def test_toda_candidata_esta_en_el_contrato_de_la_matriz():
    assert set(CANDIDATAS_IV) <= set(columnas_declaradas())


def test_la_presencia_de_cada_candidata_es_la_de_su_tabla():
    """`HAS_BUREAU_BALANCE` va sobre la de bureau, que la contiene; sobre sí misma daría cero."""
    for columna, candidata in CANDIDATAS_IV.items():
        tabla = _tabla_de(columna)
        esperada = None if tabla is None else TABLAS[tabla][0]
        if columna == "HAS_BUREAU_BALANCE":
            esperada = "HAS_BUREAU_HISTORY"
        assert candidata.presencia == esperada, columna
        assert candidata.presencia is None or candidata.presencia in PRESENCIA_AUX


def test_las_banderas_raras_son_estas_dos_y_son_candidatas():
    """Lista a mano: derivada de `BANDERAS_RARAS`, encogería con ella en vez de romperse, y la
    rara que saliera perdería su protección frente al umbral de IV sin que nada avisara.
    """
    assert set(BANDERAS_RARAS) == {"BUREAU_NEGATIVE_LIMIT_FLAG", "PREV_REFUSED_LONG_TERM_FLAG"}
    assert set(BANDERAS_RARAS) <= set(CANDIDATAS_IV)


@pytest.mark.parametrize("fuente, motivo", [("", "algo"), ("algo", " ")])
def test_una_candidata_sin_fuente_o_sin_motivo_revienta(fuente, motivo):
    with pytest.raises(ValueError, match="fuente y motivo"):
        Candidata(fuente, motivo)
    # y en la otra dirección, con las dos no revienta
    Candidata("algo", "algo")


# --- la puerta contra el dato real --------------------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["application_train"]).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)


@sin_dato_real
def test_el_iv_de_ext_source_3_sobre_train_sale_igual_por_un_camino_independiente():
    """0,3280 sobre los 245.993 de train, fuerte en la escala; el nulo va aparte con WoE +0,1562.

    El contraste no pasa por `iv.py`: bordes con `np.quantile`, tramo con `np.searchsorted`
    (intervalos cerrados por la derecha, como `qcut`) y recuentos con `np.bincount`.
    """
    train = solo_train(preparar_application(), cargar_split())
    x, y = train["EXT_SOURCE_3"], train["TARGET"]
    assert (len(train), int(x.isna().sum()), int(y.sum())) == (245_993, 48_637, 19_860)

    v, t = x.to_numpy(), y.to_numpy()
    presente = ~np.isnan(v)
    bordes = np.unique(np.quantile(v[presente], np.linspace(0, 1, 11)))
    tramo = np.where(presente, np.searchsorted(bordes[1:-1], v, side="left"), len(bordes) - 1)
    n, malos = np.bincount(tramo), np.bincount(tramo, weights=t)
    buenos, alfa = n - malos, 20
    pm = (malos + alfa * malos.sum() / n.sum()) / malos.sum()
    pb = (buenos + alfa * buenos.sum() / n.sum()) / buenos.sum()
    independiente = float(((pm - pb) * np.log(pm / pb)).sum())

    assert calcular_iv(x, y) == pytest.approx(independiente, abs=1e-12)
    assert calcular_iv(x, y) == pytest.approx(0.3280, abs=5e-5)
    assert tabla_woe(tramos(x), y)["n"].tolist() == n.tolist()


# --- informe_iv() en sintético, para que corra en CI y no solo en la puerta local --------------


def test_informe_iv_sin_presencia_quita_la_senal_del_grupo_ausente():
    """Una candidata constante donde hay historial y ausente donde no: el sin presencia sale en
    cero (un único tramo dentro de cada nivel de la presencia) y el marginal se queda con toda la
    señal de la bandera, que es justo la que el 5.5 decidió no dejar pasar a la lectura.
    """
    rng = np.random.default_rng(0)
    n = 4_000
    presencia = rng.integers(0, 2, n)
    x = np.where(presencia == 1, 5.0, np.nan)
    p_base = np.where(presencia == 1, 0.05, 0.15)
    y = pd.Series((rng.random(n) < p_base).astype(int))
    train = pd.DataFrame({"X": x, "HAS_X": presencia, "TARGET": y})

    informe = informe_iv(train, {"X": Candidata("test", "sintética", "HAS_X")})
    assert informe.loc["X", "iv"] > 0.1, "el fixture perdió la señal de la bandera de presencia"
    assert informe.loc["X", "iv_sin_presencia"] < 1e-6
    assert informe.loc["X", "iv_lectura"] == informe.loc["X", "iv_sin_presencia"]
    assert not informe.loc["X", "llega"]


def test_informe_iv_no_cuenta_el_tramo_vacio_del_nulo():
    """n_tramos cuenta los tramos con dato, no los declarados: una continua sin NaN no arrastra
    el hueco de tramos() para el nulo, y una bandera sin NaN se queda en sus dos niveles."""
    rng = np.random.default_rng(1)
    n = 2_000
    continua = rng.normal(size=n)
    bandera = rng.integers(0, 2, n)
    y = pd.Series((rng.random(n) < 0.08).astype(int))
    train = pd.DataFrame({"X": continua, "B": bandera, "TARGET": y})

    informe = informe_iv(
        train, {"X": Candidata("test", "sintética"), "B": Candidata("test", "sintética")}
    )
    assert informe.loc["X", "n_tramos"] == valor("n_bins_max")
    assert informe.loc["B", "n_tramos"] == 2


def test_informe_iv_revienta_con_una_candidata_ausente_del_frame():
    train = pd.DataFrame({"X": [0, 1, 0, 1], "TARGET": [0, 1, 0, 1]})
    with pytest.raises(KeyError, match="Y"):
        informe_iv(train, {"Y": Candidata("test", "sintética")})
    # la otra dirección: presente en el frame, no revienta
    informe_iv(train, {"X": Candidata("test", "sintética")})


def test_informe_iv_revienta_sin_target():
    train = pd.DataFrame({"X": [0, 1, 0, 1]})
    with pytest.raises(KeyError, match="TARGET"):
        informe_iv(train, {"X": Candidata("test", "sintética")})


def test_informe_iv_marca_las_raras_aunque_no_lleguen():
    rng = np.random.default_rng(3)
    n = 2_000
    train = pd.DataFrame(
        {
            "BUREAU_NEGATIVE_LIMIT_FLAG": rng.integers(0, 2, n),
            "TARGET": rng.integers(0, 2, n),
        }
    )
    informe = informe_iv(train, {"BUREAU_NEGATIVE_LIMIT_FLAG": Candidata("test", "sintética")})
    assert informe.loc["BUREAU_NEGATIVE_LIMIT_FLAG", "iv_lectura"] < valor("min_iv")
    assert not informe.loc["BUREAU_NEGATIVE_LIMIT_FLAG", "llega"]
    assert informe.loc["BUREAU_NEGATIVE_LIMIT_FLAG", "rara"]


# --- la puerta del informe de IV sobre el dato real -------------------------------------------

sin_dato_real_ensamblado = pytest.mark.skipif(
    not all(
        (ruta("raw_data") / TABLE_FILES[t]).exists()
        for t in ("bureau", "bureau_balance", "previous_application", "application_train")
    )
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO_CORTES).exists(),
    reason="data/raw, el split o cortes.json no viajan con el repo",
)

# El `iv_lectura` de cada candidata (el sin presencia si la declara, si no el marginal) y si llega
# a `min_iv` (0,02). Recomputados por un camino que no pasa por `iv.py`.
PUERTA_IV = {
    "HAS_BUREAU_FINANCIAL_DETAIL": (0.0005, False),
    "HAS_BEEN_PROLONGED": (0.0009, False),
    "BUREAU_DAYS_CREDIT_ENDDATE_MAX": (0.0282, True),
    "BUREAU_ANNUITY_ACTIVE_RATIO": (0.0078, False),
    "BUREAU_CREDIT_TYPE_NUNIQUE": (0.0001, False),
    "BB_MANY_CREDITS_FLAG": (0.0007, False),
    "BB_MONTHS_TOTAL": (0.0124, False),
    "BB_DPD_MONTHS_COUNT": (0.0076, False),
    "BB_CREDITS_WITH_DPD_COUNT": (0.0052, False),
    "PREV_EARLY_HOUR_RATIO": (0.0139, False),
    "PREV_DAYS_DECISION_MAX": (0.0093, False),
    "LIVE_CITY_NOT_WORK_CITY": (0.0138, False),
    "REG_REGION_NOT_WORK_REGION": (0.0008, False),
    "REG_REGION_NOT_LIVE_REGION": (0.0003, False),
    "LIVE_REGION_NOT_WORK_REGION": (0.0002, False),
    "FLAG_PHONE": (0.0078, False),
    "FLAG_WORK_PHONE": (0.0105, False),
    "FLAG_EMAIL": (0.0, False),
    "NAME_HOUSING_TYPE": (0.0155, False),
    "FONDKAPREMONT_MODE": (0.0115, False),
    "HOUSETYPE_MODE": (0.0215, True),
    "WALLSMATERIAL_MODE": (0.0268, True),
    "EMERGENCYSTATE_MODE": (0.0236, True),
    "BUILDING_INFO_COUNT": (0.0212, True),
    "BUREAU_HAS_FOREIGN_CURRENCY": (0.0002, False),
    "BUREAU_HAS_CURRENT_OVERDUE": (0.0099, False),
    "BUREAU_COUNT_COLA": (0.0016, False),
    "BB_MONTHS_REPORTED": (0.0021, False),
    "BB_TRAJECTORY": (0.0121, False),
    "PREV_RELACION_CORTA_ACTIVA": (0.0175, False),
    "BB_STATUS_WORST": (0.0010, False),
    "BB_RECOVERED_DPD_FLAG": (0.0003, False),
    "BB_N_CREDITS_WBAL": (0.0017, False),
    "HAS_BUREAU_BALANCE": (0.0015, False),
    "FLAG_CONT_MOBILE": (0.0, False),
    "DEF_60_CNT_SOCIAL_CIRCLE": (0.0015, False),
    "BUREAU_NEGATIVE_LIMIT_FLAG": (0.0020, False),
    "PREV_REFUSED_LONG_TERM_FLAG": (0.0004, False),
}


def _tramo_codigo(serie, n_bins_max):
    """El tramo de `tramos()`, pero con `np.searchsorted`/`pd.factorize` y no `qcut`/`groupby`."""
    numerica = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    if numerica and serie.nunique() > 2:
        v = serie.to_numpy(dtype=float)
        presente = ~np.isnan(v)
        bordes = np.unique(np.quantile(v[presente], np.linspace(0, 1, n_bins_max + 1)))
        codigo = np.where(presente, np.searchsorted(bordes[1:-1], v, side="left"), -1)
        return codigo + 1  # el 0 queda para el NULO
    codigo, _ = pd.factorize(serie.astype(object).where(serie.notna(), NULO))
    return codigo


def _iv_de_codigos(codigo, y, alfa, mask=None):
    if mask is not None:
        codigo, y = codigo[mask], y[mask]
    if len(np.unique(y)) < 2:
        return 0.0, 0
    n = np.bincount(codigo)
    malos = np.bincount(codigo, weights=y)
    buenos = n - malos
    prior = alfa * malos.sum() / n.sum()
    pm = (malos + prior) / malos.sum()
    pb = (buenos + alfa - prior) / buenos.sum()
    return float(((pm - pb) * np.log(pm / pb)).sum()), int(n.sum())


@sin_dato_real_ensamblado
def test_el_informe_de_iv_reproduce_la_puerta_del_5_5():
    """Los 38 `iv_lectura` sobre los 245.993 de train, cada uno por un camino que no pasa por
    `iv.py`: `np.quantile` + `np.searchsorted` + `np.bincount` para el marginal (`pd.factorize`
    en las categóricas y banderas), y la ponderación de `iv_condicionado()` repetida a mano para
    el sin presencia.
    """
    split = cargar_split()
    base = preparar_application()
    bureau = load_table("bureau", reduce_memory=False)
    bb = load_table("bureau_balance")
    prev = load_table("previous_application", reduce_memory=False)
    cargar_cortes(split, sobrescribir=True)
    matriz = ensamblar_auxiliares(base, bureau, bb, prev)
    matriz = matriz.merge(split[["SK_ID_CURR", "split"]], on="SK_ID_CURR", how="left")
    train = solo_train(matriz, split)
    assert len(train) == 245_993

    informe = informe_iv(train)
    assert len(informe) == 38
    alfa, n_bins_max = valor("suavizado_woe"), valor("n_bins_max")
    y = train["TARGET"].to_numpy()

    for nombre, (esperado_lectura, esperado_llega) in PUERTA_IV.items():
        candidata = CANDIDATAS_IV[nombre]
        codigo = _tramo_codigo(train[nombre], n_bins_max)
        marginal, _ = _iv_de_codigos(codigo, y, alfa)
        if candidata.presencia is None:
            lectura = marginal
        else:
            grupo = _tramo_codigo(train[candidata.presencia], n_bins_max)
            total = 0.0
            for nivel in np.unique(grupo):
                iv_nivel, n_nivel = _iv_de_codigos(codigo, y, alfa, mask=(grupo == nivel))
                total += n_nivel * iv_nivel
            lectura = total / len(train)
        assert lectura == pytest.approx(esperado_lectura, abs=5e-5), nombre
        assert lectura == pytest.approx(informe.loc[nombre, "iv_lectura"], abs=1e-9), nombre
        assert bool(lectura >= valor("min_iv")) == esperado_llega, nombre
        assert bool(informe.loc[nombre, "llega"]) == esperado_llega, nombre
