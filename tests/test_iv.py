"""Tests de iv.py: el binning, la tabla de WoE, el IV y el IV condicionado."""

import numpy as np
import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES
from src.features.build_features import preparar_application
from src.features.iv import NULO, calcular_iv, iv_condicionado, tabla_woe, tramos
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
