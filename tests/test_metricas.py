"""Pruebas de src.models.metricas (0.2): cada métrica contra un valor conocido y su inverso."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from src.models.metricas import (
    TASAS_CURVA,
    Resumen,
    auc_parcial,
    calibracion,
    cumple_criterio,
    curva_estrategia,
    elegir,
    por_segmento,
    ranking,
    tabla_calibracion,
)


@pytest.fixture
def datos():
    """2.000 clientes con un 10% de impago y un score que separa a medias."""
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.10).astype(int)
    score = rng.normal(size=2000) + 1.0 * y
    return y, score


def test_la_guarda_revienta_con_etiquetas_como_score():
    y = [0, 1, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="etiquetas"):
        ranking(y, [0, 1, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "y, score",
    [
        ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]),
        ([0, 1, 2, 1], [0.1, 0.2, 0.3, 0.4]),
        ([0, 1, 0, 1], [0.1, np.nan, 0.3, 0.4]),
        ([0, 1, 0, 1], [0.1, np.inf, 0.3, 0.4]),
        ([0, 1, 0], [0.1, 0.2, 0.3, 0.4]),
    ],
)
def test_la_guarda_revienta_con_entradas_invalidas(y, score):
    with pytest.raises(ValueError):
        ranking(y, score)


def test_un_score_continuo_pasa_la_guarda(datos):
    assert ranking(*datos)["auc"] > 0.5


def test_score_perfecto_e_invertido():
    y = np.array([0] * 90 + [1] * 10)
    score = np.arange(100, dtype=float)
    perfecto = ranking(y, score)
    assert perfecto["auc"] == 1 and perfecto["gini"] == 1 and perfecto["ks"] == 1
    assert perfecto["captura_decil"] == 1
    invertido = ranking(y, -score)
    assert invertido["auc"] == 0 and invertido["gini"] == -1


def test_el_ks_coincide_con_el_de_dos_muestras(datos):
    y, score = datos
    esperado = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert ranking(y, score)["ks"] == pytest.approx(esperado)


def test_captura_del_decil_a_mano():
    # 20 clientes, 4 impagos: los dos de mayor score lo son y otros dos caen fuera del decil
    y = np.zeros(20, dtype=int)
    y[[19, 18, 5, 2]] = 1
    captura = ranking(y, np.arange(20, dtype=float))["captura_decil"]
    assert captura == pytest.approx(2 / 4)


def test_captura_del_decil_reparte_el_empate_en_el_corte():
    # 10 clientes, el decil es 1 y los dos de mayor score empatan con un impago y un bueno
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9])
    assert ranking(y, score)["captura_decil"] == pytest.approx(0.5)


def test_el_orden_de_las_filas_no_cambia_nada(datos):
    y, score = datos
    orden = np.random.default_rng(1).permutation(y.size)
    assert ranking(y[orden], score[orden]) == pytest.approx(ranking(y, score))


def test_curva_a_mano_con_importes_que_cambian_la_lectura():
    # 10 clientes por orden de score; el segundo impago es el grande (100 frente a 10)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    importe = np.array([10, 10, 10, 10, 10, 10, 10, 100, 10, 10])
    curva = curva_estrategia(y, np.arange(10.0), importe, tasas=(0.5, 0.8, 1.0))
    assert curva["mora"].tolist() == pytest.approx([0.2, 0.25, 0.2])
    assert curva["mora_importe"].tolist() == pytest.approx([10 / 50, 110 / 170, 110 / 190])
    assert curva["reduccion_mora"].tolist() == pytest.approx([0.0, 0.25, 0.0])
    # al aprobar al 80% la mora en recuento sube el 25% pero en dinero el 12%: no coinciden
    assert curva["reduccion_mora_importe"].tolist() == pytest.approx(
        [(10 / 50) / (110 / 190) - 1, (110 / 170) / (110 / 190) - 1, 0.0]
    )


def test_curva_score_al_azar_da_la_tasa_base():
    rng = np.random.default_rng(3)
    y = (rng.random(50000) < 0.08).astype(int)
    curva = curva_estrategia(y, rng.random(50000), rng.uniform(1, 5, 50000))
    assert curva["mora"].to_numpy() == pytest.approx(y.mean(), abs=0.006)
    assert curva["mora_importe"].to_numpy() == pytest.approx(y.mean(), abs=0.01)


def test_curva_es_monotona_con_un_score_util(datos):
    y, score = datos
    curva = curva_estrategia(y, score, np.ones(y.size))
    assert curva["mora"].is_monotonic_increasing
    assert curva["mora"].iloc[0] < y.mean()


def test_curva_con_empates_no_depende_del_orden():
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.2).astype(int)
    score = rng.integers(0, 5, 300).astype(float)
    importe = rng.uniform(1, 9, 300)
    orden = rng.permutation(300)
    a = curva_estrategia(y, score, importe)
    b = curva_estrategia(y[orden], score[orden], importe[orden])
    pd.testing.assert_frame_equal(a, b)


def test_curva_revienta_con_importe_invalido(datos):
    y, score = datos
    with pytest.raises(ValueError):
        curva_estrategia(y, score, -np.ones(y.size))
    with pytest.raises(ValueError):
        curva_estrategia(y, score, np.ones(y.size - 1))


def test_auc_parcial_perfecto_azar_e_invertido():
    y = np.array([0] * 90 + [1] * 10)
    assert auc_parcial(y, np.arange(100.0)) == 1
    rng = np.random.default_rng(5)
    y = (rng.random(20000) < 0.1).astype(int)
    assert auc_parcial(y, rng.random(20000)) == pytest.approx(0.5, abs=0.03)
    util = rng.normal(size=y.size) + 2.0 * y
    assert auc_parcial(y, util) > 0.5 > auc_parcial(y, -util)


@pytest.fixture
def pd_calibrada():
    """PD con la tasa de impago exactamente igual a su valor: calibrada por construcción."""
    rng = np.random.default_rng(6)
    prob = rng.beta(1, 11, 60000)
    return (rng.random(60000) < prob).astype(int), prob


def test_calibracion_de_una_pd_calibrada(pd_calibrada):
    y, prob = pd_calibrada
    cal = calibracion(y, prob)
    assert cal["pendiente"] == pytest.approx(1.0, abs=0.05)
    assert cal["ordenada"] == pytest.approx(0.0, abs=0.05)
    assert cal["brier"] == pytest.approx(np.mean((prob - y) ** 2))


def test_calibracion_detecta_pd_sobreconfiada(pd_calibrada):
    y, prob = pd_calibrada
    z = np.log(prob / (1 - prob))
    extrema = 1 / (1 + np.exp(-2 * z))
    assert calibracion(y, extrema)["pendiente"] == pytest.approx(0.5, abs=0.05)
    timida = 1 / (1 + np.exp(-0.5 * z))
    assert calibracion(y, timida)["pendiente"] > 1.5


def test_calibracion_detecta_pd_desplazada(pd_calibrada):
    y, prob = pd_calibrada
    z = np.log(prob / (1 - prob))
    baja = 1 / (1 + np.exp(-(z - 0.5)))
    # la PD infravalora el impago, así que la ordenada sale positiva y cercana a 0,5
    assert calibracion(y, baja)["ordenada"] == pytest.approx(0.5, abs=0.06)


def test_calibracion_exige_probabilidades(datos):
    y, score = datos
    with pytest.raises(ValueError, match="entre 0 y 1"):
        calibracion(y, score)


def test_tabla_de_calibracion_por_deciles(pd_calibrada):
    y, prob = pd_calibrada
    tabla = tabla_calibracion(y, prob)
    assert len(tabla) == 10 and tabla["n"].sum() == y.size
    assert tabla["pd_media"].is_monotonic_increasing
    assert (tabla["pd_media"] - tabla["tasa_observada"]).abs().max() < 0.02


def test_por_segmento_es_calcular_sobre_el_subconjunto(pd_calibrada):
    y, prob = pd_calibrada
    importe = np.random.default_rng(7).uniform(1, 9, y.size)
    mascara = np.random.default_rng(8).random(y.size) < 0.3
    res = por_segmento(y, prob, importe, {"sin_historial": mascara, "todos": np.ones(y.size, bool)})
    sub = res["sin_historial"]
    assert sub["n"] == mascara.sum() and sub["impagos"] == y[mascara].sum()
    assert sub["ranking"] == pytest.approx(ranking(y[mascara], prob[mascara]))
    pd.testing.assert_frame_equal(
        sub["curva"], curva_estrategia(y[mascara], prob[mascara], importe[mascara])
    )
    assert res["todos"]["ranking"]["auc"] == pytest.approx(ranking(y, prob)["auc"])
    with pytest.raises(ValueError, match="longitud"):
        por_segmento(y, prob, importe, {"mala": mascara[:-1]})


TASAS_ZONA = (0.70, 0.75, 0.80, 0.85, 0.90)


def _resumen(auc=0.75, mora=0.06, mora_imp=0.06, sin_hist=0.70, pend=1.0, ord_=0.0):
    curva = pd.DataFrame({"tasa": TASAS_CURVA, "mora": mora, "mora_importe": mora_imp})
    return Resumen(auc, curva, sin_hist, pend, ord_)


@pytest.fixture
def simple():
    return _resumen()


def test_complejo_que_cumple_todo(simple):
    ok = cumple_criterio(simple, _resumen(auc=0.76), p_auc=0.01)
    assert all(ok.values())


@pytest.mark.parametrize(
    "rota, condicion",
    [
        ({"auc": 0.754}, "auc"),
        ({"mora": 0.0611}, "curva"),
        ({"mora_imp": 0.0611}, "curva"),
        ({"sin_hist": 0.694}, "sin_historial"),
        ({"pend": 0.85}, "calibracion"),
        ({"pend": 1.15}, "calibracion"),
        ({"ord_": 0.11}, "calibracion"),
        ({"ord_": -0.11}, "calibracion"),
    ],
)
def test_cada_condicion_rota_por_separado_falla_sola(simple, rota, condicion):
    complejo = _resumen(**{"auc": 0.76, **rota})
    ok = cumple_criterio(simple, complejo, p_auc=0.01)
    assert [k for k, v in ok.items() if not v] == [condicion]


def test_p_sin_significacion_falla_el_auc_aunque_mejore_mucho(simple):
    assert not cumple_criterio(simple, _resumen(auc=0.80), p_auc=0.05)["auc"]


def test_los_limites_exactos_cuentan_como_cumplidos(simple):
    limite = _resumen(auc=0.755, mora=0.061, mora_imp=0.061, sin_hist=0.695, pend=1.1, ord_=0.1)
    assert all(cumple_criterio(simple, limite, p_auc=0.01).values())


def test_la_curva_fuera_de_la_zona_no_decide(simple):
    complejo = _resumen(auc=0.76)
    complejo.curva.loc[complejo.curva["tasa"] < 0.70, "mora"] = 0.5
    assert cumple_criterio(simple, complejo, p_auc=0.01)["curva"]


def test_las_curvas_de_distinta_zona_revientan(simple):
    otra = _resumen(auc=0.76)
    with pytest.raises(ValueError, match="mismas tasas"):
        cumple_criterio(simple, Resumen(0.76, otra.curva.iloc[:3], 0.7, 1.0, 0.0), 0.01)


def test_elegir_gana_el_complejo_si_cumple_y_el_simple_si_no():
    p = {("logistica", "random_forest"): 0.01, ("logistica", "lightgbm"): 0.01}
    logistica = _resumen()
    assert elegir({"logistica": logistica, "lightgbm": _resumen(auc=0.76)}, p) == "lightgbm"
    assert elegir({"logistica": logistica, "lightgbm": _resumen(auc=0.752)}, p) == "logistica"


def test_elegir_compara_lightgbm_contra_el_bosque_si_este_gana():
    bosque = _resumen(auc=0.76)
    lgbm = _resumen(auc=0.764)
    p = {
        ("logistica", "random_forest"): 0.01,
        ("random_forest", "lightgbm"): 0.01,
        ("logistica", "lightgbm"): 0.0,
    }
    resumenes = {"logistica": _resumen(), "random_forest": bosque, "lightgbm": lgbm}
    # el bosque gana a la logística, pero lightgbm no mejora al bosque en el delta mínimo
    assert elegir(resumenes, p) == "random_forest"
    resumenes["lightgbm"] = _resumen(auc=0.77)
    assert elegir(resumenes, p) == "lightgbm"


def test_elegir_compara_lightgbm_contra_la_logistica_si_el_bosque_no_pasa():
    # el bosque no mejora, y lightgbm se compara contra la logística, no contra el bosque
    p = {("logistica", "random_forest"): 0.01, ("logistica", "lightgbm"): 0.01}
    resumenes = {
        "logistica": _resumen(),
        "random_forest": _resumen(auc=0.751),
        "lightgbm": _resumen(auc=0.76),
    }
    assert elegir(resumenes, p) == "lightgbm"


def test_elegir_no_acumula_tolerancias_de_un_escalon_a_otro():
    # cada escalón empeora la mora 0,08 puntos (dentro de la tolerancia), pero lightgbm
    # acumularía 0,16 frente a la logística
    p = {
        ("logistica", "random_forest"): 0.01,
        ("random_forest", "lightgbm"): 0.01,
        ("logistica", "lightgbm"): 0.01,
    }
    resumenes = {
        "logistica": _resumen(),
        "random_forest": _resumen(auc=0.756, mora=0.0608),
        "lightgbm": _resumen(auc=0.762, mora=0.0616),
    }
    assert elegir(resumenes, p) == "random_forest"


def test_elegir_revienta_sin_logistica_sin_p_o_con_modelo_ajeno():
    with pytest.raises(ValueError):
        elegir({"lightgbm": _resumen()}, {})
    with pytest.raises(ValueError):
        elegir({"logistica": _resumen(), "svm": _resumen()}, {})
    with pytest.raises(KeyError):
        elegir({"logistica": _resumen(), "lightgbm": _resumen(auc=0.8)}, {})


def test_un_complejo_con_mejor_curva_no_se_penaliza_por_la_distancia(simple):
    mejor = _resumen(auc=0.76, mora=0.04, mora_imp=0.04)
    assert cumple_criterio(simple, mejor, p_auc=0.01)["curva"]


def test_auc_parcial_usa_el_fpr_de_la_zona_de_operacion(datos):
    y, score = datos
    assert auc_parcial(y, score) == pytest.approx(roc_auc_score(y, score, max_fpr=0.30))
    assert auc_parcial(y, score) != pytest.approx(roc_auc_score(y, score, max_fpr=0.10))
