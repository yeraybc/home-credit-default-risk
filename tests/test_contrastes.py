"""Pruebas de src.models.contrastes (0.3): cada test detecta la diferencia y no la inventa."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_sm

from src.models.contrastes import (
    N_BOOTSTRAP,
    bonferroni,
    bootstrap_estratificado,
    delong,
    mcnemar,
    mcnemar_aprobados,
    nadeau_bengio,
    pares,
)
from src.models.metricas import TASAS_CURVA, Resumen, elegir


def test_nadeau_bengio_a_mano():
    # media 0,02 y varianza 2,5e-4: t = 0,02 / sqrt((1/5 + 1/4) · 2,5e-4) = 1,8856
    res = nadeau_bengio([0.01, 0.02, 0.03, 0.00, 0.04])
    assert res["t"] == pytest.approx(0.02 / np.sqrt(0.45 * 2.5e-4))
    assert res["t"] == pytest.approx(1.8856, abs=1e-4)
    assert res["p"] == pytest.approx(2 * stats.t.sf(1.8856, 4), abs=1e-4)


def test_nadeau_bengio_sin_correccion_es_el_t_pareado():
    d = np.random.default_rng(0).normal(0.004, 0.01, 15)
    ingenuo = stats.ttest_1samp(d, 0)
    res = nadeau_bengio(d, ratio_test_train=0)
    assert res["t"] == pytest.approx(ingenuo.statistic)
    assert res["p"] == pytest.approx(ingenuo.pvalue)


def test_nadeau_bengio_nunca_es_mas_laxo_que_el_ingenuo():
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = rng.normal(rng.uniform(-0.01, 0.01), 0.01, 15)
        assert nadeau_bengio(d)["p"] >= stats.ttest_1samp(d, 0).pvalue


def test_nadeau_bengio_detecta_una_mejora_sistematica():
    d = np.random.default_rng(2).normal(0.01, 0.003, 15)
    assert nadeau_bengio(d)["p"] < 0.001
    assert nadeau_bengio(-d)["p"] < 0.001 and nadeau_bengio(-d)["t"] < 0


def test_nadeau_bengio_no_inventa_diferencias_en_ruido():
    rng = np.random.default_rng(3)
    p = np.array([nadeau_bengio(rng.normal(0, 0.01, 15))["p"] for _ in range(1000)])
    # con folds independientes la corrección es conservadora: rechaza menos del 5%
    assert (p < 0.05).mean() < 0.05


def test_nadeau_bengio_con_varianza_cero():
    assert nadeau_bengio([0.0] * 15)["p"] == 1
    assert nadeau_bengio([0.01] * 15)["p"] == 0


@pytest.mark.parametrize("dif", [[0.01], [0.01, np.nan, 0.02], [[0.01, 0.02], [0.03, 0.04]]])
def test_nadeau_bengio_revienta_con_entradas_invalidas(dif):
    with pytest.raises(ValueError):
        nadeau_bengio(dif)


def _par(n=3000, sep_a=1.0, sep_b=1.0, seed=0):
    """Dos scores correlados (comparten una parte de señal) sobre el mismo y."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.10).astype(int)
    comun = rng.normal(size=n)
    a = 0.7 * comun + 0.7 * rng.normal(size=n) + sep_a * y
    b = 0.7 * comun + 0.7 * rng.normal(size=n) + sep_b * y
    return y, a, b


def test_delong_los_auc_coinciden_con_sklearn():
    y, a, b = _par()
    res = delong(y, a, b)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert res["dif"] == pytest.approx(res["auc_a"] - res["auc_b"])


def test_delong_los_auc_con_empates_coinciden_con_sklearn():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    a = rng.integers(0, 8, 2000) + 2.0 * y
    b = rng.integers(0, 5, 2000) + 1.0 * y
    res = delong(y, a, b)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, b))


def test_delong_el_mismo_score_no_da_diferencia():
    y, a, _ = _par()
    res = delong(y, a, a)
    assert res["dif"] == 0 and res["p"] == 1
    # una transformación monótona tampoco cambia el ranking
    assert delong(y, a, 3 * a + 1)["p"] == 1


def test_delong_es_antisimetrico():
    y, a, b = _par(sep_a=1.4)
    ab, ba = delong(y, a, b), delong(y, b, a)
    assert ab["dif"] == pytest.approx(-ba["dif"]) and ab["p"] == pytest.approx(ba["p"])
    assert ab["z"] == pytest.approx(-ba["z"])


def test_delong_detecta_una_diferencia_conocida():
    y, a, b = _par(n=20000, sep_a=1.2, sep_b=0.9)
    res = delong(y, a, b)
    assert res["dif"] > 0.02 and res["p"] < 1e-4


def test_delong_no_inventa_diferencias():
    p = np.array([delong(*_par(n=1500, seed=s))["p"] for s in range(200)])
    # sin diferencia real el p es uniforme: el rechazo al 5% ronda el 5%
    assert 0.01 < (p < 0.05).mean() < 0.10


def test_delong_su_error_estandar_casa_con_el_bootstrap():
    y, a, b = _par(n=4000, sep_a=1.1)
    rng = np.random.default_rng(2)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    difs = []
    for _ in range(400):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        difs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    assert delong(y, a, b)["ee"] == pytest.approx(np.std(difs, ddof=1), rel=0.15)


def test_delong_revienta_con_etiquetas_como_score():
    y, a, _ = _par()
    with pytest.raises(ValueError, match="etiquetas"):
        delong(y, a, (a > 0).astype(int))


def test_bootstrap_conserva_la_prevalencia_en_cada_replica():
    y, _, _ = _par()
    prevalencias = []
    bootstrap_estratificado(y, lambda i: prevalencias.append(y[i].mean()) or 0.0, n_rep=50)
    assert np.ptp(prevalencias) == 0 and prevalencias[0] == pytest.approx(y.mean())


def test_bootstrap_es_reproducible_con_la_semilla():
    y, a, _ = _par()
    auc = lambda i: roc_auc_score(y[i], a[i])  # noqa: E731
    uno = bootstrap_estratificado(y, auc, n_rep=100)
    assert bootstrap_estratificado(y, auc, n_rep=100) == uno
    assert bootstrap_estratificado(y, auc, n_rep=100, semilla=7) != uno


def test_bootstrap_del_auc_casa_con_el_error_de_delong():
    y, a, b = _par(n=4000)
    auc = lambda i: roc_auc_score(y[i], a[i])  # noqa: E731
    ic = bootstrap_estratificado(y, auc, n_rep=500)
    # el error estándar del AUC solo sale de DeLong comparando el score con una copia ruidosa
    ancho_delong = 2 * 1.96 * _ee_auc(y, a)
    assert ic["inferior"] < roc_auc_score(y, a) < ic["superior"]
    assert ic["superior"] - ic["inferior"] == pytest.approx(ancho_delong, rel=0.15)


def _ee_auc(y, score):
    """Error estándar de un AUC por la varianza de los componentes de DeLong."""
    from src.models.contrastes import _componentes

    _, v01, v10 = _componentes(y, score)
    return np.sqrt(v01.var(ddof=1) / v01.size + v10.var(ddof=1) / v10.size)


def test_bootstrap_acepta_un_estadistico_vectorial():
    y, a, _ = _par()
    mora = lambda i: [y[i][np.argsort(a[i])[: int(t * i.size)]].mean() for t in (0.7, 0.8, 0.9)]  # noqa: E731
    ic = bootstrap_estratificado(y, mora, n_rep=50)
    assert ic["inferior"].shape == ic["superior"].shape == (3,)
    assert (ic["inferior"] < ic["superior"]).all()


def test_bootstrap_de_la_diferencia_pareada_detecta_y_no_inventa():
    y, a, b = _par(n=20000, sep_a=1.2, sep_b=0.9)
    dif = lambda i: roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])  # noqa: E731
    assert bootstrap_estratificado(y, dif, n_rep=100)["inferior"] > 0
    nula = lambda i: roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], a[i])  # noqa: E731
    ic = bootstrap_estratificado(y, nula, n_rep=20)
    assert ic["inferior"] == ic["superior"] == 0


def test_bootstrap_revienta_con_entradas_invalidas():
    with pytest.raises(ValueError, match="dos clases"):
        bootstrap_estratificado([0, 0, 0], lambda i: 0.0)
    with pytest.raises(ValueError):
        bootstrap_estratificado([0, 1, 0, 1], lambda i: 0.0, nivel=1.5)
    assert N_BOOTSTRAP == 2000


def test_mcnemar_a_mano():
    # 1 = aprueba. Clientes: no impago, no impago, impago, impago, no impago, impago
    y = np.array([0, 0, 1, 1, 0, 1])
    a = np.array([1, 1, 0, 0, 0, 1])  # acierta en 0, 1, 2, 3 y falla en 4, 5
    b = np.array([1, 0, 0, 1, 1, 1])  # acierta en 0, 2, 4 y falla en 1, 3, 5
    res = mcnemar(y, a, b)
    assert (res["b"], res["c"]) == (2, 1)  # solo A: clientes 1 y 3; solo B: cliente 4
    assert res["p"] == pytest.approx(stats.binomtest(1, 3, 0.5).pvalue) == 1.0


def test_mcnemar_coincide_con_statsmodels():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    a = (rng.random(3000) < 0.8).astype(int)
    b = np.where(rng.random(3000) < 0.9, a, 1 - a)
    res = mcnemar(y, a, b)
    ok_a, ok_b = (a == 1) == (y == 0), (b == 1) == (y == 0)
    tabla = [
        [(ok_a & ok_b).sum(), (ok_a & ~ok_b).sum()],
        [(~ok_a & ok_b).sum(), (~ok_a & ~ok_b).sum()],
    ]
    assert res["p"] == pytest.approx(mcnemar_sm(tabla, exact=True).pvalue)


def test_mcnemar_decisiones_iguales_dan_p_uno():
    y, a, _ = _par()
    aprueba = (a < np.quantile(a, 0.8)).astype(int)
    assert mcnemar(y, aprueba, aprueba) == {"b": 0, "c": 0, "p": 1.0}


def test_mcnemar_detecta_el_mejor_y_no_inventa_diferencias():
    y, bueno, malo = _par(n=20000, sep_a=1.5, sep_b=0.2)
    corte = lambda s: (s < np.quantile(s, 0.8)).astype(int)  # noqa: E731
    res = mcnemar(y, corte(bueno), corte(malo))
    assert res["p"] < 1e-6 and res["b"] > res["c"]
    # sin diferencia real nunca rechaza más del 5%; con la misma tasa de aprobación rechaza mucho
    # menos (ver la docstring), así que aquí solo se exige que no se pase
    p = np.array([_mcnemar_nulo(s)["p"] for s in range(300)])
    assert (p < 0.05).mean() <= 0.05


def _mcnemar_nulo(semilla):
    y, a, b = _par(n=1500, seed=semilla)
    corte = lambda s: (s < np.quantile(s, 0.8)).astype(int)  # noqa: E731
    return mcnemar(y, corte(a), corte(b))


def test_mcnemar_es_simetrico_en_el_p():
    y, a, b = _par(sep_a=1.4)
    corte = lambda s: (s < np.quantile(s, 0.8)).astype(int)  # noqa: E731
    ab, ba = mcnemar(y, corte(a), corte(b)), mcnemar(y, corte(b), corte(a))
    assert (ab["b"], ab["c"]) == (ba["c"], ba["b"]) and ab["p"] == pytest.approx(ba["p"])


def test_mcnemar_revienta_con_entradas_invalidas():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        mcnemar(y, [1, 0, 1], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="booleanas"):
        mcnemar(y, [0.2, 0.9, 0.1, 0.7], [1, 0, 1, 0])


def _corte(s, q=0.8):
    return (s < np.quantile(s, q)).astype(int)


def test_mcnemar_aprobados_a_mano():
    # A aprueba a los clientes 0 a 3 y B a los 2 a 5; solo A: 0 y 1; solo B: 4 y 5
    y = np.array([1, 0, 0, 0, 1, 1])
    a = np.array([1, 1, 1, 1, 0, 0])
    b = np.array([0, 0, 1, 1, 1, 1])
    res = mcnemar_aprobados(y, a, b)
    assert res == {**res, "solo_a": 2, "solo_b": 2, "impagos_a": 1, "impagos_b": 2}
    assert res["p"] == pytest.approx(stats.binomtest(1, 3, 0.5).pvalue)


def test_mcnemar_aprobados_pondera_por_el_tamano_de_cada_grupo():
    # A aprueba a 3 clientes que B no y B a 1 que A no: la probabilidad nula es 3/4
    y = np.array([1, 0, 0, 1])
    res = mcnemar_aprobados(y, [1, 1, 1, 0], [0, 0, 0, 1])
    assert res["p"] == pytest.approx(stats.binomtest(1, 2, 0.75).pvalue)


def test_mcnemar_aprobados_detecta_el_mejor_y_con_tasa_igual_es_mas_potente():
    y, bueno, malo = _par(n=20000, sep_a=1.5, sep_b=0.2)
    res = mcnemar_aprobados(y, _corte(bueno), _corte(malo))
    assert res["impagos_a"] < res["impagos_b"] and res["p"] < 1e-6
    # con diferencia moderada ve lo que mcnemar() conservador no llega a ver
    y, a, b = _par(n=20000, sep_a=1.05, sep_b=0.9, seed=3)
    assert mcnemar_aprobados(y, _corte(a), _corte(b))["p"] < mcnemar(y, _corte(a), _corte(b))["p"]


@pytest.mark.parametrize("q_a, q_b", [(0.8, 0.8), (0.8, 0.7)])
def test_mcnemar_aprobados_no_inventa_diferencias(q_a, q_b):
    # sin diferencia real, con la misma tasa y con tasas distintas (donde el score B aprueba menos)
    def nulo(s):
        y, a, b = _par(n=1500, seed=s)
        return mcnemar_aprobados(y, _corte(a, q_a), _corte(a, q_b))["p"]

    assert (np.array([nulo(s) for s in range(300)]) < 0.05).mean() <= 0.07


def test_mcnemar_aprobados_decisiones_iguales_y_entradas_invalidas():
    y, a, _ = _par()
    assert mcnemar_aprobados(y, _corte(a), _corte(a))["p"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_aprobados([0, 1, 0], [1, 0], [1, 0, 1])


def test_pares_en_orden_de_simplicidad_y_con_el_recuento_de_la_familia():
    assert pares(["lightgbm", "logistica", "random_forest"]) == [
        ("logistica", "random_forest"),
        ("logistica", "lightgbm"),
        ("random_forest", "lightgbm"),
    ]
    assert pares(["lightgbm", "logistica"]) == [("logistica", "lightgbm")]
    with pytest.raises(ValueError):
        pares(["logistica", "svm"])
    with pytest.raises(ValueError):
        pares(["logistica", "logistica"])


def test_bonferroni_multiplica_por_la_familia_y_recorta_a_uno():
    p = {("logistica", "random_forest"): 0.01, ("logistica", "lightgbm"): 0.4,
         ("random_forest", "lightgbm"): 0.0}
    assert bonferroni(p, 3) == {
        ("logistica", "random_forest"): pytest.approx(0.03),
        ("logistica", "lightgbm"): 1.0,
        ("random_forest", "lightgbm"): 0.0,
    }


@pytest.mark.parametrize("declarados", [2, 4, 0])
def test_bonferroni_revienta_si_el_recuento_no_coincide(declarados):
    p = dict.fromkeys(pares(["logistica", "random_forest", "lightgbm"]), 0.01)
    with pytest.raises(ValueError, match="contrastes"):
        bonferroni(p, declarados)


def _resumen(auc):
    curva = pd.DataFrame({"tasa": TASAS_CURVA, "mora": 0.06, "mora_importe": 0.06})
    return Resumen(auc, curva, 0.70, 1.0, 0.0)


def test_la_familia_corregida_entra_en_elegir_sin_traducir():
    modelos = {"logistica": _resumen(0.75), "random_forest": _resumen(0.76),
               "lightgbm": _resumen(0.77)}
    claves = pares(list(modelos))
    # un p crudo de 0,02 solo es significativo sin corregir: con tres pares queda en 0,06
    sin_corregir = dict.fromkeys(claves, 0.02)
    assert elegir(modelos, sin_corregir) == "lightgbm"
    assert elegir(modelos, bonferroni(sin_corregir, len(claves))) == "logistica"
    assert elegir(modelos, bonferroni(dict.fromkeys(claves, 0.01), len(claves))) == "lightgbm"
