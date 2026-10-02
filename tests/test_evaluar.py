"""Tests de la evaluación común y los suelos (src/models/evaluar.py, punto 0.5).

Casi todos sobre un frame sintético con señal conocida, así que corren en CI. Lo que protegen es que
la columna de probabilidad sea la del impago, que ningún ajuste vea su parte de evaluación y que el
OOF cubra a cada cliente una vez por semilla. El del final contra el dato real se salta sin los CSV
o sin la caché de folds.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import RAIZ, ruta
from src.models.evaluar import (
    Evaluacion,
    evaluar_en_folds,
    resumir,
    suelo_dummy,
    suelo_ext_source,
)
from src.models.folds import (
    NOMBRE_MANIFIESTO,
    Fold,
    cargar_contexto,
    cargar_folds,
    particionar,
)
from src.models.metricas import TASAS_CURVA

N = 400


@pytest.fixture
def poblacion():
    """400 clientes con un 8% de impago, una señal clara en `f1` y el contexto del 0.4."""
    rng = np.random.default_rng(0)
    indice = pd.Index(np.arange(1000, 1000 + N), name="SK_ID_CURR")
    y = pd.Series((np.arange(N) % 25 == 0).astype(int), index=indice, name="TARGET")
    X = pd.DataFrame(
        {"f1": y * 1.5 + rng.normal(size=N), "f2": rng.normal(size=N)}, index=indice
    )
    contexto = pd.DataFrame(
        {
            "AMT_CREDIT": rng.uniform(1e4, 1e6, N),
            "HAS_BUREAU_HISTORY": rng.integers(0, 2, N),
            "TARGET": y,
        },
        index=indice,
    )
    return X, y, contexto


def _folds(X, y, semillas=(0, 1)):
    return list(
        particionar(X, y, lambda: Pipeline([("id", FunctionTransformer())]), semillas=semillas)
    )


class _Invertido(BaseEstimator, ClassifierMixin):
    """Devuelve las columnas de `predict_proba` al revés, con `classes_` como corresponde."""

    def fit(self, X, y):
        self.interno_ = LogisticRegression().fit(X, y)
        self.classes_ = self.interno_.classes_
        return self

    def predict_proba(self, X):
        return self.interno_.predict_proba(X)[:, ::-1]


class _ClasesAlReves(_Invertido):
    """Con la columna bien puesta pero `classes_` como [1, 0]."""

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, X):
        return self.interno_.predict_proba(X)


class _Espia(LogisticRegression):
    """Anota los índices que ve cada `fit`, en una lista de clase para que sobreviva a `clone()`."""

    vistos: list = []

    def fit(self, X, y, sample_weight=None):
        type(self).vistos.append(set(X.index))
        return super().fit(X, y, sample_weight)


# --- la función contra un caso con señal conocida --------------------------------------------


def test_el_auc_por_fold_coincide_con_recalcularlo_a_mano(poblacion):
    X, y, _ = poblacion
    folds = _folds(X, y)

    ev = evaluar_en_folds(LogisticRegression(), folds)

    assert len(ev.por_fold) == len(folds)
    for fold in folds:
        # camino independiente: ajustar a mano y puntuar con roc_auc_score
        a_mano = LogisticRegression().fit(fold.X_ajuste, fold.y_ajuste)
        auc = roc_auc_score(fold.y_eval, a_mano.predict_proba(fold.X_eval)[:, 1])
        fila = ev.por_fold.query("semilla == @fold.semilla and k == @fold.k").iloc[0]
        assert fila["auc"] == pytest.approx(auc)
        assert fila["n_eval"] == len(fold.X_eval)
    assert ev.por_fold["auc"].min() > 0.5
    assert (ev.por_fold["gap"] == ev.por_fold["auc_ajuste"] - ev.por_fold["auc"]).all()


def test_el_score_de_ajuste_es_el_del_ajuste_y_se_mide_el_tiempo(poblacion):
    """Un bosque de arboles sin podar memoriza el ajuste: su AUC de ajuste casi 1 y el de
    evaluacion no."""
    X, y, _ = poblacion

    ev = evaluar_en_folds(RandomForestClassifier(50, random_state=0), _folds(X, y, semillas=(0,)))

    assert (ev.por_fold["auc_ajuste"] > 0.99).all()
    assert (ev.por_fold["gap"] > 0.05).all()
    assert (ev.por_fold["t_ajuste"] > 0).all() and (ev.por_fold["t_prediccion"] > 0).all()


def test_el_oof_cubre_a_cada_cliente_una_vez_por_semilla(poblacion):
    X, y, _ = poblacion

    ev = evaluar_en_folds(LogisticRegression(), _folds(X, y))

    assert list(ev.oof.columns) == [0, 1]
    assert ev.oof.index.equals(X.index.sort_values())
    assert not ev.oof.isna().any().any()
    assert ((ev.oof >= 0) & (ev.oof <= 1)).all().all()


def test_una_sola_semilla_el_caso_del_tuning_se_resume_sin_reventar(poblacion):
    """Con una semilla pandas no ordena al unir, y el OOF saldria en el orden de los folds."""
    X, y, contexto = poblacion

    ev = evaluar_en_folds(LogisticRegression(), _folds(X, y, semillas=(0,)))

    assert ev.oof.index.is_monotonic_increasing
    assert resumir(ev, contexto).auc_cv == pytest.approx(ev.por_fold["auc"].mean())


def test_el_estimador_de_entrada_no_se_ajusta(poblacion):
    X, y, _ = poblacion
    estimador = LogisticRegression()

    evaluar_en_folds(estimador, _folds(X, y))

    assert not hasattr(estimador, "coef_")


# --- las guardas, en las dos direcciones -----------------------------------------------------


def test_columnas_distintas_entre_ajuste_y_evaluacion_revienta(poblacion):
    X, y, _ = poblacion
    f = _folds(X, y, semillas=(0,))[0]
    roto = Fold(f.semilla, f.k, f.X_ajuste, f.y_ajuste, f.X_eval[["f2", "f1"]], f.y_eval)

    with pytest.raises(ValueError, match="columnas de ajuste y evaluación"):
        evaluar_en_folds(LogisticRegression(), [roto])


def test_classes_al_reves_revienta(poblacion):
    X, y, _ = poblacion

    with pytest.raises(ValueError, match=r"classes_ tiene que ser \[0, 1\]"):
        evaluar_en_folds(_ClasesAlReves(), _folds(X, y, semillas=(0,)))


def test_la_columna_equivocada_revienta_por_el_auc(poblacion):
    """Con `[:, 0]` el AUC sale a 1 menos el real: el modelo bueno se ve peor que el azar."""
    X, y, _ = poblacion

    with pytest.raises(ValueError, match="no pasa de 0,5"):
        evaluar_en_folds(_Invertido(), _folds(X, y, semillas=(0,)))


def test_un_cliente_repetido_en_el_oof_de_una_semilla_revienta(poblacion):
    X, y, _ = poblacion
    folds = _folds(X, y, semillas=(0,))

    with pytest.raises(ValueError, match="sale dos veces"):
        evaluar_en_folds(LogisticRegression(), [*folds, folds[0]])


def test_ningun_ajuste_ve_su_parte_de_evaluacion(poblacion):
    X, y, _ = poblacion
    folds = _folds(X, y)
    _Espia.vistos.clear()

    evaluar_en_folds(_Espia(), folds)

    assert len(_Espia.vistos) == len(folds)
    for visto, fold in zip(_Espia.vistos, folds):
        assert visto == set(fold.X_ajuste.index)
        assert not visto & set(fold.X_eval.index)


# --- resumir ---------------------------------------------------------------------------------


def test_resumir_con_una_pd_casi_perfecta_da_los_valores_conocidos(poblacion):
    """PD en 0,02 y 0,9 según el impago real, con un impago y un buen cliente intercambiados.
    El impago pasa a 0,001 y el buen cliente a 0,95.

    Sin ese solape la calibración no está identificada (separación perfecta). Los valores se
    calculan a mano: el impago colado entre los aprobados es el único que cuenta en la curva.
    """
    _, y, contexto = poblacion
    rng = np.random.default_rng(1)
    pd_casi = np.where(y == 1, 0.9, 0.02) + rng.uniform(0, 0.001, N)
    pos, neg = y.index[y == 1][0], y.index[y == 0][0]
    pd_casi = pd.Series(pd_casi, index=y.index)
    pd_casi[pos], pd_casi[neg] = 0.001, 0.95  # el impago colado queda el primero de los aprobados
    oof = pd.DataFrame({0: pd_casi, 1: pd_casi})
    por_fold = pd.DataFrame({"auc": [0.8, 0.9], "gap": [0.1, 0.2]})

    r = resumir(Evaluacion(por_fold, oof), contexto)

    sub = contexto["HAS_BUREAU_HISTORY"].eq(0)
    assert r.auc_cv == pytest.approx(0.85) and r.gap == pytest.approx(0.15)
    assert r.auc_sin_historial == pytest.approx(roc_auc_score(y[sub], pd_casi[sub]))
    # a mano: entre los 280 de menor PD solo hay un impago, el colado; el importe pesa lo suyo
    fila = r.curva.set_index("tasa").loc[0.70]
    aprobados = contexto.loc[pd_casi.sort_values().index[: int(0.70 * N)]]
    assert fila["mora"] == pytest.approx(1 / 280)
    assert fila["mora_importe"] == pytest.approx(
        contexto.loc[pos, "AMT_CREDIT"] / aprobados["AMT_CREDIT"].sum()
    )


def test_resumir_promedia_las_semillas_y_los_folds(poblacion):
    """Tres semillas distintas, tres folds distintos: la media no es la primera ni la mediana."""
    _, y, contexto = poblacion
    rng = np.random.default_rng(3)
    # cada semilla con mas senal que la anterior, para que ninguna media coincida con una sola
    logits = {k - 1: 0.4 * k * y.to_numpy() + rng.normal(size=N) - 2 for k in (1, 2, 3)}
    oof = pd.DataFrame({s: 1 / (1 + np.exp(-z)) for s, z in logits.items()}, index=y.index)
    por_fold = pd.DataFrame({"auc": [0.6, 0.7, 0.95], "gap": [0.1, 0.2, 0.6]})

    r = resumir(Evaluacion(por_fold, oof), contexto)

    sin = contexto["HAS_BUREAU_HISTORY"].eq(0).to_numpy()
    sueltos = [resumir(Evaluacion(por_fold, oof[[s]]), contexto) for s in oof.columns]
    assert r.auc_cv == pytest.approx(0.75) and r.gap == pytest.approx(0.3)  # media, no mediana
    for campo in ("pendiente", "ordenada", "auc_sin_historial"):
        valores = [getattr(x, campo) for x in sueltos]
        assert len(set(np.round(valores, 6))) == 3  # el caso separa las semillas
        assert getattr(r, campo) == pytest.approx(np.mean(valores))
    assert r.auc_sin_historial == pytest.approx(
        np.mean([roc_auc_score(y[sin], oof[s][sin]) for s in oof.columns])
    )
    mora = [x.curva["mora"] for x in sueltos]
    pd.testing.assert_series_equal(r.curva["mora"], sum(mora) / 3)


def test_resumir_deja_las_tasas_de_la_curva_exactas_con_tres_semillas(poblacion):
    """Promediar las curvas con sum()/3 movia la tasa (0,7 a 0,7000000000000001) y
    `cumple_criterio()` compara las tasas con igualdad exacta."""
    _, y, contexto = poblacion
    prob = pd.Series(np.linspace(0.01, 0.99, N), index=y.index)
    ev = Evaluacion(
        pd.DataFrame({"auc": [0.7], "gap": [0.0]}), pd.DataFrame({0: prob, 1: prob, 2: prob})
    )

    r = resumir(ev, contexto)

    assert list(r.curva["tasa"]) == list(TASAS_CURVA)


def test_resumir_revienta_si_el_oof_no_cubre_el_contexto(poblacion):
    _, y, contexto = poblacion
    prob = pd.Series(np.linspace(0.01, 0.99, N), index=y.index)
    por_fold = pd.DataFrame({"auc": [0.7], "gap": [0.0]})

    with pytest.raises(ValueError, match="no cubre exactamente"):
        resumir(Evaluacion(por_fold, pd.DataFrame({0: prob}).iloc[:-1]), contexto)
    con_hueco = pd.DataFrame({0: prob, 1: prob.where(prob.index != prob.index[3])})
    with pytest.raises(ValueError, match="no cubre exactamente"):
        resumir(Evaluacion(por_fold, con_hueco), contexto)


# --- los suelos ------------------------------------------------------------------------------


def test_el_dummy_da_la_prevalencia_del_ajuste_y_su_brier(poblacion):
    X, y, _ = poblacion
    folds = _folds(X, y, semillas=(0,))

    suelo = suelo_dummy(folds)

    assert len(suelo) == len(folds)
    for fila, fold in zip(suelo.itertuples(), folds):
        p = fold.y_ajuste.mean()
        assert fila.pd == pytest.approx(p)
        assert fila.brier == pytest.approx(((fold.y_eval - p) ** 2).mean())


def test_el_suelo_ext_source_va_sin_pesos_de_clase():
    """Excepcion declarada a la regla del desbalance: el liston es sin ninguna tecnica."""
    assert suelo_ext_source().named_steps["modelo"].class_weight is None


def test_el_suelo_ext_source_solo_lee_las_tres_ext_source(poblacion):
    X, y, _ = poblacion
    X = X.rename(columns={"f1": "EXT_SOURCE_1", "f2": "EXT_SOURCE_2"}).assign(
        EXT_SOURCE_3=lambda d: d["EXT_SOURCE_2"] * 0.5, otra=y * 100.0
    )

    ev = evaluar_en_folds(suelo_ext_source(), _folds(X, y, semillas=(0,)))

    # `otra` delata el impago al 100%: si el suelo la leyera, el AUC sería 1
    assert ev.por_fold["auc"].max() < 0.99


# --- contra el dato real ---------------------------------------------------------------------

sin_cache = pytest.mark.skipif(
    not (ruta("processed_data") / "folds" / NOMBRE_MANIFIESTO).exists()
    or not (RAIZ / "data" / "raw" / "application_train.csv").exists(),
    reason="la caché de folds no está construida o faltan los CSV, que no viajan con el repo",
)


@sin_cache
def test_el_suelo_ext_source_real_queda_donde_el_02_lo_dejo():
    """Puerta del 0.5: la logística de las tres EXT_SOURCE sobre los 15 folds reales."""
    ev = evaluar_en_folds(suelo_ext_source(), cargar_folds())
    r = resumir(ev, cargar_contexto())

    assert len(ev.por_fold) == 15
    # la media de las tres sin ajustar daba 0,7168 en el 0.2: ajustar los pesos solo puede mejorar
    assert 0.71 < r.auc_cv < 0.76
    assert r.gap < 0.01
    assert 0.9 <= r.pendiente <= 1.1
