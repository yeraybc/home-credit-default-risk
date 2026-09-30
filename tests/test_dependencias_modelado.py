"""Compatibilidad del stack de modelado (0.1): imports y remuestreo dentro del pipeline."""

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def _datos(n=600, seed=0):
    rng = np.random.default_rng(seed)
    binaria = rng.integers(0, 2, n).astype(float)
    numericas = rng.normal(size=(n, 3))
    y = (rng.random(n) < 0.1 + 0.1 * binaria).astype(int)
    return np.column_stack([binaria, numericas]), y


def test_imports_del_stack():
    import lightgbm
    import optuna
    import statsmodels

    assert lightgbm.__version__ == "4.6.0"
    assert optuna.__version__ == "5.0.0"
    assert statsmodels.__version__ == "0.14.6"


def test_smotenc_en_pipeline_solo_actua_en_fit():
    X, y = _datos()
    pipe = Pipeline(
        [("s", SMOTENC([0], random_state=0)), ("m", LogisticRegression())],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*_get_tags.*", category=DeprecationWarning)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_smotenc_sinteticas_binarias_solo_0_o_1():
    X, y = _datos()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*_get_tags.*", category=DeprecationWarning)
        Xr, yr = SMOTENC([0], random_state=0).fit_resample(X, y)
    assert len(Xr) > len(X)
    assert set(np.unique(Xr[:, 0])) <= {0.0, 1.0}


def test_avisos_de_imblearn_son_solo_el_de_tags():
    X, y = _datos()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        Pipeline([("s", SMOTENC([0], random_state=0)), ("m", LogisticRegression())]).fit(X, y)
    # los RuntimeWarning de matmul son de numpy en macOS arm64, ajenos a imblearn
    ajenos = [
        a
        for a in avisos
        if issubclass(a.category, DeprecationWarning)
        and "_get_tags" not in str(a.message)
        and "_more_tags" not in str(a.message)
    ]
    if ajenos:
        pytest.fail(f"avisos nuevos de imblearn: {[str(a.message) for a in ajenos]}")
