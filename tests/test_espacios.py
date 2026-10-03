"""Tests de las estrategias de desbalance (src/models/espacios.py, punto 0.7).

Casi todos sobre un frame sintético con binarias, una constante y continuas, así que corren en CI.
Protegen que el remuestreo solo actúe en `fit`, que las sintéticas no inventen valores en las
binarias, que nunca haya sampler y pesos a la vez y que la regla de las finalistas lea bien la
tabla de contrastes. El del final contra el dato real se salta sin la caché de folds.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import ruta
from src.models import evaluar as evaluar_modulo
from src.models.espacios import (
    ESTRATEGIAS,
    FINALISTAS,
    NOMBRE_DECISION,
    PREFERENCIA,
    SINTETICAS,
    SMOTENCBinarias,
    avisos_imblearn,
    construir_estimador,
    contrastar,
    escribir_decision,
    filas_tras_remuestrear,
    finalistas,
    medir,
    miembros_de_folds,
    versiones,
)
from src.models.folds import NOMBRE_MANIFIESTO, Fold, cargar_folds, huella_cache, particionar
from src.models.metricas import ALFA

N = 1200
TODAS = [(m, e) for m, es in ESTRATEGIAS.items() for e in es]


@pytest.fixture
def datos():
    """1.200 clientes con un 8% de impago, tres binarias (una constante) y tres continuas."""
    rng = np.random.default_rng(0)
    y = pd.Series((np.arange(N) % 25 == 0).astype(int))
    X = pd.DataFrame(
        {
            "b1": rng.integers(0, 2, N).astype(float),
            "Spouse, partner": rng.integers(0, 2, N).astype(float),
            "const": np.ones(N),
            "c1": rng.normal(size=N) + y,
            "c2": rng.normal(size=N) * 100,
            "c3": rng.exponential(size=N),
        }
    )
    return X, y


def _ajustar(estimador, X, y):
    with avisos_imblearn():
        return estimador.fit(X, y)


def _proba(estimador, X):
    with avisos_imblearn():
        return estimador.predict_proba(X)


@pytest.mark.parametrize("modelo,estrategia", TODAS)
def test_evaluacion_conserva_filas_y_no_ve_sinteticos(datos, modelo, estrategia):
    X, y = datos
    est = _ajustar(construir_estimador(modelo, estrategia), X, y)
    nuevo = X.iloc[:100]
    assert _proba(est, nuevo).shape == (100, 2)
    with avisos_imblearn():
        assert len(est.predict(nuevo)) == 100


def test_remuestreo_solo_actua_en_fit_y_deja_la_entrada_intacta(datos):
    X, y = datos
    antes = X.copy()
    est = _ajustar(construir_estimador("logistica", "smotenc"), X, y)
    n, prev = filas_tras_remuestrear(est, Fold(0, 0, X, y, X, y))
    assert n > len(X) and prev == pytest.approx(0.5)
    pd.testing.assert_frame_equal(X, antes)
    assert len(_proba(est, X)) == len(X)


@pytest.mark.parametrize("modelo,estrategia", TODAS)
def test_filas_tras_remuestrear_en_todas_las_combinaciones(datos, modelo, estrategia):
    X, y = datos
    n, prev = filas_tras_remuestrear(
        construir_estimador(modelo, estrategia), Fold(0, 0, X, y, X, y)
    )
    if estrategia in ("submuestreo",) + SINTETICAS:
        assert prev == pytest.approx(0.5, abs=0.02) and n != len(X)
    else:
        assert n == len(X) and prev == pytest.approx(float(y.mean()))


def test_sinteticas_solo_toman_los_valores_de_cada_binaria(datos):
    X, y = datos
    with avisos_imblearn():
        Xr, yr = SMOTENCBinarias(random_state=0).fit_resample(X, y)
    assert len(Xr) > len(X)
    for col in ("b1", "Spouse, partner", "const"):
        assert set(Xr[col].unique()) <= set(X[col].unique())
    assert set(Xr["c1"].round(6)) - set(X["c1"].round(6))  # las continuas sí se interpolan


def test_smote_simple_si_inventa_valores_en_las_binarias(datos):
    X, y = datos
    with avisos_imblearn():
        Xr, _ = SMOTE(random_state=0).fit_resample(X, y)
    assert not set(Xr["b1"].unique()) <= {0.0, 1.0}


def test_smotenc_tras_el_escalado_tampoco_inventa_valores(datos):
    X, y = datos
    est = _ajustar(construir_estimador("logistica", "smotenc"), X, y)
    escalada = est[:1].transform(X)
    with avisos_imblearn():
        Xr, _ = est[:-1].fit_resample(X, y)
    for col in ("b1", "Spouse, partner"):
        assert set(Xr[col].unique()) <= set(escalada[col].unique())


def test_binarias_detectadas_por_valores_distintos():
    X = pd.DataFrame(
        {
            "binaria": [0.0, 1.0] * 50,
            "const": 1.0,
            "ternaria": [0.0, 1.0, 2.0, 0.0] * 25,
            "cont": np.linspace(0, 1, 100),
        }
    )
    smote = SMOTENCBinarias()
    smote._validate_column_types(X)
    assert list(smote.categorical_features_) == [0, 1]
    assert list(smote.continuous_features_) == [2, 3]


def test_binarias_detectadas_con_numpy_y_con_frame(datos):
    X, _ = datos
    a, b = SMOTENCBinarias(), SMOTENCBinarias()
    a._validate_column_types(X)
    b._validate_column_types(X.to_numpy())
    assert list(a.categorical_features_) == list(b.categorical_features_) == [0, 1, 2]


def test_clone_conserva_la_semilla_y_no_trae_categoricas_fijas():
    c = clone(SMOTENCBinarias(random_state=7, k_neighbors=3))
    assert c.random_state == 7 and c.k_neighbors == 3 and c.get_params()["k_neighbors"] == 3


@pytest.mark.parametrize("modelo,estrategia", TODAS)
def test_sampler_y_pesos_nunca_a_la_vez(modelo, estrategia):
    est = construir_estimador(modelo, estrategia)
    modelo_final = est.named_steps["modelo"]
    pesos = getattr(modelo_final, "class_weight", None) is not None
    sampler = "remuestreo" in est.named_steps
    assert not (pesos and sampler)
    assert pesos == (estrategia == "pesos")
    if estrategia == "balanced_rf":
        assert not sampler and not pesos


@pytest.mark.parametrize("modelo,estrategia", TODAS)
def test_escalado_antes_del_remuestreo_y_solo_donde_toca(modelo, estrategia):
    pasos = [n for n, _ in construir_estimador(modelo, estrategia).steps]
    if "remuestreo" in pasos:
        assert estrategia in ("submuestreo",) + SINTETICAS
    if estrategia in SINTETICAS:
        assert pasos.index("escalado") < pasos.index("remuestreo") < pasos.index("modelo")
    assert ("escalado" in pasos) == (modelo == "logistica" or estrategia in SINTETICAS)
    assert pasos[-1] == "modelo"


def test_fabrica_rechaza_estrategia_que_el_modelo_no_tiene():
    with pytest.raises(ValueError, match="no es una estrategia"):
        construir_estimador("logistica", "balanced_rf")
    with pytest.raises(ValueError, match="no es una estrategia"):
        construir_estimador("svm", "pesos")


def test_estrategias_cubren_los_tres_modelos_con_pesos_de_referencia():
    assert {m for m, _ in TODAS} == {"logistica", "random_forest", "lightgbm"}
    assert all("pesos" in es and "nada" in es for es in ESTRATEGIAS.values())
    assert set(PREFERENCIA) == {e for _, e in TODAS}


@pytest.mark.parametrize("modelo", ["logistica", "random_forest", "lightgbm"])
def test_dos_ajustes_con_la_misma_semilla_dan_lo_mismo(datos, modelo):
    X, y = datos
    p = [_proba(_ajustar(construir_estimador(modelo, "smotenc"), X, y), X)[:, 1] for _ in range(2)]
    np.testing.assert_array_equal(p[0], p[1])


@pytest.mark.parametrize("modelo", ["logistica", "lightgbm"])
def test_cache_de_remuestreo_no_cambia_la_prediccion_y_se_comparte(datos, tmp_path, modelo):
    X, y = datos
    sin = _proba(_ajustar(construir_estimador(modelo, "smotenc"), X, y), X)
    con = construir_estimador(modelo, "smotenc", str(tmp_path))
    np.testing.assert_array_equal(_proba(_ajustar(con, X, y), X), sin)
    guardado = {p.name for p in tmp_path.rglob("*") if p.is_dir()}
    # otro modelo con el mismo caché reutiliza lo remuestreado: no añade entradas del sampler
    otro = "random_forest" if modelo == "logistica" else "logistica"
    _ajustar(construir_estimador(otro, "smotenc", str(tmp_path)), X, y)
    nuevo = {p.name for p in tmp_path.rglob("*") if p.is_dir()} - guardado
    assert not any("fit_resample_one" in n for n in nuevo)


def test_lightgbm_ve_los_mismos_nombres_al_ajustar_y_al_predecir(datos):
    X, y = datos
    est = construir_estimador("lightgbm", "smotenc")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _proba(_ajustar(est, X, y), X)
    assert list(est.named_steps["modelo"].feature_name_) == [
        "b1",
        "Spouse__partner",
        "const",
        "c1",
        "c2",
        "c3",
    ]


def test_imblearn_no_deja_avisos_propios_con_las_seis_estrategias(datos):
    X, y = datos
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        for modelo, estrategia in TODAS:
            _proba(_ajustar(construir_estimador(modelo, estrategia), X, y), X)
    # los RuntimeWarning de matmul son de numpy en macOS arm64 con la logística (0.6)
    ajenos = [a for a in avisos if "matmul" not in str(a.message)]
    assert not ajenos, [str(a.message) for a in ajenos]


def test_filtro_de_avisos_es_solo_el_de_imblearn():
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        with avisos_imblearn():
            warnings.warn("otro aviso cualquiera", FutureWarning)
            warnings.warn("`BaseEstimator._validate_data` is deprecated", FutureWarning)
    assert [str(a.message) for a in avisos] == ["otro aviso cualquiera"]


def test_balanced_rf_equilibra_cada_arbol_sin_el_aviso_de_la_013(datos):
    X, y = datos
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with avisos_imblearn():
            construir_estimador("random_forest", "balanced_rf").fit(X, y)


def _semillas_del_remuestreo(estimador):
    """Los `random_state` del paso de remuestreo, incluidos los de los samplers que contiene."""
    paso = estimador.named_steps["remuestreo"]
    return [v for k, v in paso.get_params().items() if k.endswith("random_state")]


SIN_SEMILLA = ("pesos", "nada", "balanced_rf")


@pytest.mark.parametrize(
    ("modelo", "estrategia"), [(m, e) for m, e in TODAS if e not in SIN_SEMILLA]
)
def test_todo_remuestreo_lleva_semilla_en_cada_sampler(modelo, estrategia):
    semillas = _semillas_del_remuestreo(construir_estimador(modelo, estrategia))
    assert semillas and all(s is not None for s in semillas)


# la tabla por fold que alimenta los contrastes ----------------------------------------------


def test_medir_da_por_fold_el_auc_y_la_media_de_mora_de_la_zona_calculados_a_mano(monkeypatch):
    rng = np.random.default_rng(0)
    n = 1000
    indice = pd.Index(np.arange(1000, 1000 + n), name="SK_ID_CURR")
    y = pd.Series((np.arange(n) % 25 == 0).astype(int), index=indice, name="TARGET")
    X = pd.DataFrame({"f1": y * 1.5 + rng.normal(size=n), "f2": rng.normal(size=n)}, index=indice)
    contexto = pd.DataFrame(
        {
            "AMT_CREDIT": rng.uniform(1e4, 1e6, n),
            "HAS_BUREAU_HISTORY": rng.integers(0, 2, n),
            "TARGET": y,
        },
        index=indice,
    )
    split = pd.DataFrame({"SK_ID_CURR": indice, "TARGET": y.to_numpy(), "split": "train"})
    monkeypatch.setattr(evaluar_modulo, "cargar_split", lambda: split)
    identidad = lambda: SkPipeline([("id", FunctionTransformer())])  # noqa: E731

    def folds():
        yield from particionar(X, y, identidad, semillas=(0,))

    por_fold = medir("logistica", "nada", folds, contexto, miembros_de_folds(folds()))[1]

    assert len(por_fold) == 5
    for fold in folds():
        est = construir_estimador("logistica", "nada").fit(fold.X_ajuste, fold.y_ajuste)
        p = est.predict_proba(fold.X_eval)[:, 1]
        ye = fold.y_eval.to_numpy()[np.argsort(p, kind="stable")]
        moras = [ye[: round(t * len(ye))].mean() for t in (0.70, 0.75, 0.80, 0.85, 0.90)]
        fila = por_fold.query("k == @fold.k").iloc[0]
        assert fila["auc"] == pytest.approx(roc_auc_score(fold.y_eval, p))
        assert fila["mora_zona"] == pytest.approx(np.mean(moras))
        assert np.mean(moras) < max(moras)  # guardián: media y máximo se distinguen


# contrastes y finalistas ----------------------------------------------------------------------


def _por_fold(modelo, difs, ruido=0.0005, estrategias=("pesos", "nada", "smotenc")):
    """Folds sintéticos: `pesos` en 0,76 de AUC y 5% de mora, el resto desplazado por `difs`."""
    rng = np.random.default_rng(1)
    filas = []
    for e in estrategias:
        d_auc, d_mora = difs.get(e, (0.0, 0.0))
        for k in range(5):
            filas.append(
                {
                    "modelo": modelo,
                    "estrategia": e,
                    "semilla": 0,
                    "k": k,
                    "auc": 0.76 + 0.001 * k + d_auc + rng.normal(0, ruido),
                    "mora_zona": 0.05 - d_mora + rng.normal(0, ruido),
                }
            )
    return pd.DataFrame(filas)


def test_contrastar_detecta_la_mejora_y_la_ausencia_de_diferencia():
    por_fold = _por_fold(
        "lightgbm", {"smotenc": (0.02, 0.01)}, estrategias=("pesos", "nada", "smotenc")
    )
    # la familia de lightgbm son 4 contrastes: se rellenan con réplicas iguales a `nada`
    extra = [
        por_fold[por_fold["estrategia"] == "nada"].assign(estrategia=e)
        for e in ("submuestreo", "smotenc_tomek")
    ]
    c = contrastar(pd.concat([por_fold, *extra])).set_index("estrategia")
    assert c.loc["smotenc", "p_auc"] < ALFA and c.loc["smotenc", "media_auc"] > 0
    assert c.loc["smotenc", "p_mora"] < ALFA and c.loc["smotenc", "media_mora"] > 0
    assert c.loc["nada", "p_auc"] > ALFA


def test_contrastar_revienta_si_falta_una_estrategia_de_la_familia():
    with pytest.raises(ValueError, match="se declaran 4 contrastes y se han emitido 2"):
        contrastar(_por_fold("lightgbm", {}))


def test_contrastar_revienta_sin_la_referencia_o_con_otros_folds():
    sin_pesos = _por_fold("lightgbm", {})
    with pytest.raises(ValueError, match="falta la referencia"):
        contrastar(sin_pesos[sin_pesos["estrategia"] != "pesos"])
    otros = sin_pesos.copy()
    otros.loc[otros["estrategia"] == "nada", "k"] += 10
    with pytest.raises(ValueError, match="no son los de la referencia"):
        contrastar(otros)


def _tabla(modelo, filas):
    """`filas`: estrategia a (auc_media, media_auc, p_auc, media_mora, p_mora); `pesos` va sola."""
    base = {"modelo": modelo, "estrategia": "pesos", "auc_media": 0.76}
    columnas = ("auc_media", "media_auc", "p_auc", "media_mora", "p_mora")
    return pd.DataFrame(
        [base]
        + [{"modelo": modelo, "estrategia": e, **dict(zip(columnas, v))} for e, v in filas.items()]
    )


NS = (0.76, 0.0, 0.9, 0.0, 0.9)  # sin diferencia significativa


def test_finalistas_sin_diferencias_son_pesos_y_nada_por_preferencia():
    t = _tabla("lightgbm", {"nada": NS, "submuestreo": NS, "smotenc": NS, "smotenc_tomek": NS})
    assert finalistas(t) == {"lightgbm": ["pesos", "nada"]}


def test_finalistas_una_sintetica_que_gana_adelanta_a_pesos():
    gana = (0.78, 0.02, 0.001, 0.0, 0.9)
    t = _tabla("lightgbm", {"nada": NS, "smotenc": gana})
    assert finalistas(t) == {"lightgbm": ["smotenc", "pesos"]}


def test_finalistas_gana_por_la_mora_aunque_el_auc_no_se_mueva():
    t = _tabla("lightgbm", {"nada": NS, "smotenc": (0.76, 0.0, 0.9, 0.01, 0.001)})
    assert finalistas(t) == {"lightgbm": ["smotenc", "pesos"]}


def test_finalistas_todas_peores_que_pesos_queda_pesos_solo():
    peor = (0.70, -0.06, 0.001, 0.0, 0.9)
    t = _tabla("logistica", {"nada": peor, "smotenc": peor})
    assert finalistas(t) == {"logistica": ["pesos"]}


def test_finalistas_la_que_empeora_una_metrica_queda_fuera_aunque_mejore_la_otra():
    mixta = (0.78, 0.02, 0.001, -0.01, 0.001)
    t = _tabla("lightgbm", {"nada": NS, "smotenc": mixta})
    assert finalistas(t) == {"lightgbm": ["pesos", "nada"]}


def test_finalistas_adelantan_ordenadas_por_auc_medio():
    a = (0.77, 0.01, 0.001, 0.0, 0.9)
    b = (0.78, 0.02, 0.001, 0.0, 0.9)
    t = _tabla("random_forest", {"submuestreo": a, "smotenc": b})
    assert finalistas(t) == {"random_forest": ["smotenc", "submuestreo"]}


def test_finalistas_una_diferencia_no_significativa_no_adelanta():
    t = _tabla("lightgbm", {"nada": NS, "smotenc": (0.78, 0.02, ALFA, 0.0, 0.9)})
    assert finalistas(t) == {"lightgbm": ["pesos", "nada"]}


def test_finalistas_revienta_sin_la_referencia():
    t = _tabla("lightgbm", {"nada": NS})
    with pytest.raises(ValueError, match="falta la referencia"):
        finalistas(t[t["estrategia"] != "pesos"])


# la decisión escrita ---------------------------------------------------------------------------


def _decision():
    return json.loads((ruta("models") / NOMBRE_DECISION).read_text())


def test_finalistas_son_dos_por_modelo_de_su_espacio_y_sin_sinteticas():
    assert set(FINALISTAS) == set(ESTRATEGIAS)
    for modelo, elegidas in FINALISTAS.items():
        assert len(elegidas) == 2 and len(set(elegidas)) == 2
        assert set(elegidas) <= set(ESTRATEGIAS[modelo])
        assert not set(elegidas) & set(SINTETICAS)


def test_la_decision_escrita_lleva_la_huella_de_los_folds_y_las_versiones(tmp_path):
    estrategias = ESTRATEGIAS["logistica"]
    por_fold = _por_fold("logistica", {}, estrategias=estrategias)
    resumen = pd.DataFrame(
        {"modelo": "logistica", "estrategia": list(estrategias), "auc_media": 0.76}
    )
    destino = tmp_path / "decision.json"

    escribir_decision(resumen, por_fold, "huella-de-prueba", destino)

    d = json.loads(destino.read_text())
    assert d["huella_folds"] == "huella-de-prueba"
    assert d["versiones"] == versiones()
    assert set(d["versiones"]) == {"scikit-learn", "imbalanced-learn", "lightgbm"}
    assert all(v for v in d["versiones"].values())


def test_finalistas_escritas_salen_de_aplicar_la_regla_a_la_evidencia_guardada():
    d = _decision()
    tabla = pd.DataFrame(d["tabla"]).merge(
        pd.DataFrame(d["contrastes"]), on=["modelo", "estrategia"], how="left"
    )
    assert {m: tuple(e) for m, e in finalistas(tabla).items()} == FINALISTAS
    assert {m: tuple(e) for m, e in d["finalistas"].items()} == FINALISTAS


def test_evidencia_guardada_cubre_las_seis_estrategias_y_las_familias_de_bonferroni():
    d = _decision()
    celdas = {(r["modelo"], r["estrategia"]) for r in d["tabla"]}
    assert celdas == {(m, e) for m, es in ESTRATEGIAS.items() for e in es}
    emitidos = pd.DataFrame(d["contrastes"]).groupby("modelo").size().to_dict()
    assert emitidos == {m: len(es) - 1 for m, es in ESTRATEGIAS.items()}


# dato real ------------------------------------------------------------------------------------

sin_cache = pytest.mark.skipif(
    not (ruta("processed_data") / "folds" / NOMBRE_MANIFIESTO).exists(),
    reason="la caché de folds no está construida, que no viaja con el repo",
)


@sin_cache
def test_fold_real_conserva_su_tamano_y_su_prevalencia_tras_ajustar():
    """La parte de evaluación no se toca: mismas filas, mismo frame y el 8,07% de impago."""
    fold = next(cargar_folds(semillas=(0,)))
    antes = fold.X_eval.copy()
    est = _ajustar(
        construir_estimador("lightgbm", "smotenc"),
        fold.X_ajuste.iloc[:20000],
        fold.y_ajuste.iloc[:20000],
    )
    assert len(_proba(est, fold.X_eval)) == len(fold.y_eval)
    pd.testing.assert_frame_equal(fold.X_eval, antes)
    assert float(fold.y_eval.mean()) == pytest.approx(0.0807, abs=0.0005)


@sin_cache
def test_la_decision_guardada_es_de_los_folds_y_las_versiones_actuales():
    d = _decision()
    assert d["huella_folds"] == huella_cache()
    assert d["versiones"] == versiones()
