"""Tests de los folds con el pipeline reajustado por fold (src/models/folds.py, punto 0.4).

Casi todos sobre un frame sintético con un pipeline de identidad o un espía, así que corren en CI.
Lo que protegen es que **ningún ajuste ve su parte de evaluación**, que la caché no sobrevive a un
cambio de split, de cortes o de código, y que los folds cubren train sin pisarse. Los dos del
final contra el dato real se saltan sin los CSV o sin la caché construida.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import RAIZ, ruta
from src.features.pipeline import construir_pipeline
from src.features.selection import SEMILLAS_ESTABILIDAD
from src.features.split import cargar_split
from src.models import folds as modulo
from src.models.folds import (
    N_FOLDS,
    NOMBRE_MANIFIESTO,
    SEMILLA_TUNING,
    SEMILLAS,
    cargar_contexto,
    cargar_folds,
    escribir_folds,
    huella_folds,
    informe_folds,
    informe_identidad,
    particionar,
    poblacion_train,
)

N = 400


@pytest.fixture
def poblacion():
    """400 clientes con un 8% de impago, tres columnas y el contexto que pide `metricas.py`."""
    rng = np.random.default_rng(0)
    indice = pd.Index(np.arange(1000, 1000 + N), name="SK_ID_CURR")
    y = pd.Series((np.arange(N) % 25 == 0).astype(int), index=indice, name="TARGET")
    X = pd.DataFrame(rng.normal(size=(N, 3)), index=indice, columns=["f1", "f2", "f3"])
    contexto = pd.DataFrame(
        {"AMT_CREDIT": rng.uniform(1e4, 1e6, N), "HAS_BUREAU_HISTORY": rng.integers(0, 2, N)},
        index=indice,
    )
    return X, y, contexto


def _identidad():
    return Pipeline([("id", FunctionTransformer())])


class _Espia(BaseEstimator, TransformerMixin):
    """Anota los índices que ve cada `fit` y cada `transform`, en una lista de clase para que
    sobreviva a `clone()`."""

    ajustados: list = []
    transformados: list = []

    def fit(self, datos, y=None):
        type(self).ajustados.append(set(datos.index))
        return self

    def transform(self, datos):
        type(self).transformados.append(set(datos.index))
        return datos


@pytest.fixture
def split(poblacion):
    X, y, _ = poblacion
    return pd.DataFrame({"SK_ID_CURR": X.index, "TARGET": y.to_numpy(), "split": "train"})


# --- el ajuste de cada fold no ve su evaluación ----------------------------------------------


def test_el_ajuste_de_cada_fold_no_ve_ningun_indice_de_su_evaluacion(poblacion):
    X, y, _ = poblacion
    _Espia.ajustados, _Espia.transformados = [], []

    todos = list(particionar(X, y, lambda: Pipeline([("espia", _Espia())])))

    assert len(todos) == len(SEMILLAS) * N_FOLDS == len(_Espia.ajustados)
    for i, fold in enumerate(todos):
        ajustado = _Espia.ajustados[i]
        assert ajustado == set(fold.X_ajuste.index)
        assert not ajustado & set(fold.X_eval.index), f"el fold {i} ajustó con su evaluación"
        # dos transform por fold: el de la matriz de ajuste y el de evaluación, que va el último
        assert _Espia.transformados[2 * i + 1] == set(fold.X_eval.index)


def test_el_espia_delata_un_ajuste_con_todos_los_clientes(poblacion):
    """El otro sentido: si el pipeline se ajustara con todo train, la comprobación saltaría."""
    X, y, _ = poblacion
    _Espia.ajustados, _Espia.transformados = [], []
    _Espia().fit(X)
    fold = next(particionar(X, y, _identidad))

    assert _Espia.ajustados[0] & set(fold.X_eval.index)


def test_la_matriz_de_ajuste_sale_de_fit_transform(poblacion):
    """La ocupación cruzada solo es honesta con `fit_transform`: con `fit` y `transform` cada fila
    de ajuste recibiría la media de su categoría con ella dentro."""
    X, y, _ = poblacion
    registro = []

    class _Marca(_Espia):
        def fit_transform(self, datos, y=None, **kw):
            registro.append("fit_transform")
            return super().fit(datos, y).transform(datos)

    list(particionar(X, y, lambda: Pipeline([("m", _Marca())]), semillas=(0,)))

    assert registro == ["fit_transform"] * N_FOLDS


def test_folds_disjuntos_que_cubren_la_poblacion_y_estratificados(poblacion):
    X, y, _ = poblacion
    for semilla in SEMILLAS:
        dados = [f for f in particionar(X, y, _identidad, (semilla,))]
        evals = [set(f.X_eval.index) for f in dados]

        assert set().union(*evals) == set(X.index)
        assert sum(len(e) for e in evals) == len(X), "algún cliente cae en dos folds"
        for f in dados:
            assert not set(f.X_ajuste.index) & set(f.X_eval.index)
            assert abs(f.y_eval.mean() - y.mean()) < 0.015
            assert abs(f.y_ajuste.mean() - y.mean()) < 0.015


def test_los_folds_de_cada_semilla_son_los_de_estabilidad_banda(poblacion):
    """Las tres semillas, cada una con sus folds: con una sola fija cambiaría la medida."""
    X, y, _ = poblacion
    assert SEMILLAS == SEMILLAS_ESTABILIDAD == (0, 1, 2)
    assert SEMILLA_TUNING == SEMILLAS[0]

    reales = {}
    for semilla in SEMILLAS:
        kfold = StratifiedKFold(N_FOLDS, shuffle=True, random_state=semilla)
        esperados = [set(X.index[i_eval]) for _, i_eval in kfold.split(X, y)]
        reales[semilla] = [set(f.X_eval.index) for f in particionar(X, y, _identidad, (semilla,))]
        assert reales[semilla] == esperados
    assert reales[0] != reales[1] != reales[2], "dos semillas repartieron igual"


# --- la caché --------------------------------------------------------------------------------


def test_la_cache_vuelve_en_float32_y_con_los_mismos_folds(poblacion, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)

    leidos = list(cargar_folds(origen=tmp_path, huella="h"))
    originales = list(particionar(X, y, _identidad))

    assert len(leidos) == len(originales) == len(SEMILLAS) * N_FOLDS
    for leido, original in zip(leidos, originales):
        assert (leido.semilla, leido.k) == (original.semilla, original.k)
        assert (leido.X_eval.dtypes == "float32").all()
        pd.testing.assert_frame_equal(leido.X_ajuste, original.X_ajuste.astype("float32"))
        pd.testing.assert_series_equal(leido.y_eval, original.y_eval)
    pd.testing.assert_frame_equal(
        cargar_contexto(tmp_path, "h").drop(columns="TARGET"), contexto
    )


def test_cargar_folds_filtra_por_semilla(poblacion, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)

    dos = list(cargar_folds(semillas=SEMILLAS[1:], origen=tmp_path, huella="h"))

    assert {f.semilla for f in dos} == set(SEMILLAS[1:])


def test_no_pisa_una_cache_existente_sin_sobrescribir(poblacion, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)

    with pytest.raises(FileExistsError, match="sobrescribir=True"):
        escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    escribir_folds(X, y, contexto, "h2", tmp_path, sobrescribir=True, fabrica=_identidad)

    assert json.loads((tmp_path / NOMBRE_MANIFIESTO).read_text())["huella"] == "h2"


def test_una_reconstruccion_interrumpida_no_deja_el_manifiesto_viejo(poblacion, tmp_path):
    """Con `sobrescribir=True` el manifiesto anterior se retira antes de tocar nada: si el ajuste
    se cae a medias, la caché mezcla folds viejos y nuevos y no debe poder leerse."""
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    llamadas = iter(range(100))

    def se_cae():
        if next(llamadas) == 3:
            raise RuntimeError("ajuste interrumpido")
        return _identidad()

    with pytest.raises(RuntimeError, match="interrumpido"):
        escribir_folds(X, y, contexto, "h2", tmp_path, sobrescribir=True, fabrica=se_cae)

    assert not (tmp_path / NOMBRE_MANIFIESTO).exists()
    with pytest.raises(FileNotFoundError):
        next(cargar_folds(origen=tmp_path, huella="h"))


def test_una_huella_distinta_revienta_al_cargar(poblacion, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)

    with pytest.raises(ValueError, match="no casa"):
        next(cargar_folds(origen=tmp_path, huella="otra"))
    with pytest.raises(ValueError, match="no casa"):
        cargar_contexto(tmp_path, "otra")


def test_una_cache_sin_manifiesto_no_se_lee(poblacion, tmp_path):
    """El manifiesto se escribe el último: sin él, la caché quedó a medias."""
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    (tmp_path / NOMBRE_MANIFIESTO).unlink()

    with pytest.raises(FileNotFoundError, match="construir_folds"):
        next(cargar_folds(origen=tmp_path, huella="h"))


def test_la_huella_cambia_con_el_split_y_con_los_cortes(split, tmp_path):
    cortes = tmp_path / "cortes.json"
    cortes.write_text('{"cortes": {}}')
    base = huella_folds(split, cortes)

    otro_split = split.assign(split=["valid"] + ["train"] * (N - 1))
    cortes_distintos = tmp_path / "otros.json"
    cortes_distintos.write_text('{"cortes": {"a": 1}}')

    assert huella_folds(split, cortes) == base
    assert huella_folds(otro_split, cortes) != base
    assert huella_folds(split, cortes_distintos) != base


def test_la_huella_cambia_con_el_codigo_de_features(split, tmp_path, monkeypatch):
    """Si cambia cualquier fuente del pipeline, la caché deja de valer."""
    cortes = tmp_path / "cortes.json"
    cortes.write_text("{}")
    fuente = tmp_path / "src" / "features"
    fuente.mkdir(parents=True)
    (fuente / "a.py").write_text("x = 1")
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    antes = huella_folds(split, cortes)

    (fuente / "a.py").write_text("x = 2")

    assert huella_folds(split, cortes) != antes


# --- el informe, que es la puerta ------------------------------------------------------------


def test_el_informe_de_un_fold_sano(poblacion, split, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)

    informe = informe_folds(split, tmp_path, "h")

    assert len(informe) == len(SEMILLAS) * N_FOLDS
    assert informe["disjuntos"].all() and informe["mismas_columnas"].all()
    assert (informe["n_ajuste"] + informe["n_eval"] == N).all()
    assert (informe["n_columnas"] == 3).all()
    assert informe["faltan"].map(len).eq(0).all()
    assert (informe[["pct_ajuste", "pct_eval"]].sub(y.mean() * 100).abs() < 1.5).all().all()


def test_el_informe_revienta_con_un_cliente_que_no_es_de_train(poblacion, split, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    con_valid = split.assign(split=["valid"] + ["train"] * (N - 1))

    with pytest.raises(ValueError, match="no es el de train"):
        informe_folds(con_valid, tmp_path, "h")


def test_el_informe_delata_un_fold_con_solapamiento(poblacion, split, tmp_path):
    """Una fila de ajuste repetida en la evaluación: cubre train, pero ya no es disjunto."""
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    ajuste = pd.read_parquet(tmp_path / "s0_k0_ajuste.parquet")
    evaluacion = pd.read_parquet(tmp_path / "s0_k0_eval.parquet")
    pd.concat([evaluacion, ajuste.iloc[:1]]).to_parquet(tmp_path / "s0_k0_eval.parquet")

    informe = informe_folds(split, tmp_path, "h").set_index(["semilla", "k"])

    assert not informe.loc[(0, 0), "disjuntos"]
    assert informe["disjuntos"].drop((0, 0)).all()


def test_el_informe_revienta_si_un_fold_no_cubre_train(poblacion, split, tmp_path):
    X, y, contexto = poblacion
    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=_identidad)
    pd.read_parquet(tmp_path / "s0_k0_ajuste.parquet").to_parquet(tmp_path / "s0_k0_eval.parquet")

    with pytest.raises(ValueError, match="de train ausentes"):
        informe_folds(split, tmp_path, "h")


def test_el_informe_delata_las_columnas_que_un_fold_pierde(poblacion, split, tmp_path):
    """El `SelectorIV` puede decidir distinto en cada fold: el informe dice quién se cae."""
    X, y, contexto = poblacion
    llamadas = iter(range(100))

    def fabrica():
        quita = next(llamadas) == 3
        return Pipeline(
            [("f", FunctionTransformer((lambda d: d.drop(columns=["f2"])) if quita else None))]
        )

    escribir_folds(X, y, contexto, "h", tmp_path, fabrica=fabrica)
    informe = informe_folds(split, tmp_path, "h")

    raro = informe[informe["n_columnas"] == 2]
    assert len(raro) == 1 and list(raro["faltan"].iloc[0]) == ["f2"]
    assert (informe["n_columnas"] == 3).sum() == len(informe) - 1


# --- la población de train -------------------------------------------------------------------


def _falsear_capa1(monkeypatch, split, solo_train=None, cortes=None):
    """Sustituye la lectura de CSV por un frame de capa 1 de seis clientes."""
    capa1 = pd.DataFrame(
        {
            "SK_ID_CURR": split["SK_ID_CURR"],
            "TARGET": split["TARGET"],
            "AMT_CREDIT": 100.0,
            "HAS_BUREAU_HISTORY": 1.0,
            "f1": 1.0,
        }
    )
    monkeypatch.setattr(modulo, "cargar_split", lambda: split)
    monkeypatch.setattr(modulo, "preparar_application", lambda: capa1)
    monkeypatch.setattr(
        modulo, "cargar_cortes", lambda *a, **k: None if cortes is None else cortes.append((a, k))
    )
    monkeypatch.setattr(modulo, "cargar_auxiliares", lambda: (None, None, None))
    monkeypatch.setattr(modulo, "ensamblar_auxiliares", lambda base, *_: base)
    if solo_train is not None:
        monkeypatch.setattr(modulo, "solo_train", solo_train)


def test_la_poblacion_son_los_clientes_de_train_sin_id_ni_etiqueta(monkeypatch):
    split = pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3, 4, 5, 6], "TARGET": [0, 1, 0, 0, 1, 0]}
    ).assign(split=["train", "train", "valid", "train", "valid", "train"])
    cortes = []
    _falsear_capa1(monkeypatch, split, cortes=cortes)

    X, y, contexto = poblacion_train()

    # los cortes se recargan con sobrescribir=True: sin él, la segunda llamada del proceso revienta
    assert cortes == [((split,), {"sobrescribir": True})]
    assert list(X.index) == [1, 2, 4, 6] and X.index.name == "SK_ID_CURR"
    assert list(X.columns) == ["AMT_CREDIT", "HAS_BUREAU_HISTORY", "f1"]
    assert list(y) == [0, 1, 0, 0]
    assert list(contexto.columns) == ["AMT_CREDIT", "HAS_BUREAU_HISTORY"]


def test_la_poblacion_revienta_si_se_cuela_un_cliente_de_valid(monkeypatch):
    split = pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "split": ["train", "train", "valid"]}
    )
    _falsear_capa1(monkeypatch, split, solo_train=lambda df, s=None: df)

    with pytest.raises(ValueError, match="no es el de train"):
        poblacion_train()


# --- la misma persona con dos solicitudes ----------------------------------------------------


def test_informe_identidad_separa_la_coincidencia_exacta_de_la_invariante():
    """Un par exacto que cruza train y valid, y un par desplazado 100 días en train."""
    base = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "CODE_GENDER": ["F", "F", "M", "M", "F", "M"],
            "DAYS_BIRTH": [-10000, -10000, -12000, -12100, -9000, -8000],
            "DAYS_ID_PUBLISH": [-3000, -3000, -4000, -4100, -2000, -1000],
            "DAYS_REGISTRATION": [-2000.0, -2000.0, -5000.0, -5100.0, -100.0, -50.0],
        }
    )
    split = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "split": ["train", "valid", "train", "train", "train", "valid"],
        }
    )

    informe = informe_identidad(base, split)

    assert informe.loc["exacta"].to_dict() == {
        "clientes_en_grupo": 2,
        "grupos": 1,
        "grupos_cruzan": 1,
        "valid_con_par_en_train": 1,
        "grupos_solo_train": 0,
    }
    assert informe.loc["invariante"].to_dict() == {
        "clientes_en_grupo": 4,
        "grupos": 2,
        "grupos_cruzan": 1,
        "valid_con_par_en_train": 1,
        "grupos_solo_train": 1,
    }


def test_informe_identidad_sin_nadie_repetido_sale_a_cero():
    base = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "CODE_GENDER": ["F", "M"],
            "DAYS_BIRTH": [-10000, -12000],
            "DAYS_ID_PUBLISH": [-3000, -4000],
            "DAYS_REGISTRATION": [-2000.0, -5000.0],
        }
    )
    split = pd.DataFrame({"SK_ID_CURR": [1, 2], "split": ["train", "valid"]})

    assert (informe_identidad(base, split) == 0).all().all()


# --- contra el dato real ---------------------------------------------------------------------

DATOS = RAIZ / "data"
sin_datos = pytest.mark.skipif(
    not all(
        (DATOS / r).exists()
        for r in (
            "raw/application_train.csv",
            "raw/bureau.csv",
            "raw/bureau_balance.csv",
            "raw/previous_application.csv",
            "processed/split.parquet",
            "processed/cortes.json",
            "processed/X_train.parquet",
        )
    ),
    reason="faltan los CSV o los artefactos de la Fase 3, que no viajan con el repo",
)
sin_cache = pytest.mark.skipif(
    not (ruta("processed_data") / "folds" / NOMBRE_MANIFIESTO).exists(),
    reason="la caché de folds no está construida (construir_folds())",
)


@sin_datos
def test_reajustado_sobre_el_80_entero_reproduce_x_train():
    """Parada 3: el pipeline ajustado con todo train sobre la población de `poblacion_train()` da
    exactamente la matriz persistida. Con la ocupación sin semilla, esta columna no coincidía."""
    X, y, _ = poblacion_train()

    matriz = construir_pipeline().fit_transform(X, y)

    persistida = pd.read_parquet(ruta("processed_data") / "X_train.parquet")
    pd.testing.assert_frame_equal(matriz, persistida)


@sin_datos
@sin_cache
def test_la_cache_real_cubre_train_con_la_prevalencia_de_train():
    """Parada 1 contra la caché real: disjuntos, solo train y la tasa de default en cada parte."""
    split = cargar_split()
    train = split[split["split"].eq("train")]
    informe = informe_folds(split)

    assert len(informe) == len(SEMILLAS) * N_FOLDS
    assert informe["disjuntos"].all() and informe["mismas_columnas"].all()
    assert (informe["n_ajuste"] + informe["n_eval"] == len(train)).all()
    for parte in ("pct_ajuste", "pct_eval"):
        assert (informe[parte] - train["TARGET"].mean() * 100).abs().max() < 0.01
