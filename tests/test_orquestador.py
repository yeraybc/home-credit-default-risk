"""Tests del orquestador del bloque 5 (punto 5.9): `construir_artefactos()`, sus guardas de
sobrescribir y de "solo train", y el ensamblado de la reconciliación contra el EDA.

Las guardas y `informe_reconciliacion()` corren en CI, sobre datos sintéticos o con
`preparar_application`/`_medir_reconciliacion` sustituidos: no hace falta ningún CSV para
comprobar que una guarda revienta antes de leerlos. El resto necesita las tres tablas,
`application_train.csv` y el split, y se salta sin ellos.
"""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES
from src.features import build_features
from src.features.build_features import (
    ANCLAJES,
    _exigir_train,
    construir_artefactos,
    informe_reconciliacion,
)
from src.features.params import PARAMS
from src.features.split import NOMBRE_FICHERO
from src.features.transformers import _resolver_origen

# --- _exigir_train, en las dos direcciones ----------------------------------------------------

SPLIT_SINTETICO = pd.DataFrame(
    {"SK_ID_CURR": [1, 2, 3, 4], "split": ["train", "train", "valid", "valid"]}
)


def test_exigir_train_pasa_con_exactamente_los_ids_de_train():
    _exigir_train(pd.Series([2, 1]), SPLIT_SINTETICO)  # no revienta; el orden no importa


def test_exigir_train_revienta_con_un_cliente_de_valid_colado():
    with pytest.raises(ValueError, match="no es el de train"):
        _exigir_train(pd.Series([1, 3]), SPLIT_SINTETICO)


def test_exigir_train_revienta_si_falta_uno_de_train():
    with pytest.raises(ValueError, match="de train ausentes"):
        _exigir_train(pd.Series([1]), SPLIT_SINTETICO)


# --- la guarda de sobrescribir, en las dos direcciones y antes de tocar ningún dato ------------


class _LlegoACargarDatosError(AssertionError):
    """Marca que `construir_artefactos()` pasó de la guarda a cargar datos."""


def _preparar_application_centinela():
    raise _LlegoACargarDatosError("construir_artefactos() llegó a cargar datos")


def test_sobrescribir_false_revienta_antes_de_cargar_nada(tmp_path, monkeypatch):
    datos, modelos = tmp_path / "datos", tmp_path / "modelos"
    datos.mkdir()
    (datos / "X_train.parquet").write_bytes(b"lo que sea, no se lee")
    monkeypatch.setattr(build_features, "preparar_application", _preparar_application_centinela)

    with pytest.raises(FileExistsError, match="X_train"):
        construir_artefactos(destino_datos=datos, destino_modelos=modelos)


def test_sobrescribir_true_deja_pasar_a_cargar_los_datos(tmp_path, monkeypatch):
    datos, modelos = tmp_path / "datos", tmp_path / "modelos"
    datos.mkdir()
    (datos / "X_train.parquet").write_bytes(b"lo que sea, no se lee")
    monkeypatch.setattr(build_features, "preparar_application", _preparar_application_centinela)

    with pytest.raises(_LlegoACargarDatosError):
        construir_artefactos(destino_datos=datos, destino_modelos=modelos, sobrescribir=True)


def test_sin_nada_persistido_tambien_deja_pasar(tmp_path, monkeypatch):
    monkeypatch.setattr(build_features, "preparar_application", _preparar_application_centinela)

    with pytest.raises(_LlegoACargarDatosError):
        construir_artefactos(destino_datos=tmp_path / "datos", destino_modelos=tmp_path / "modelos")


# --- informe_reconciliacion ensambla ANCLAJES contra lo medido, sin construir un Pipeline ------


def test_informe_reconciliacion_marca_cuadra_falso_cuando_una_medida_no_coincide(monkeypatch):
    referencia = {medida: esperada for _, medida, _, esperada, _ in ANCLAJES}
    medida_rota = ANCLAJES[0][1]
    falseado = {**referencia, medida_rota: -1}
    monkeypatch.setattr(build_features, "_medir_reconciliacion", lambda *a, **k: falseado)

    informe = informe_reconciliacion(None, None, None, None).set_index("medida")

    assert not informe.loc[medida_rota, "cuadra"]
    assert informe["cuadra"].drop(medida_rota).all()


def test_informe_reconciliacion_reproduce_todas_las_esperadas_cuando_coinciden(monkeypatch):
    referencia = {medida: esperada for _, medida, _, esperada, _ in ANCLAJES}
    monkeypatch.setattr(build_features, "_medir_reconciliacion", lambda *a, **k: referencia)

    informe = informe_reconciliacion(None, None, None, None)

    assert informe["cuadra"].all()


def test_informe_reconciliacion_revienta_si_falta_una_medida(monkeypatch):
    incompleto = {medida: esperada for _, medida, _, esperada, _ in ANCLAJES}
    del incompleto[ANCLAJES[0][1]]
    monkeypatch.setattr(build_features, "_medir_reconciliacion", lambda *a, **k: incompleto)

    with pytest.raises(KeyError):
        informe_reconciliacion(None, None, None, None)


# --- la puerta contra el dato real, el orquestador entero ---------------------------------------

sin_dato_real = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["application_train"]).exists()
    or not (ruta("raw_data") / TABLE_FILES["bureau"]).exists()
    or not (ruta("raw_data") / TABLE_FILES["bureau_balance"]).exists()
    or not (ruta("raw_data") / TABLE_FILES["previous_application"]).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)


@pytest.fixture(scope="module")
def artefactos_reales(tmp_path_factory):
    """Corre `construir_artefactos(refijar=True)` entero, hacia un directorio de scratch.

    Snapshot y restauración de `PARAMS` a la escala del módulo, como `train_ensamblado` en
    `test_iv.py`: los ocho cortes que refija aquí no pueden quedarse fijados para el resto de la
    sesión. El split que lee es siempre el real de `data/processed/`, nunca el de scratch:
    `construir_artefactos()` no lo rehace.
    """
    snapshot = dict(PARAMS)
    destino = tmp_path_factory.mktemp("artefactos_5_9")
    rutas, reconciliacion = construir_artefactos(
        refijar=True, destino_datos=destino / "datos", destino_modelos=destino / "modelos"
    )
    yield rutas, reconciliacion
    PARAMS.clear()
    PARAMS.update(snapshot)


@sin_dato_real
def test_la_reconciliacion_cuadra_entera(artefactos_reales):
    _, reconciliacion = artefactos_reales
    # antes del .all(), que sobre una tabla vacia seria vacuamente cierto si ANCLAJES se quedara
    # sin filas por accidente
    assert len(reconciliacion) == len(ANCLAJES)
    assert reconciliacion["cuadra"].all(), reconciliacion.loc[~reconciliacion["cuadra"]].to_string()


@sin_dato_real
def test_las_matrices_salen_con_las_cifras_de_la_puerta(artefactos_reales):
    rutas, _ = artefactos_reales
    X_train = pd.read_parquet(rutas["X_train"])
    X_valid = pd.read_parquet(rutas["X_valid"])
    y_train = pd.read_parquet(rutas["y_train"])
    y_valid = pd.read_parquet(rutas["y_valid"])

    assert X_train.shape == (245_993, 164)
    assert X_valid.shape == (61_499, 164)
    assert X_train.isna().sum().sum() == 0
    assert X_valid.isna().sum().sum() == 0
    assert list(X_train.columns) == list(X_valid.columns)
    assert X_train.index.name == "SK_ID_CURR"
    assert X_train.index.equals(y_train.index)
    assert X_valid.index.equals(y_valid.index)
    assert not set(X_train.index) & set(X_valid.index)
    assert y_train["TARGET"].mean() * 100 == pytest.approx(8.0734, abs=1e-4)
    assert y_valid["TARGET"].mean() * 100 == pytest.approx(8.0733, abs=1e-4)


@sin_dato_real
def test_los_cortes_refijados_coinciden_con_los_ya_persistidos(artefactos_reales):
    rutas, _ = artefactos_reales
    nuevo = json.loads(rutas["cortes"].read_text())
    viejo = json.loads((ruta("processed_data") / "cortes.json").read_text())
    assert nuevo == viejo


@sin_dato_real
def test_el_registro_de_seleccion_reproduce_exactamente_la_matriz_final(artefactos_reales):
    rutas, _ = artefactos_reales
    registro = pd.read_csv(rutas["seleccion"], index_col=0)
    X_train = pd.read_parquet(rutas["X_train"])

    assert len(registro) == 180
    columnas_que_quedan = set()
    for origen in registro[registro["queda"]].index:
        resuelto = _resolver_origen(origen, X_train.columns)
        assert resuelto is not None, f"{origen} queda pero no resuelve en la matriz final"
        columnas_que_quedan.update(resuelto)
    assert columnas_que_quedan == set(X_train.columns)


# El script que un proceso nuevo correría para servir a un cliente: nada de lo que ya está en
# memoria del fixture, solo lo persistido (`cortes.json` y el joblib) más los CSV de siempre. Es
# el camino de la API y no una reimplementación: usa `ensamblar_auxiliares()` y `cargar_cortes()`
# tal cual, con el pipeline ya ajustado.
_SCRIPT_PROCESO_NUEVO = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import joblib
    from src.data.loader import load_table
    from src.features.build_features import (
        cargar_cortes,
        ensamblar_auxiliares,
        matriz_de_features,
        preparar_application,
    )
    from src.features.split import cargar_split, solo_valid

    ruta_cortes, ruta_pipeline, ruta_salida = sys.argv[1], sys.argv[2], sys.argv[3]
    pipeline = joblib.load(ruta_pipeline)
    base = preparar_application()
    split = cargar_split()
    cargar_cortes(split, Path(ruta_cortes))
    bureau = load_table("bureau", reduce_memory=False)
    bb = load_table("bureau_balance")
    prev = load_table("previous_application", reduce_memory=False)
    ensamblada = ensamblar_auxiliares(base, bureau, bb, prev)
    valid = solo_valid(ensamblada, split)
    matriz = pipeline.transform(matriz_de_features(valid))
    matriz.set_axis(valid["SK_ID_CURR"]).to_parquet(ruta_salida)
    """
)


@sin_dato_real
def test_el_pipeline_serializado_transforma_valid_igual_en_un_proceso_nuevo(
    artefactos_reales, tmp_path
):
    import subprocess
    import sys

    rutas, _ = artefactos_reales
    guion = tmp_path / "transformar_valid.py"
    guion.write_text(_SCRIPT_PROCESO_NUEVO)
    salida = tmp_path / "x_valid_proceso_nuevo.parquet"

    subprocess.run(
        [sys.executable, str(guion), str(rutas["cortes"]), str(rutas["pipeline"]), str(salida)],
        check=True,
    )

    x_valid_original = pd.read_parquet(rutas["X_valid"])
    x_valid_proceso_nuevo = pd.read_parquet(salida)
    pd.testing.assert_frame_equal(x_valid_proceso_nuevo, x_valid_original)
