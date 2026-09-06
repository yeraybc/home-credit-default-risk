"""Tests de la partición de entrenamiento y validación (src/features/split.py).

El split es la capa 0: si se rehace con otra semilla, todo lo ajustado sobre él queda inválido
sin que nada avise. Estos tests fijan sus cifras de anclaje y su reproducibilidad.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import cargar_config
from src.features.split import (
    COLUMNAS,
    cargar_split,
    construir_split,
    mascara,
    resumen_split,
)

# cifras del EDA que la partición tiene que reproducir
CLIENTES = 307_511
POSITIVOS = 24_825
TASA = 8.0729

RUTA_SPLIT = Path(__file__).resolve().parents[1] / "data" / "processed" / "split.parquet"
sin_split = pytest.mark.skipif(
    not RUTA_SPLIT.exists(), reason="no hay split persistido; construir_split() no se ha corrido"
)


@pytest.fixture(scope="module")
def split():
    return cargar_split()


@sin_split
def test_el_split_reproduce_las_cifras_del_eda(split):
    assert len(split) == CLIENTES
    assert int(split["TARGET"].sum()) == POSITIVOS
    assert round(split["TARGET"].mean() * 100, 4) == TASA


@sin_split
def test_estructura_y_unicidad(split):
    assert list(split.columns) == COLUMNAS
    assert split["SK_ID_CURR"].is_unique
    assert set(split["split"]) == {"train", "valid"}
    assert not split.isna().any().any()


@sin_split
def test_las_dos_partes_particionan_el_conjunto(split):
    train, valid = mascara(split, "train"), mascara(split, "valid")
    assert (train ^ valid).all(), "hay clientes en las dos partes o en ninguna"
    assert int(train.sum()) + int(valid.sum()) == CLIENTES


@sin_split
def test_la_proporcion_de_la_particion_es_la_declarada(split):
    esperado = cargar_config()["dataset"]["test_size"]
    assert round(mascara(split, "valid").mean(), 3) == esperado


@sin_split
def test_la_estratificacion_conserva_la_tasa_en_las_dos_partes(split):
    tasas = resumen_split(split).set_index("parte")["% default"]
    assert abs(tasas["train"] - tasas["valid"]) < 0.01, tasas.to_dict()
    assert abs(tasas["conjunto"] - TASA) < 1e-9


@sin_split
def test_ningun_cliente_de_validacion_esta_en_entrenamiento(split):
    ids_train = set(split.loc[mascara(split, "train"), "SK_ID_CURR"])
    ids_valid = set(split.loc[mascara(split, "valid"), "SK_ID_CURR"])
    assert not (ids_train & ids_valid)


# --- lógica de la partición, sobre datos sintéticos ----------------------------------------
# Los CSV de data/raw no viajan con el repo, así que la lógica se verifica con un frame
# inyectado y las cifras reales quedan en los tests de arriba, que se saltan si no hay split.


@pytest.fixture
def clientes():
    """2.000 clientes con la misma tasa de default que la tabla real."""
    n, positivos = 2_000, 161  # 8,05%
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100_000, 100_000 + n),
            "TARGET": [1] * positivos + [0] * (n - positivos),
        }
    )


def test_la_misma_semilla_da_la_misma_particion(tmp_path, clientes):
    a = construir_split(destino=tmp_path / "a.parquet", app=clientes)
    b = construir_split(destino=tmp_path / "b.parquet", app=clientes)
    assert a.equals(b)


def test_otra_semilla_da_otra_particion(tmp_path, clientes):
    a = construir_split(destino=tmp_path / "a.parquet", app=clientes)
    b = construir_split(random_state=7, destino=tmp_path / "b.parquet", app=clientes)
    assert not a["split"].equals(b["split"])
    # pero la estratificación aguanta con cualquier semilla
    tasa = clientes["TARGET"].mean() * 100
    assert abs(b.loc[b["split"].eq("valid"), "TARGET"].mean() * 100 - tasa) < 0.5


def test_la_particion_sintetica_es_exhaustiva_y_estratificada(tmp_path, clientes):
    s = construir_split(destino=tmp_path / "s.parquet", app=clientes)
    assert len(s) == len(clientes)
    assert s["SK_ID_CURR"].is_unique
    assert round(s["split"].eq("valid").mean(), 3) == cargar_config()["dataset"]["test_size"]
    tasas = resumen_split(s).set_index("parte")["% default"]
    assert abs(tasas["train"] - tasas["valid"]) < 0.5, tasas.to_dict()


def test_ida_y_vuelta_por_parquet(tmp_path, clientes):
    destino = tmp_path / "s.parquet"
    escrito = construir_split(destino=destino, app=clientes)
    assert pd.read_parquet(destino).equals(escrito)


def test_un_frame_inyectado_sin_las_columnas_falla(tmp_path):
    with pytest.raises(AssertionError, match="faltan columnas"):
        construir_split(destino=tmp_path / "s.parquet", app=pd.DataFrame({"SK_ID_CURR": [1, 2, 3]}))


def test_identificadores_duplicados_fallan(tmp_path):
    duplicados = pd.DataFrame({"SK_ID_CURR": [1, 1, 2, 3], "TARGET": [0, 1, 0, 1]})
    with pytest.raises(AssertionError, match="duplicados"):
        construir_split(destino=tmp_path / "s.parquet", app=duplicados)


def test_cargar_un_split_que_no_existe_falla_con_instrucciones(tmp_path):
    with pytest.raises(FileNotFoundError, match="constrúyelo"):
        cargar_split(origen=tmp_path / "no_existe.parquet")


def test_parte_desconocida_revienta():
    df = pd.DataFrame({"split": ["train", "valid"]})
    with pytest.raises(ValueError, match="parte desconocida"):
        mascara(df, "test")
