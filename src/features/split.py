"""Partición de entrenamiento y validación, la capa 0 del pipeline de features.

Se hace lo primero de todo y sobre SK_ID_CURR, antes de cualquier limpieza, agregación o
transformación: todo lo que estime un parámetro a partir de datos se ajusta después y solo
sobre la parte de entrenamiento. Se persiste para que el notebook, los scripts y los tests
usen exactamente la misma partición.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cargar_config, ruta
from src.data.loader import load_table

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "split.parquet"
COLUMNAS = ["SK_ID_CURR", "TARGET", "split"]


def _destino(raiz: Path | None = None) -> Path:
    """Ruta del fichero de split."""
    return (raiz or ruta("processed_data")) / NOMBRE_FICHERO


def construir_split(
    test_size: float | None = None,
    random_state: int | None = None,
    destino: Path | None = None,
    persistir: bool = True,
    app: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Split estratificado por TARGET sobre los SK_ID_CURR de application_train.

    Sin argumentos toma `test_size` y `random_state` de config.yaml y lee la tabla del disco;
    `app` permite inyectar el frame de identificador y objetivo, que es lo que usan los tests.
    """
    cfg = cargar_config()["dataset"]
    test_size = cfg["test_size"] if test_size is None else test_size
    random_state = cfg["random_state"] if random_state is None else random_state
    id_col, target_col = cfg["id_col"], cfg["target_col"]

    if app is None:
        app = load_table("application_train", reduce_memory=False, usecols=[id_col, target_col])
    else:
        faltan = {id_col, target_col} - set(app.columns)
        assert not faltan, f"al frame inyectado le faltan columnas: {sorted(faltan)}"
        app = app[[id_col, target_col]].copy()
    assert app[id_col].is_unique, "application_train trae SK_ID_CURR duplicados"

    ids_train, ids_valid = train_test_split(
        app[id_col],
        test_size=test_size,
        random_state=random_state,
        stratify=app[target_col],
    )

    split = app.assign(split="train")
    split.loc[split[id_col].isin(set(ids_valid)), "split"] = "valid"
    split = split[COLUMNAS].sort_values(id_col).reset_index(drop=True)

    if persistir:
        ruta_destino = destino or _destino()
        ruta_destino.parent.mkdir(parents=True, exist_ok=True)
        split.to_parquet(ruta_destino, index=False)
        logger.info("split escrito en %s", ruta_destino)

    return split


def cargar_split(origen: Path | None = None) -> pd.DataFrame:
    """Lee el split persistido. Falla si no existe, en vez de rehacerlo con otra semilla."""
    ruta_origen = origen or _destino()
    if not ruta_origen.exists():
        raise FileNotFoundError(
            f"no hay split en {ruta_origen}. constrúyelo con construir_split() una sola vez: "
            "rehacerlo por accidente con otra semilla invalida todo lo ajustado sobre él"
        )
    return pd.read_parquet(ruta_origen)


def mascara(split: pd.DataFrame, parte: str) -> pd.Series:
    """Máscara booleana de una de las dos partes, alineada al frame de split."""
    if parte not in {"train", "valid"}:
        raise ValueError(f"parte desconocida: {parte!r}. válidas: 'train' y 'valid'")
    return split["split"].eq(parte)


def resumen_split(split: pd.DataFrame) -> pd.DataFrame:
    """Clientes, positivos y tasa de default de cada parte y del conjunto."""
    filas = []
    for etiqueta, sub in [
        ("conjunto", split),
        ("train", split[mascara(split, "train")]),
        ("valid", split[mascara(split, "valid")]),
    ]:
        filas.append(
            {
                "parte": etiqueta,
                "clientes": len(sub),
                "positivos": int(sub["TARGET"].sum()),
                "% default": round(sub["TARGET"].mean() * 100, 4),
            }
        )
    return pd.DataFrame(filas)
