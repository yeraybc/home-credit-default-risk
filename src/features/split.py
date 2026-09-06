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


def _destino() -> Path:
    """Ruta del fichero de split."""
    return ruta("processed_data") / NOMBRE_FICHERO


def construir_split(
    test_size: float | None = None,
    random_state: int | None = None,
    destino: Path | None = None,
    persistir: bool = True,
    app: pd.DataFrame | None = None,
    sobrescribir: bool = False,
) -> pd.DataFrame:
    """Split estratificado por TARGET sobre los SK_ID_CURR de application_train.

    Sin argumentos toma `test_size` y `random_state` de config.yaml y lee la tabla del disco;
    `app` permite inyectar el frame de identificador y objetivo, que es lo que usan los tests.
    Si ya hay un split persistido no lo pisa: hay que pedirlo con `sobrescribir=True`.
    """
    cfg = cargar_config()["dataset"]
    test_size = cfg["test_size"] if test_size is None else test_size
    random_state = cfg["random_state"] if random_state is None else random_state
    id_col, target_col = cfg["id_col"], cfg["target_col"]

    # se comprueba antes de leer los 166MB del csv: si va a fallar, que falle barato
    ruta_destino = destino or _destino()
    if persistir and ruta_destino.exists() and not sobrescribir:
        raise FileExistsError(
            f"ya hay un split en {ruta_destino}. Rehacerlo cambia la partición y deja "
            "inválido en silencio todo lo ajustado sobre ella: medianas, percentiles, WoE, "
            "IV y los cortes refijados. Léelo con cargar_split(), o pasa sobrescribir=True "
            "si de verdad quieres una partición nueva."
        )

    if app is None:
        app = load_table("application_train", reduce_memory=False, usecols=[id_col, target_col])
    else:
        faltan = {id_col, target_col} - set(app.columns)
        if faltan:
            raise ValueError(f"al frame inyectado le faltan columnas: {sorted(faltan)}")
        app = app[[id_col, target_col]].copy()
    if not app[id_col].is_unique:
        raise ValueError(f"{id_col} trae duplicados: la partición sería ambigua")

    ids_train, ids_valid = train_test_split(
        app[id_col],
        test_size=test_size,
        random_state=random_state,
        stratify=app[target_col],
    )

    split = app.assign(split="train")
    split.loc[split[id_col].isin(ids_valid), "split"] = "valid"
    split = split[COLUMNAS].sort_values(id_col).reset_index(drop=True)

    if persistir:
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


PARTES = ("train", "valid")


def mascara(split: pd.DataFrame, parte: str) -> pd.Series:
    """Máscara booleana de una de las dos partes, alineada al frame de split."""
    if parte not in PARTES:
        raise ValueError(f"parte desconocida: {parte!r}. válidas: {PARTES}")
    if "split" not in split.columns:
        raise ValueError(
            f"el frame no tiene columna 'split'; columnas: {sorted(split.columns)[:10]}"
        )
    return split["split"].eq(parte)


def filtrar(df: pd.DataFrame, parte: str, split: pd.DataFrame | None = None) -> pd.DataFrame:
    """Se queda con las filas de una parte, buscando SK_ID_CURR en columna o en el índice.

    Es el accesor que usan los consumidores del pipeline, para no tener que acordarse de
    aplicar la máscara a mano cada vez y arriesgarse a ajustar algo sobre el conjunto entero.
    """
    # repetida y no delegada en mascara(): falla antes de pagar el cargar_split() de abajo
    if parte not in PARTES:
        raise ValueError(f"parte desconocida: {parte!r}. válidas: {PARTES}")
    split = cargar_split() if split is None else split
    id_col = cargar_config()["dataset"]["id_col"]
    ids = set(split.loc[mascara(split, parte), id_col])

    if id_col in df.columns:
        seleccion = df[id_col].isin(ids).to_numpy()
    elif df.index.name == id_col:
        seleccion = df.index.isin(ids)
    else:
        raise ValueError(
            f"no encuentro {id_col} ni en las columnas ni en el índice del frame a filtrar"
        )
    return df.loc[seleccion]


def solo_train(df: pd.DataFrame, split: pd.DataFrame | None = None) -> pd.DataFrame:
    """Las filas de entrenamiento. Todo lo que estime un parámetro se ajusta sobre esto."""
    return filtrar(df, "train", split)


def solo_valid(df: pd.DataFrame, split: pd.DataFrame | None = None) -> pd.DataFrame:
    """Las filas de validación. Solo se transforman, nunca se ajusta nada sobre ellas."""
    return filtrar(df, "valid", split)


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
