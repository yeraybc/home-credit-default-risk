"""Folds estratificados con el pipeline de features reajustado por fold, cacheados con huella.

El CV no puede hacerse sobre `X_train.parquet`: sale de un `fit_transform` sobre el 80% entero, así
que la mediana, el winsorizador, el WoE y el `SelectorIV` ya han visto la parte de evaluación de
cada fold. Aquí cada fold ajusta su propio pipeline con su parte de ajuste y solo transforma la de
evaluación. Lo que no se reajusta por fold son los ocho cortes de `cortes.json`, medidos sobre el
80% entero: es el abierto (a) del 4.3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import RAIZ, cargar_config, ruta
from src.features.build_features import (
    NOMBRE_FICHERO_CORTES,
    _exigir_train,
    cargar_auxiliares,
    cargar_cortes,
    ensamblar_auxiliares,
    huella_split,
    matriz_de_features,
    preparar_application,
)
from src.features.pipeline import construir_pipeline
from src.features.selection import N_FOLDS_ESTABILIDAD, SEMILLAS_ESTABILIDAD
from src.features.split import cargar_split, solo_train

logger = logging.getLogger(__name__)

# las mismas semillas y folds con los que la Fase 3 midió la inestabilidad de las colas; el tuning
# usa la primera y el benchmark informa aparte de las otras dos, que el tuning no vio
SEMILLAS = SEMILLAS_ESTABILIDAD
N_FOLDS = N_FOLDS_ESTABILIDAD
SEMILLA_TUNING = SEMILLAS[0]

NOMBRE_DIRECTORIO = "folds"
NOMBRE_MANIFIESTO = "manifiesto.json"
NOMBRE_CONTEXTO = "contexto.parquet"
# lo que `metricas.py` necesita de cada cliente y que la matriz no trae crudo: el importe sin
# winsorizar y la presencia de historial de buró
COLUMNAS_CONTEXTO = ("AMT_CREDIT", "HAS_BUREAU_HISTORY")

# features de identidad con las que se busca a la misma persona con dos solicitudes; la segunda
# lectura es la invariante a la fecha de solicitud, porque todos los DAYS_* cuentan desde ella
LECTURAS_IDENTIDAD = {
    "exacta": ("CODE_GENDER", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"),
    "invariante": ("CODE_GENDER", "EDAD_A_ID", "EDAD_A_REGISTRO"),
}


@dataclass(frozen=True, eq=False)
class Fold:
    """Un fold: el pipeline se ajustó con `X_ajuste` y `X_eval` solo se transformó."""

    semilla: int
    k: int
    X_ajuste: pd.DataFrame
    y_ajuste: pd.Series
    X_eval: pd.DataFrame
    y_eval: pd.Series


def particionar(
    X: pd.DataFrame,
    y: pd.Series,
    fabrica: Callable[[], Pipeline] = construir_pipeline,
    semillas: tuple[int, ...] = SEMILLAS,
) -> Iterator[Fold]:
    """Los folds de cada semilla con el pipeline ajustado dentro de cada uno.

    La matriz de ajuste sale de `fit_transform`, no de `fit` y `transform`: la codificación
    cruzada de la ocupación solo es honesta así (`ajustar_pipeline()` lo explica). `fabrica` existe
    para probar con un espía sin montar el esquema entero.
    """
    for semilla in semillas:
        kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=semilla)
        for k, (i_ajuste, i_eval) in enumerate(kfold.split(X, y)):
            pipeline = clone(fabrica())
            X_ajuste = pipeline.fit_transform(X.iloc[i_ajuste], y.iloc[i_ajuste])
            X_eval = pipeline.transform(X.iloc[i_eval])
            yield Fold(semilla, k, X_ajuste, y.iloc[i_ajuste], X_eval, y.iloc[i_eval])


def poblacion_train() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """El 80% de entrenamiento con la capa 1 completa: `(X, y, contexto)`, indexados por cliente.

    `X` es lo que consumen las capas 2, sin identificador ni etiqueta. Los cortes se cargan siempre
    de `cortes.json` y con `sobrescribir=True`, que es lo que vuelve a comprobar su huella contra
    el split en cada llamada.
    """
    cfg = cargar_config()["dataset"]
    id_col, target_col = cfg["id_col"], cfg["target_col"]
    split = cargar_split()
    base = preparar_application()
    cargar_cortes(split, sobrescribir=True)
    ensamblada = ensamblar_auxiliares(base, *cargar_auxiliares())
    train = solo_train(ensamblada, split)
    _exigir_train(train[id_col], split)
    indice = pd.Index(train[id_col], name=id_col)
    X = matriz_de_features(train).set_axis(indice)
    y = train[target_col].set_axis(indice)
    contexto = train[list(COLUMNAS_CONTEXTO)].set_axis(indice)
    return X, y, contexto


def _sha256_ficheros(rutas: list[Path]) -> str:
    resumen = hashlib.sha256()
    for r in sorted(rutas):
        resumen.update(r.name.encode())
        resumen.update(r.read_bytes())
    return resumen.hexdigest()


def huella_folds(split: pd.DataFrame, cortes: Path | None = None) -> str:
    """Sha256 de todo lo que decide el contenido de la caché.

    El split, `cortes.json`, el código de `src/features/` y la configuración (que incluye las
    recetas), la versión de scikit-learn, las semillas, el número de folds y el `float32`. El
    pipeline no tiene huella propia, así que cualquier cambio en su código invalida la caché: es
    lo conservador. Los CSV no entran, y se asume que no cambian.
    """
    cortes = cortes or ruta("processed_data") / NOMBRE_FICHERO_CORTES
    fuentes = [*(RAIZ / "src" / "features").glob("*.py"), *(RAIZ / "config").glob("*.yaml")]
    partes = {
        "split": huella_split(split),
        "cortes": _sha256_ficheros([cortes]),
        "fuentes": _sha256_ficheros(fuentes),
        "sklearn": sklearn.__version__,
        "semillas": list(SEMILLAS),
        "folds": N_FOLDS,
        "dtype": "float32",
    }
    return hashlib.sha256(json.dumps(partes, sort_keys=True).encode()).hexdigest()


def _destino() -> Path:
    return ruta("processed_data") / NOMBRE_DIRECTORIO


def _nombre(semilla: int, k: int, parte: str) -> str:
    return f"s{semilla}_k{k}_{parte}.parquet"


def _exigir_libre(destino: Path, sobrescribir: bool) -> None:
    """No pisa una caché con manifiesto sin `sobrescribir=True`."""
    if (destino / NOMBRE_MANIFIESTO).exists() and not sobrescribir:
        raise FileExistsError(
            f"ya hay folds en {destino}. Reconstruirlos son 15 ajustes del pipeline; pasa "
            "sobrescribir=True si de verdad quieres reemplazarlos."
        )


def escribir_folds(
    X: pd.DataFrame,
    y: pd.Series,
    contexto: pd.DataFrame,
    huella: str,
    destino: Path,
    sobrescribir: bool = False,
    fabrica: Callable[[], Pipeline] = construir_pipeline,
) -> Path:
    """Construye los folds y los escribe en `float32`, con el manifiesto al final.

    El manifiesto va el último: una caché a medias no tiene manifiesto y `cargar_folds()` no la
    lee. No pisa una caché existente sin `sobrescribir=True`.
    """
    _exigir_libre(destino, sobrescribir)
    manifiesto = destino / NOMBRE_MANIFIESTO
    destino.mkdir(parents=True, exist_ok=True)
    manifiesto.unlink(missing_ok=True)
    contexto.assign(TARGET=y).to_parquet(destino / NOMBRE_CONTEXTO)
    registro = []
    for fold in particionar(X, y, fabrica):
        for parte, matriz in (("ajuste", fold.X_ajuste), ("eval", fold.X_eval)):
            matriz.astype("float32").to_parquet(destino / _nombre(fold.semilla, fold.k, parte))
        registro.append({"semilla": fold.semilla, "k": fold.k})
        logger.info("fold s%s k%s escrito", fold.semilla, fold.k)
    bytes_disco = sum(f.stat().st_size for f in destino.glob("*.parquet"))
    manifiesto.write_text(
        json.dumps({"huella": huella, "folds": registro, "bytes": bytes_disco}, indent=2)
    )
    return destino


def construir_folds(destino: Path | None = None, sobrescribir: bool = False) -> Path:
    """De los CSV a la caché de folds en `data/processed/folds/`.

    Comprueba que no pisa nada antes de leer ningún CSV.
    """
    ruta_destino = destino or _destino()
    _exigir_libre(ruta_destino, sobrescribir)
    X, y, contexto = poblacion_train()
    huella = huella_folds(cargar_split())
    return escribir_folds(X, y, contexto, huella, ruta_destino, sobrescribir)


def _leer_manifiesto(origen: Path, huella: str | None) -> dict:
    """El manifiesto, si la huella de la caché casa con la actual; si no, revienta."""
    ruta_manifiesto = origen / NOMBRE_MANIFIESTO
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(
            f"no hay folds en {origen}. Constrúyelos con construir_folds(), una sola vez."
        )
    manifiesto: dict = json.loads(ruta_manifiesto.read_text())
    actual = huella or huella_folds(cargar_split())
    if manifiesto["huella"] != actual:
        raise ValueError(
            f"la huella de {origen} no casa con la actual: cambió el split, los cortes, el "
            "código de features o la configuración desde que se construyó, y esos folds ya no "
            "son los de este pipeline. Reconstrúyelos con sobrescribir=True."
        )
    return manifiesto


def cargar_contexto(origen: Path | None = None, huella: str | None = None) -> pd.DataFrame:
    """`TARGET`, `AMT_CREDIT` crudo y `HAS_BUREAU_HISTORY` de cada cliente de train."""
    ruta_origen = origen or _destino()
    _leer_manifiesto(ruta_origen, huella)
    return pd.read_parquet(ruta_origen / NOMBRE_CONTEXTO)


def cargar_folds(
    semillas: tuple[int, ...] = SEMILLAS, origen: Path | None = None, huella: str | None = None
) -> Iterator[Fold]:
    """Los folds cacheados de uno en uno, para no tener los 15 en memoria a la vez.

    Revienta si la huella no casa, como `cargar_cortes()`. Las matrices vuelven en `float32`.
    """
    ruta_origen = origen or _destino()
    manifiesto = _leer_manifiesto(ruta_origen, huella)
    y = pd.read_parquet(ruta_origen / NOMBRE_CONTEXTO)["TARGET"]
    for entrada in manifiesto["folds"]:
        if entrada["semilla"] not in semillas:
            continue
        s, k = entrada["semilla"], entrada["k"]
        X_ajuste = pd.read_parquet(ruta_origen / _nombre(s, k, "ajuste"))
        X_eval = pd.read_parquet(ruta_origen / _nombre(s, k, "eval"))
        yield Fold(s, k, X_ajuste, y.loc[X_ajuste.index], X_eval, y.loc[X_eval.index])


def informe_folds(
    split: pd.DataFrame, origen: Path | None = None, huella: str | None = None
) -> pd.DataFrame:
    """Puerta del 0.4: una fila por fold con lo que tiene que cumplir.

    Filas y prevalencia de cada parte, si ajuste y evaluación son disjuntos y suman exactamente
    train, y el número de columnas con las que faltan frente a la unión de todos los folds (el
    `SelectorIV` puede decidir distinto en cada uno). Revienta si algún cliente no es de train.
    """
    filas = []
    columnas = {}
    for fold in cargar_folds(origen=origen, huella=huella):
        ids = fold.X_ajuste.index.append(fold.X_eval.index)
        _exigir_train(pd.Series(ids), split)
        columnas[(fold.semilla, fold.k)] = set(fold.X_ajuste.columns)
        filas.append(
            {
                "semilla": fold.semilla,
                "k": fold.k,
                "n_ajuste": len(fold.X_ajuste),
                "n_eval": len(fold.X_eval),
                "pct_ajuste": fold.y_ajuste.mean() * 100,
                "pct_eval": fold.y_eval.mean() * 100,
                "disjuntos": fold.X_ajuste.index.intersection(fold.X_eval.index).empty,
                "n_columnas": fold.X_ajuste.shape[1],
                "mismas_columnas": set(fold.X_eval.columns) == columnas[(fold.semilla, fold.k)],
            }
        )
    union = set().union(*columnas.values())
    informe = pd.DataFrame(filas)
    clave = zip(informe["semilla"], informe["k"])
    informe["faltan"] = [sorted(union - columnas[c]) for c in clave]
    return informe


def informe_identidad(base: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Busca a la misma persona con dos `SK_ID_CURR` en las features de identidad.

    Una fila por lectura (`LECTURAS_IDENTIDAD`): clientes en un grupo de coincidencia, grupos que
    cruzan train y valid, clientes de valid con una coincidencia en train y grupos que quedan solo
    en train, que es lo que afecta al CV. Solo informa: si aparecen, se declara como limitación.
    """
    id_col = cargar_config()["dataset"]["id_col"]
    datos = base.assign(
        EDAD_A_ID=base["DAYS_BIRTH"] - base["DAYS_ID_PUBLISH"],
        EDAD_A_REGISTRO=base["DAYS_BIRTH"] - base["DAYS_REGISTRATION"],
    ).merge(split[[id_col, "split"]], on=id_col)
    filas = []
    for lectura, claves in LECTURAS_IDENTIDAD.items():
        d = datos.dropna(subset=list(claves))
        grupo = d.groupby(list(claves), observed=True)["split"]
        en_grupo = d[grupo.transform("size").gt(1)].assign(
            grupo=lambda f: f.groupby(list(claves), observed=True).ngroup()
        )
        partes = en_grupo.groupby("grupo")["split"].agg(lambda s: set(s))
        cruzan = partes[partes.map(lambda p: p == {"train", "valid"})].index
        filas.append(
            {
                "lectura": lectura,
                "clientes_en_grupo": len(en_grupo),
                "grupos": en_grupo["grupo"].nunique(),
                "grupos_cruzan": len(cruzan),
                "valid_con_par_en_train": int(
                    en_grupo[en_grupo["grupo"].isin(cruzan)]["split"].eq("valid").sum()
                ),
                "grupos_solo_train": int(partes.map(lambda p: p == {"train"}).sum()),
            }
        )
    return pd.DataFrame(filas).set_index("lectura")
