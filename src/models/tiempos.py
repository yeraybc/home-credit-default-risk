"""Cronómetro reproducible y línea base de tiempos de entrenamiento.

La línea base es el proceso ingenuo, sin optimizar nada: el pipeline de features reajustado dentro
de cada trial, `float64` tal como sale, sin pruning ni early stopping y los estimadores por
defecto. Es contra lo que el 4.5 cifra la reducción de cada palanca, en el mismo hardware.
"""

from __future__ import annotations

import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import ruta
from src.features.pipeline import construir_pipeline
from src.models.evaluar import _proba_impago
from src.models.folds import N_FOLDS, SEMILLA_TUNING, particionar, poblacion_train
from src.models.metricas import ORDEN_SIMPLICIDAD, ranking

# tres repeticiones, mediana y dispersión, como pide el catálogo F del plan: una mejora de tiempo
# que es ruido del sistema no cuenta. Si (máx menos mín) / mediana supera el 10% la medida no se
# da por buena y se busca la causa (temperatura, un proceso en segundo plano)
REPETICIONES = 3
DISPERSION_MAX = 0.10
# provisional, hasta que el 1.5, el 2.2 y el 3.2 fijen los reales: la rejilla de la logística (7
# valores de C por 3 penalizaciones por 2 estrategias de desbalance), los "unos 40 trials" del 2.2
# y 100 para LightGBM. El 4.5 recalcula el total con los trials que se hicieron de verdad
PRESUPUESTO_PROVISIONAL = {"logistica": 42, "random_forest": 40, "lightgbm": 100}
# modelos en los que, además del fold 0, se mide una pasada por cada otro fold para comprobar que
# todos cuestan lo mismo; el bosque queda fuera por lo que tarda
MODELOS_TODOS_LOS_FOLDS = ("logistica", "lightgbm")
ETAPAS = ("features", "ajuste", "prediccion")
NOMBRE_FICHERO = "linea_base_tiempos.json"

# los caracteres de JSON que LightGBM rechaza en un nombre de columna; en la matriz solo aparece
# la coma, en NAME_TYPE_SUITE_Spouse, partner y WALLSMATERIAL_MODE_Stone, brick (0.4)
_CARACTERES_JSON = re.compile(r'[",:\[\]{}]')


def hardware() -> dict:
    """Chip, núcleos, memoria, sistema y versiones: lo que tiene que repetirse en el 4.5."""
    chip = platform.processor()
    if sys.platform == "darwin":
        sysctl = ["sysctl", "-n", "machdep.cpu.brand_string"]
        chip = subprocess.run(sysctl, capture_output=True, text=True).stdout.strip() or chip
    memoria = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "chip": chip,
        "nucleos": os.cpu_count(),
        "memoria_gb": round(memoria / 2**30, 1),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def sanear_nombres(X: pd.DataFrame) -> pd.DataFrame:
    """Cambia por `_` los caracteres que LightGBM rechaza, o revienta si dos nombres chocan."""
    saneado = X.rename(columns=lambda c: _CARACTERES_JSON.sub("_", c))
    if saneado.columns.has_duplicates:
        raise ValueError("el saneado de nombres deja dos columnas con el mismo nombre")
    return saneado


def modelos_ingenuos() -> dict[str, BaseEstimator]:
    """Los tres modelos por defecto, con la semilla del tuning y dos excepciones declaradas.

    La logística escala y sube `max_iter` a 1000, como el suelo del 0.5, porque con 100 no converge;
    LightGBM sanea los nombres con coma y calla su log. `n_jobs` queda por defecto.
    """
    return {
        "logistica": Pipeline(
            [
                ("escalado", StandardScaler()),
                ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA_TUNING)),
            ]
        ),
        "random_forest": RandomForestClassifier(random_state=SEMILLA_TUNING),
        "lightgbm": Pipeline(
            [
                ("nombres", FunctionTransformer(sanear_nombres)),
                ("modelo", LGBMClassifier(random_state=SEMILLA_TUNING, verbose=-1)),
            ]
        ),
    }



def _pico_mb() -> float:
    """Memoria residente máxima del proceso hasta ahora, en MB (bytes en macOS, KB en Linux)."""
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pico / 2**20 if sys.platform == "darwin" else pico / 2**10


def _nan_canonico(X: pd.DataFrame) -> pd.DataFrame:
    """Devuelve los NaN de las columnas object al objeto `np.nan` único, como están en memoria.

    El pickle con el que viajan al hijo los convierte en un objeto distinto cada uno, y en Python
    3.9 todos los NaN tienen hash 0: el `set` del `OneHotEncoder` se vuelve cuadrático y un fold
    pasa de segundos a más de diez minutos.
    """
    canonico = X.copy()
    for columna in canonico.select_dtypes("object"):
        valores = canonico[columna].to_numpy(copy=True)
        valores[pd.isna(valores)] = np.nan
        canonico[columna] = valores
    return canonico


def _medir(
    modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series, k: int, fabrica: Callable[[], Pipeline]
) -> dict:
    """Un fold ingenuo cronometrado por etapas; corre dentro del proceso hijo."""
    X_canonico = _nan_canonico(X)
    base = _pico_mb()
    marcas = [(time.perf_counter(), time.process_time())]
    fold = next(particionar(X_canonico, y, fabrica, semillas=(SEMILLA_TUNING,), ks=(k,)))
    marcas.append((time.perf_counter(), time.process_time()))
    mem_features = _pico_mb() - base
    modelo.fit(fold.X_ajuste, fold.y_ajuste)
    marcas.append((time.perf_counter(), time.process_time()))
    prob = _proba_impago(modelo, fold.X_eval)
    marcas.append((time.perf_counter(), time.process_time()))
    auc = ranking(fold.y_eval, prob)["auc"]
    if auc <= 0.5:
        raise ValueError(f"fold k{k}: AUC {auc:.4f}, no pasa de 0,5")
    fila = {"k": k, "n_ajuste": len(fold.X_ajuste), "n_columnas": fold.X_ajuste.shape[1]}
    for etapa, (antes, despues) in zip(
        ("features", "ajuste", "prediccion"), zip(marcas, marcas[1:])
    ):
        fila[f"pared_{etapa}"] = despues[0] - antes[0]
        fila[f"cpu_{etapa}"] = despues[1] - antes[1]
    return {**fila, "mem_features_mb": mem_features, "mem_pico_mb": _pico_mb() - base, "auc": auc}


def medir_fold(
    modelo: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 0,
    fabrica: Callable[[], Pipeline] = construir_pipeline,
) -> dict:
    """Pared, CPU y memoria pico de un fold ingenuo de la semilla del tuning, por etapas.

    Corre en un proceso hijo nuevo, así que cada medida arranca en frío y su memoria pico es solo
    suya: el incremento sobre la base del hijo con los datos ya recibidos. La CPU suma todos los
    hilos. El AUC comprueba que lo medido es un ajuste de verdad.
    """
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as hijo:
        return hijo.submit(_medir, clone(modelo), X, y, k, fabrica).result()


def resumir_tiempos(
    medidas: pd.DataFrame, presupuesto: dict[str, int] = PRESUPUESTO_PROVISIONAL
) -> pd.DataFrame:
    """Una fila por modelo con la mediana y la dispersión de las repeticiones del fold 0.

    `medidas` lleva una fila por medida (`modelo`, `k` y las columnas de `_medir()`). El trial es
    `N_FOLDS` veces la mediana del fold y el total es `presupuesto` veces el trial: ambos son una
    extrapolación, y la columna `extrapolado` lo dice. La dispersión es (máx menos mín) / mediana
    del tiempo de pared del fold.
    """
    filas = []
    for modelo in ORDEN_SIMPLICIDAD:
        m = medidas[(medidas["modelo"] == modelo) & (medidas["k"] == 0)]
        if m.empty:
            continue
        pared = m[[f"pared_{e}" for e in ETAPAS]].sum(axis=1)
        cpu = m[[f"cpu_{e}" for e in ETAPAS]].sum(axis=1)
        fila = {"modelo": modelo, "repeticiones": len(m)}
        for etapa in ETAPAS:
            fila[f"pared_{etapa}"] = m[f"pared_{etapa}"].median()
        fila |= {
            "pared_fold": pared.median(),
            "dispersion": (pared.max() - pared.min()) / pared.median(),
            "cpu_sobre_pared": cpu.median() / pared.median(),
            "mem_pico_mb": m["mem_pico_mb"].median(),
            "auc": m["auc"].median(),
            "trial_s": N_FOLDS * pared.median(),
            "total_s": presupuesto[modelo] * N_FOLDS * pared.median(),
            "trials": presupuesto[modelo],
            "extrapolado": True,
        }
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def homogeneidad_folds(medidas: pd.DataFrame) -> pd.DataFrame:
    """Tiempo de pared de cada fold medido una vez frente a la mediana del fold 0, por modelo.

    Sirve para comprobar que extrapolar el trial como 5 veces el fold 0 es razonable: cada
    `razon` tiene que quedar dentro de `1 +- DISPERSION_MAX`.
    """
    medidas = medidas.assign(pared=medidas[[f"pared_{e}" for e in ETAPAS]].sum(axis=1))
    mediana_k0 = medidas[medidas["k"] == 0].groupby("modelo")["pared"].median()
    otros = medidas[medidas["k"] != 0][["modelo", "k", "pared"]].copy()
    otros["razon"] = otros["pared"] / otros["modelo"].map(mediana_k0)
    otros["dentro"] = (otros["razon"] - 1).abs() <= DISPERSION_MAX
    return otros.reset_index(drop=True)


def linea_base(X: pd.DataFrame, y: pd.Series, repeticiones: int = REPETICIONES) -> pd.DataFrame:
    """Las medidas crudas: las repeticiones del fold 0 intercaladas entre modelos y una pasada
    por cada otro fold en `MODELOS_TODOS_LOS_FOLDS`.

    Intercalar (logística, bosque, LightGBM, logística, ...) evita que la deriva térmica del
    portátil cargue sobre un solo modelo.
    """
    modelos = modelos_ingenuos()
    pedidos = [(m, 0) for _ in range(repeticiones) for m in ORDEN_SIMPLICIDAD]
    pedidos += [(m, k) for m in MODELOS_TODOS_LOS_FOLDS for k in range(1, N_FOLDS)]
    filas = []
    for modelo, k in pedidos:
        filas.append({"modelo": modelo, **medir_fold(modelos[modelo], X, y, k)})
    return pd.DataFrame(filas)


def escribir_linea_base(
    medidas: pd.DataFrame, resumen: pd.DataFrame, destino: Path | None = None
) -> str:
    """`models/linea_base_tiempos.json`: hardware, criterio, medidas crudas y tabla, para el 4.5."""
    destino = destino or ruta("models") / NOMBRE_FICHERO
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido = {
        "hardware": hardware(),
        "criterio": {
            "repeticiones": REPETICIONES,
            "dispersion_max": DISPERSION_MAX,
            "semilla": SEMILLA_TUNING,
            "n_folds_por_trial": N_FOLDS,
            "presupuesto_provisional": PRESUPUESTO_PROVISIONAL,
            "nota": "trial y total son una extrapolacion de la mediana del fold 0",
        },
        "resumen": json.loads(resumen.reset_index().to_json(orient="records")),
        "medidas": json.loads(medidas.to_json(orient="records")),
    }
    destino.write_text(json.dumps(contenido, indent=2))
    return str(destino)


def main() -> None:
    """`python -m src.models.tiempos`: mide la línea base sobre train y la escribe en models/."""
    X, y, _ = poblacion_train()
    medidas = linea_base(X, y)
    resumen = resumir_tiempos(medidas)
    print(resumen.T.to_string())
    print(homogeneidad_folds(medidas).to_string())
    print(escribir_linea_base(medidas, resumen))


if __name__ == "__main__":
    main()
