"""evaluar_en_folds(), la única forma de puntuar un estimador, y los dos suelos de referencia.

Se puntúa siempre sobre los folds cacheados del 0.4: el estimador se ajusta con `X_ajuste`, y la
parte de evaluación solo predice. Las guardas del catálogo C del plan revientan con `ValueError` y
no con `assert`, para que sigan vivas con `python -O`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import _exigir_train
from src.features.split import cargar_split, mascara
from src.models.folds import Fold
from src.models.metricas import Resumen, calibracion, curva_estrategia, ranking

COLUMNAS_EXT_SOURCE = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
SEMILLA_SUELO = 0


@dataclass(frozen=True, eq=False)
class Evaluacion:
    """`por_fold` lleva una fila por fold y `oof` la PD fuera de fold, una columna por semilla."""

    por_fold: pd.DataFrame
    oof: pd.DataFrame


def _proba_impago(modelo: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """La columna de la clase 1, o revienta si `classes_` no es `[0, 1]`.

    Con `[:, 0]` el AUC sale a 1 menos el real y nadie lo ve sin esta guarda.
    """
    if list(modelo.classes_) != [0, 1]:
        raise ValueError(f"classes_ tiene que ser [0, 1] y es {list(modelo.classes_)}")
    return np.asarray(modelo.predict_proba(X)[:, 1])


def _clientes_de_train(split: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Index]:
    """El split (el persistido si no se da) y los clientes de train, ordenados."""
    split = cargar_split() if split is None else split
    return split, pd.Index(split.loc[mascara(split, "train"), "SK_ID_CURR"]).sort_values()


def _exigir_particion_de_train(fold: Fold, split: pd.DataFrame, esperados: pd.Index) -> None:
    """Ajuste y evaluación son disjuntos y suman exactamente train, o revienta.

    Camino rápido por ordenación (9 ms frente a 59 ms de `_exigir_train()` por fold, que importa
    en el tuning); si falla, `_exigir_train()` dice quién sobra o falta.
    """
    ids = fold.X_ajuste.index.append(fold.X_eval.index)
    if ids.sort_values().equals(esperados):
        return
    _exigir_train(pd.Series(ids.unique()), split)
    raise ValueError(f"fold s{fold.semilla} k{fold.k}: ajuste y evaluación comparten clientes")


def evaluar_en_folds(
    estimador: BaseEstimator, folds: Iterable[Fold], split: pd.DataFrame | None = None
) -> Evaluacion:
    """Ajusta un clon del estimador en cada fold y puntúa su parte de evaluación.

    Revienta si ajuste y evaluación de un fold no son disjuntos y suman exactamente train (contra
    `split`, el persistido por defecto: un fold con clientes de valid o repetidos no es de aquí),
    si las columnas no coinciden, si `classes_` no es `[0, 1]`, si el AUC de un fold no pasa de 0,5
    (columna o signo al revés) o si un cliente sale dos veces en el OOF de una misma semilla. El
    score de ajuste solo alimenta el gap.
    """
    split, esperados = _clientes_de_train(split)
    filas = []
    oof: dict[int, list[pd.Series]] = {}
    for fold in folds:
        _exigir_particion_de_train(fold, split, esperados)
        if not fold.X_eval.columns.equals(fold.X_ajuste.columns):
            raise ValueError(f"fold s{fold.semilla} k{fold.k}: columnas de ajuste y evaluación")
        modelo = clone(estimador)
        t0 = time.perf_counter()
        modelo.fit(fold.X_ajuste, fold.y_ajuste)
        t1 = time.perf_counter()
        prob = _proba_impago(modelo, fold.X_eval)
        t2 = time.perf_counter()
        metricas = ranking(fold.y_eval, prob)
        if metricas["auc"] <= 0.5:
            raise ValueError(
                f"fold s{fold.semilla} k{fold.k}: AUC {metricas['auc']:.4f}, no pasa de 0,5"
            )
        auc_ajuste = ranking(fold.y_ajuste, _proba_impago(modelo, fold.X_ajuste))["auc"]
        filas.append(
            {
                "semilla": fold.semilla,
                "k": fold.k,
                "n_eval": len(fold.X_eval),
                **metricas,
                "auc_ajuste": auc_ajuste,
                "gap": auc_ajuste - metricas["auc"],
                "t_ajuste": t1 - t0,
                "t_prediccion": t2 - t1,
            }
        )
        oof.setdefault(fold.semilla, []).append(pd.Series(prob, index=fold.X_eval.index))
    columnas = {}
    for semilla, partes in oof.items():
        serie = pd.concat(partes)
        if serie.index.has_duplicates:
            raise ValueError(f"semilla {semilla}: un cliente sale dos veces fuera de fold")
        columnas[semilla] = serie
    return Evaluacion(pd.DataFrame(filas), pd.DataFrame(columnas).sort_index())


def resumir(
    evaluacion: Evaluacion, contexto: pd.DataFrame, split: pd.DataFrame | None = None
) -> Resumen:
    """Lo que lee el criterio de elección, promediado entre las semillas.

    `auc_cv` y `gap` son la media de los folds. La curva de estrategia, el AUC sin historial y la
    calibración se miden sobre el OOF completo de cada semilla, que es de lo que hay una PD por
    cliente, y se promedian. `contexto` es el de `cargar_contexto()`; revienta si no es el de train
    (contra `split`, el persistido por defecto) o si el OOF no cubre exactamente a sus clientes.
    """
    split = cargar_split() if split is None else split
    _exigir_train(pd.Series(contexto.index), split)
    oof = evaluacion.oof
    if not oof.index.equals(contexto.index.sort_values()) or oof.isna().any().any():
        raise ValueError("el OOF no cubre exactamente a los clientes del contexto")
    contexto = contexto.loc[oof.index]
    y, importe = contexto["TARGET"], contexto["AMT_CREDIT"]
    sin_historial = contexto["HAS_BUREAU_HISTORY"].eq(0).to_numpy()
    curvas, aucs, cals = [], [], []
    for semilla in oof.columns:
        prob = oof[semilla]
        curvas.append(curva_estrategia(y, prob, importe))
        aucs.append(ranking(y[sin_historial], prob[sin_historial])["auc"])
        cals.append(calibracion(y, prob))
    por_fold = evaluacion.por_fold
    curva = pd.concat(curvas).groupby("tasa", sort=True, as_index=False).mean()
    return Resumen(
        auc_cv=float(por_fold["auc"].mean()),
        curva=curva,
        auc_sin_historial=float(np.mean(aucs)),
        pendiente=float(np.mean([c["pendiente"] for c in cals])),
        ordenada=float(np.mean([c["ordenada"] for c in cals])),
        gap=float(por_fold["gap"].mean()),
    )


def suelo_ext_source() -> Pipeline:
    """Logística de las tres `EXT_SOURCE`: el listón sin feature engineering.

    Va sin pesos de clase a propósito, como excepción declarada a la regla del desbalance: es el
    listón sin ninguna técnica, que la decide el 0.7, y así su PD sale calibrada de forma natural.
    """
    return Pipeline(
        [
            ("columnas", ColumnTransformer([("ext", "passthrough", list(COLUMNAS_EXT_SOURCE))])),
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA_SUELO)),
        ]
    )


def suelo_dummy(folds: Iterable[Fold]) -> pd.DataFrame:
    """Brier y log loss de la PD constante (la prevalencia del ajuste), una fila por fold.

    Fuera de `evaluar_en_folds()`, que rechaza un score constante: su AUC es 0,5 y su curva de
    estrategia es plana en la prevalencia por construcción, así que solo se mide la calibración.
    """
    filas = []
    for fold in folds:
        modelo = DummyClassifier(strategy="prior").fit(fold.X_ajuste, fold.y_ajuste)
        prob = _proba_impago(modelo, fold.X_eval)
        filas.append(
            {
                "semilla": fold.semilla,
                "k": fold.k,
                "pd": float(prob[0]),
                "brier": brier_score_loss(fold.y_eval, prob),
                "log_loss": log_loss(fold.y_eval, prob, labels=[0, 1]),
            }
        )
    return pd.DataFrame(filas)
