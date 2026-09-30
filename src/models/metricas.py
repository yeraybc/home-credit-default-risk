"""Ranking, calibración, curva de estrategia en recuento e importe y segmentos.

Funciones puras sobre `(y, score)`: no ajustan nada ni saben de dónde sale la predicción. El score
es de riesgo, así que más alto es más probable que impague, y aprobar es quedarse con los de abajo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

# Fijados el 2026-09-30, antes de medir ningún modelo.
# fracción de más riesgo en la que se mide la captura de impagos
DECIL = 0.10
# tasas de aprobación en las que el banco opera, y las que recorre la curva de estrategia
ZONA_OPERACION = (0.70, 0.90)
TASAS_CURVA = tuple(round(t, 2) for t in np.arange(0.50, 0.951, 0.05))
# Criterio de elección: el modelo más complejo gana solo si cumple las cuatro condiciones frente al
# más simple. El gap entre train y CV se informa en `Resumen` y no decide.
DELTA_AUC_MIN = 0.005  # mejora mínima del AUC de CV
ALFA = 0.05  # sobre el p ya corregido por Bonferroni (0.3)
TOL_CURVA = 0.001  # 0,1 puntos de mora por tasa de la zona, en recuento y en importe
TOL_AUC_SIN_HISTORIAL = 0.005  # caída máxima del AUC en los clientes sin historial de buró
PENDIENTE_CALIBRACION = (0.9, 1.1)
ORDENADA_CALIBRACION_MAX = 0.1  # en logit
ORDEN_SIMPLICIDAD = ("logistica", "random_forest", "lightgbm")
# la probabilidad se recorta aquí antes del logit, que no está definido en 0 ni en 1
EPS = 1e-6


def _validar(y: ArrayLike, score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """`y` y `score` como arrays, o revienta si no son un objetivo binario y un score continuo.

    Un score con dos valores distintos es casi siempre la salida de `predict` en vez de la de
    `predict_proba`, y da un AUC que parece válido pero mide otra cosa.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError(f"y y score deben ser vectores iguales: {y.shape} frente a {score.shape}")
    if not np.isin(y, (0, 1)).all() or np.unique(y).size < 2:
        raise ValueError("y tiene que ser 0 y 1, con las dos clases presentes")
    if not np.isfinite(score).all():
        raise ValueError("score con NaN o infinitos")
    if np.unique(score).size < 3:
        raise ValueError("score con menos de tres valores distintos: parecen etiquetas")
    return y, score


def _suma_aprobada(score: np.ndarray, valores: np.ndarray, aprobados: ArrayLike) -> np.ndarray:
    """Suma de `valores` sobre los `aprobados` clientes de menor score, para cada tamaño pedido.

    El grupo empatado en el corte entra en fracción: es el valor esperado de desempatar al azar,
    y así el resultado no depende del orden de las filas.
    """
    orden = np.argsort(score, kind="stable")
    _, inicio, n_grupo = np.unique(score[orden], return_index=True, return_counts=True)
    suma_grupo = np.add.reduceat(valores[orden], inicio)
    n_acum = np.concatenate([[0], np.cumsum(n_grupo)])
    v_acum = np.concatenate([[0.0], np.cumsum(suma_grupo)])
    k = np.asarray(aprobados, dtype=float)
    g = np.clip(np.searchsorted(n_acum, k, side="right") - 1, 0, n_grupo.size - 1)
    return np.asarray(v_acum[g] + (k - n_acum[g]) / n_grupo[g] * suma_grupo[g])


def ranking(y: ArrayLike, score: ArrayLike) -> dict[str, float]:
    """AUC, Gini, KS, captura de impagos en el decil de más riesgo y PR-AUC como apoyo."""
    y, score = _validar(y, score)
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    impagos_aprobados = _suma_aprobada(score, y, (1 - DECIL) * y.size)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "captura_decil": float(1 - impagos_aprobados / y.sum()),
        "pr_auc": average_precision_score(y, score),
    }


def curva_estrategia(
    y: ArrayLike, score: ArrayLike, importe: ArrayLike, tasas: tuple[float, ...] = TASAS_CURVA
) -> pd.DataFrame:
    """Por tasa de aprobación, la mora entre aprobados en recuento y en importe.

    Se aprueba a los de menor score. La tasa en importe es el importe en impago entre el importe
    aprobado, con el `importe` crudo de capa 1 (no el winsorizado de la matriz). `reduccion_*`
    es el cambio relativo frente a aprobar a todos.
    """
    y, score = _validar(y, score)
    importe = np.asarray(importe, dtype=float)
    if importe.shape != y.shape or not np.isfinite(importe).all() or (importe < 0).any():
        raise ValueError("importe debe ser finito, no negativo y de la misma longitud que y")
    aprobados = np.array(tasas) * y.size
    n_impago = _suma_aprobada(score, y, aprobados)
    imp_aprobado = _suma_aprobada(score, importe, aprobados)
    imp_impago = _suma_aprobada(score, importe * y, aprobados)
    mora = n_impago / aprobados
    mora_imp = imp_impago / imp_aprobado
    return pd.DataFrame(
        {
            "tasa": tasas,
            "mora": mora,
            "mora_importe": mora_imp,
            "reduccion_mora": mora / y.mean() - 1,
            "reduccion_mora_importe": mora_imp / ((importe * y).sum() / importe.sum()) - 1,
        }
    )


def auc_parcial(y: ArrayLike, score: ArrayLike) -> float:
    """AUC estandarizado (McClish) hasta el FPR que deja la zona de operación.

    Con un 92% de buenos, el FPR es casi la tasa de rechazo, así que aprobar al menos el 70%
    corresponde a `max_fpr` de 0,30. Da 0,5 para un score al azar y 1 para el perfecto.
    """
    y, score = _validar(y, score)
    return float(roc_auc_score(y, score, max_fpr=1 - ZONA_OPERACION[0]))


def _logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, EPS, 1 - EPS)
    return np.asarray(np.log(p / (1 - p)))


def _validar_prob(y: ArrayLike, prob: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y, prob = _validar(y, prob)
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError("la probabilidad de impago tiene que estar entre 0 y 1")
    return y, prob


def calibracion(y: ArrayLike, prob: ArrayLike) -> dict[str, float]:
    """Brier, log loss de apoyo, pendiente y ordenada de calibración de la PD.

    - Pendiente: coeficiente de `logit(p)` en una logística de `y` sobre él. 1 es calibrada, menos
      de 1 es una PD sobreconfiada (demasiado extrema) y más de 1 es una PD demasiado tímida.
    - Ordenada (calibración en el grande): intercepto de una logística con `logit(p)` como offset.
      0 es que la PD media acierta; positiva, que la PD infravalora el impago.
    """
    y, prob = _validar_prob(y, prob)
    z = _logit(prob)
    pendiente = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit().params[1]
    ordenada = sm.GLM(y, np.ones_like(z), family=sm.families.Binomial(), offset=z).fit().params[0]
    return {
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, prob),
        "pendiente": float(pendiente),
        "ordenada": float(ordenada),
    }


def tabla_calibracion(y: ArrayLike, prob: ArrayLike, n_tramos: int = 10) -> pd.DataFrame:
    """PD media frente a tasa observada por tramos de igual tamaño de PD, para el gráfico."""
    y, prob = _validar_prob(y, prob)
    tramo = pd.qcut(pd.Series(prob).rank(method="first"), n_tramos, labels=False)
    tabla = pd.DataFrame({"y": y, "prob": prob}).groupby(tramo, observed=True).agg(
        n=("y", "size"), pd_media=("prob", "mean"), tasa_observada=("y", "mean")
    )
    return tabla.reset_index(drop=True)


def por_segmento(
    y: ArrayLike, prob: ArrayLike, importe: ArrayLike, segmentos: dict[str, ArrayLike]
) -> dict[str, dict]:
    """Ranking, curva de estrategia y calibración dentro de cada segmento (máscara booleana).

    El de sin historial de buró es `HAS_BUREAU_HISTORY == 0`. El mínimo de impagos por segmento
    para no leer ruido como hallazgo lo exige el 5.2, no esta función.
    """
    y, prob = _validar_prob(y, prob)
    importe = np.asarray(importe, dtype=float)
    salida = {}
    for nombre, mascara in segmentos.items():
        m = np.asarray(mascara, dtype=bool)
        if m.shape != y.shape:
            raise ValueError(f"la máscara de {nombre!r} no tiene la longitud de y")
        salida[nombre] = {
            "n": int(m.sum()),
            "impagos": int(y[m].sum()),
            "ranking": ranking(y[m], prob[m]),
            "curva": curva_estrategia(y[m], prob[m], importe[m]),
            "calibracion": calibracion(y[m], prob[m]),
        }
    return salida


@dataclass(frozen=True, eq=False)
class Resumen:
    """Lo que el criterio lee de un modelo, medido fuera de fold y con la PD ya calibrada.

    `curva` es la de `curva_estrategia()`; `gap` (AUC de train menos el de CV) solo se informa.
    """

    auc_cv: float
    curva: pd.DataFrame
    auc_sin_historial: float
    pendiente: float
    ordenada: float
    gap: float | None = None


def _zona(curva: pd.DataFrame) -> pd.DataFrame:
    """Las filas de la curva dentro de la zona de operación, por tasa."""
    lo, hi = ZONA_OPERACION
    en_zona = curva[curva["tasa"].between(lo - 1e-9, hi + 1e-9)]
    return en_zona.set_index("tasa")


def cumple_criterio(simple: Resumen, complejo: Resumen, p_auc: float) -> dict[str, bool]:
    """Las cuatro condiciones del complejo frente al simple, cada una por separado.

    - `auc`: mejora de al menos `DELTA_AUC_MIN` y `p_auc` (ya con Bonferroni) por debajo de `ALFA`.
    - `curva`: en cada tasa de la zona, la mora del complejo no supera a la del simple en más de
      `TOL_CURVA`, ni en recuento ni en importe.
    - `sin_historial`: el AUC sin historial de buró no baja más de `TOL_AUC_SIN_HISTORIAL`.
    - `calibracion`: pendiente y ordenada del complejo dentro de sus rangos.
    """
    zs, zc = _zona(simple.curva), _zona(complejo.curva)
    if zs.empty or not zs.index.equals(zc.index):
        raise ValueError("las curvas no cubren las mismas tasas de la zona de operación")
    peor = (zc[["mora", "mora_importe"]] - zs[["mora", "mora_importe"]]).to_numpy().max()
    eps = 1e-12  # un límite exacto cuenta como cumplido
    return {
        "auc": bool(complejo.auc_cv - simple.auc_cv >= DELTA_AUC_MIN - eps and p_auc < ALFA),
        "curva": bool(peor <= TOL_CURVA + eps),
        "sin_historial": bool(
            simple.auc_sin_historial - complejo.auc_sin_historial <= TOL_AUC_SIN_HISTORIAL + eps
        ),
        "calibracion": bool(
            PENDIENTE_CALIBRACION[0] <= complejo.pendiente <= PENDIENTE_CALIBRACION[1]
            and abs(complejo.ordenada) <= ORDENADA_CALIBRACION_MAX
        ),
    }


def elegir(resumenes: dict[str, Resumen], p_valores: dict[tuple[str, str], float]) -> str:
    """El modelo elegido: el más simple, salvo que uno más complejo cumpla las cuatro condiciones.

    Recorre `ORDEN_SIMPLICIDAD` con un campeón que empieza en la logística. Cada aspirante tiene
    que cumplir todo frente a la logística (el suelo, para que las tolerancias no se acumulen de
    un escalón a otro) y frente al campeón actual (para no desplazar a un intermedio que ya le
    iguala). `p_valores` lleva, por cada par `(simple, aspirante)` que se compare, el p de AUC
    ya corregido; si falta alguno revienta.
    """
    desconocidos = set(resumenes) - set(ORDEN_SIMPLICIDAD)
    suelo = ORDEN_SIMPLICIDAD[0]
    if desconocidos or suelo not in resumenes:
        raise ValueError(f"los modelos tienen que ser de {ORDEN_SIMPLICIDAD}, con la logística")
    campeon = suelo
    for aspirante in (m for m in ORDEN_SIMPLICIDAD[1:] if m in resumenes):
        rivales = dict.fromkeys((suelo, campeon))  # sin repetir cuando el campeón es el suelo
        veredictos = (
            cumple_criterio(resumenes[r], resumenes[aspirante], p_valores[(r, aspirante)])
            for r in rivales
        )
        if all(all(v.values()) for v in veredictos):
            campeon = aspirante
    return campeon
