"""Nadeau y Bengio, DeLong, bootstrap, McNemar y Bonferroni.

Funciones puras sobre arrays, como `metricas.py`: no ajustan nada ni saben de dónde sale la
predicción. La hipótesis nula es siempre que no hay diferencia, y todos los tests son bilaterales;
la dirección de la mejora ya la exige `cumple_criterio()` con `DELTA_AUC_MIN`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from src.models.metricas import ORDEN_SIMPLICIDAD, _validar

# réplicas del bootstrap, fijadas el 2026-10-01 antes de medir
N_BOOTSTRAP = 2000


def nadeau_bengio(dif: ArrayLike, ratio_test_train: float = 1 / 4) -> dict[str, float]:
    """t pareado corregido (Nadeau y Bengio, en la versión repetida de Bouckaert y Frank).

    `dif` es la diferencia de la métrica por fold (complejo menos simple), alineada por fold. Los
    folds comparten entrenamiento, así que la varianza se infla con `ratio_test_train`
    (1/4 con 5 folds); con 0 es el t pareado de siempre.
    """
    d = np.asarray(dif, dtype=float)
    if d.ndim != 1 or d.size < 2 or not np.isfinite(d).all():
        raise ValueError("dif tiene que ser un vector finito de al menos dos folds")
    n, media, var = d.size, d.mean(), d.var(ddof=1)
    if var == 0:  # todos los folds iguales: diferencia segura si no es cero
        t = np.copysign(np.inf, media) if media else 0.0
    else:
        t = media / np.sqrt((1 / n + ratio_test_train) * var)
    p = 2 * stats.t.sf(abs(t), n - 1)
    return {"t": float(t), "p": float(p), "media": float(media), "n": n}


def _componentes(y: np.ndarray, score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC y componentes estructurales de DeLong (por impago y por no impago), con rangos medios."""
    pos, neg = score[y == 1], score[y == 0]
    m, n = pos.size, neg.size
    r_todos = stats.rankdata(score)
    r_pos, r_neg = stats.rankdata(pos), stats.rankdata(neg)
    auc = r_todos[y == 1].sum() / (m * n) - (m + 1) / (2 * n)
    v01 = (r_todos[y == 1] - r_pos) / n
    v10 = 1 - (r_todos[y == 0] - r_neg) / m
    return float(auc), v01, v10


def delong(y: ArrayLike, score_a: ArrayLike, score_b: ArrayLike) -> dict[str, float]:
    """Test de DeLong para la diferencia de AUC entre dos scores del mismo conjunto.

    Algoritmo rápido de Sun y Xu (2014). `dif` es `auc_a - auc_b` y `ee` su error estándar.
    """
    y, score_a = _validar(y, score_a)
    _, score_b = _validar(y, score_b)
    auc_a, a01, a10 = _componentes(y, score_a)
    auc_b, b01, b10 = _componentes(y, score_b)
    m, n = a01.size, a10.size
    cov = np.cov([a01, b01]) / m + np.cov([a10, b10]) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    dif = auc_a - auc_b
    if var <= 0:  # sin variación en la diferencia (mismo ranking): o es cero o es segura
        igual = abs(dif) < 1e-12
        z, p, ee = (0.0 if igual else np.copysign(np.inf, dif)), float(igual), 0.0
    else:
        ee = float(np.sqrt(var))
        z = dif / ee
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "dif": dif, "ee": ee, "z": float(z), "p": p}


def bootstrap_estratificado(
    y: ArrayLike,
    estadistico: Callable[[np.ndarray], ArrayLike],
    n_rep: int = N_BOOTSTRAP,
    nivel: float = 0.95,
    semilla: int = 42,
) -> dict[str, np.ndarray]:
    """Intervalo por percentiles de `estadistico(idx)`, remuestreando cada clase por separado.

    Cada réplica conserva la prevalencia exacta. `estadistico` recibe los índices remuestreados y
    devuelve un escalar o un vector (la curva de estrategia entera); si calcula la diferencia de
    dos modelos sobre el mismo `idx`, el intervalo es el pareado.
    """
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all() or np.unique(y).size < 2:
        raise ValueError("y tiene que ser 0 y 1, con las dos clases presentes")
    if n_rep < 2 or not 0 < nivel < 1:
        raise ValueError(f"n_rep {n_rep} o nivel {nivel} no válidos")
    rng = np.random.default_rng(semilla)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    replicas = np.array(
        [
            estadistico(np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]))
            for _ in range(n_rep)
        ],
        dtype=float,
    )
    cola = 100 * (1 - nivel) / 2
    inferior, superior = np.percentile(replicas, [cola, 100 - cola], axis=0)
    return {"inferior": inferior, "superior": superior}


def _decisiones(
    y: ArrayLike, aprueba_a: ArrayLike, aprueba_b: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`y` y las dos decisiones (True es aprobar) como arrays, o revienta si no son 0 y 1."""
    y = np.asarray(y)
    decisiones = [np.asarray(d) for d in (aprueba_a, aprueba_b)]
    if not np.isin(y, (0, 1)).all() or any(d.shape != y.shape for d in decisiones):
        raise ValueError("y tiene que ser 0 y 1, y las decisiones de su misma longitud")
    if any(not np.isin(d, (0, 1)).all() for d in decisiones):
        raise ValueError("las decisiones tienen que ser booleanas (aprobar o rechazar)")
    return y, decisiones[0] == 1, decisiones[1] == 1


def mcnemar(y: ArrayLike, aprueba_a: ArrayLike, aprueba_b: ArrayLike) -> dict[str, float]:
    """McNemar exacto sobre el acierto de las decisiones de aprobar o rechazar de dos modelos.

    Acierta quien aprueba a un no impago o rechaza a un impago. Se mide el acierto y no la
    decisión cruda: con la misma tasa de aprobación las dos tienen el mismo marginal por
    construcción. `b` son los clientes que solo acierta A y `c` los que solo acierta B.

    Con las dos decisiones cortadas a la misma tasa de aprobación el test es conservador (sus
    discordantes no son independientes: aprueban el mismo número), así que nunca rechaza de más
    pero pierde potencia. Con umbrales distintos, que dan tasas distintas, es el McNemar de siempre.
    """
    y, a, b = _decisiones(y, aprueba_a, aprueba_b)
    acierta_a, acierta_b = (a == (y == 0)), (b == (y == 0))
    b, c = int((acierta_a & ~acierta_b).sum()), int((~acierta_a & acierta_b).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p": p}


def mcnemar_aprobados(y: ArrayLike, aprueba_a: ArrayLike, aprueba_b: ArrayLike) -> dict[str, float]:
    """Entre los clientes que aprueba solo uno de los dos, ¿qué modelo deja pasar más impagos?

    Binomial exacto sobre los impagos de cada grupo, con probabilidad nula igual al peso del grupo
    de A. Mide la mora entre aprobados y es válido con tasas de aprobación iguales o distintas,
    donde `mcnemar()` es conservador. `impagos_a` son los impagos que solo aprueba A. Si las
    decisiones de un modelo contienen las del otro no hay grupo con el que contrastar y da p = 1.
    """
    y, a, b = _decisiones(y, aprueba_a, aprueba_b)
    solo_a, solo_b = a & ~b, b & ~a
    n_a, n_b = int(solo_a.sum()), int(solo_b.sum())
    imp_a, imp_b = int(y[solo_a].sum()), int(y[solo_b].sum())
    if imp_a + imp_b == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(imp_a, imp_a + imp_b, n_a / (n_a + n_b)).pvalue)
    return {"solo_a": n_a, "solo_b": n_b, "impagos_a": imp_a, "impagos_b": imp_b, "p": p}


def pares(modelos: list[str]) -> list[tuple[str, str]]:
    """Los pares `(simple, complejo)` por `ORDEN_SIMPLICIDAD`, que son las claves de `elegir()`."""
    desconocidos = set(modelos) - set(ORDEN_SIMPLICIDAD)
    if desconocidos or len(set(modelos)) != len(modelos):
        raise ValueError(f"los modelos tienen que ser distintos y de {ORDEN_SIMPLICIDAD}")
    ordenados = [m for m in ORDEN_SIMPLICIDAD if m in modelos]
    return [(s, c) for i, s in enumerate(ordenados) for c in ordenados[i + 1 :]]


def bonferroni(p_valores: dict[tuple[str, str], float], n_contrastes: int) -> dict:
    """Cada p multiplicado por el tamaño de la familia y recortado a 1, el que lee `elegir()`.

    La familia son los contrastes emitidos de un mismo test (un par por modelo con tres modelos);
    `n_contrastes` es lo declarado y revienta si no coincide con lo emitido, de más o de menos.
    """
    if len(p_valores) != n_contrastes or n_contrastes < 1:
        raise ValueError(f"se declaran {n_contrastes} contrastes y se han emitido {len(p_valores)}")
    return {par: min(1.0, p * n_contrastes) for par, p in p_valores.items()}
