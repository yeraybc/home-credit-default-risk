"""Espacios de búsqueda y estrategias de desbalance, con su procedencia.

El 0.7 mide qué técnica de entrenamiento gana en cada modelo sin tunear. El remuestreo y los pesos
van dentro de un `imblearn.pipeline.Pipeline`, que solo remuestrea en `fit`: la parte de evaluación
de cada fold y la API nunca ven un cliente sintético. `construir_estimador()` es la única fábrica,
la que reutilizan el tuning y el reentreno final.
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import ruta
from src.models.contrastes import bonferroni, nadeau_bengio
from src.models.evaluar import evaluar_en_folds, resumir
from src.models.folds import SEMILLA_TUNING, Fold, cargar_contexto, cargar_folds
from src.models.metricas import ALFA, ORDEN_SIMPLICIDAD, _zona, curva_estrategia
from src.models.tiempos import sanear_nombres

# de más a menos preferida a igualdad, por coste y por no inventar datos; es también el orden en
# que entran las finalistas cuando ninguna adelanta a los pesos (criterio escrito antes de medir)
PREFERENCIA = ("pesos", "nada", "submuestreo", "balanced_rf", "smotenc", "smotenc_tomek")
SINTETICAS = ("smotenc", "smotenc_tomek")
REFERENCIA = "pesos"
# Borderline-SMOTE queda fuera: imbalanced-learn no tiene versión NC y crearía valores imposibles
# en las binarias y one-hot. balanced_rf solo existe para el bosque, que equilibra por árbol
ESTRATEGIAS = {
    "logistica": tuple(e for e in PREFERENCIA if e != "balanced_rf"),
    "random_forest": PREFERENCIA,
    "lightgbm": tuple(e for e in PREFERENCIA if e != "balanced_rf"),
}
TASAS_TABLA = (0.70, 0.80, 0.90)
NOMBRE_TABLA = "comparacion_desbalance.csv"
NOMBRE_POR_FOLD = "comparacion_desbalance_por_fold.csv"
NOMBRE_DECISION = "decision_desbalance.json"

# decidido con visto bueno el 2026-10-03 por `finalistas()` sobre los 5 folds de la semilla 0 y los
# tres modelos sin tunear; la evidencia está en `models/decision_desbalance.json`. Entran a la
# búsqueda como hiperparámetro categórico. Logística y LightGBM empatan entre `pesos` y `nada`
# (`nada` sale calibrada), y el bosque se queda con `balanced_rf` y `submuestreo`. Las sintéticas
# quedan fuera en los tres con p corregido por debajo de 0,002 (AUC -0,03 a -0,09)
FINALISTAS = {
    "logistica": ("pesos", "nada"),
    "random_forest": ("balanced_rf", "submuestreo"),
    "lightgbm": ("pesos", "nada"),
}

if set(ESTRATEGIAS) != set(ORDEN_SIMPLICIDAD):
    raise RuntimeError("ESTRATEGIAS tiene que cubrir exactamente los modelos de ORDEN_SIMPLICIDAD")


@contextmanager
def avisos_imblearn() -> Generator[None, None, None]:
    """Calla solo los avisos de imbalanced-learn 0.12.4 con scikit-learn 1.6.1.

    Son los tags (`DeprecationWarning`) y cuatro usos de la API privada de scikit-learn que ya
    estaba deprecada (`FutureWarning`). Se corrigen subiendo imbalanced-learn, que pide Python 3.10.
    """
    with warnings.catch_warnings():
        for texto in (".*_get_tags.*", ".*_more_tags.*"):
            warnings.filterwarnings("ignore", message=texto, category=DeprecationWarning)
        for texto in (
            ".*force_all_finite.*",
            ".*BaseEstimator._check_feature_names.*",
            ".*BaseEstimator._check_n_features.*",
            ".*BaseEstimator._validate_data.*",
        ):
            warnings.filterwarnings("ignore", message=texto, category=FutureWarning)
        yield


class SMOTENCBinarias(SMOTENC):
    """SMOTENC que declara categóricas las columnas con dos valores o menos en lo que ve en `fit`.

    Es una regla estructural, sin mirar el TARGET, y vale igual con 162, 163 o 165 columnas por
    fold y antes o después del escalado. Un SMOTE simple interpolaría esas columnas y crearía
    clientes con un 0,37 de "es pensionista".
    """

    def __init__(
        self,
        *,
        sampling_strategy: str = "auto",
        random_state: int | None = None,
        k_neighbors: int = 5,
    ) -> None:
        super().__init__(
            categorical_features=[],
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    def _validate_column_types(self, X: pd.DataFrame | np.ndarray) -> None:
        binarias = (pd.DataFrame(X).nunique() <= 2).to_numpy()
        self.categorical_features_ = np.flatnonzero(binarias)
        self.continuous_features_ = np.flatnonzero(~binarias)


def _remuestreo(estrategia: str) -> BaseEstimator | None:
    """El sampler de la estrategia, o `None` si no remuestrea. Siempre a 1:1 (`auto`)."""
    if estrategia == "submuestreo":
        return RandomUnderSampler(random_state=SEMILLA_TUNING)
    if estrategia == "smotenc":
        return SMOTENCBinarias(random_state=SEMILLA_TUNING)
    if estrategia == "smotenc_tomek":
        return SMOTETomek(
            smote=SMOTENCBinarias(random_state=SEMILLA_TUNING),
            tomek=TomekLinks(n_jobs=-1),
            random_state=SEMILLA_TUNING,
        )
    return None


def _modelo(modelo: str, estrategia: str) -> BaseEstimator:
    """El estimador por defecto con su semilla. `pesos` es `class_weight` en los tres."""
    pesos = "balanced" if estrategia == "pesos" else None
    if modelo == "logistica":
        return LogisticRegression(max_iter=1000, random_state=SEMILLA_TUNING, class_weight=pesos)
    if modelo == "lightgbm":
        return LGBMClassifier(random_state=SEMILLA_TUNING, verbose=-1, class_weight=pesos)
    if estrategia == "balanced_rf":
        # los tres parámetros fijan el comportamiento de la 0.13 y evitan su FutureWarning
        return BalancedRandomForestClassifier(
            random_state=SEMILLA_TUNING,
            n_jobs=-1,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
        )
    return RandomForestClassifier(random_state=SEMILLA_TUNING, n_jobs=-1, class_weight=pesos)


def construir_estimador(modelo: str, estrategia: str, memoria: str | None = None) -> Pipeline:
    """`[escalado] + [remuestreo] + [nombres] + modelo`, la única fábrica de estimadores.

    Sampler y pesos son excluyentes por construcción (nunca los dos a la vez). El escalado va en
    la logística y antes de SMOTE, que mide distancias, también en los árboles, y devuelve un frame
    para que LightGBM vea los mismos nombres al ajustar y al predecir. LightGBM sanea los nombres
    con coma justo antes del modelo, así escalado y sampler reciben lo mismo en los tres modelos.
    `memoria` es el directorio de caché de joblib de imblearn: con él el remuestreo, que cuesta
    minutos por fold, se hace una vez y lo comparten los tres modelos.
    """
    if estrategia not in ESTRATEGIAS.get(modelo, ()):
        raise ValueError(f"{estrategia!r} no es una estrategia de {modelo!r}")
    pasos: list[tuple[str, BaseEstimator]] = []
    if modelo == "logistica" or estrategia in SINTETICAS:
        pasos.append(("escalado", StandardScaler().set_output(transform="pandas")))
    sampler = _remuestreo(estrategia)
    if sampler is not None:
        pasos.append(("remuestreo", sampler))
    if modelo == "lightgbm":
        pasos.append(("nombres", FunctionTransformer(sanear_nombres)))
    pasos.append(("modelo", _modelo(modelo, estrategia)))
    return Pipeline(pasos, memory=memoria)


def filas_tras_remuestrear(estimador: Pipeline, fold: Fold) -> tuple[int, float]:
    """Filas y prevalencia de la parte de ajuste del fold tras el remuestreo del estimador."""
    if "remuestreo" not in estimador.named_steps:
        return len(fold.y_ajuste), float(fold.y_ajuste.mean())
    with avisos_imblearn():
        hasta = [n for n, _ in estimador.steps].index("remuestreo") + 1
        previos = Pipeline(estimador.steps[:hasta], memory=estimador.memory)
        _, y = previos.fit_resample(fold.X_ajuste, fold.y_ajuste)
    return len(y), float(np.mean(y))


def miembros_de_folds(folds: Iterable[Fold]) -> dict[tuple[int, int], pd.Index]:
    """Los clientes de la parte de evaluación de cada fold, para medir la mora fold a fold."""
    return {(f.semilla, f.k): f.X_eval.index for f in folds}


def medir(
    modelo: str,
    estrategia: str,
    fabrica: Callable[[], Iterable[Fold]],
    contexto: pd.DataFrame,
    miembros: dict[tuple[int, int], pd.Index],
    memoria: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Una celda de la comparación: la fila de resumen y la tabla por fold (AUC y mora de zona)."""
    estimador = construir_estimador(modelo, estrategia, memoria)
    with avisos_imblearn():
        evaluacion = evaluar_en_folds(estimador, fabrica())
    resumen = resumir(evaluacion, contexto)
    por_fold, oof = evaluacion.por_fold, evaluacion.oof
    y, importe = contexto["TARGET"], contexto["AMT_CREDIT"]
    filas = []
    for _, f in por_fold.iterrows():
        idx = miembros[(int(f["semilla"]), int(f["k"]))]
        curva = curva_estrategia(y.loc[idx], oof.loc[idx, int(f["semilla"])], importe.loc[idx])
        filas.append(
            {
                "modelo": modelo,
                "estrategia": estrategia,
                "semilla": int(f["semilla"]),
                "k": int(f["k"]),
                "auc": f["auc"],
                "mora_zona": float(_zona(curva)["mora"].mean()),
            }
        )
    curva_resumen = resumen.curva.set_index("tasa")
    n_filas, prevalencia = filas_tras_remuestrear(estimador, next(iter(fabrica())))
    fila = {
        "modelo": modelo,
        "estrategia": estrategia,
        "auc_media": resumen.auc_cv,
        "auc_std": float(por_fold["auc"].std()),
        "pr_auc": float(por_fold["pr_auc"].mean()),
        "ks": float(por_fold["ks"].mean()),
        "gap": resumen.gap,
        "auc_sin_historial": resumen.auc_sin_historial,
        "brier": float(np.mean([brier_score_loss(y, oof[s]) for s in oof.columns])),
        "pendiente": resumen.pendiente,
        "ordenada": resumen.ordenada,
        "t_ajuste": float(por_fold["t_ajuste"].mean()),
        "filas_ajuste": n_filas,
        "prevalencia_ajuste": prevalencia,
    }
    for tasa in TASAS_TABLA:
        sufijo = f"{round(tasa * 100)}"
        fila[f"mora_{sufijo}"] = float(curva_resumen.loc[tasa, "mora"])
        fila[f"mora_importe_{sufijo}"] = float(curva_resumen.loc[tasa, "mora_importe"])
    return fila, pd.DataFrame(filas)


def contrastar(por_fold: pd.DataFrame) -> pd.DataFrame:
    """Cada estrategia frente a `pesos`, por modelo, con Nadeau y Bengio y Bonferroni.

    Dos métricas por fold, las dos con el signo de "mejora" positivo: el AUC (estrategia menos
    pesos) y la mora media de la zona de operación (pesos menos estrategia). La familia es un
    modelo y una métrica, y revienta si falta alguna estrategia. `p_*` ya va corregido.
    """
    filas = []
    for modelo, grupo in por_fold.groupby("modelo", sort=False):
        celdas = {
            e: g.set_index(["semilla", "k"]).sort_index() for e, g in grupo.groupby("estrategia")
        }
        if REFERENCIA not in celdas:
            raise ValueError(f"{modelo}: falta la referencia {REFERENCIA!r}")
        base = celdas[REFERENCIA]
        pruebas = {}
        for e, c in celdas.items():
            if e == REFERENCIA:
                continue
            if not c.index.equals(base.index):
                raise ValueError(f"{modelo}/{e}: los folds no son los de la referencia")
            pruebas[e] = {
                "auc": nadeau_bengio((c["auc"] - base["auc"]).to_numpy()),
                "mora": nadeau_bengio((base["mora_zona"] - c["mora_zona"]).to_numpy()),
            }
        n = len(ESTRATEGIAS[modelo]) - 1
        p_auc = bonferroni({(e, REFERENCIA): t["auc"]["p"] for e, t in pruebas.items()}, n)
        p_mora = bonferroni({(e, REFERENCIA): t["mora"]["p"] for e, t in pruebas.items()}, n)
        for e, t in pruebas.items():
            filas.append(
                {
                    "modelo": modelo,
                    "estrategia": e,
                    "media_auc": t["auc"]["media"],
                    "p_auc": p_auc[(e, REFERENCIA)],
                    "media_mora": t["mora"]["media"],
                    "p_mora": p_mora[(e, REFERENCIA)],
                }
            )
    return pd.DataFrame(filas)


def finalistas(tabla: pd.DataFrame) -> dict[str, list[str]]:
    """Las dos estrategias que entran a la búsqueda de cada modelo, por el criterio escrito.

    `tabla` lleva `auc_media` por celda y, salvo en `pesos`, `media_auc`, `p_auc`, `media_mora` y
    `p_mora` de `contrastar()`. Una estrategia adelanta a `pesos` si lo mejora con significación
    en el AUC o en la mora; la que lo empeora con significación en cualquiera de las dos queda
    fuera. Adelantan primero, por AUC medio; después `pesos` y el resto por `PREFERENCIA`.
    """
    salida = {}
    for modelo, grupo in tabla.groupby("modelo", sort=False):
        t = grupo.set_index("estrategia")
        if REFERENCIA not in t.index:
            raise ValueError(f"{modelo}: falta la referencia {REFERENCIA!r}")
        adelantan, resto = [], []
        for e in (e for e in PREFERENCIA if e in t.index and e != REFERENCIA):
            r = t.loc[e]
            signo = {
                m: (np.sign(r[f"media_{m}"]) if r[f"p_{m}"] < ALFA else 0) for m in ("auc", "mora")
            }
            if min(signo.values()) < 0:
                continue
            (adelantan if max(signo.values()) > 0 else resto).append(e)
        adelantan.sort(key=lambda e: -t.loc[e, "auc_media"])
        salida[modelo] = (adelantan + [REFERENCIA] + resto)[:2]
    return salida


def _acumular(nuevo: pd.DataFrame, destino: Path, claves: list[str]) -> pd.DataFrame:
    """Añade filas al CSV sustituyendo las de la misma celda, para medir una estrategia cada vez."""
    if destino.exists():
        previo = pd.read_csv(destino)
        celdas = nuevo[claves].drop_duplicates().apply(tuple, axis=1)
        previo = previo[~previo[claves].apply(tuple, axis=1).isin(celdas)]
        nuevo = pd.concat([previo, nuevo], ignore_index=True)
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(destino, index=False)
    return nuevo


def escribir_decision(
    resumen: pd.DataFrame, por_fold: pd.DataFrame, destino: Path | None = None
) -> dict[str, list[str]]:
    """`models/decision_desbalance.json`: tabla, contrastes y finalistas (viaja con el repo)."""
    contrastes = contrastar(por_fold)
    propuestas = finalistas(resumen.merge(contrastes, on=["modelo", "estrategia"], how="left"))
    destino = destino or ruta("models") / NOMBRE_DECISION
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido = {
        "criterio": "escrito antes de medir (plan.md, 0.7): contra pesos, ALFA tras Bonferroni",
        "folds": "5 de la semilla 0, modelos sin tunear",
        "finalistas": propuestas,
        "tabla": json.loads(resumen.to_json(orient="records")),
        "contrastes": json.loads(contrastes.to_json(orient="records")),
    }
    destino.write_text(json.dumps(contenido, indent=2))
    return propuestas


def main(argv: list[str] | None = None) -> None:
    """`python -m src.models.espacios [estrategia ...]`: mide las estrategias y acumula en models/.

    Sin argumentos mide las seis. Con la tabla completa imprime los contrastes y las finalistas
    propuestas, que solo pasan a `FINALISTAS` con visto bueno.
    """
    pedidas = (sys.argv[1:] if argv is None else argv) or list(PREFERENCIA)
    desconocidas = set(pedidas) - set(PREFERENCIA)
    if desconocidas:
        raise ValueError(f"estrategias desconocidas: {sorted(desconocidas)}")

    def fabrica() -> Iterable[Fold]:
        return cargar_folds(semillas=(SEMILLA_TUNING,))

    contexto = cargar_contexto()
    miembros = miembros_de_folds(fabrica())
    carpeta = ruta("models")
    claves = ["modelo", "estrategia"]
    for estrategia in pedidas:
        filas, por_fold = [], []
        # un caché por estrategia, borrado al terminar: no sobrevive a un cambio de código
        with tempfile.TemporaryDirectory() as memoria:
            for modelo in ORDEN_SIMPLICIDAD:
                if estrategia not in ESTRATEGIAS[modelo]:
                    continue
                fila, pf = medir(modelo, estrategia, fabrica, contexto, miembros, memoria)
                print(pd.Series(fila).to_string(), flush=True)
                filas.append(fila)
                por_fold.append(pf)
        resumen = _acumular(pd.DataFrame(filas), carpeta / NOMBRE_TABLA, claves)
        por_fold = _acumular(
            pd.concat(por_fold), carpeta / NOMBRE_POR_FOLD, claves + ["semilla", "k"]
        )
    hechas = set(zip(resumen["modelo"], resumen["estrategia"]))
    if all((m, e) in hechas for m in ESTRATEGIAS for e in ESTRATEGIAS[m]):
        print(escribir_decision(resumen, por_fold))


if __name__ == "__main__":
    main()
