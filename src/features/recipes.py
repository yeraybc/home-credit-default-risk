"""Exportación de la receta de features de una tabla auxiliar a config/<tabla>_features.yaml.

La receta se genera desde el ranking consolidado del notebook, así que no puede
desincronizarse de las decisiones; lleva la decisión y su justificación, no cómo se construye
cada feature, que es código y vive en el pipeline de FE.

El esquema es el mismo en las tres tablas (auditoría transversal de septiembre de 2026), que
antes divergían en cuatro cosas: `estado` era texto libre sin vocabulario cerrado, la
distinción firme/provisional no estaba en ningún campo pese a decidir qué se remide sobre el
split, `efecto` no declaraba unidad ni n, y las claves eran expresiones (`X > 0`, `X | historial`)
en dos ficheros e identificadores limpios en el tercero.

Uso:
    exportar_receta("bureau", "SK_ID_CURR", nota, rank_flags, rank_cont,
                    alfa_bonferroni=alpha_bonf, firmes={"BUREAU_MAX_DAYS_OVERDUE"})
"""

from __future__ import annotations  # PEP 604 en anotaciones: el entorno es Python 3.9

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# vocabulario cerrado de decisiones, para que el pipeline pueda ramificar sin parsear prosa
DECISIONES = ("conservar", "iv", "degradada", "descartar", "control", "referencia")

# unidad de `efecto` según el tipo: las dos no son comparables entre sí
UNIDADES = {"flag": "pp", "continua": "rank_biserial", "control": None}

_ESQUEMA = {
    "tipo": "flag (efecto en pp) o continua (efecto en rank-biserial)",
    "codificacion": "cómo entra la feature cuando no es la columna cruda ('> 0', '| historial')",
    "poblacion_medicion": (
        "sobre qué clientes se midió; los efectos NO son comparables entre poblaciones distintas"
    ),
    "n": "tamaño de la población de medición (solo continuas)",
    "n_marcados": "clientes con la bandera a 1 (solo flags)",
    "efecto": "tamaño del efecto, en la unidad que declara `unidad`",
    "unidad": "pp para banderas, rank_biserial para continuas",
    "p": "p-valor del contraste (z-test en banderas, Mann-Whitney en continuas)",
    "significativa_bonferroni": "si p supera el alfa corregido que declara `metodologia`",
    "decision": " | ".join(DECISIONES),
    "firmeza": (
        "firme = target-independiente (redundancia estructural), no hay que rehacerlo sobre el "
        "split; provisional = decidido contra el TARGET sobre train completo, se reconfirma"
    ),
    "control": (
        "true si la feature se queda como control pase lo que pase con su capacidad predictiva"
    ),
    "estado": "la cadena original del ranking, verbatim, con el matiz que no cabe en `decision`",
}


def clasificar_estado(estado: str) -> str:
    """Traduce la cadena de estado del ranking al vocabulario cerrado de `DECISIONES`."""
    e = estado.strip().lower()
    if e.startswith("conservar"):
        return "conservar"
    if e.startswith("iv"):
        return "iv"
    if e.startswith("descartar"):
        return "descartar"
    if e.startswith("control"):
        return "control"
    if e.startswith("referencia"):
        return "referencia"
    if "denominador" in e or "fuente de la bandera" in e:
        return "degradada"
    raise ValueError(f"estado sin decisión en el vocabulario cerrado: {estado!r}")


def separar_codificacion(feature: str) -> tuple[str, str | None]:
    """Parte 'BUREAU_X > 0' o 'HAS_X | historial' en (nombre limpio, codificación)."""
    for sep in (" > ", " | "):
        if sep in feature:
            nombre, resto = feature.split(sep, 1)
            return nombre.strip(), f"{sep.strip()} {resto.strip()}"
    return feature.strip(), None


def _registro(
    fila: Any,  # fila del ranking: pd.Series o mapping
    alfa: float,
    firmes: set[str],
    controles: set[str],
) -> dict[str, Any]:
    """Un registro de la receta a partir de una fila del ranking."""
    nombre, codificacion = separar_codificacion(str(fila["feature"]))
    tipo = str(fila["tipo"])
    estado = str(fila["estado"])
    efecto = fila.get("efecto")
    p = fila.get("p")

    reg: dict[str, Any] = {"nombre": nombre}
    if codificacion is not None:
        reg["codificacion"] = codificacion
    reg["tipo"] = tipo
    reg["poblacion_medicion"] = fila["poblacion"]

    # n_pos significa cosas distintas en cada acumulador: marcados en flags, población en
    # continuas. Se exporta con nombre distinto en cada caso en vez de con uno ambiguo.
    n = fila.get("n_pos")
    if n is not None and not pd.isna(n):
        reg["n_marcados" if tipo == "flag" else "n"] = int(n)

    reg["efecto"] = None if efecto is None or pd.isna(efecto) else round(float(efecto), 4)
    reg["unidad"] = UNIDADES[tipo]
    if p is not None and not pd.isna(p):
        reg["p"] = float(f"{float(p):.3e}")
        reg["significativa_bonferroni"] = float(p) < alfa

    reg["decision"] = clasificar_estado(estado)
    reg["firmeza"] = "firme" if fila["feature"] in firmes or nombre in firmes else "provisional"
    if fila["feature"] in controles or nombre in controles:
        reg["control"] = True
    reg["estado"] = estado
    return reg


def construir_receta(
    tabla: str,
    nivel: str,
    nota: str,
    rankings: Iterable[pd.DataFrame],
    alfa_bonferroni: float,
    firmes: Iterable[str] = (),
    controles: Iterable[str] = (),
    extra: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Receta completa como dict, lista para volcar a yaml.

    `firmes` son las features cuyo descarte es target-independiente (redundancia estructural):
    todo lo demás se decidió contra el TARGET sobre train completo y es provisional.
    """
    firmes, controles = set(firmes), set(controles)
    features: list[dict[str, Any]] = []
    for rank in rankings:
        for _, fila in rank.iterrows():
            features.append(_registro(fila, alfa_bonferroni, firmes, controles))
    for fila in extra:
        features.append(_registro(fila, alfa_bonferroni, firmes, controles))

    claves = [(f["nombre"], f.get("codificacion")) for f in features]
    assert len(claves) == len(set(claves)), "hay dos registros con el mismo nombre y codificación"
    declarados = {f["nombre"] for f in features} | {
        f"{f['nombre']} {f['codificacion']}" for f in features if f.get("codificacion")
    }
    sin_usar = (firmes | controles) - declarados
    assert not sin_usar, f"firmes/controles que no existen en el ranking: {sorted(sin_usar)}"

    return {
        "tabla": tabla,
        "nivel": nivel,
        "nota": nota,
        "metodologia": {
            "efectos_sobre": "train completo, no sobre el split de entrenamiento",
            "alfa_bonferroni": float(f"{alfa_bonferroni:.3e}"),
            "n_features": len(features),
        },
        "esquema": _ESQUEMA,
        "features": features,
    }


def exportar_receta(
    tabla: str,
    nivel: str,
    nota: str,
    rankings: Iterable[pd.DataFrame],
    alfa_bonferroni: float,
    firmes: Iterable[str] = (),
    controles: Iterable[str] = (),
    extra: Iterable[Mapping[str, Any]] = (),
    raiz: Path | None = None,
) -> Path:
    """Escribe config/<tabla>_features.yaml y devuelve la ruta relativa a la raíz del repo."""
    receta = construir_receta(
        tabla, nivel, nota, rankings, alfa_bonferroni, firmes, controles, extra
    )
    raiz = raiz or Path(__file__).resolve().parents[2]
    destino = raiz / "config" / f"{tabla}_features.yaml"
    destino.write_text(yaml.safe_dump(receta, allow_unicode=True, sort_keys=False))
    return destino.relative_to(raiz)
