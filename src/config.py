"""Acceso a config/config.yaml y a las rutas del proyecto.

Único sitio que resuelve la raíz del repo, para que no se repita el parents[N] en cada módulo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config() -> dict[str, Any]:
    """Contenido de config/config.yaml."""
    return yaml.safe_load((RAIZ / "config" / "config.yaml").read_text())


def ruta(clave: str) -> Path:
    """Ruta absoluta de una de las declaradas en el bloque `paths`."""
    paths = cargar_config()["paths"]
    if clave not in paths:
        raise KeyError(f"ruta no declarada en config.yaml: {clave!r}. válidas: {sorted(paths)}")
    return RAIZ / paths[clave]
