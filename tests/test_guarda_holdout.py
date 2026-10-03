"""Guarda del holdout (catálogo A del plan): ningún módulo de src/models lee la parte de valid.

El holdout de valid se usa una sola vez, en el 5.1. Hasta entonces nadie en `src/models/` puede
pedirlo: ni sus matrices (`X_valid`, `y_valid`) ni sus filas (`solo_valid`, o `mascara`/`filtrar`
con "valid"). Al llegar el 5.1 su módulo entra en `PERMITIDOS`, con el visto bueno.
"""

import ast
from pathlib import Path

import pytest

from src.config import RAIZ

PERMITIDOS: set[str] = set()  # ficheros de src/models con permiso, uno solo en el 5.1
NOMBRES = {"X_valid", "y_valid", "solo_valid"}
CADENAS = {"X_valid", "y_valid", "X_valid.parquet", "y_valid.parquet"}
LLAMADAS_CON_PARTE = {"mascara", "filtrar"}


def lecturas_de_valid(fuente: str) -> list[str]:
    """Lo que en `fuente` pide la parte de valid: nombres, rutas y llamadas con la parte."""
    hallazgos = []
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Name) and nodo.id in NOMBRES:
            hallazgos.append(nodo.id)
        elif isinstance(nodo, ast.Attribute) and nodo.attr in NOMBRES:
            hallazgos.append(nodo.attr)
        elif isinstance(nodo, ast.alias) and nodo.name in NOMBRES:
            hallazgos.append(nodo.name)
        elif isinstance(nodo, ast.Constant) and nodo.value in CADENAS:
            hallazgos.append(repr(nodo.value))
        elif isinstance(nodo, ast.Call):
            llamada = getattr(nodo.func, "id", "") or getattr(nodo.func, "attr", "")
            argumentos = [*nodo.args, *(k.value for k in nodo.keywords)]
            if llamada in LLAMADAS_CON_PARTE and any(
                isinstance(a, ast.Constant) and a.value == "valid" for a in argumentos
            ):
                hallazgos.append(f"{llamada}(..., 'valid')")
    return hallazgos


def test_ningun_modulo_de_src_models_lee_el_holdout():
    modulos = sorted((RAIZ / "src" / "models").glob("*.py"))
    assert len(modulos) >= 8  # el glob encuentra los módulos y el test no pasa por vacuidad
    culpables = {
        m.name: lecturas_de_valid(m.read_text())
        for m in modulos
        if m.name not in PERMITIDOS and lecturas_de_valid(m.read_text())
    }
    assert not culpables, f"el holdout solo se lee en el 5.1: {culpables}"


@pytest.mark.parametrize(
    "fuente",
    [
        "X = pd.read_parquet(ruta / 'X_valid.parquet')",
        "from src.features.split import solo_valid",
        "v = solo_valid(df)",
        "y = datos.y_valid",
        "v = mascara(split, 'valid')",
        "v = filtrar(df, parte='valid')",
        "import src.features.split as s\ns.solo_valid(df)",
    ],
)
def test_el_detector_ve_cada_forma_de_pedir_el_holdout(fuente):
    assert lecturas_de_valid(fuente)


@pytest.mark.parametrize(
    "fuente",
    [
        "t = solo_train(df, split)",
        "m = mascara(split, 'train')",
        "grupos = {'train', 'valid'}",  # comparar etiquetas del split no es leer el holdout
        "d = split['split'].eq('valid')",
        "validar(y, score)",
    ],
)
def test_el_detector_no_se_queja_de_lo_que_no_es_el_holdout(fuente):
    assert not lecturas_de_valid(fuente)


def test_el_escaneo_cae_con_un_modulo_que_lee_valid(tmp_path: Path):
    malo = tmp_path / "malo.py"
    malo.write_text("def f(df):\n    return solo_valid(df)\n")
    assert lecturas_de_valid(malo.read_text()) == ["solo_valid"]
