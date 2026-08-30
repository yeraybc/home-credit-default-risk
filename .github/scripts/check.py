#!/usr/bin/env python3
"""Verificación de sintaxis, kernel y dependencias para CI.

No instala el stack del proyecto: ast.parse valida la sintaxis sin evaluar los
import, así que no hace falta descargar pandas para saber si el código está bien.

Es un verificador estático, así que corre en la versión de Python del runner y no
en la del proyecto (3.9): necesita sys.stdlib_module_names, que es 3.10+.

Uso: python .github/scripts/check.py
"""

import ast
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Un kernelspec con nombre propio apunta a un venv registrado a nivel de usuario y
# solo existe en la maquina de origen: en un clon limpio da NoSuchKernel. El kernel
# "python3" lo provee cualquier venv con ipykernel.
KERNEL_PORTABLE = "python3"

# El nombre que se importa no siempre es el del paquete que se instala.
IMPORT_A_PAQUETE = {
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
}

# Convenciones de redacción de los markdown del EDA (CLAUDE.md 9 y 11.2). Se revisan
# porque son errores que no se ven a ojo: en la ultima pasada manual quedaron 353
# decimales con punto sin detectar en un solo notebook.
TIPOGRAFIA = [
    # los miles espanoles van en grupos de 3, asi que 1.670.214 es correcto y 0.21 no
    (r"\d\.(\d{1,2}(?!\d)|\d{4,})", "decimal con punto"),
    (r"(?<!0),\d{3}(?!\d)", "miles a la inglesa"),
    (r"—|–", "em dash"),
    (r"->|→", "flecha"),
    (r"(?<=\w) - (?=\w)", "guion como conector"),
    (r"={3,}|·", "separador decorativo"),
    (r"(?<![\d,.])\d{1,3}-\d", "rango con guion"),
]

# Nombres de variables del codigo que el lector de una conclusion no puede interpretar.
VARIABLES_INTERNAS = ["prev_t", "ev_cli", "tiene_prev", "bureau_t", "bureau_client", "map_bc",
                      "fb_flags", "fb_cont", "rank_flags", "df["]

# La conclusion no habla del documento que la contiene (CLAUDE.md 6.3.6b).
META_ESTRUCTURA = [r"secci[óo]n", r"Fase [AB3]", r"notebook", r"§", r"apartado"]

# Arrastran redaccion anterior a la unificacion y se limpian en una pasada aparte:
# se reportan pero no bloquean, y este conjunto debe acabar vacio.
TIPOGRAFIA_PENDIENTE = {"01_eda_application_train.ipynb", "02_eda_bureau.ipynb"}

fallos = []


def seccion(titulo):
    print(f"\n{titulo}")


def celdas_de_codigo(nb):
    """Devuelve el código de cada celda, sin las líneas mágicas de IPython."""
    for celda in nb.get("cells", []):
        if celda.get("cell_type") != "code":
            continue
        lineas = [l for l in "".join(celda["source"]).split("\n")
                  if not l.strip().startswith(("%", "!"))]
        yield "\n".join(lineas)


def imports_de(codigo):
    for nodo in ast.walk(ast.parse(codigo)):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                yield alias.name.split(".")[0]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            yield nodo.module.split(".")[0]


def comprobar_sintaxis_python():
    """Valida los módulos y devuelve el código de los que se pudieron parsear."""
    seccion("Sintaxis de los módulos")
    legibles = []
    rutas = sorted(RAIZ.glob("src/**/*.py")) + sorted(RAIZ.glob("scripts/*.py"))
    for ruta in rutas:
        rel = ruta.relative_to(RAIZ)
        try:
            codigo = ruta.read_text(encoding="utf-8")
            ast.parse(codigo)
            print(f"  {rel}: ok")
            legibles.append(codigo)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  {rel}: {e}")
            fallos.append(f"no parsea: {rel}")
    return legibles


def comprobar_notebooks():
    """Valida los notebooks y devuelve sus celdas de código y los que se pudieron leer."""
    seccion("Notebooks")
    legibles, leidos = [], []
    for ruta in sorted(RAIZ.glob("notebooks/*.ipynb")):
        rel = ruta.relative_to(RAIZ)
        try:
            nb = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  {rel}: JSON inválido, {e}")
            fallos.append(f"JSON inválido: {rel}")
            continue

        leidos.append((rel, nb))
        problemas = []
        kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name")
        if kernel != KERNEL_PORTABLE:
            problemas.append(f"kernel '{kernel}', se espera '{KERNEL_PORTABLE}'")
            fallos.append(f"kernel no portable: {rel}")

        celdas = list(celdas_de_codigo(nb))
        try:
            for codigo in celdas:
                ast.parse(codigo)
            legibles.append(celdas)
        except SyntaxError as e:
            problemas.append(f"celda con error de sintaxis, {e}")
            fallos.append(f"celda no parsea: {rel}")

        print(f"  {rel}: " + ("; ".join(problemas) or f"ok, {len(celdas)} celdas de código"))
    return legibles, leidos


def avisos_de_tipografia(nb):
    """Avisos de redacción de los markdown de un notebook, como (etiqueta, contexto)."""
    comprobaciones = (TIPOGRAFIA
                      + [(re.escape(v), "variable interna") for v in VARIABLES_INTERNAS]
                      + [(w, "meta-estructura") for w in META_ESTRUCTURA])
    for i, celda in enumerate(nb.get("cells", [])):
        if celda.get("cell_type") != "markdown":
            continue
        texto = re.sub(r"`[^`]*`", "", "".join(celda["source"]))  # fuera nombres de columna y código
        # las listas, las tablas y los títulos usan guiones y numeración de forma legítima
        texto = "\n".join(l for l in texto.split("\n") if not l.lstrip().startswith(("-", "|", "#")))

        for patron, etiqueta in comprobaciones:
            for m in re.finditer(patron, texto, re.M | re.I):
                yield etiqueta, f"celda {i}: ...{texto[max(0, m.start() - 40):m.end() + 40]}..."


def comprobar_tipografia(leidos):
    """Las conclusiones siguen las convenciones de redacción del proyecto.

    Los notebooks de TIPOGRAFIA_PENDIENTE se reportan sin bloquear: arrastran redacción
    anterior a la unificación y se limpian aparte.
    """
    seccion("Tipografía de los markdown")
    for rel, nb in leidos:
        avisos = list(avisos_de_tipografia(nb))
        if not avisos:
            print(f"  {rel}: ok")
            continue

        tipos = {}
        for etiqueta, ctx in avisos:
            tipos.setdefault(etiqueta, ctx)
        detalle = ", ".join(f"{e} ({sum(1 for a, _ in avisos if a == e)})" for e in tipos)
        if rel.name in TIPOGRAFIA_PENDIENTE:
            print(f"  {rel}: {len(avisos)} avisos pendientes de limpieza, no bloquean: {detalle}")
            continue

        print(f"  {rel}: {len(avisos)} avisos: {detalle}")
        for etiqueta, ctx in tipos.items():
            print(f"      {etiqueta}: {ctx}".replace("\n", " "))
        fallos.append(f"tipografía: {rel}")


def leer_requirements():
    """Lee requirements.txt exigiendo UTF-8 y devuelve los paquetes declarados."""
    seccion("Formato de requirements.txt")
    ruta = RAIZ / "requirements.txt"
    try:
        # Se exige UTF-8: un requirements.txt en UTF-16 lo rompe todo aguas abajo.
        contenido = ruta.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"  no decodifica como UTF-8: {e}")
        fallos.append("requirements.txt no está en UTF-8")
        return set()

    lineas = [l.strip() for l in contenido.splitlines()]
    lineas = [l for l in lineas if l and not l.startswith("#")]
    patron = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*((==|>=|<=|~=)[\w.]+)?$")
    invalidas = [l for l in lineas if not patron.match(l)]
    if invalidas:
        print(f"  líneas inválidas: {', '.join(invalidas)}")
        fallos.append("requirements.txt mal formado")
    else:
        print(f"  ok, {len(lineas)} dependencias declaradas")

    return {re.split(r"==|>=|<=|~=", l)[0].lower().replace("_", "-") for l in lineas}


def comprobar_dependencias(declaradas, modulos, notebooks):
    """Toda librería importada debe estar declarada.

    Este check existe porque el requirements.txt se escribe a mano, declarando solo
    lo que se usa: sin él, un import nuevo en un notebook se cuela sin declarar y
    rompe el entorno del siguiente que clone.
    Solo mira el código que parseó: lo que no parsea ya se reportó antes.
    """
    seccion("Dependencias declaradas")
    usadas = set()
    for codigo in modulos:
        usadas.update(imports_de(codigo))
    for celdas in notebooks:
        for codigo in celdas:
            usadas.update(imports_de(codigo))

    externas = {m for m in usadas if m not in sys.stdlib_module_names and m != "src"}
    normalizadas = {IMPORT_A_PAQUETE.get(m, m).lower().replace("_", "-") for m in externas}
    faltan = sorted(normalizadas - declaradas)

    if faltan:
        print(f"  usadas pero NO declaradas: {', '.join(faltan)}")
        fallos.append("faltan dependencias en requirements.txt")
    else:
        print(f"  ok, {len(externas)} librerías usadas y todas declaradas")


def main():
    if sys.version_info < (3, 10):
        sys.exit("Este script necesita Python 3.10+ (usa sys.stdlib_module_names)")

    modulos = comprobar_sintaxis_python()
    notebooks, leidos = comprobar_notebooks()
    comprobar_tipografia(leidos)
    declaradas = leer_requirements()
    comprobar_dependencias(declaradas, modulos, notebooks)

    print()
    if fallos:
        print("FALLO:")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    print("Todo correcto.")


if __name__ == "__main__":
    main()
