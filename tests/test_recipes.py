"""Tests de la exportación de recetas de features (src/features/recipes.py)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.features.recipes import (
    DECISIONES,
    clasificar_estado,
    construir_receta,
    exportar_receta,
    separar_codificacion,
)


@pytest.mark.parametrize(
    "estado, esperado",
    [
        ("conservar (1er orden)", "conservar"),
        ("conservar", "conservar"),
        ("IV (banda debil)", "iv"),
        ("IV (banda débil)", "iv"),
        ("IV (no pasa Bonferroni)", "iv"),
        ("IV (descarte ya no estructural)", "iv"),
        ("descartar (redundancia)", "descartar"),
        ("descartar (no supera ACTIVE_COUNT)", "descartar"),
        ("denominador de ratios", "degradada"),
        ("denominador del ratio", "degradada"),
        ("fuente de la bandera de actividad", "degradada"),
        ("control de composición", "control"),
        ("control (desplaza la recencia, no es predictor)", "control"),
        ("referencia (feature de bureau)", "referencia"),
    ],
)
def test_clasificar_estado_cubre_el_vocabulario(estado, esperado):
    assert clasificar_estado(estado) == esperado
    assert esperado in DECISIONES


def test_clasificar_estado_falla_en_vez_de_inventar():
    # una decisión nueva tiene que romper el export, no colarse como categoría silenciosa
    with pytest.raises(ValueError, match="vocabulario cerrado"):
        clasificar_estado("quizás la dejamos")


@pytest.mark.parametrize(
    "clave, nombre, codificacion",
    [
        ("BUREAU_MAX_DAYS_OVERDUE > 0", "BUREAU_MAX_DAYS_OVERDUE", "> 0"),
        ("HAS_BUREAU_ANNUITY | historial", "HAS_BUREAU_ANNUITY", "| historial"),
        ("PREV_REFUSED_RATIO", "PREV_REFUSED_RATIO", None),
    ],
)
def test_separar_codificacion(clave, nombre, codificacion):
    assert separar_codificacion(clave) == (nombre, codificacion)


def _rank(filas):
    return pd.DataFrame(filas)


FLAGS = _rank(
    [
        {
            "feature": "X_COUNT > 0",
            "tipo": "flag",
            "poblacion": "con historial",
            "efecto": 8.273,
            "p": 1e-30,
            "n_pos": 3397,
            "estado": "descartar (redundancia)",
        },
        {
            "feature": "HAS_X",
            "tipo": "flag",
            "poblacion": "global (nulo=0)",
            "efecto": -2.3949,
            "p": 1e-60,
            "n_pos": 263491,
            "estado": "conservar (1er orden)",
        },
    ]
)

CONT = _rank(
    [
        {
            "feature": "X_COUNT",
            "tipo": "continua",
            "poblacion": "con historial",
            "efecto": 0.0098,
            "p": 0.5,
            "n_pos": 263491,
            "estado": "denominador de ratios",
        }
    ]
)


def _receta(**kw):
    kw.setdefault("alfa_bonferroni", 1.67e-03)
    return construir_receta("t", "SK_ID_CURR", "nota", [FLAGS, CONT], **kw)


def test_n_se_exporta_con_nombre_distinto_segun_el_tipo():
    # n_pos son marcados en las banderas y población en las continuas: el defecto que se corrige
    # es exportarlos bajo la misma clave
    feats = {f["nombre"] + str(f.get("codificacion")): f for f in _receta()["features"]}
    assert feats["X_COUNT> 0"]["n_marcados"] == 3397
    assert "n" not in feats["X_COUNT> 0"]
    assert feats["X_COUNTNone"]["n"] == 263491
    assert "n_marcados" not in feats["X_COUNTNone"]


def test_unidad_declarada_por_tipo():
    unidades = {f["tipo"]: f["unidad"] for f in _receta()["features"]}
    assert unidades == {"flag": "pp", "continua": "rank_biserial"}


def test_misma_feature_con_dos_codificaciones_no_colisiona():
    # 'X_COUNT > 0' y 'X_COUNT' son dos codificaciones de la misma columna: con `features` como
    # diccionario y la clave limpia, una pisaba a la otra
    receta = _receta()
    assert len(receta["features"]) == 3
    nombres = [f["nombre"] for f in receta["features"]]
    assert nombres.count("X_COUNT") == 2


def test_firmeza_por_defecto_provisional_y_firme_solo_lo_declarado():
    receta = _receta(firmes={"X_COUNT > 0"})
    firmeza = {(f["nombre"], f.get("codificacion")): f["firmeza"] for f in receta["features"]}
    assert firmeza[("X_COUNT", "> 0")] == "firme"
    assert firmeza[("HAS_X", None)] == "provisional"


def test_control_solo_donde_se_declara():
    receta = _receta(controles={"HAS_X"})
    marcadas = [f["nombre"] for f in receta["features"] if f.get("control")]
    assert marcadas == ["HAS_X"]


def _sig(alfa):
    receta = _receta(alfa_bonferroni=alfa)
    return {
        f["nombre"] + str(f.get("codificacion")): f["significativa_bonferroni"]
        for f in receta["features"]
    }


def test_significancia_contra_el_alfa_declarado():
    # con el alfa por debajo de los dos p-valores no pasa ninguna; con 0,05 pasan los dos
    # pequeños y sigue sin pasar la continua de p = 0,5
    estricto = _sig(1e-70)
    assert estricto["X_COUNT> 0"] is False and estricto["HAS_XNone"] is False
    laxo = _sig(0.05)
    assert laxo["X_COUNT> 0"] is True and laxo["HAS_XNone"] is True
    assert laxo["X_COUNTNone"] is False


def test_firme_inexistente_rompe_el_export():
    # protege contra el typo silencioso que dejaría una feature marcada como provisional sin querer
    with pytest.raises(AssertionError, match="no existen en el ranking"):
        _receta(firmes={"NO_EXISTE"})


def test_extra_admite_registros_sin_efecto_ni_p():
    receta = _receta(
        extra=[
            {
                "feature": "X_CENSORED_RATIO",
                "tipo": "control",
                "poblacion": "con histórico",
                "efecto": None,
                "estado": "control (desplaza la recencia, no es predictor)",
            }
        ]
    )
    ctrl = receta["features"][-1]
    assert ctrl["efecto"] is None and ctrl["unidad"] is None
    assert ctrl["decision"] == "control" and "p" not in ctrl


def test_metodologia_y_esquema_viajan_en_el_fichero():
    receta = _receta()
    assert receta["metodologia"]["n_features"] == 3
    # el aviso que las tablas de ranking sí llevan y que antes no sobrevivía a la exportación
    assert "comparables entre poblaciones distintas" in receta["esquema"]["poblacion_medicion"]
    assert set(DECISIONES) == set(receta["esquema"]["decision"].split(" | "))


# --- las tres recetas reales de config/, que el pipeline de FE va a consumir ---------------

RECETAS = {"bureau": 30, "bureau_balance": 24, "previous_application": 28}
RAIZ = Path(__file__).resolve().parents[1]


def _cargar(tabla):
    return yaml.safe_load((RAIZ / "config" / f"{tabla}_features.yaml").read_text())


@pytest.mark.parametrize("tabla, n", RECETAS.items())
def test_receta_real_tiene_el_esquema_unificado(tabla, n):
    receta = _cargar(tabla)
    assert set(receta) == {"tabla", "nivel", "nota", "metodologia", "esquema", "features"}
    assert receta["tabla"] == tabla
    assert len(receta["features"]) == n == receta["metodologia"]["n_features"]

    claves = set()
    for f in receta["features"]:
        ident = (f["nombre"], f.get("codificacion"))
        assert ident not in claves, f"registro repetido en {tabla}: {ident}"
        claves.add(ident)

        assert f["decision"] in DECISIONES, f"{tabla}/{f['nombre']}: {f['decision']}"
        assert f["firmeza"] in ("firme", "provisional")
        assert f["tipo"] in ("flag", "continua", "control")
        unidades = {"flag": "pp", "continua": "rank_biserial", "control": None}
        assert f["unidad"] == unidades[f["tipo"]]
        # el n va con nombre distinto según el tipo, que es el defecto que se corrigió
        assert not (f["tipo"] == "flag" and "n" in f)
        assert not (f["tipo"] == "continua" and "n_marcados" in f)


def test_las_tres_recetas_declaran_los_mismos_campos_de_cabecera():
    esquemas = [tuple(_cargar(t)["esquema"]) for t in RECETAS]
    assert len(set(esquemas)) == 1, "el bloque esquema difiere entre recetas"
    metod = [tuple(_cargar(t)["metodologia"]) for t in RECETAS]
    assert len(set(metod)) == 1, "el bloque metodologia difiere entre recetas"


def test_solo_son_firmes_los_descartes_target_independientes():
    """La firmeza decide qué NO hay que remedir sobre el split, así que no puede inflarse."""
    esperado = {
        "bureau": {"BUREAU_DAYS_ENDDATE_FACT_MIN", "BUREAU_MAX_DAYS_OVERDUE"},
        "bureau_balance": {"BB_MONTHS_MAX"},
        "previous_application": {
            "PREV_EARLY_SETTLED_COUNT",
            "PREV_EARLY_SETTLED_RATIO",
            "PREV_IMPLIED_COST_MAX",
        },
    }
    for tabla, nombres in esperado.items():
        firmes = {f["nombre"] for f in _cargar(tabla)["features"] if f["firmeza"] == "firme"}
        assert firmes == nombres, f"{tabla}: firmes {firmes}"
        # y todo lo firme es un descarte: conservar algo "en firme" no tendría sentido
        for f in _cargar(tabla)["features"]:
            if f["firmeza"] == "firme":
                assert f["decision"] == "descartar", f"{tabla}/{f['nombre']}"


def test_exportar_escribe_yaml_valido(tmp_path):
    (tmp_path / "config").mkdir()
    destino = exportar_receta(
        "t", "SK_ID_CURR", "nota", [FLAGS, CONT], alfa_bonferroni=1.67e-03, raiz=tmp_path
    )
    assert str(destino) == "config/t_features.yaml"
    cargado = yaml.safe_load((tmp_path / destino).read_text())
    assert [f["nombre"] for f in cargado["features"]] == ["X_COUNT", "HAS_X", "X_COUNT"]
    assert cargado["tabla"] == "t"
