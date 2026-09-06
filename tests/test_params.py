"""Tests del registro de parámetros del pipeline (src/features/params.py).

Lo que protegen: que ningún corte entre sin declarar de dónde sale, que el mismo umbral no se
escriba dos veces con valores distintos (el patrón del 52.500 frente al 52.497 de bureau), y que
usar un parámetro todavía sin fijar falle en vez de colarse como None.
"""

import pytest
import yaml

from src.config import RAIZ, cargar_config
from src.features.params import (
    PARAMS,
    PROCEDENCIAS,
    REAJUSTABLES,
    Parametro,
    parametro,
    por_procedencia,
    reajustables,
    sin_fijar,
    valor,
)


def test_toda_procedencia_esta_en_el_vocabulario_cerrado():
    assert set(REAJUSTABLES) < set(PROCEDENCIAS)
    for nombre, p in PARAMS.items():
        assert p.procedencia in PROCEDENCIAS, nombre


def test_todo_parametro_declara_descripcion_y_fuente():
    for nombre, p in PARAMS.items():
        assert p.descripcion.strip(), f"{nombre} sin descripción"
        assert p.fuente.strip(), f"{nombre} sin fuente"


def test_un_parametro_de_dominio_no_puede_quedar_sin_valor():
    with pytest.raises(ValueError, match="dominio"):
        Parametro(None, "dominio", "d", "f")


def test_procedencia_fuera_del_vocabulario_revienta():
    with pytest.raises(ValueError, match="procedencia"):
        Parametro(1, "heredado", "d", "f")


def test_solo_estimado_y_medido_pueden_estar_sin_fijar():
    for nombre, p in sin_fijar().items():
        assert p.procedencia in REAJUSTABLES, nombre


def test_usar_un_parametro_sin_fijar_falla_en_vez_de_devolver_none():
    pendientes = sin_fijar()
    assert pendientes, "si ya no queda ninguno sin fijar, este test sobra"
    nombre = next(iter(pendientes))
    with pytest.raises(ValueError, match="sin fijar"):
        valor(nombre)


def test_un_parametro_fijado_devuelve_su_valor():
    assert valor("dias_por_anio") == 365.25
    assert valor("umbral_flags_pp") == 2.0


def test_parametro_no_declarado_revienta():
    with pytest.raises(KeyError, match="no declarado"):
        parametro("umbral_inventado")


def test_las_procedencias_particionan_el_registro():
    total = sum(len(por_procedencia(p)) for p in PROCEDENCIAS)
    assert total == len(PARAMS)
    assert set(reajustables()) == set(por_procedencia("estimado")) | set(por_procedencia("medido"))


def test_el_centinela_del_dataset_esta_declarado_una_sola_vez():
    """365243 es el mismo código en application_train y en previous_application.

    Declararlo por tabla es el patrón que en bureau dio 52.500 en un sitio y 52.497 en otro
    para el mismo control.
    """
    con_ese_valor = [n for n, p in PARAMS.items() if p.valor == 365243]
    assert con_ese_valor == ["centinela_365243"], con_ese_valor


def test_no_hay_dos_parametros_con_la_misma_descripcion():
    """Misma descripción es copia y pega: el mismo corte con dos nombres."""
    vistas = {}
    for nombre, p in PARAMS.items():
        d = p.descripcion.strip()
        assert d not in vistas, f"{nombre} repite la descripción de {vistas.get(d)}"
        vistas[d] = nombre


def test_params_no_redeclara_nada_de_config_yaml():
    """El umbral de IV y los del split viven en config.yaml y no se copian aquí."""
    cfg = cargar_config()
    del cfg["paths"]
    claves_config = {k for bloque in cfg.values() for k in bloque}
    assert not (set(PARAMS) & claves_config)
    for prohibido in ("min_iv", "test_size", "random_state", "correlation_threshold"):
        assert prohibido not in PARAMS, f"{prohibido} ya está en config.yaml"


def test_los_umbrales_metodologicos_coinciden_con_las_recetas_del_eda():
    """El alfa y los umbrales que las recetas ya publicaron no se pueden mover por descuido."""
    assert valor("redundancia_pearson") == 0.70
    assert valor("redundancia_cramer") == 0.50
    assert valor("banda_revision_pearson") < valor("redundancia_pearson")
    assert valor("umbral_flags_pp") == 2.0


@pytest.mark.parametrize("tabla", ["bureau", "bureau_balance", "previous_application"])
def test_toda_feature_provisional_tiene_su_corte_reajustable(tabla):
    """Las recetas declaran qué se remide sobre el split; el registro tiene que poder hacerlo."""
    receta = yaml.safe_load((RAIZ / "config" / f"{tabla}_features.yaml").read_text())
    provisionales = [f for f in receta["features"] if f["firmeza"] == "provisional"]
    assert provisionales, f"{tabla} sin nada provisional, revisar la receta"
    assert reajustables(), "no hay ni un parámetro reajustable declarado"
