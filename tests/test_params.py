"""Tests del registro de parámetros del pipeline (src/features/params.py).

Lo que protegen: que ningún corte entre sin declarar de dónde sale, que el mismo umbral no se
escriba dos veces con valores distintos (el patrón del 52.500 frente al 52.497 de bureau), y que
usar un parámetro todavía sin fijar falle en vez de colarse como None.
"""

import pytest
import yaml

from src.config import RAIZ, cargar_config
from src.features.params import (
    CORTES_POR_FEATURE,
    PARAMS,
    PROCEDENCIAS,
    REAJUSTABLES,
    Parametro,
    con_contraste_pendiente,
    cortes_de,
    features_con_corte,
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


@pytest.mark.parametrize("nombre", sorted(sin_fijar()))
def test_usar_un_parametro_sin_fijar_falla_en_vez_de_devolver_none(nombre):
    """Uno por uno, no solo el primero: cada pendiente tiene que bloquear a quien lo use."""
    with pytest.raises(ValueError, match="sin fijar"):
        valor(nombre)


def test_los_pendientes_son_los_tres_declarados_de_la_capa_2b():
    """Deuda declarada, no olvidos. Si aparece uno nuevo, que se note aquí."""
    assert set(sin_fijar()) == {
        "umbral_continuas_rb",
        "min_denominador_proporcion",
        "prev_ratio_rechazo_min_solicitudes",
    }


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


# --- frontera entre config.yaml y params.py -------------------------------------------------
# config.yaml lleva infraestructura y params.py lleva cortes de modelado. La frontera se
# comprueba en las dos direcciones: que no haya la misma clave en los dos sitios, y que las
# claves concretas que se migraron o se eliminaron no vuelvan a colarse en config.yaml.

CORTES_QUE_NO_VUELVEN_A_CONFIG = (
    "min_iv",
    "n_bins_max",
    "correlation_threshold",
    "missing_threshold",
)


def _claves_de_config():
    cfg = cargar_config()
    cfg.pop("paths", None)
    return {k for bloque in cfg.values() for k in bloque}


def test_ninguna_clave_vive_en_los_dos_ficheros():
    solapadas = set(PARAMS) & _claves_de_config()
    assert not solapadas, f"mismo concepto declarable en dos sitios: {sorted(solapadas)}"


def test_los_cortes_de_modelado_ya_no_estan_en_config_yaml():
    """El 0,95 de correlation_threshold contra el 0,70 de aquí era el mismo control duplicado."""
    presentes = set(CORTES_QUE_NO_VUELVEN_A_CONFIG) & _claves_de_config()
    assert not presentes, f"volvieron a config.yaml: {sorted(presentes)}"


def test_los_dos_cortes_migrados_estan_declarados_aqui():
    for nombre in ("min_iv", "n_bins_max"):
        assert nombre in PARAMS, f"{nombre} se sacó de config.yaml y no llegó a params.py"
    assert valor("min_iv") == 0.02
    assert valor("n_bins_max") == 10


def test_el_umbral_de_redundancia_es_uno_solo_y_es_el_del_eda():
    assert valor("redundancia_pearson") == 0.70
    assert "correlation_threshold" not in PARAMS


def test_el_umbral_unico_de_nulos_no_se_migro():
    """El EDA decide los nulos columna a columna con IV, no con un umbral único."""
    assert "missing_threshold" not in PARAMS


def test_config_yaml_conserva_la_infraestructura():
    """Migrar los cortes no puede llevarse por delante lo que el split y la API necesitan."""
    cfg = cargar_config()
    assert set(cfg["dataset"]) == {"target_col", "id_col", "test_size", "random_state"}
    for bloque in ("paths", "modeling", "monitoring", "api", "mlflow"):
        assert cfg[bloque], f"bloque {bloque} vacío o ausente"


def test_los_umbrales_metodologicos_coinciden_con_las_recetas_del_eda():
    """El alfa y los umbrales que las recetas ya publicaron no se pueden mover por descuido."""
    assert valor("redundancia_pearson") == 0.70
    assert valor("redundancia_cramer") == 0.50
    assert valor("banda_revision_pearson") < valor("redundancia_pearson")
    assert valor("umbral_flags_pp") == 2.0


# --- cruce entre las recetas del EDA y el registro de cortes --------------------------------

TABLAS = ["bureau", "bureau_balance", "previous_application"]


def _receta(tabla):
    return yaml.safe_load((RAIZ / "config" / f"{tabla}_features.yaml").read_text())


@pytest.mark.parametrize("tabla", TABLAS)
def test_cada_feature_del_mapa_existe_en_su_receta(tabla):
    """El mapa no puede referirse a features que la receta no tiene."""
    de_la_receta = {f["nombre"] for f in _receta(tabla)["features"]}
    del_mapa = set(CORTES_POR_FEATURE[tabla])
    assert del_mapa <= de_la_receta, f"features inventadas: {sorted(del_mapa - de_la_receta)}"


@pytest.mark.parametrize("tabla", TABLAS)
def test_cada_feature_con_corte_apunta_a_un_parametro_declarado(tabla):
    """Cada corte que lleva dentro una feature tiene que existir en el registro y ser accionable.

    Accionable significa reajustable sobre el split, o de dominio pero con su contraste
    declarado; un corte de dominio a secas dentro de una feature provisional sería un valor
    que nadie va a revisar.
    """
    for feature, cortes in CORTES_POR_FEATURE[tabla].items():
        assert cortes, f"{tabla}.{feature} está en el mapa sin ningún corte"
        for corte in cortes:
            p = parametro(corte)  # revienta si no está declarado
            accionable = p.procedencia in REAJUSTABLES or p.contraste_pendiente is not None
            assert accionable, f"{tabla}.{feature} depende de {corte}, que nadie revisa"


@pytest.mark.parametrize("tabla", TABLAS)
def test_una_feature_con_corte_solo_es_firme_si_su_descarte_es_estructural(tabla):
    """Si su construcción depende de un umbral que se puede mover, casi siempre es provisional.

    La excepción legítima es el descarte por redundancia estructural contra una hermana
    construida desde el mismo corte: ahí la redundancia se mantiene con cualquier valor del
    umbral, porque las dos salen del mismo booleano. Caso de PREV_EARLY_SETTLED_COUNT y
    _RATIO, que correlacionan 0,8586 y 0,8539 con la bandera que sí se conserva.
    """
    receta = {f["nombre"]: f for f in _receta(tabla)["features"]}
    for feature, cortes in CORTES_POR_FEATURE[tabla].items():
        if receta[feature]["firmeza"] == "provisional":
            continue
        assert (
            receta[feature]["decision"] == "descartar"
        ), f"{tabla}.{feature} es firme, depende de {cortes} y no es un descarte"
        hermanas = [
            otra
            for otra, sus_cortes in CORTES_POR_FEATURE[tabla].items()
            if otra != feature
            and set(sus_cortes) & set(cortes)
            and receta[otra]["decision"] != "descartar"
        ]
        assert hermanas, (
            f"{tabla}.{feature} se descarta en firme por redundancia, pero ninguna feature "
            f"viva comparte sus cortes {cortes}"
        )


def test_todo_corte_reajustable_lo_usa_alguna_feature_o_es_transversal():
    """Un corte reajustable que no usa nadie es un valor huérfano.

    Los transversales no aparecen en el mapa porque no pertenecen a una feature concreta:
    son los percentiles de application_train y los tres pendientes de la capa 2b.
    """
    usados = set(features_con_corte())
    transversales = {n for n in PARAMS if n.startswith("app_")} | {
        "umbral_continuas_rb",
        "min_denominador_proporcion",
    }
    huerfanos = set(reajustables()) - usados - transversales
    assert not huerfanos, f"cortes reajustables que no usa ninguna feature: {sorted(huerfanos)}"


def test_el_suelo_del_denominador_es_dominio_con_su_contraste_declarado():
    """Reclasificarlo no puede llevarse por delante el contraste que la receta pide."""
    p = parametro("suelo_anios_denominador")
    assert p.procedencia == "dominio"
    assert p.contraste_pendiente is not None
    assert "suelo_anios_denominador" in con_contraste_pendiente()
    # las dos features que lo llevan dentro, una por tabla
    assert cortes_de("bureau", "BUREAU_CREDITS_PER_YEAR") == ("suelo_anios_denominador",)
    assert cortes_de("previous_application", "PREV_APPLICATIONS_PER_YEAR") == (
        "suelo_anios_denominador",
    )


def test_el_cap_de_la_antiguedad_del_coche_es_estimado():
    """64 era exactamente el p99, así que el criterio de negocio no sostenía el valor."""
    assert parametro("app_cap_p99_own_car_age").procedencia == "estimado"
    assert "app_own_car_age_max" not in PARAMS


def test_cortes_de_una_tabla_desconocida_revienta():
    with pytest.raises(KeyError, match="sin mapa de cortes"):
        cortes_de("installments", "LO_QUE_SEA")


def test_una_feature_sin_corte_devuelve_tupla_vacia():
    assert cortes_de("bureau", "BUREAU_ACTIVE_COUNT") == ()
