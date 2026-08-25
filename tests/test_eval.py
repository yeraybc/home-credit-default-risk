"""Pruebas de src.features.eval.EvaluadorSenal.

El control de composición es la lección más cara del EDA de bureau: medir una feature
agregada sin condicionar al flag de presencia mete a los clientes sin filas en el grupo 0
y distorsiona la señal en dirección opuesta según el signo de la variable. Llegó a invertir
qué conteo era el primario. Estas pruebas fijan ese comportamiento para que las 4 tablas
que faltan no lo pierdan por un refactor.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.eval import EvaluadorSenal


@pytest.fixture
def escenario_composicion():
    """Reproduce en pequeño el artefacto de bureau: los sin-historial son los de más riesgo.

    100 clientes con historial (feature 1, riesgo bajo), 100 con historial (feature 0,
    riesgo bajo) y 100 SIN historial (feature NaN, riesgo alto). Si la medición mete a
    estos últimos en el grupo 0, el grupo 0 parece mucho peor de lo que es.
    """
    con_historial_alto = pd.DataFrame({"feature": np.ones(100)})
    con_historial_bajo = pd.DataFrame({"feature": np.zeros(100)})
    sin_historial = pd.DataFrame({"feature": [np.nan] * 100})
    df = pd.concat([con_historial_alto, con_historial_bajo, sin_historial], ignore_index=True)

    # riesgo: 20% con feature=1, 10% con feature=0, 50% en los sin historial
    target = np.concatenate(
        [
            np.array([1] * 20 + [0] * 80),
            np.array([1] * 10 + [0] * 90),
            np.array([1] * 50 + [0] * 50),
        ]
    )
    return df, target


def test_continua_como_string_se_autocondiciona(escenario_composicion):
    """Pasada como string, el notna() excluye a los clientes sin filas en la tabla auxiliar."""
    df, target = escenario_composicion
    ev = EvaluadorSenal(df, target)
    ev.evaluar_continuas(["feature"])

    registro = ev.fb_cont[0]
    assert registro["n_pos"] == 200, "los 100 sin historial deberían quedar fuera"
    assert registro["poblacion"] == "no nulos (auto-cond.)"


def test_continua_con_fill0_mide_global(escenario_composicion):
    """Con fill0=True los sin-historial entran en el grupo 0 y contaminan la baseline."""
    df, target = escenario_composicion
    ev = EvaluadorSenal(df, target)
    ev.evaluar_continuas([("feature", True)])

    registro = ev.fb_cont[0]
    assert registro["n_pos"] == 300, "el fill0 debería incluir a los 100 sin historial"
    assert registro["poblacion"] == "global (fill 0)"


def test_el_control_de_composicion_cambia_el_efecto(escenario_composicion):
    """El artefacto de §8.5: la misma feature da efectos distintos según cómo se mida.

    Es la prueba central del módulo. Si las dos mediciones coincidieran, el control de
    composición no estaría haciendo nada.
    """
    df, target = escenario_composicion

    ev_global = EvaluadorSenal(df, target)
    ev_global.evaluar_continuas([("feature", True)])
    efecto_global = ev_global.fb_cont[0]["efecto"]

    ev_cond = EvaluadorSenal(df, target)
    ev_cond.evaluar_continuas(["feature"])
    efecto_cond = ev_cond.fb_cont[0]["efecto"]

    assert efecto_global != pytest.approx(efecto_cond), (
        "medir global y condicionada debe dar resultados distintos cuando el grupo "
        "sin datos tiene un riesgo base diferente"
    )


def test_flag_calcula_el_delta_en_puntos_porcentuales():
    """20% frente a 10% son +10 pp, no +10% ni +0,1."""
    df = pd.DataFrame({"x": np.zeros(200)})
    valores = np.array([1] * 100 + [0] * 100)
    target = np.concatenate([np.array([1] * 20 + [0] * 80), np.array([1] * 10 + [0] * 90)])

    ev = EvaluadorSenal(df, target)
    ev.evaluar_flags([("FLAG", valores)])

    assert ev.fb_flags[0]["efecto"] == pytest.approx(10.0)
    assert ev.fb_flags[0]["n_pos"] == 100


def test_flag_respeta_la_poblacion_de_medicion():
    """Pasar `poblacion` es lo que condiciona una flag al subconjunto con historial."""
    df = pd.DataFrame({"x": np.zeros(300)})
    valores = np.array([1] * 100 + [0] * 100 + [0] * 100)
    target = np.concatenate(
        [
            np.array([1] * 20 + [0] * 80),
            np.array([1] * 10 + [0] * 90),
            np.array([1] * 50 + [0] * 50),
        ]
    )
    poblacion = np.array([True] * 200 + [False] * 100)

    ev = EvaluadorSenal(df, target)
    ev.evaluar_flags([("FLAG", valores, poblacion)])

    registro = ev.fb_flags[0]
    assert registro["efecto"] == pytest.approx(10.0), "los 100 excluidos no deben mover el delta"
    assert registro["poblacion"] == "con historial"


def test_la_medicion_condicionada_reemplaza_a_la_global():
    """El upsert por nombre evita que el ranking liste dos filas de la misma feature."""
    df = pd.DataFrame({"x": np.zeros(200)})
    valores = np.array([1] * 100 + [0] * 100)
    target = np.concatenate([np.array([1] * 20 + [0] * 80), np.array([1] * 10 + [0] * 90)])
    poblacion = np.array([True] * 200)

    ev = EvaluadorSenal(df, target)
    ev.evaluar_flags([("FLAG", valores)])
    ev.evaluar_flags([("FLAG", valores, poblacion)])

    assert len(ev.fb_flags) == 1, "debe quedar una sola entrada por feature"
    assert ev.fb_flags[0]["poblacion"] == "con historial", "manda la última medición"


def test_toda_comparacion_queda_registrada_para_bonferroni():
    """La familia de Bonferroni se cuenta sobre las comparaciones emitidas, no sobre features."""
    df = pd.DataFrame({"x": np.zeros(200)})
    valores = np.array([1] * 100 + [0] * 100)
    target = np.concatenate([np.array([1] * 20 + [0] * 80), np.array([1] * 10 + [0] * 90)])

    ev = EvaluadorSenal(df, target)
    ev.evaluar_flags([("FLAG", valores)])
    ev.evaluar_flags([("FLAG", valores, np.array([True] * 200))])

    assert len(ev.fb_comparisons) == 2, "las dos mediciones cuentan como dos comparaciones"
    assert len(ev.fb_flags) == 1, "pero el ranking sigue teniendo una sola fila"


def test_rank_biserial_esta_acotado_entre_cero_y_uno():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"feature": rng.random(400)})
    target = rng.integers(0, 2, 400)

    ev = EvaluadorSenal(df, target)
    ev.evaluar_continuas(["feature"])

    assert 0.0 <= ev.fb_cont[0]["efecto"] <= 1.0


def test_separacion_perfecta_da_rank_biserial_uno():
    """Si los defaulters tienen todos valores mayores, el efecto es máximo."""
    df = pd.DataFrame({"feature": list(range(100)) + list(range(100, 200))})
    target = np.array([0] * 100 + [1] * 100)

    ev = EvaluadorSenal(df, target)
    ev.evaluar_continuas(["feature"])

    assert ev.fb_cont[0]["efecto"] == pytest.approx(1.0)


def test_grupo_vacio_se_ignora_en_vez_de_reventar():
    """Una flag sin ningún 1 no puede compararse: se salta, no lanza."""
    df = pd.DataFrame({"x": np.zeros(100)})
    valores = np.zeros(100)
    target = np.array([1] * 10 + [0] * 90)

    ev = EvaluadorSenal(df, target)
    ev.evaluar_flags([("FLAG_SIN_POSITIVOS", valores)])

    assert ev.fb_flags == []
