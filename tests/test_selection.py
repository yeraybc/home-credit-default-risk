"""Tests de `recomendar_codificacion()` (src/features/selection.py).

La función recoge lo que concluyó el EDA y quien codifica de verdad es `pipeline.py`. Las dos
coinciden salvo en cinco entradas, donde el punto 1.4 midió sobre el split y llegó a otra
conclusión. Lo que estos tests protegen es que esas cinco lo digan: sin la marca, quien lea la
función exportada se lleva la decisión superada sin nada que se lo advierta, que es el patrón de
las dos fuentes de verdad y en este proyecto ya ha mordido tres veces.

Sintéticos, así que corren también en un clon limpio.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import columnas_declaradas
from src.features.selection import (
    DECISIONES_IV,
    PARES_DECLARADOS,
    TABLA_PRINCIPAL,
    DecisionIV,
    columnas_protegidas,
    pares_redundantes,
    recomendar_codificacion,
    tabla_de,
)

# Las seis donde el pipeline hace otra cosa, con lo que la marca tiene que anunciar. La última no
# es una entrada de `especificas` sino la rama por defecto del bloque de documento, y sirve
# cualquier FLAG_DOCUMENT_* que no tenga entrada propia.
DIVERGENTES = {
    "NAME_TYPE_SUITE": "no agrupa",
    "NAME_INCOME_TYPE": "no parte en dos",
    "NAME_HOUSING_TYPE": "no fusiona",
    "ORGANIZATION_TYPE": "nivel a nivel",
    "OCCUPATION_TYPE": "no agrupa en bandas",
    "FLAG_DOCUMENT_12": "no exceptúa",
}

# Las dos del bloque de documento que sí tienen entrada propia, y por las que la rama por defecto
# pregunta a `especificas` en vez de llevar sus dos nombres escritos.
DE_DOCUMENTO_CON_ENTRADA = ("FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6")

# Las tres de contacto que caían en la rama por defecto y salían sin decisión propia.
DE_CONTACTO = ("FLAG_PHONE", "FLAG_WORK_PHONE", "FLAG_EMAIL")

DETALLE_GENERICO = "Ya es una variable numérica binaria 0/1."


@pytest.fixture
def tabla():
    """Una fila por categoría, que es todo lo que la función mira: nombres y cardinalidad."""
    categoricas = {
        "NAME_TYPE_SUITE": ["Unaccompanied", "Family"],
        "NAME_INCOME_TYPE": ["Working", "Pensioner"],
        "NAME_HOUSING_TYPE": ["House / apartment", "With parents"],
        "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed"],
        "OCCUPATION_TYPE": ["Laborers", "Sales staff"],
    }
    documento = {c: [0, 1] for c in ("FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_12")}
    binarias = {c: [0, 1] for c in DE_CONTACTO}
    return pd.DataFrame({**categoricas, **binarias, **documento})


def _detalle(recomendaciones, variable):
    return recomendaciones.set_index("Variable").loc[variable, "Detalle"]


def test_las_divergentes_anuncian_que_el_pipeline_hace_otra_cosa(tabla):
    """El marcador va en el detalle y no en un comentario: el detalle es lo que se renderiza."""
    recomendaciones = recomendar_codificacion(tabla)

    for variable, anuncio in DIVERGENTES.items():
        detalle = _detalle(recomendaciones, variable)
        assert "El pipeline" in detalle, f"{variable} no dice que el pipeline decida otra cosa"
        assert anuncio in detalle, f"{variable} no dice qué hace el pipeline en su lugar"


def test_las_tres_de_contacto_tienen_decision_propia(tabla):
    """Caían en la rama por defecto, que da el mismo detalle para cualquier binaria."""
    recomendaciones = recomendar_codificacion(tabla)

    for variable in DE_CONTACTO:
        detalle = _detalle(recomendaciones, variable)
        assert detalle != DETALLE_GENERICO, f"{variable} sigue sin decisión propia"
        assert "TARGET" in detalle, f"{variable} no declara su correlación"


def test_las_dos_de_documento_con_entrada_propia_se_salen_de_la_rama_generica(tabla):
    """La rama pregunta a `especificas` y no lleva sus dos nombres escritos.

    Es el par que también declara `pipeline.COLUMNAS_PROTEGIDAS_DE_VARIANZA`, o sea el patrón de
    las dos fuentes de verdad. Preguntando al diccionario, meter otro FLAG_DOCUMENT_* con
    decisión propia funciona sin tocar la condición.
    """
    recomendaciones = recomendar_codificacion(tabla).set_index("Variable")

    for variable in DE_DOCUMENTO_CON_ENTRADA:
        assert recomendaciones.loc[variable, "Estrategia Recomendada"] == "Conservar como binaria"
        assert "correlación" in recomendaciones.loc[variable, "Detalle"]
    assert (
        recomendaciones.loc["FLAG_DOCUMENT_12", "Estrategia Recomendada"]
        == "Filtrar con VarianceThreshold"
    )


def test_la_rama_por_defecto_sigue_existiendo_para_lo_no_declarado(tabla):
    """Guardián: si nadie cayera ya en ella, el test de arriba no distinguiría nada."""
    recomendaciones = recomendar_codificacion(tabla.assign(FLAG_INVENTADA=[0, 1]))

    assert _detalle(recomendaciones, "FLAG_INVENTADA") == DETALLE_GENERICO


# --- el registro de selección del 5.6 -----------------------------------------------------------


def test_toda_decision_iv_que_no_descarta_apunta_a_una_columna_de_la_matriz():
    """Un `conservar`, `iv` o `degradada` sin columna real dejaría un nombre suelto en el
    registro que nadie podría reconciliar con la matriz. Un `descartar` no tiene por qué apuntar
    a nada, que es justo el caso de la tripartita de la mora: no se construyó columna."""
    declaradas = columnas_declaradas()
    for feature, decision in DECISIONES_IV.items():
        if decision.decision != "descartar":
            assert feature in declaradas, feature


def test_una_decision_iv_sin_criterio_o_sin_motivo_revienta():
    with pytest.raises(ValueError, match="declara"):
        DecisionIV("conservar", "", "algo")
    with pytest.raises(ValueError, match="declara"):
        DecisionIV("conservar", "algo", " ")


# --- el detector de pares redundantes del 5.7 ---------------------------------------------------

N = 20_000


def _correlada(base, r, rng):
    """Una normal con correlación `r` contra `base`, que va estandarizada."""
    return r * base + np.sqrt(1 - r**2) * rng.standard_normal(len(base))


@pytest.fixture
def plantado():
    """Un par por cada rama del detector, con nombres reales para que `tabla_de` los reparta.

    Los pares entre tablas: AMT_CREDIT con BB_MONTHS_TOTAL a -0,80 (por encima, y negativo para
    que el umbral mire el valor absoluto), con BUREAU_DAYS_CREDIT_MIN a -0,65 (banda) y con
    PREV_CNT_PAYMENT_MEAN a 0,45 (fuera). Dos
    banderas a unos 0,55, que pasan por ser banderas y no pasarían como magnitudes. Una bandera
    contra una magnitud que es el mismo evento con una escala que no escala con él. Y dentro de
    bureau, dos columnas copiadas, que no salen; más una descartada de su receta copiada de otra
    tabla, que tampoco.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(N)
    evento = rng.random(N) < 0.3
    otra = np.where(rng.random(N) < 0.55, evento, rng.random(N) < 0.3)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(N),
            "TARGET": rng.integers(0, 2, N),
            "AMT_CREDIT": base,
            "BB_MONTHS_TOTAL": _correlada(base, -0.80, rng),
            "BUREAU_DAYS_CREDIT_MIN": _correlada(base, -0.65, rng),
            "PREV_CNT_PAYMENT_MEAN": _correlada(base, 0.45, rng),
            "HAS_BUREAU_HISTORY": evento.astype("int8"),
            "HAS_PREV_APPLICATION": otra.astype(float),
            "BB_OVERDUE_UNION": evento.astype(float),
            "PREV_REFUSED_COUNT": evento * rng.lognormal(0, 2, N),
            "BUREAU_CLOSED_COUNT": base * 2,
            "BUREAU_LOAN_COUNT": base * 3,
            "BB_N_CREDITS_WBAL": base * 4,
            "BB_TRAJECTORY": pd.Categorical(rng.choice(["a", "b"], N)),
        }
    )


def test_el_fixture_del_detector_planta_lo_que_dice(plantado):
    """Guardián: si el fixture pierde un caso de borde, los tests de abajo pasan sin probar nada."""
    r = plantado.drop(columns="BB_TRAJECTORY").corr()
    assert r.loc["AMT_CREDIT", "BB_MONTHS_TOTAL"] == pytest.approx(-0.80, abs=0.02)
    assert r.loc["AMT_CREDIT", "BUREAU_DAYS_CREDIT_MIN"] == pytest.approx(-0.65, abs=0.02)
    assert r.loc["AMT_CREDIT", "PREV_CNT_PAYMENT_MEAN"] == pytest.approx(0.45, abs=0.02)
    assert 0.50 < r.loc["HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION"] < 0.60
    # el mismo evento, pero la magnitud no llega ni a la banda en bruto
    assert abs(r.loc["BB_OVERDUE_UNION", "PREV_REFUSED_COUNT"]) < 0.60
    assert tabla_de("BUREAU_CLOSED_COUNT") == tabla_de("BUREAU_LOAN_COUNT") == "bureau"
    assert tabla_de("BB_N_CREDITS_WBAL") == "bureau_balance"
    assert tabla_de("AMT_CREDIT") == TABLA_PRINCIPAL


def test_el_detector_saca_por_encima_y_banda_y_deja_fuera_lo_de_debajo(plantado):
    pares = pares_redundantes(plantado)
    assert pares.loc[("AMT_CREDIT", "BB_MONTHS_TOTAL"), "zona"] == "por encima"
    assert pares.loc[("AMT_CREDIT", "BUREAU_DAYS_CREDIT_MIN"), "zona"] == "banda"
    assert ("AMT_CREDIT", "PREV_CNT_PAYMENT_MEAN") not in pares.index


def test_dos_banderas_cruzan_con_la_v_y_no_con_el_umbral_de_magnitudes(plantado):
    pares = pares_redundantes(plantado)
    fila = pares.loc[("HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION")]
    assert (fila["tipo"], fila["zona"]) == ("FF", "por encima")


def test_una_bandera_y_una_magnitud_cruzan_por_el_evento_binarizado(plantado):
    fila = pares_redundantes(plantado).loc[("BB_OVERDUE_UNION", "PREV_REFUSED_COUNT")]
    assert fila["tipo"] == "FM"
    assert fila["r_bin"] == pytest.approx(1.0)
    assert fila["zona"] == "por encima"


def test_una_magnitud_con_negativos_no_se_binariza(plantado):
    """`> 0` no es un evento en una columna de días con signo."""
    datos = plantado.assign(PREV_REFUSED_COUNT=plantado["PREV_REFUSED_COUNT"] - 1)
    pares = pares_redundantes(datos)
    assert ("BB_OVERDUE_UNION", "PREV_REFUSED_COUNT") not in pares.index


def test_dentro_de_una_misma_tabla_solo_salen_los_declarados(plantado):
    pares = pares_redundantes(plantado)
    assert ("BUREAU_CLOSED_COUNT", "BUREAU_LOAN_COUNT") not in pares.index
    # el mismo par entre tablas sí sale: lo que lo deja fuera es la tabla, no la cifra
    assert ("AMT_CREDIT", "BUREAU_CLOSED_COUNT") in pares.index


def test_un_par_declarado_sale_aunque_no_cruce_nada():
    rng = np.random.default_rng(1)
    datos = pd.DataFrame(
        {
            "BUREAU_ACTIVE_COUNT": rng.standard_normal(N),
            "BUREAU_LOAN_COUNT": rng.standard_normal(N),
        }
    )
    pares = pares_redundantes(datos)
    assert pares.loc[("BUREAU_ACTIVE_COUNT", "BUREAU_LOAN_COUNT"), "zona"] == "declarado"
    assert pares["declarado"].all()


def test_una_columna_descartada_por_su_receta_no_entra(plantado):
    pares = pares_redundantes(plantado)
    assert "BB_N_CREDITS_WBAL" not in set(pares.index.get_level_values("a")) | set(
        pares.index.get_level_values("b")
    )
    # guardián: con la lista de columnas explícita sí entra, así que la deja fuera la receta
    columnas = ["AMT_CREDIT", "BB_N_CREDITS_WBAL"]
    assert ("AMT_CREDIT", "BB_N_CREDITS_WBAL") in pares_redundantes(plantado, columnas).index


def test_los_pares_declarados_van_en_orden_alfabetico_y_dentro_de_bureau():
    for a, b in PARES_DECLARADOS:
        assert a < b
        assert tabla_de(a) == tabla_de(b) == "bureau"


def test_las_protegidas_incluyen_las_de_control_y_las_tres_de_presencia():
    protegidas = columnas_protegidas()
    assert {"HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE", "HAS_PREV_APPLICATION"} <= protegidas
    # las cuatro `control: true` de las recetas, leídas y no escritas a mano
    assert {"PREV_HISTORIAL_RECORTADO", "PREV_DAYS_DECISION_MAX"} <= protegidas
    assert len(protegidas) == 10
