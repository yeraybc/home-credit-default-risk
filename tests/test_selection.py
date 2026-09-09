"""Tests de `recomendar_codificacion()` (src/features/selection.py).

La función recoge lo que concluyó el EDA y quien codifica de verdad es `pipeline.py`. Las dos
coinciden salvo en cinco entradas, donde el punto 1.4 midió sobre el split y llegó a otra
conclusión. Lo que estos tests protegen es que esas cinco lo digan: sin la marca, quien lea la
función exportada se lleva la decisión superada sin nada que se lo advierta, que es el patrón de
las dos fuentes de verdad y en este proyecto ya ha mordido tres veces.

Sintéticos, así que corren también en un clon limpio.
"""

import pandas as pd
import pytest

from src.features.selection import recomendar_codificacion

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
