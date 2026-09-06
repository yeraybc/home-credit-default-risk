"""Tests de las features de capa 1 de la tabla principal (src/features/application.py).

Lo que protegen: que el cómputo de completitud del bloque edificio no se coma las cuatro
categóricas (metiéndolas la tripartita deja de dar las cifras del EDA), que las banderas de
los tres bloques sigan siendo tres ejes y no uno, y que los ratios propaguen el nulo en vez
de inventar un cero.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.application import (
    CATEGORICAS_EDIFICIO,
    anadir_banderas_ausencia,
    anadir_ratios,
    columnas_buro,
    columnas_edificio_numericas,
    construir_features_capa1,
    informe_capa1,
)
from src.features.params import valor


@pytest.fixture
def app():
    """Seis clientes: uno completo, uno todo nulo, y cuatro parciales de distinto tipo."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, 7),
            "TARGET": [0, 1, 0, 0, 1, 0],
            # bloque edificio: dos numéricas y una categórica
            "APARTMENTS_AVG": [0.1, np.nan, 0.3, np.nan, 0.5, np.nan],
            "TOTALAREA_MODE": [0.2, np.nan, np.nan, 0.4, 0.6, np.nan],
            "HOUSETYPE_MODE": ["block", np.nan, "block", np.nan, np.nan, "block"],
            # buró: el nulo es idéntico en las seis en el dato real
            "AMT_REQ_CREDIT_BUREAU_DAY": [0.0, np.nan, 1.0, 0.0, 0.0, 0.0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0, np.nan, 2.0, 1.0, 1.0, 1.0],
            # círculo social
            "OBS_30_CNT_SOCIAL_CIRCLE": [2.0, 1.0, np.nan, 0.0, 3.0, 1.0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, 0.0, np.nan, 0.0, 1.0, 0.0],
            "EXT_SOURCE_1": [0.5, np.nan, 0.4, np.nan, 0.6, 0.3],
            "EXT_SOURCE_3": [0.5, 0.4, np.nan, 0.3, 0.6, 0.2],
            # ratios
            "DAYS_BIRTH": [-14610, -21915, -10958, -18263, -25568, -12775],
            "DAYS_EMPLOYED": [-1000.0, np.nan, -500.0, -2000.0, -3000.0, 0.0],
            "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "AMT_GOODS_PRICE": [100.0, 100.0, 150.0, np.nan, 250.0, 0.0],
        }
    )


# --- bloque edificio -------------------------------------------------------------------------


def test_las_categoricas_del_edificio_no_entran_en_la_completitud(app):
    cols = columnas_edificio_numericas(app)
    assert cols == ["APARTMENTS_AVG", "TOTALAREA_MODE"]
    assert not set(cols) & set(CATEGORICAS_EDIFICIO)


def test_el_conteo_de_completitud_cuenta_solo_las_numericas(app):
    con = anadir_banderas_ausencia(app)
    # cliente 1 tiene las dos, el 2 ninguna, el 3 y el 4 una, el 5 las dos, el 6 ninguna
    assert list(con["BUILDING_INFO_COUNT"]) == [2, 0, 1, 1, 2, 0]


def test_la_bandera_del_edificio_es_binaria_y_separa_el_todo_nulo(app):
    """La señal está en no tener ningún dato, no en tenerlo incompleto."""
    con = anadir_banderas_ausencia(app)
    assert list(con["HAS_BUILDING_INFO"]) == [1, 0, 1, 1, 1, 0]
    # el cliente 6 tiene la categórica del bloque y aun así cuenta como sin información
    assert con.loc[5, "HOUSETYPE_MODE"] == "block"
    assert con.loc[5, "HAS_BUILDING_INFO"] == 0


# --- los otros dos bloques -------------------------------------------------------------------


def test_la_bandera_del_buro_exige_las_seis_columnas(app):
    assert columnas_buro(app) == [
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    con = anadir_banderas_ausencia(app)
    assert list(con["HAS_BUREAU_INFO"]) == [1, 0, 1, 1, 1, 1]


def test_la_bandera_social_exige_las_dos_columnas(app):
    con = anadir_banderas_ausencia(app)
    assert list(con["HAS_SOCIAL_INFO"]) == [1, 1, 0, 1, 1, 1]


def test_las_banderas_de_los_scores_marcan_el_nulo(app):
    con = anadir_banderas_ausencia(app)
    assert list(con["FLAG_EXT_SOURCE_1_NULL"]) == [0, 1, 0, 1, 0, 0]
    assert list(con["FLAG_EXT_SOURCE_3_NULL"]) == [0, 0, 1, 0, 0, 0]


def test_los_tres_bloques_son_columnas_distintas(app):
    """Si dos banderas salieran idénticas, estaríamos midiendo el mismo eje dos veces."""
    con = anadir_banderas_ausencia(app)
    banderas = con[["HAS_BUILDING_INFO", "HAS_BUREAU_INFO", "HAS_SOCIAL_INFO"]]
    assert not banderas["HAS_BUILDING_INFO"].equals(banderas["HAS_BUREAU_INFO"])
    assert not banderas["HAS_BUREAU_INFO"].equals(banderas["HAS_SOCIAL_INFO"])


# --- ratios ----------------------------------------------------------------------------------


def test_la_edad_sale_del_parametro_y_no_de_un_literal(app):
    con = anadir_ratios(app)
    esperado = -app["DAYS_BIRTH"] / valor("dias_por_anio")
    assert np.allclose(con["AGE_YEARS"], esperado)
    assert (con["AGE_YEARS"] > 0).all()


def test_el_ratio_de_empleo_propaga_el_nulo_del_centinela(app):
    """Sin dato de antigüedad el ratio es nulo, no cero: no es que lleve cero tiempo empleado."""
    con = anadir_ratios(app)
    assert con.loc[1, "EMPLOYED_TO_AGE_RATIO"] != con.loc[1, "EMPLOYED_TO_AGE_RATIO"]  # NaN
    assert int(con["EMPLOYED_TO_AGE_RATIO"].isna().sum()) == 1
    assert (con["EMPLOYED_TO_AGE_RATIO"].dropna() >= 0).all()


def test_el_ltv_no_produce_infinitos_con_denominador_cero(app):
    """El cliente 6 tiene precio 0: tiene que salir nulo, no infinito."""
    con = anadir_ratios(app)
    assert not np.isinf(con["LTV"].dropna()).any()
    assert con["LTV"].isna().sum() == 2  # el nulo del cliente 4 y el cero del 6
    assert con.loc[0, "LTV"] == 1.0


# --- contrato general ------------------------------------------------------------------------


def test_no_toca_ninguna_columna_de_entrada(app):
    con = construir_features_capa1(app)
    for c in app.columns:
        assert con[c].equals(app[c]), f"{c} cambió al construir las features"


def test_es_idempotente(app):
    una = construir_features_capa1(app)
    dos = construir_features_capa1(una)
    assert una.equals(dos)


def test_no_revienta_si_faltan_bloques_enteros(app):
    """application_test y la API pueden llegar sin la mitad de estas columnas."""
    pelado = app[["SK_ID_CURR", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE"]]
    con = construir_features_capa1(pelado)
    assert "AGE_YEARS" in con.columns and "LTV" in con.columns
    assert "HAS_BUILDING_INFO" not in con.columns
    assert "EMPLOYED_TO_AGE_RATIO" not in con.columns


def test_el_informe_declara_el_denominador_de_cada_bandera(app):
    inf = informe_capa1(app, construir_features_capa1(app)).set_index("feature")
    assert inf.loc["BUILDING_INFO_COUNT", "columnas de origen"] == 2
    assert inf.loc["HAS_BUREAU_INFO", "columnas de origen"] == 2
    assert inf.loc["HAS_SOCIAL_INFO", "columnas de origen"] == 2
    assert inf.loc["AGE_YEARS", "% cobertura"] == 100.0


# --- disciplina de capas ---------------------------------------------------------------------


def test_la_capa1_solo_consume_parametros_de_dominio():
    """Un `estimado` o un `medido` dentro de la capa 1 sería un parámetro sin ajustar.

    La capa 1 corre sobre la tabla entera, antes del split, así que si consumiera un corte
    reajustable estaría usando la cifra del EDA medida sobre el conjunto completo. Hoy
    `valor()` revienta con esos, pero eso protege por accidente: esto lo comprueba a propósito.
    """
    import re

    from src.config import RAIZ
    from src.features.params import parametro

    usados = set()
    for modulo in ("cleaning.py", "application.py"):
        fuente = (RAIZ / "src" / "features" / modulo).read_text()
        usados |= set(re.findall(r'valor\("([a-z0-9_]+)"\)', fuente))

    assert usados, "no se encontró ninguna llamada a valor(), revisar el patrón"
    for nombre in sorted(usados):
        assert parametro(nombre).procedencia == "dominio", (
            f"la capa 1 consume {nombre!r}, que es "
            f"{parametro(nombre).procedencia!r} y no está ajustado sobre el split"
        )
