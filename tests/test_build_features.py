"""Test de integración de la capa 1 contra el dato real (src/features/build_features.py).

Los demás tests del bloque 1 corren sobre frames sintéticos, que verifican la lógica pero no
las cifras. Este corre el camino completo, carga, limpieza y partición, sobre
application_train de verdad, y fija la puerta de salida del punto 1.1.

Lo que protege por encima de las cifras es **el orden de las capas**. La limpieza es capa 1 y
va antes del split: si alguien los invierte, el fichero de partición pasa a declarar 19
clientes que no existen en la matriz, y eso es lo que detecta el test de cobertura exacta.

Se salta entero si no está el csv, que no viaja con el repo.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import RAIZ
from src.data.loader import load_table
from src.features.build_features import cargar_y_limpiar, informe_base, preparar_application
from src.features.cleaning import filas_a_eliminar
from src.features.split import cargar_split, mascara

# --- puerta de salida del punto 1.1, medida contra el dato real ------------------------------
FILAS_CRUDAS = 307_511
FILAS_LIMPIAS = 307_492
COLUMNAS_CRUDAS = 122
COLUMNAS_LIMPIAS = 90
CENTINELA_MARCADO = 55_374
POSITIVOS = 24_825
TASA_CRUDA = 8.0729
TASA_LIMPIA = 8.0734

# desglose real de las filas eliminadas, que no es el del frame sintético de test_cleaning
MOTIVOS = 5
FILAS_BRUTAS = 21
FILAS_NETAS = 19
SOLAPES = 2

sin_csv = pytest.mark.skipif(
    not (RAIZ / "data" / "raw" / "application_train.csv").exists(),
    reason="data/raw no viaja con el repo",
)
pytestmark = sin_csv


@pytest.fixture(scope="module")
def cruda():
    return load_table("application_train", reduce_memory=False)


@pytest.fixture(scope="module")
def limpia():
    """Solo limpieza: es lo que mide la puerta del punto 1.1."""
    return cargar_y_limpiar()


@pytest.fixture(scope="module")
def base():
    """Capa 1 entera, limpieza más features: la puerta del punto 1.2."""
    return preparar_application()


def test_la_limpieza_reproduce_la_puerta_de_salida(cruda, limpia):
    assert len(cruda) == FILAS_CRUDAS
    assert cruda.shape[1] == COLUMNAS_CRUDAS
    assert len(limpia) == FILAS_LIMPIAS
    assert limpia.shape[1] == COLUMNAS_LIMPIAS


def test_el_desglose_de_filas_eliminadas_es_el_del_dato_real(cruda):
    """Cinco motivos, 21 brutas y 19 netas con 2 solapes."""
    from src.features.cleaning import FILAS_POR_CATEGORIA, FILAS_POR_NULO

    bruto = sum(
        int(cruda[c].eq(v).sum())
        for c, v in [("CODE_GENDER", "XNA"), ("NAME_FAMILY_STATUS", "Unknown")]
    ) + sum(int(cruda[c].isna().sum()) for c in FILAS_POR_NULO)
    neto = int(filas_a_eliminar(cruda).sum())

    assert len(FILAS_POR_CATEGORIA) + len(FILAS_POR_NULO) == MOTIVOS
    assert bruto == FILAS_BRUTAS
    assert neto == FILAS_NETAS
    assert bruto - neto == SOLAPES


def test_el_centinela_marca_los_mismos_de_siempre(limpia):
    assert int(limpia["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()) == CENTINELA_MARCADO
    assert int(limpia["DAYS_EMPLOYED"].isna().sum()) == CENTINELA_MARCADO
    assert not limpia["DAYS_EMPLOYED"].eq(365243).any()
    assert pd.api.types.is_numeric_dtype(limpia["DAYS_EMPLOYED"]), "no puede volverse object"


def test_la_limpieza_no_se_lleva_ni_un_positivo(cruda, limpia):
    """Las 19 filas eliminadas son todas TARGET a 0, así que solo baja el denominador."""
    assert int(cruda["TARGET"].sum()) == POSITIVOS
    assert int(limpia["TARGET"].sum()) == POSITIVOS
    assert round(cruda["TARGET"].mean() * 100, 4) == TASA_CRUDA
    assert round(limpia["TARGET"].mean() * 100, 4) == TASA_LIMPIA


def test_el_informe_de_la_puerta_cuadra(cruda, limpia):
    inf = informe_base(cruda, limpia).set_index("medida")
    assert inf.loc["filas", "limpia"] == FILAS_LIMPIAS
    assert inf.loc["columnas", "limpia"] == COLUMNAS_LIMPIAS
    assert inf.loc["filas eliminadas (netas)", "cruda"] == FILAS_NETAS
    assert inf.loc["% default", "limpia"] == TASA_LIMPIA
    assert inf.loc["centinela marcado", "limpia"] == CENTINELA_MARCADO


# --- lo que de verdad protege el orden -------------------------------------------------------


def test_el_split_cubre_exactamente_la_poblacion_limpia(limpia):
    """Si alguien parte antes de limpiar, el split declara 19 clientes que no están en la matriz."""
    split = cargar_split()
    assert set(split["SK_ID_CURR"]) == set(limpia["SK_ID_CURR"]), (
        "la partición y la tabla limpia no cubren los mismos clientes: probablemente el split "
        "se construyó sobre la tabla cruda, o sea antes de la limpieza"
    )
    assert len(split) == FILAS_LIMPIAS


def test_las_dos_partes_suman_la_poblacion_limpia():
    split = cargar_split()
    train, valid = mascara(split, "train"), mascara(split, "valid")
    assert int(train.sum()) + int(valid.sum()) == FILAS_LIMPIAS
    assert int(split["TARGET"].sum()) == POSITIVOS


def test_ninguna_fila_eliminada_sobrevive_en_el_split(cruda):
    """Los 19 identificadores que la limpieza quita no pueden estar en la partición."""
    eliminados = set(cruda.loc[filas_a_eliminar(cruda), "SK_ID_CURR"])
    assert len(eliminados) == FILAS_NETAS
    assert not eliminados & set(cargar_split()["SK_ID_CURR"])


# --- puerta de salida del punto 1.2, las features de capa 1 ----------------------------------
# La tripartita del bloque edificio es la cifra que la fija. El EDA la midió sobre 43 columnas
# numéricas y tras la limpieza solo sobreviven 15, así que lo que se comprueba aquí es que la
# reducción no cambia ni un grupo.
COMPLETO, PARCIAL, TODO_NULO = 6.96, 7.03, 9.23
N_COMPLETO, N_PARCIAL, N_TODO_NULO = 81_548, 77_787, 148_157
COLUMNAS_CON_FEATURES = 99


def test_la_tripartita_del_bloque_edificio_reproduce_las_cifras_del_eda(base):
    n_cols = int(base["BUILDING_INFO_COUNT"].max())
    grupo = np.select(
        [base["BUILDING_INFO_COUNT"].eq(n_cols), base["BUILDING_INFO_COUNT"].eq(0)],
        ["completo", "todo nulo"],
        default="parcial",
    )
    tab = pd.DataFrame({"g": grupo, "t": base["TARGET"]}).groupby("g")["t"].agg(["size", "mean"])
    tasas = (tab["mean"] * 100).round(2)

    assert n_cols == 15, "el bloque edificio ya no tiene las 15 columnas que sobreviven"
    assert [tasas["completo"], tasas["parcial"], tasas["todo nulo"]] == [
        COMPLETO,
        PARCIAL,
        TODO_NULO,
    ]
    assert [
        tab.loc["completo", "size"],
        tab.loc["parcial", "size"],
        tab.loc["todo nulo", "size"],
    ] == [
        N_COMPLETO,
        N_PARCIAL,
        N_TODO_NULO,
    ]


def test_las_banderas_de_ausencia_reproducen_su_cobertura(base):
    assert int(base["HAS_BUILDING_INFO"].eq(0).sum()) == N_TODO_NULO
    assert int(base["HAS_BUREAU_INFO"].eq(0).sum()) == 41_516
    assert int(base["FLAG_EXT_SOURCE_1_NULL"].sum()) == 173_370
    assert int(base["FLAG_EXT_SOURCE_3_NULL"].sum()) == 60_962
    assert round(base["FLAG_EXT_SOURCE_1_NULL"].mean() * 100, 2) == 56.38
    assert round(base["FLAG_EXT_SOURCE_3_NULL"].mean() * 100, 2) == 19.83


def test_el_nulo_de_las_seis_consultas_al_buro_es_el_mismo(base):
    """Es lo que justifica una sola bandera para el bloque y no seis."""
    from src.features.application import columnas_buro

    cols = columnas_buro(base)
    assert len(cols) == 6
    assert base[cols].isna().nunique(axis=1).eq(1).all()


def test_los_ratios_de_capa1_no_inventan_datos(base):
    assert int(base["EMPLOYED_TO_AGE_RATIO"].isna().sum()) == CENTINELA_MARCADO
    assert not np.isinf(base["LTV"].dropna()).any()
    assert base["AGE_YEARS"].between(20, 70).all()
    assert base.shape[1] == COLUMNAS_CON_FEATURES


def test_la_capa1_no_toca_el_recuento_de_filas(limpia, base):
    """Las features añaden columnas y no quitan clientes."""
    assert len(base) == len(limpia) == FILAS_LIMPIAS
    assert base.shape[1] == limpia.shape[1] + 9
