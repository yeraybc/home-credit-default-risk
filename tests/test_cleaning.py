"""Tests de la limpieza determinista de application_train (src/features/cleaning.py).

La limpieza es capa 1: corre fuera del split y sin estimar nada. Lo que estos tests protegen
es que siga siendo determinista (nada de percentiles colándose), que el recuento cuadre, y que
las tres cosas que el bloque edificio tiene que conservar no se vayan por delante con el resto.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.cleaning import (
    COLUMNAS_FIRMES,
    COLUMNAS_PROVISIONALES,
    FILAS_POR_CATEGORIA,
    FILAS_POR_NULO,
    aplicar_centinela,
    columnas_a_eliminar,
    columnas_edificio_redundantes,
    columnas_pendientes_de_decidir,
    filas_a_eliminar,
    informe_limpieza,
    limpiar_application,
    limpiar_application_entrenamiento,
)
from src.features.params import valor

CENTINELA = 365243


@pytest.fixture
def app():
    """Frame sintético con un caso de cada cosa que la limpieza tiene que tratar."""
    n = 12
    base = pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n + 1),
            "TARGET": [0] * n,
            "CODE_GENDER": ["M"] * n,
            "NAME_FAMILY_STATUS": ["Married"] * n,
            "CNT_FAM_MEMBERS": [2.0] * n,
            "DAYS_LAST_PHONE_CHANGE": [-100.0] * n,
            "AMT_ANNUITY": [1000.0] * n,
            "DAYS_EMPLOYED": [-500] * n,
            "AMT_REQ_CREDIT_BUREAU_DAY": [0.0] * n,
            # columnas que se eliminan
            "FLAG_MOBIL": [1] * n,
            "FLAG_CONT_MOBILE": [1] * n,
            "FLAG_EMP_PHONE": [1] * n,
            "OBS_60_CNT_SOCIAL_CIRCLE": [0.0] * n,
            "DEF_60_CNT_SOCIAL_CIRCLE": [0.0] * n,
            "OBS_30_CNT_SOCIAL_CIRCLE": [0.0] * n,
            "DEF_30_CNT_SOCIAL_CIRCLE": [0.0] * n,
            # bloque edificio: un concepto con las tres versiones
            "APARTMENTS_AVG": [0.1] * n,
            "APARTMENTS_MODE": [0.1] * n,
            "APARTMENTS_MEDI": [0.1] * n,
            # el concepto 15, sin pareja, y una categórica del bloque
            "TOTALAREA_MODE": [0.2] * n,
            "HOUSETYPE_MODE": ["block of flats"] * n,
        }
    )
    base.loc[0, "CODE_GENDER"] = "XNA"
    base.loc[1, "NAME_FAMILY_STATUS"] = "Unknown"
    base.loc[2, "CNT_FAM_MEMBERS"] = np.nan
    base.loc[3, "DAYS_LAST_PHONE_CHANGE"] = np.nan
    base.loc[4, "AMT_ANNUITY"] = np.nan
    base.loc[5, "CODE_GENDER"] = "XNA"  # solape: además tiene un nulo
    base.loc[5, "AMT_ANNUITY"] = np.nan
    base.loc[6, "DAYS_EMPLOYED"] = CENTINELA
    base.loc[7, "AMT_REQ_CREDIT_BUREAU_DAY"] = 9.0
    return base


def test_las_filas_a_eliminar_resuelven_el_solape(app):
    """En este frame sintético, siete motivos repartidos en seis filas: una lo es por dos.

    Sobre application_train el reparto es otro y son cinco motivos, 21 filas brutas y 19
    netas con 2 solapes; lo fija el test de integración contra el dato real.
    """
    fuera = filas_a_eliminar(app)
    assert int(fuera.sum()) == 6
    assert set(app.loc[fuera, "SK_ID_CURR"]) == {1, 2, 3, 4, 5, 6}


def test_el_informe_declara_por_separado_lo_bruto_y_lo_neto(app):
    inf = informe_limpieza(app)
    bruto = inf[(inf.tipo == "fila") & (inf.objeto != "unión, con solapes resueltos")]["n"].sum()
    neto = inf[inf.objeto == "unión, con solapes resueltos"]["n"].iloc[0]
    assert bruto == 7 and neto == 6, "el informe tiene que enseñar el solape, no esconderlo"


def test_el_bloque_edificio_conserva_avg_totalarea_y_las_categoricas(app):
    limpio = limpiar_application(app)
    assert "APARTMENTS_AVG" in limpio.columns
    assert "APARTMENTS_MODE" not in limpio.columns
    assert "APARTMENTS_MEDI" not in limpio.columns
    # los dos que terminan en _MODE y no son del grupo de tres versiones
    assert "TOTALAREA_MODE" in limpio.columns, "el concepto sin pareja no se elimina"
    assert "HOUSETYPE_MODE" in limpio.columns, "las categóricas del bloque esperan a su IV"


def test_solo_se_marcan_como_redundantes_los_conceptos_que_tienen_avg(app):
    redundantes = columnas_edificio_redundantes(app)
    assert redundantes == ["APARTMENTS_MEDI", "APARTMENTS_MODE"]


def test_el_centinela_se_marca_antes_de_borrarse(app):
    marcado = aplicar_centinela(app)
    fila = marcado["DAYS_EMPLOYED"].isna()
    assert int(fila.sum()) == 1
    assert marcado.loc[fila, "FLAG_DAYS_EMPLOYED_ANOMALY"].eq(1).all()
    assert int(marcado["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()) == 1
    assert not (marcado["DAYS_EMPLOYED"] == CENTINELA).any()


def test_el_centinela_sale_de_params_y_no_de_un_literal():
    assert valor("centinela_365243") == CENTINELA


def test_el_cap_de_dominio_recorta_por_arriba_y_no_toca_lo_demas(app):
    limpio = limpiar_application(app)
    assert limpio["AMT_REQ_CREDIT_BUREAU_DAY"].max() == valor("app_amt_req_bureau_day_max")
    assert limpio["AMT_REQ_CREDIT_BUREAU_DAY"].min() == 0


def test_el_recuento_de_columnas_cuadra(app):
    limpio = limpiar_application(app)
    esperado = app.shape[1] - len(columnas_a_eliminar(app)) + 1  # la bandera del centinela
    assert limpio.shape[1] == esperado


def test_la_limpieza_no_reordena_ni_duplica(app):
    limpio = limpiar_application_entrenamiento(app)
    assert limpio["SK_ID_CURR"].is_unique
    assert limpio["SK_ID_CURR"].is_monotonic_increasing
    assert list(limpio.index) == list(range(len(limpio)))


def test_es_idempotente(app):
    """Volver a limpiar lo ya limpio no puede quitar nada más ni borrar la bandera.

    En la segunda pasada el centinela ya es NaN, así que recalcular la bandera la pondría a
    cero: la información de qué cliente estaba inactivo se perdería sin que nada fallase.
    """
    una = limpiar_application(app)
    dos = limpiar_application(una)
    assert una.shape == dos.shape
    assert int(una["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()) == 1
    assert int(dos["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()) == 1
    assert una.equals(dos)


def test_no_revienta_si_faltan_columnas(app):
    """application_test no trae TARGET, y otras tablas no traen la mitad de estas columnas."""
    sin_target = app.drop(columns=["TARGET", "FLAG_MOBIL", "OBS_60_CNT_SOCIAL_CIRCLE"])
    limpio = limpiar_application(sin_target)
    assert "FLAG_DAYS_EMPLOYED_ANOMALY" in limpio.columns
    assert len(limpio) == len(sin_target), "la limpieza de inferencia no puede perder clientes"


def test_todo_lo_que_se_elimina_declara_su_motivo(app):
    for col, motivo in columnas_a_eliminar(app).items():
        assert motivo.strip(), f"{col} se elimina sin motivo declarado"
    for motivo in {**FILAS_POR_CATEGORIA, **FILAS_POR_NULO, **COLUMNAS_FIRMES}.values():
        assert motivo.strip()


# --- la ruta de inferencia no puede perder clientes ------------------------------------------


def test_la_limpieza_base_no_toca_el_numero_de_filas(app):
    """Es la que corre sobre application_test y sobre la petición de la API."""
    limpio = limpiar_application(app)
    assert len(limpio) == len(app)
    assert set(limpio["SK_ID_CURR"]) == set(app["SK_ID_CURR"])


def test_la_limpieza_base_conserva_el_indice_de_quien_llama(app):
    """En serving la predicción se alinea de vuelta contra la petición por el índice."""
    desplazado = app.set_index(pd.Index(range(500, 500 + len(app))))
    limpio = limpiar_application(desplazado)
    assert list(limpio.index) == list(desplazado.index)


def test_la_limpieza_de_entrenamiento_si_quita_las_filas(app):
    limpio = limpiar_application_entrenamiento(app)
    assert len(limpio) == len(app) - 6


def test_quitar_filas_sin_target_revienta_en_vez_de_perder_clientes(app):
    """El caso real: application_test trae 24 nulos de AMT_ANNUITY y se irían en silencio."""
    with pytest.raises(ValueError, match="ningún cliente puede desaparecer"):
        limpiar_application_entrenamiento(app.drop(columns=["TARGET"]))


# --- descartes provisionales: los decide la capa 2b, no esta ---------------------------------


def test_las_provisionales_sobreviven_a_la_capa_1(app):
    """Si se eliminaran aquí, la 2b no tendría la columna que tiene que juzgar."""
    limpio = limpiar_application(app)
    for col in COLUMNAS_PROVISIONALES:
        assert col in limpio.columns, f"{col} se descartó contra el TARGET y no puede caer aquí"


def test_ninguna_provisional_se_declara_tambien_como_firme():
    assert not set(COLUMNAS_PROVISIONALES) & set(COLUMNAS_FIRMES)


def test_cada_provisional_declara_que_hay_que_remedir(app):
    pendientes = columnas_pendientes_de_decidir(app)
    assert set(pendientes) == set(COLUMNAS_PROVISIONALES)
    for col, descarte in pendientes.items():
        assert "solo_train" in descarte.remedir, f"{col} no dice qué remedir sobre el split"


def test_el_informe_declara_las_aplazadas_aparte_de_las_eliminadas(app):
    inf = informe_limpieza(app)
    aplazadas = set(inf.loc[inf.tipo == "columna aplazada", "objeto"])
    eliminadas = set(inf.loc[inf.tipo == "columna", "objeto"])
    assert aplazadas == set(COLUMNAS_PROVISIONALES)
    assert not aplazadas & eliminadas


def test_el_informe_no_revienta_si_faltan_columnas_de_fila(app):
    """La limpieza tolera que falten; el informe tenía guardas solo en una de las dos listas."""
    inf = informe_limpieza(app.drop(columns=["CODE_GENDER", "AMT_ANNUITY"]))
    assert not inf.empty
