"""Tests de la limpieza determinista de application_train (src/features/cleaning.py).

La limpieza es capa 1: corre fuera del split y sin estimar nada. Lo que estos tests protegen
es que siga siendo determinista (nada de percentiles colándose), que el recuento cuadre, y que
las tres cosas que el bloque edificio tiene que conservar no se vayan por delante con el resto.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.cleaning import (
    CAPS_BUREAU,
    COL_BB_DPD,
    COL_BB_IS_DPD,
    COL_BB_IS_X,
    COL_MONEDA_EXTRANJERA,
    COLUMNAS_FIRMES,
    COLUMNAS_PROVISIONALES,
    DTYPE_STATUS,
    FILAS_POR_CATEGORIA,
    FILAS_POR_NULO,
    IMPORTES_BUREAU,
    IMPORTES_CON_FOTO,
    MESES_BB,
    STATUS_DPD,
    SUFIJO_SIGNO,
    acotar_ventanas_bureau,
    aplicar_caps_bureau,
    aplicar_centinela,
    capar_deuda_al_credito,
    columnas_a_eliminar,
    columnas_edificio_redundantes,
    columnas_pendientes_de_decidir,
    filas_a_eliminar,
    fotografiar_signo,
    informe_limpieza,
    limpiar_application,
    limpiar_application_entrenamiento,
    limpiar_bureau,
    limpiar_bureau_balance,
    marcar_moneda_extranjera,
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


# las dos que se quedan sin cap, y no es lo mismo en las dos: YEAR porque su cola tiene señal
# real, HOUR porque su máximo es 4 y no llega al cap de 5 que lleva su gemela DAY
SIN_CAP_DECLARADO = {"AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_YEAR"}


def test_cada_ventana_del_buro_tiene_una_disposicion_y_solo_una():
    """Son seis columnas gemelas y HOUR no tenía decisión escrita en ninguna de las listas.

    El riesgo no es el cap que falta, que hoy no recortaría nada: es que una de las seis se
    caiga de su lista sin que nadie lo note, porque se parecen entre sí.
    """
    from src.features.application import COLUMNAS_BURO
    from src.features.cleaning import CAPS_DE_DOMINIO
    from src.features.transformers import CORTES_WINSOR

    winsorizadas = set(CORTES_WINSOR) & set(COLUMNAS_BURO)
    capadas = set(CAPS_DE_DOMINIO) & set(COLUMNAS_BURO)

    assert winsorizadas == {
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
    }
    assert capadas == {"AMT_REQ_CREDIT_BUREAU_DAY"}
    assert not winsorizadas & capadas, "una misma columna con dos tratamientos"
    assert set(COLUMNAS_BURO) - winsorizadas - capadas == SIN_CAP_DECLARADO


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


# --- bureau -------------------------------------------------------------------
# La limpieza de bureau es la misma clase de cosa que la de arriba, con dos diferencias que los
# tests vigilan: aquí no se borra ni una fila, porque la tabla no tiene TARGET propio y el mismo
# frame lo recorren entrenamiento y la API, y todo va a nivel fila **antes** de agregar, porque
# capar la deuda después de sumarla no arregla la suma.

# Los tres salen de params, no de un literal, por lo mismo que los caps de importe: escritos a
# mano serían el mismo corte en dos sitios, que es lo que dejó 52.500 frente a 52.497 en bureau.
# Se evalúan al importar y `test_los_cortes_de_ventana_del_fixture_salen_de_params` comprueba que
# siguen cuadrando, que es lo que un módulo de constantes no puede hacer por sí solo.
DIAS_POR_ANIO = valor("dias_por_anio")
TOPE_ENDDATE = valor("bureau_enddate_max_anios") * DIAS_POR_ANIO
SUELO_CIERRE = -valor("bureau_cierre_max_anios") * DIAS_POR_ANIO


@pytest.fixture
def bureau():
    """Un crédito por cada rama de la limpieza, con su vecino justo al otro lado del corte.

    Las parejas son deliberadas: por cada fila que cruza un corte hay otra que se queda
    exactamente en él y no se puede tocar. Sin ellas un `>=` por un `>` pasaría en verde.
    """
    filas = [
        # cliente 1: nacional normal, y su segunda fila en moneda extranjera (mezcla)
        (1, "currency 1", 100_000.0, 50_000.0, 0.0, 1_000.0, 0.0, 365.0, -100.0, "Active"),
        (1, "currency 2", 9_000_000.0, 8_000_000.0, 0.0, 500_000.0, 0.0, 200.0, -50.0, "Closed"),
        # cliente 2: su único crédito es extranjero, no puede desaparecer
        (2, "currency 3", 5_000_000.0, 1_000.0, 0.0, 100.0, 0.0, 100.0, -10.0, "Active"),
        # cliente 3: importe que cruza los 50M, y el que se queda justo en el corte
        (3, "currency 1", 60_000_000.0, 1_000.0, 0.0, 1_000.0, 0.0, 300.0, -20.0, "Active"),
        (3, "currency 1", 50_000_000.0, 1_000.0, 0.0, 1_000.0, 0.0, 300.0, -20.0, "Active"),
        # cliente 4: cuota que cruza los 10M, y la que se queda justo en el corte
        (4, "currency 1", 100_000.0, 1_000.0, 0.0, 20_000_000.0, 0.0, 300.0, -20.0, "Active"),
        (4, "currency 1", 100_000.0, 1_000.0, 0.0, 10_000_000.0, 0.0, 300.0, -20.0, "Active"),
        # cliente 5: exceso grosero de deuda (ratio 10), exceso leve (ratio 2) y el ratio exacto
        (5, "currency 1", 1_000.0, 10_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Active"),
        (5, "currency 1", 1_000.0, 2_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Active"),
        (5, "currency 1", 1_000.0, 3_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Active"),
        # cliente 6: denominador cero con deuda positiva, que no se toca
        (6, "currency 1", 0.0, 5_000.0, 0.0, 100.0, 0.0, 300.0, SUELO_CIERRE, "Closed"),
        # cliente 7: vencimiento fuera de ventana y límite negativo, el sobregiro que se conserva
        (
            7,
            "currency 1",
            100_000.0,
            1_000.0,
            -5_000.0,
            100.0,
            0.0,
            TOPE_ENDDATE + 1,
            -20.0,
            "Active",
        ),
        # cliente 8: mora que cruza los 50M, cierre fuera de ventana y vencimiento justo en el tope
        (
            8,
            "currency 1",
            100_000.0,
            1_000.0,
            0.0,
            100.0,
            6e7,
            TOPE_ENDDATE,
            SUELO_CIERRE - 1,
            "Closed",
        ),
        # cliente 9: deuda rota que cruza los 50M y a la vez el ratio, que es donde el orden decide
        (9, "currency 1", 1_000.0, 60_000_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Active"),
        # cliente 10: deuda negativa, o sea sobrepago; son 8.418 filas reales y no se tocan
        (10, "currency 1", 100_000.0, -5_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Closed"),
        # cliente 11: vencimiento por debajo del suelo, la cola que enciende en falso
        # BUREAU_CLOSED_AFTER_ENDDATE. Cierra de verdad, así que sin acotar el vencimiento el
        # cierre le queda posterior y la bandera sale a 1 por un dato roto
        (11, "currency 1", 100_000.0, 1_000.0, 0.0, 100.0, 0.0, SUELO_CIERRE - 1, -20.0, "Closed"),
        # cliente 12: el vencimiento justo en el suelo, que no se toca
        (12, "currency 1", 100_000.0, 1_000.0, 0.0, 100.0, 0.0, SUELO_CIERRE, -20.0, "Closed"),
        # cliente 13: crédito negativo con un sobrepago que lo quintuplica. Es la única forma de
        # separar el denominador bueno del que solo excluye el cero: con los dos importes negativos
        # el ratio sale +5 y cruza el corte, así que un `!= 0` capa la deuda al crédito y reescribe
        # un sobrepago apoyándose en un principal roto. En bureau.csv no hay ni una fila así, que
        # es justo por lo que hace falta aquí: por la API sí puede llegar
        (13, "currency 1", -1_000.0, -5_000.0, 0.0, 100.0, 0.0, 300.0, -20.0, "Closed"),
    ]
    columnas = [
        "SK_ID_CURR",
        "CREDIT_CURRENCY",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_ANNUITY",
        "AMT_CREDIT_MAX_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "CREDIT_ACTIVE",
    ]
    df = pd.DataFrame(filas, columns=columnas)
    df.insert(1, "SK_ID_BUREAU", range(1001, 1001 + len(df)))
    df["AMT_CREDIT_SUM_OVERDUE"] = 0.0
    df["DAYS_CREDIT"] = -500.0
    # La actualización del registro va fuera de la tupla porque solo dos filas la ejercitan: por
    # defecto es reciente, y el par que cruza el suelo y el que se queda en él se pone a mano.
    df["DAYS_CREDIT_UPDATE"] = -30.0
    df.loc[df.SK_ID_CURR == 11, "DAYS_CREDIT_UPDATE"] = SUELO_CIERRE - 1
    df.loc[df.SK_ID_CURR == 12, "DAYS_CREDIT_UPDATE"] = SUELO_CIERRE
    return df


def credito_de(df, cliente):
    """Las filas de un cliente, exigiendo que haya alguna.

    Existe porque `df[df.SK_ID_CURR == n].X.isna().all()` es **verdadero sobre selección vacía**:
    el día que el fixture pierda a ese cliente, media docena de tests de abajo pasarían sin medir
    nada. Hoy lo cazaría el guardián, pero eso los deja apoyados en otro test en vez de en sí
    mismos, y es una línea evitarlo.
    """
    filas = df[df.SK_ID_CURR == cliente]
    assert not filas.empty, f"el cliente {cliente} no está en el frame: el test no mide nada"
    return filas


def test_el_fixture_de_bureau_ejercita_cada_rama(bureau):
    """Guardián: si el fixture pierde un caso, los tests de abajo dejan de probar lo que dicen."""
    extranjera = bureau.CREDIT_CURRENCY.ne("currency 1")
    assert (
        extranjera.sum() == 2
    ), "hacen falta la fila de cliente con mezcla y la del cliente solo extranjero"
    assert bureau.loc[extranjera, "SK_ID_CURR"].nunique() == 2
    assert (bureau.AMT_CREDIT_SUM > valor("bureau_importe_max")).sum() == 1
    assert (
        bureau.AMT_CREDIT_SUM == valor("bureau_importe_max")
    ).sum() == 1, "falta el borde del importe"
    assert (bureau.AMT_ANNUITY > valor("bureau_cuota_max")).sum() == 1
    assert (
        bureau.AMT_ANNUITY == valor("bureau_cuota_max")
    ).sum() == 1, "falta el borde de la cuota"
    assert (bureau.AMT_CREDIT_MAX_OVERDUE > valor("bureau_importe_max")).sum() == 1
    ratio = bureau.AMT_CREDIT_SUM_DEBT / bureau.AMT_CREDIT_SUM.where(bureau.AMT_CREDIT_SUM > 0)
    # las dos filas que cruzan el ratio se separan por si la deuda es además un importe roto:
    # la sana se capa al crédito y la rota se anula, y son ramas distintas
    rota = bureau.AMT_CREDIT_SUM_DEBT > valor("bureau_importe_max")
    ratio_max = valor("bureau_ratio_deuda_credito_max")
    assert ((ratio > ratio_max) & ~rota).sum() == 1, "falta el exceso grosero que sí se capa"
    assert (
        (ratio > ratio_max) & rota
    ).sum() == 1, "falta la fila donde el orden de los dos caps decide"
    assert (ratio == ratio_max).sum() == 1, "falta el borde del ratio de deuda"
    assert ((ratio > 1) & (ratio < ratio_max)).sum() == 1, "falta el exceso leve, que se conserva"
    assert ((bureau.AMT_CREDIT_SUM == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0)).sum() == 1
    assert (bureau.DAYS_CREDIT_ENDDATE > TOPE_ENDDATE).sum() == 1
    assert (bureau.DAYS_CREDIT_ENDDATE == TOPE_ENDDATE).sum() == 1, "falta el borde del vencimiento"
    assert (bureau.DAYS_ENDDATE_FACT < SUELO_CIERRE).sum() == 1
    assert (bureau.DAYS_ENDDATE_FACT == SUELO_CIERRE).sum() == 1, "falta el borde del cierre"
    # las dos colas por abajo que la primera versión del punto se dejó fuera
    assert (
        bureau.DAYS_CREDIT_ENDDATE < SUELO_CIERRE
    ).sum() == 1, "falta el vencimiento pasado, que enciende la bandera de cierre tardío en falso"
    assert (
        bureau.DAYS_CREDIT_ENDDATE == SUELO_CIERRE
    ).sum() == 1, "falta el borde del vencimiento pasado"
    assert (bureau.DAYS_CREDIT_UPDATE < SUELO_CIERRE).sum() == 1
    assert (
        bureau.DAYS_CREDIT_UPDATE == SUELO_CIERRE
    ).sum() == 1, "falta el borde de la actualización"
    # el cierre tardío que el vencimiento roto fabrica: sin acotarlo la bandera sale a 1
    roto = bureau.DAYS_CREDIT_ENDDATE < SUELO_CIERRE
    assert (
        bureau.loc[roto, "DAYS_ENDDATE_FACT"] > bureau.loc[roto, "DAYS_CREDIT_ENDDATE"]
    ).all(), "el vencimiento roto tiene que ir con un cierre real posterior, o no prueba nada"
    assert (bureau.AMT_CREDIT_SUM_LIMIT < 0).sum() == 1, "falta el sobregiro, la bandera más fuerte"
    sano = bureau.AMT_CREDIT_SUM > 0
    assert (
        (bureau.AMT_CREDIT_SUM_DEBT < 0) & sano
    ).sum() == 1, "falta el sobrepago, que tampoco se toca"
    # el denominador del ratio solo cuenta donde es positivo, y sin una fila de crédito negativo
    # cuyo ratio crudo cruce el corte, excluir solo el cero pasa en verde: en bureau.csv no hay
    # ninguna, así que el caso solo existe si el fixture lo trae
    negativo = bureau.AMT_CREDIT_SUM < 0
    assert (
        negativo & (bureau.AMT_CREDIT_SUM_DEBT / bureau.AMT_CREDIT_SUM > ratio_max)
    ).sum() == 1, "falta el crédito negativo, cuyo ratio no significa nada"


def test_los_cortes_de_ventana_del_fixture_salen_de_params():
    """Las constantes del módulo se evalúan al importar, así que su valor se fija aquí.

    Sin esto, cambiar el corte en PARAMS movería a la vez el fixture y lo que se le exige, y el
    test seguiría verde midiendo otra cosa: el fixture se adaptaría al corte nuevo en silencio.
    """
    assert DIAS_POR_ANIO == 365.25
    assert TOPE_ENDDATE == 20 * 365.25
    assert SUELO_CIERRE == -30 * 365.25


def test_la_limpieza_de_bureau_no_borra_ni_una_fila(bureau):
    limpio = limpiar_bureau(bureau)
    assert len(limpio) == len(bureau)
    assert limpio.SK_ID_CURR.tolist() == bureau.SK_ID_CURR.tolist()
    assert limpio.SK_ID_BUREAU.tolist() == bureau.SK_ID_BUREAU.tolist()


def test_la_moneda_extranjera_vacia_los_importes_y_conserva_lo_demas(bureau):
    limpio = limpiar_bureau(bureau)
    extranjera = limpio[COL_MONEDA_EXTRANJERA] == 1
    assert extranjera.sum() == 2
    presentes = [c for c in IMPORTES_BUREAU if c in limpio.columns]
    assert limpio.loc[extranjera, presentes].isna().all().all()
    # lo que no es importe sobrevive: es la razón de no borrar la fila ni vaciarla entera
    assert limpio.loc[extranjera, "CREDIT_ACTIVE"].notna().all()
    assert limpio.loc[extranjera, "DAYS_CREDIT"].notna().all()
    assert limpio.loc[extranjera, "DAYS_CREDIT_ENDDATE"].notna().all()


def test_la_bandera_de_moneda_no_marca_a_quien_va_en_nacional(bureau):
    limpio = limpiar_bureau(bureau)
    nacional = limpio.CREDIT_CURRENCY.eq("currency 1")
    assert (limpio.loc[nacional, COL_MONEDA_EXTRANJERA] == 0).all()
    assert limpio.loc[nacional, "AMT_CREDIT_SUM"].notna().sum() > 0


def test_una_moneda_no_informada_cae_del_lado_extranjero(bureau):
    """Sin saber la unidad no se puede sumar, así que el nulo pierde sus importes igual.

    En `bureau.csv` no hay ni un nulo de moneda, o sea que esto no lo ejercita el dato real: la
    ruta que lo alcanza es la API. Va con test para que la disposición sea una decisión y no un
    efecto de que `ne()` diga verdad sobre un NaN.
    """
    con_nulo = bureau.copy()
    con_nulo.loc[con_nulo.SK_ID_CURR == 3, "CREDIT_CURRENCY"] = np.nan
    limpio = limpiar_bureau(con_nulo)
    cliente3 = credito_de(limpio, 3)
    assert cliente3[COL_MONEDA_EXTRANJERA].eq(1).all()
    assert cliente3[[c for c in IMPORTES_BUREAU if c in limpio.columns]].isna().all().all()


def test_el_cap_de_importe_anula_lo_que_cruza_y_respeta_el_borde(bureau):
    limpio = limpiar_bureau(bureau)
    cliente3 = credito_de(limpio, 3).sort_values("AMT_CREDIT_SUM", na_position="first")
    assert cliente3.AMT_CREDIT_SUM.isna().sum() == 1, "el de 60M tiene que irse a NaN"
    assert cliente3.AMT_CREDIT_SUM.max() == valor(
        "bureau_importe_max"
    ), "el de 50M exactos se queda"
    cliente4 = credito_de(limpio, 4)
    assert cliente4.AMT_ANNUITY.isna().sum() == 1
    assert cliente4.AMT_ANNUITY.max() == valor("bureau_cuota_max")
    assert credito_de(limpio, 8).AMT_CREDIT_MAX_OVERDUE.isna().all()


def test_el_cap_de_importe_lee_el_corte_de_params_y_no_un_literal(bureau):
    """Bajar el corte en PARAMS tiene que anular más filas. El conftest lo restaura."""
    from src.features.params import PARAMS, Parametro

    antes = int(limpiar_bureau(bureau).AMT_CREDIT_SUM.isna().sum())
    PARAMS["bureau_importe_max"] = Parametro(1_000.0, "dominio", "corte de prueba", "test")
    despues = int(limpiar_bureau(bureau).AMT_CREDIT_SUM.isna().sum())
    assert despues > antes


def test_el_exceso_grosero_de_deuda_se_capa_y_el_leve_se_conserva(bureau):
    limpio = limpiar_bureau(bureau)
    cliente5 = credito_de(limpio, 5).sort_values("SK_ID_BUREAU")
    deuda = cliente5.AMT_CREDIT_SUM_DEBT.tolist()
    assert deuda[0] == 1_000.0, "el ratio 10 se capa al propio crédito, no se anula"
    assert deuda[1] == 2_000.0, "el exceso leve es deuda real y se conserva"
    assert deuda[2] == 3_000.0, "el ratio exactamente 3 no cruza el corte"
    assert cliente5.AMT_CREDIT_SUM_DEBT.notna().all(), "capar no es anular"


def test_la_deuda_rota_se_anula_y_no_se_capa_al_credito(bureau):
    """Fija el orden: el cap de importe va antes que el de ratio, y no al revés.

    Sobre la tabla de hoy las dos reglas no se solapan en ninguna fila, así que invertirlas no
    mueve ni una cifra y ninguna puerta lo delataría. La propiedad es del código igual: una
    deuda de 60M sobre un crédito de 1.000 es dato roto y tiene que quedar en NaN, no capada a
    1.000, que la dejaría leyéndose como una deuda plausible que iguala al principal.
    """
    limpio = limpiar_bureau(bureau)
    cliente9 = credito_de(limpio, 9)
    assert cliente9.AMT_CREDIT_SUM_DEBT.isna().all(), "capada al crédito en vez de anulada"


def test_el_sobrepago_sobrevive_a_la_limpieza(bureau):
    """Deuda negativa son 8.418 filas reales sobre 5.886 clientes, y ninguna regla las toca."""
    limpio = limpiar_bureau(bureau)
    assert credito_de(limpio, 10).AMT_CREDIT_SUM_DEBT.iloc[0] == -5_000.0


def test_el_denominador_cero_no_capa_nada(bureau):
    limpio = limpiar_bureau(bureau)
    cliente6 = credito_de(limpio, 6)
    assert cliente6.AMT_CREDIT_SUM_DEBT.iloc[0] == 5_000.0
    assert cliente6.AMT_CREDIT_SUM.iloc[0] == 0.0


def test_el_denominador_negativo_tampoco_capa_nada(bureau):
    """Con el principal negativo el ratio no significa nada, aunque el cociente crudo cruce.

    Es la fila que separa `where(credito > 0)` de `where(credito != 0)`: con deuda y crédito
    negativos el cociente sale +5, y excluyendo solo el cero la deuda se caparía al crédito, o sea
    que un sobrepago de 5.000 se reescribiría a 1.000 por un principal que ya es dato roto.
    """
    limpio = limpiar_bureau(bureau)
    cliente13 = credito_de(limpio, 13)
    assert cliente13.AMT_CREDIT_SUM_DEBT.iloc[0] == -5_000.0, "capada contra un principal negativo"
    assert cliente13.AMT_CREDIT_SUM.iloc[0] == -1_000.0


def test_las_ventanas_anulan_lo_que_cae_fuera_y_respetan_el_borde(bureau):
    limpio = limpiar_bureau(bureau)
    assert credito_de(limpio, 7).DAYS_CREDIT_ENDDATE.isna().all()
    assert credito_de(limpio, 8).DAYS_CREDIT_ENDDATE.iloc[0] == TOPE_ENDDATE
    assert credito_de(limpio, 8).DAYS_ENDDATE_FACT.isna().all()
    assert credito_de(limpio, 6).DAYS_ENDDATE_FACT.iloc[0] == SUELO_CIERRE


def test_las_ventanas_acotan_tambien_por_abajo_el_vencimiento_y_la_actualizacion(bureau):
    """Las dos colas del inventario del EDA que el punto se dejó fuera en su primera versión.

    Son 146 y 95 filas sobre la tabla real, contra las 62.604 del tope y la 1 del cierre. El
    volumen no es el argumento: 95 de las 146 encienden `BUREAU_CLOSED_AFTER_ENDDATE` en falso.
    """
    limpio = limpiar_bureau(bureau)
    assert credito_de(limpio, 11).DAYS_CREDIT_ENDDATE.isna().all()
    assert credito_de(limpio, 11).DAYS_CREDIT_UPDATE.isna().all()
    assert credito_de(limpio, 12).DAYS_CREDIT_ENDDATE.iloc[0] == SUELO_CIERRE
    assert credito_de(limpio, 12).DAYS_CREDIT_UPDATE.iloc[0] == SUELO_CIERRE


def test_el_vencimiento_roto_deja_de_encender_la_bandera_de_cierre_tardio(bureau):
    """El motivo por el que la cola de abajo del vencimiento es bloqueante y no cosmética.

    `BUREAU_CLOSED_AFTER_ENDDATE` compara el cierre real contra el vencimiento planificado, así
    que un vencimiento de hace 115 años la enciende siempre. Sobre la tabla real son 95 filas y
    43 clientes que llevan una bandera de riesgo puesta por un error de captura.
    """

    def cierre_tardio(df):
        return df.CREDIT_ACTIVE.eq("Closed") & (df.DAYS_ENDDATE_FACT > df.DAYS_CREDIT_ENDDATE)

    assert cierre_tardio(bureau)[
        bureau.SK_ID_CURR == 11
    ].all(), "el fixture no monta el falso positivo"
    assert not cierre_tardio(limpiar_bureau(bureau))[bureau.SK_ID_CURR == 11].any()


def test_las_tres_fechas_con_suelo_comparten_el_mismo_corte(bureau):
    """Un solo umbral declarado una vez, que es lo que evita los 52.500 frente a 52.497.

    Bajarlo en PARAMS tiene que alcanzar a las tres a la vez. Si alguna leyera su propio literal,
    se quedaría con la cifra vieja sin que nada fallase.
    """
    from src.features.cleaning import FECHAS_CON_SUELO
    from src.features.params import PARAMS, Parametro

    antes = {c: int(limpiar_bureau(bureau)[c].isna().sum()) for c in FECHAS_CON_SUELO}
    PARAMS["bureau_cierre_max_anios"] = Parametro(0.05, "dominio", "corte de prueba", "test")
    despues = {c: int(limpiar_bureau(bureau)[c].isna().sum()) for c in FECHAS_CON_SUELO}
    for c in FECHAS_CON_SUELO:
        assert despues[c] > antes[c], f"{c} no lee el suelo de params"


def test_el_sobregiro_sobrevive_a_la_limpieza(bureau):
    """AMT_CREDIT_SUM_LIMIT negativo es BUREAU_NEGATIVE_LIMIT_FLAG, +12,72pp, la más fuerte."""
    limpio = limpiar_bureau(bureau)
    assert (limpio.AMT_CREDIT_SUM_LIMIT < 0).sum() == 1


def test_la_limpieza_de_bureau_es_idempotente(bureau):
    """Protege la foto de presencia: refotografiar sobre el importe ya anulado la borraría."""
    una = limpiar_bureau(bureau)
    dos = limpiar_bureau(una)
    pd.testing.assert_frame_equal(una, dos)


def test_la_limpieza_de_bureau_no_muta_el_frame_de_entrada(bureau):
    copia = bureau.copy()
    limpiar_bureau(bureau)
    pd.testing.assert_frame_equal(bureau, copia)


@pytest.mark.parametrize(
    "regla",
    [
        fotografiar_signo,
        marcar_moneda_extranjera,
        aplicar_caps_bureau,
        capar_deuda_al_credito,
        acotar_ventanas_bureau,
    ],
)
def test_ninguna_regla_por_separado_muta_el_frame_de_entrada(bureau, regla):
    """El test de arriba solo protege el `.copy()` de la primera regla, y son cinco.

    Encadenadas, la primera copia y escuda a las demás, así que quitarle el `.copy()` a
    cualquiera de las otras cuatro dejaba la suite entera en verde. Y no es teórico: dos de las
    cinco **retornan sin copiar** cuando les falta su columna (la moneda sin `CREDIT_CURRENCY`,
    la deuda sin los dos importes), que son las rutas parciales que la API sí recorre. Ahí el
    escudo es la siguiente que copie, y eso no puede depender del orden.
    """
    copia = bureau.copy()
    regla(bureau)
    pd.testing.assert_frame_equal(bureau, copia)


def test_la_limpieza_de_bureau_da_lo_mismo_fila_a_fila_que_sobre_la_tabla_entera(bureau):
    """La premisa que permite que la capa 1 corra fuera del split: no cruza filas.

    Hoy se cumple sola, porque las cinco reglas son comparaciones contra constantes. Justo por
    eso el test se escribe ahora: es el que se pondrá rojo el día que alguien meta una mediana,
    un percentil o un `groupby` aquí dentro, que es la fuga estructural de la que este módulo
    tiene que estar libre para poder correr sobre la tabla entera.
    """
    entera = limpiar_bureau(bureau)
    fila_a_fila = pd.concat([limpiar_bureau(bureau.iloc[[i]]) for i in range(len(bureau))])
    pd.testing.assert_frame_equal(entera, fila_a_fila)


def test_la_limpieza_de_bureau_no_revienta_si_faltan_columnas(bureau):
    """A la API puede llegar un frame parcial: quien exige el contrato es la frontera, no esto."""
    parcial = bureau[["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT"]]
    assert limpiar_bureau(parcial).shape == parcial.shape
    sin_moneda = bureau.drop(columns=["CREDIT_CURRENCY"])
    assert COL_MONEDA_EXTRANJERA not in limpiar_bureau(sin_moneda).columns


def test_cada_importe_con_cap_declara_cual():
    """Los cuatro con cap salen de CAPS_BUREAU, y los dos sin él están en IMPORTES_BUREAU igual."""
    assert set(CAPS_BUREAU).issubset(set(IMPORTES_BUREAU))
    sin_cap = set(IMPORTES_BUREAU) - set(CAPS_BUREAU)
    assert sin_cap == {"AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"}
    for corte in CAPS_BUREAU.values():
        assert valor(corte) > 0


# --- la presencia del importe, que la limpieza de magnitud se lleva por delante --------------


def test_la_presencia_sobrevive_a_la_limpieza_de_moneda(bureau):
    """Una cuota en moneda extranjera se reportó: el importe se va, la foto de que hubo dato no.

    Es la propiedad que 2.2 tiene que leer en vez de `notna()`. La limpieza anula 432 cuotas de
    341 clientes, y con `notna()` sobre la tabla limpia 27 cambian de nivel en la tripartita.
    """
    limpio = limpiar_bureau(bureau)
    extranjera = limpio[COL_MONEDA_EXTRANJERA] == 1
    con_cuota = extranjera & bureau.AMT_ANNUITY.notna()
    assert con_cuota.sum() >= 1, "el fixture no tiene ninguna fila extranjera con cuota"
    assert limpio.loc[con_cuota, "AMT_ANNUITY"].isna().all(), "el importe sí se va"
    assert limpio.loc[con_cuota, "AMT_ANNUITY" + SUFIJO_SIGNO].eq(1).all(), "la foto no"


def test_la_presencia_no_se_inventa_donde_no_habia_dato(bureau):
    """El fallo del primer intento, que reconstruía con la bandera y no con la foto.

    La bandera marca las 1.408 filas extranjeras y solo 412 traían cuota: `notna() | bandera`
    habría dado por reportadas las otras 996. La foto distingue las dos: sin dato, NaN.
    """
    sin_cuota = bureau.copy()
    sin_cuota.loc[sin_cuota.CREDIT_CURRENCY.ne("currency 1"), "AMT_ANNUITY"] = np.nan
    limpio = limpiar_bureau(sin_cuota)
    extranjera = limpio[COL_MONEDA_EXTRANJERA] == 1
    assert extranjera.sum() >= 1, "el fixture no tiene ninguna fila extranjera"
    assert limpio.loc[extranjera, "AMT_ANNUITY" + SUFIJO_SIGNO].isna().all()


def test_la_foto_guarda_el_cero_ademas_de_la_presencia(bureau):
    """Reportada a cero y reportada con valor son dos niveles de la tripartita, de riesgo contrario.

    Una foto de solo presencia los funde: de las 432 cuotas que la limpieza anula, 191 eran cero,
    y con ella 20 clientes seguían cambiando de nivel.
    """
    a_cero = bureau.copy()
    extranjera = a_cero.CREDIT_CURRENCY.ne("currency 1")
    a_cero.loc[extranjera, "AMT_ANNUITY"] = 0.0
    limpio = limpiar_bureau(a_cero)
    assert extranjera.sum() >= 1, "el fixture no tiene ninguna fila extranjera"
    assert limpio.loc[extranjera, "AMT_ANNUITY"].isna().all(), "el importe se va"
    assert limpio.loc[extranjera, "AMT_ANNUITY" + SUFIJO_SIGNO].eq(0).all(), "el cero no"


def test_la_presencia_tambien_se_pierde_cuando_el_importe_esta_roto(bureau):
    """La foto es de antes de la limpieza, así que un valor por encima del cap sí contaba.

    Y tiene que seguir contando: el cap dice que no se sabe cuánto vale, no que no lo reportaran.
    Son las 24 filas del cap frente a las 412 de moneda, y las dos se recuperan igual.
    """
    limpio = limpiar_bureau(bureau)
    cliente4 = credito_de(limpio, 4).index  # la cuota de 20M, por encima del cap, nacional
    rota = limpio.loc[cliente4, "AMT_ANNUITY"].isna()
    assert rota.sum() == 1, "el fixture no monta la cuota rota en moneda nacional"
    assert limpio.loc[cliente4[rota], "AMT_ANNUITY" + SUFIJO_SIGNO].eq(1).all()


def test_cada_importe_con_foto_declara_quien_la_lee(bureau):
    """Los cinco son importes de verdad, y al que queda fuera solo se le lee la magnitud."""
    assert set(IMPORTES_CON_FOTO).issubset(set(IMPORTES_BUREAU))
    assert set(IMPORTES_BUREAU) - set(IMPORTES_CON_FOTO) == {"AMT_CREDIT_SUM"}
    for col, quien in IMPORTES_CON_FOTO.items():
        assert quien.strip(), f"{col} guarda su foto sin decir quién la lee"
    limpio = limpiar_bureau(bureau)
    for col in IMPORTES_CON_FOTO:
        assert col + SUFIJO_SIGNO in limpio.columns


# --- bureau_balance -----------------------------------------------------------
# Es el caso opuesto a las dos de arriba: aquí la limpieza no retira nada, porque el EDA no
# encontró ni un valor fuera de dominio. Lo que estos tests protegen es la decodificación, que
# `C` y `X` no se lean como mora cero, y que el dominio roto pare en vez de colarse.


@pytest.fixture
def bb():
    """Los ocho códigos, los dos bordes de la ventana y el crédito que no reporta ningún estado.

    El crédito 2 trae los códigos con espacios y en minúscula, que es lo único que la
    normalización tiene que arreglar, y el 3 es el que se queda sin ningún estado numérico: son
    130.368 créditos reales de esta tabla y es donde "sin dato" se lee como "sin mora".

    El 1 y el 2 traen además el mismo código en limpio y en sucio (`C` y ` c `, `X` y `x`), que es
    lo que deja montar las dos ramas de la normalización sobre una categórica: por separado no
    colisionan al limpiarse y juntos sí.
    """
    filas = [
        # crédito 1: los dos bordes de la ventana, y el cierre y el sin información al lado
        (1, 0, "C"),
        (1, -1, "X"),
        (1, -2, "0"),
        (1, MESES_BB[0], "5"),
        # crédito 2: los mismos códigos sucios, que es lo que la normalización arregla
        (2, -3, " c "),
        (2, -4, "x"),
        (2, -5, "1"),
        (2, -6, "4"),
        # crédito 3: solo cierre y sin información, o sea ningún estado numérico
        (3, -10, "C"),
        (3, -11, "X"),
        # crédito 4: los dos códigos de mora que faltaban
        (4, -20, "2"),
        (4, -21, "3"),
    ]
    return pd.DataFrame(filas, columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])


def test_el_fixture_de_bureau_balance_ejercita_cada_rama(bb):
    """Guardián: si el fixture pierde un caso, los tests de abajo dejan de probar lo que dicen."""
    codigos = bb.STATUS.str.strip().str.upper()
    assert set(codigos) == set(STATUS_DPD), "los ocho códigos del dominio tienen que estar"
    assert set(bb.MONTHS_BALANCE) >= set(MESES_BB), "faltan los bordes de la ventana"
    assert bb.STATUS.ne(codigos).sum() >= 2, "nadie ejercita el strip ni el upper"
    sin_numerico = bb.groupby("SK_ID_BUREAU").STATUS.apply(lambda s: s.isin(list("CX")).all())
    assert sin_numerico.sum() == 1, "falta el crédito sin ningún estado numérico"


def test_el_cierre_y_el_sin_informacion_no_son_mora_cero(bb):
    """La regla de "sin dato no es sin mora", en su primera aparición y donde se rompe.

    `C` está saldado y `X` es que el buró no informó: ninguno de los dos es un mes sin mora, así
    que salen de la escala. Leerlos como 0 pondría a los 130.368 créditos sin ningún estado
    numérico a severidad cero y a intensidad cero, que es exactamente lo que el EDA prohíbe.
    """
    limpio = limpiar_bureau_balance(bb)
    fuera = limpio.STATUS.isin(["C", "X"])
    # los dos lados tienen que existir, o los `.all()` de abajo son verdaderos por vacuidad
    assert fuera.any() and (~fuera).any(), "el fixture no monta los dos lados: no se mide nada"
    assert limpio.loc[fuera, COL_BB_DPD].isna().all()
    assert limpio.loc[~fuera, COL_BB_DPD].notna().all()
    assert limpio.loc[limpio.STATUS == "0", COL_BB_DPD].eq(0).all(), "el activo sin mora sí es 0"


@pytest.mark.parametrize("codigo,severidad", sorted(STATUS_DPD.items()))
def test_cada_codigo_decodifica_a_su_severidad(bb, codigo, severidad):
    """Uno a uno, que es lo que caza una traducción desplazada y no un recuento global."""
    limpio = limpiar_bureau_balance(bb)
    dpd = limpio.loc[limpio.STATUS == codigo, COL_BB_DPD]
    assert not dpd.empty, f"el fixture no trae ningún {codigo}: el test no mide nada"
    assert dpd.isna().all() if np.isnan(severidad) else dpd.eq(severidad).all()


def test_las_dos_banderas_marcan_lo_suyo_y_nada_mas(bb):
    """`BB_IS_X` solo el sin información, y `BB_IS_DPD` la mora de 1 a 5, nunca el activo a 0."""
    limpio = limpiar_bureau_balance(bb)
    assert limpio.loc[limpio[COL_BB_IS_X] == 1, "STATUS"].eq("X").all()
    assert limpio[COL_BB_IS_X].sum() == int(limpio.STATUS.eq("X").sum())
    assert limpio.loc[limpio[COL_BB_IS_DPD] == 1, COL_BB_DPD].gt(0).all()
    assert limpio.loc[limpio.STATUS == "0", COL_BB_IS_DPD].eq(0).all()


def test_la_particion_de_estados_cierra(bb):
    """El cierre, el sin información y los meses reportados suman la ventana entera.

    Es el `assert` que el notebook 03 lleva, y el que cazó la traducción desplazada: con la
    severidad leída del orden equivocado la suma se iba en 11,7 millones de filas.
    """
    limpio = limpiar_bureau_balance(bb)
    nC = int(limpio.STATUS.eq("C").sum())
    nX = int(limpio[COL_BB_IS_X].sum())
    reportados = int(limpio[COL_BB_DPD].notna().sum())
    assert nC + nX + reportados == len(limpio)


@pytest.mark.parametrize("malo", ["Z", "", "6", None])
def test_un_status_fuera_de_dominio_revienta(bb, malo):
    """Se deja caer a NaN se leería igual que un `C`, o sea como "sin mora".

    Es el pendiente abierto de `CREDIT_ACTIVE` en `bureau`, donde un estado desconocido descuadra
    la partición sin error, y aquí se cierra por el otro lado. El nulo va en la lista porque la
    tabla no trae ninguno: la ausencia se codifica como `X`, no como NaN.
    """
    roto = bb.copy()
    roto.loc[0, "STATUS"] = malo
    with pytest.raises(ValueError, match="STATUS fuera de dominio"):
        limpiar_bureau_balance(roto)


def test_el_dominio_entero_pasa(bb):
    """La otra dirección del check de arriba: los ocho códigos válidos salen enteros y sin tocar."""
    limpio = limpiar_bureau_balance(bb)
    assert set(limpio.STATUS.dropna()) == set(STATUS_DPD)


@pytest.mark.parametrize("malo", [MESES_BB[0] - 1, MESES_BB[1] + 1])
def test_un_mes_fuera_de_ventana_revienta(bb, malo):
    """El eje del panel no admite NaN: el nivel crédito saca de él la ventana.

    Es la diferencia con `acotar_ventanas_bureau()`, donde la fecha rota es un dato entre otros
    y la fila sobrevive sin él. Aquí anularla mediría mal la ventana sin que nada avise.
    """
    roto = bb.copy()
    roto.loc[0, "MONTHS_BALANCE"] = malo
    with pytest.raises(ValueError, match="MONTHS_BALANCE fuera de dominio"):
        limpiar_bureau_balance(roto)


def test_los_bordes_de_la_ventana_pasan(bb):
    """La otra dirección: -96 y 0 están dentro, y un `>` por un `>=` los perdería."""
    limpio = limpiar_bureau_balance(bb)
    assert set(limpio.MONTHS_BALANCE) >= set(MESES_BB)


def test_un_mes_con_decimales_revienta(bb):
    """Pasa el rango y el `int8` lo truncaría a un mes que no es. Solo puede traerlo la API."""
    roto = bb.astype({"MONTHS_BALANCE": float})
    roto.loc[0, "MONTHS_BALANCE"] = -1.5
    with pytest.raises(ValueError, match="MONTHS_BALANCE fuera de dominio"):
        limpiar_bureau_balance(roto)


def test_un_mes_en_float_sin_decimales_da_lo_mismo_que_en_entero(bb):
    """La otra dirección: el float que llega de la API con meses enteros no revienta."""
    en_float = bb.astype({"MONTHS_BALANCE": float})
    pd.testing.assert_frame_equal(limpiar_bureau_balance(en_float), limpiar_bureau_balance(bb))


def como_categorica(frame):
    """El frame con `STATUS` en `category`, que es como lo deja `load_table`."""
    otro = frame.copy()
    otro["STATUS"] = otro.STATUS.astype("category")
    return otro


@pytest.mark.parametrize(
    "creditos,sucio,colision",
    [
        ([1], False, False),
        ([2], True, False),
        ([1, 2], True, True),
    ],
)
def test_la_categorica_sucia_recorre_sus_dos_ramas(bb, creditos, sucio, colision):
    """La normalización tiene dos caminos sobre una categórica y hay que pisar los dos.

    Cuando los niveles limpios siguen siendo distintos entre sí se renombran en sitio, que es lo
    barato; cuando dos colapsan al mismo (` c ` y `C`) no se puede, y cae al respaldo de `object`.
    La rama de arriba solo hace algo si los niveles vienen sucios, y una categórica ya limpia no
    la ejercita: sin estos tres casos sobrevivían quitarle el `upper`, quitarle el `strip` y
    quitar entera la guarda del colapso, las tres con la suite en verde.

    Los dos `assert` de arriba son el guardián de la rama: si el fixture deja de montarla, el
    caso falla en vez de pasar midiendo otra cosa.
    """
    frame = bb[bb.SK_ID_BUREAU.isin(creditos)]
    niveles = list(frame.STATUS.astype("category").cat.categories)
    limpios = [n.strip().upper() for n in niveles]
    assert (niveles != limpios) == sucio, "el fixture no monta esta rama"
    assert (len(set(limpios)) != len(limpios)) == colision, "el fixture no monta esta rama"
    pd.testing.assert_frame_equal(
        limpiar_bureau_balance(como_categorica(frame)), limpiar_bureau_balance(frame)
    )


def test_el_respaldo_a_object_aguanta_el_nivel_que_todavia_no_existe():
    """La colisión en la que el valor limpio no es ya un nivel, que es la que necesita el cast.

    `bureau_balance.csv` no trae ni un código sucio, así que este caso solo puede existir en un
    fixture, y sin él quitar el `astype(object)` del respaldo dejaba la suite entera en verde: en
    las colisiones del fixture grande (` c ` contra `C`) el valor limpio ya era un nivel y pandas
    lo recodifica sin protestar. Con ` c ` y `c`, que colapsan en una `C` que no está, la misma
    línea revienta con TypeError. Es el cuarto mecanismo del fixture aguado.
    """
    frame = pd.DataFrame(
        {"SK_ID_BUREAU": [9, 9], "MONTHS_BALANCE": [0, -1], "STATUS": [" c ", "c"]}
    )
    niveles = list(frame.STATUS.astype("category").cat.categories)
    assert "C" not in niveles, "el fixture no monta el nivel que todavía no existe"
    assert len({n.strip().upper() for n in niveles}) == 1, "y tienen que colisionar"
    assert limpiar_bureau_balance(como_categorica(frame)).STATUS.eq("C").all()


def test_da_igual_que_status_llegue_categorica_o_en_texto(bb):
    """Los dos caminos de la normalización tienen que dar el mismo frame, tipos incluidos.

    Este es el test que caza el fallo que la puerta encontró. La primera versión fijaba el dtype
    con `astype(DTYPE_STATUS)`, y pandas da por iguales dos categóricas no ordenadas con el mismo
    conjunto de niveles: el astype devolvía la de entrada tal cual, con el orden que trae
    `load_table`, y leída la severidad por el código salía el 71,56% de las filas en mora en vez
    del 1,26%. Con `STATUS` en texto no se veía, porque ahí el orden ya era el bueno.

    Va aparte del test de las dos ramas y no es duplicado: aquel parte el fixture por créditos, y
    el astype solo se equivoca cuando el lote trae **los ocho niveles**, que es cuando pandas da
    los dos dtypes por iguales. Con cuatro niveles reordena bien y el fallo no aparece.
    """
    como_carga = bb.copy()
    como_carga["STATUS"] = como_carga.STATUS.str.strip().str.upper().astype("category")
    orden_del_lote = list(como_carga.STATUS.cat.categories)
    assert orden_del_lote != list(DTYPE_STATUS.categories), "el orden del lote ya es el fijo"
    pd.testing.assert_frame_equal(limpiar_bureau_balance(como_carga), limpiar_bureau_balance(bb))


def test_el_orden_de_los_niveles_no_lo_pone_el_lote(bb):
    """Patrón 12: el esquema de salida no puede depender de qué códigos traiga el frame."""
    solo_cerrados = bb[bb.STATUS.str.strip().str.upper() == "C"]
    assert not solo_cerrados.empty
    for frame in (bb, solo_cerrados):
        assert limpiar_bureau_balance(frame).STATUS.dtype == DTYPE_STATUS
        assert list(limpiar_bureau_balance(frame).STATUS.cat.categories) == list(STATUS_DPD)


def test_la_limpieza_de_bureau_balance_no_borra_ni_una_fila(bb):
    """El nivel cliente cuadra su suma de meses contra las filas enlazadas: no se pierde ninguna."""
    assert len(limpiar_bureau_balance(bb)) == len(bb)


def test_la_limpieza_de_bureau_balance_es_idempotente(bb):
    """Las tres derivadas salen de `STATUS`, que se normaliza pero no se destruye.

    Es lo contrario de `fotografiar_signo()`, que sí lee una columna que sus vecinas anulan y por
    eso necesita guarda. Aquí la segunda pasada recalcula lo mismo, y este test lo fija.
    """
    una = limpiar_bureau_balance(bb)
    pd.testing.assert_frame_equal(limpiar_bureau_balance(una), una)


def test_la_limpieza_de_bureau_balance_no_muta_el_frame_de_entrada(bb):
    copia = bb.copy()
    limpiar_bureau_balance(bb)
    pd.testing.assert_frame_equal(bb, copia)


def test_la_limpieza_de_bureau_balance_da_lo_mismo_fila_a_fila_que_sobre_la_tabla_entera(bb):
    """La premisa que permite que la capa 1 corra fuera del split: no cruza filas.

    Con `assert_frame_equal` y tipos incluidos, que es lo que caza un dtype que dependa del lote:
    sin los niveles fijos, la fila suelta salía con una categórica de un solo nivel.
    """
    entera = limpiar_bureau_balance(bb)
    fila_a_fila = pd.concat([limpiar_bureau_balance(bb.iloc[[i]]) for i in range(len(bb))])
    pd.testing.assert_frame_equal(entera, fila_a_fila)


def test_el_frame_vacio_y_sin_tipos_sale_con_el_mismo_esquema(bb):
    """Es lo que llega de la API: un cliente sin panel, y construido sin tipos.

    El esquema de la salida no puede depender del lote, que es el patrón 12: las cuatro columnas
    que la limpieza fija tienen que salir con el mismo dtype que sobre la tabla entera.
    """
    vacio = pd.DataFrame({c: [] for c in bb.columns}, dtype=object)
    fijadas = ["MONTHS_BALANCE", "STATUS", COL_BB_DPD, COL_BB_IS_X, COL_BB_IS_DPD]
    esperado = limpiar_bureau_balance(bb).dtypes[fijadas]
    pd.testing.assert_series_equal(limpiar_bureau_balance(vacio).dtypes[fijadas], esperado)


def test_la_limpieza_de_bureau_balance_no_revienta_si_faltan_columnas(bb):
    """A la API puede llegar un frame parcial: quien exige el contrato es la frontera, no esto."""
    sin_status = bb.drop(columns=["STATUS"])
    assert COL_BB_DPD not in limpiar_bureau_balance(sin_status).columns
    sin_meses = bb.drop(columns=["MONTHS_BALANCE"])
    assert COL_BB_DPD in limpiar_bureau_balance(sin_meses).columns


# --- la puerta del 3.1 contra el dato real ----------------------------------------------------

sin_bureau_balance = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["bureau_balance"]).exists(),
    reason="data/raw no viaja con el repo",
)

# La puerta sobre `bureau_balance.csv` completo, que es la única población de este punto: la
# limpieza es capa 1, no ve el TARGET y no necesita ni puente ni split, así que aquí no hay las
# dos poblaciones del bloque 2.
PUERTA_BB = {
    "filas": 27_299_925,
    "columnas": 3,
    "créditos": 817_395,
    "duplicados del par crédito-mes": 0,
    "meses fuera de ventana": 0,
    "códigos fuera de dominio": 0,
    "C": 13_646_993,
    "X": 5_810_482,
    "0": 7_499_507,
    "mora": 342_943,
}

# Los porcentajes que publica el EDA (5B.2), al lado de los conteos y no en vez de ellos: son la
# cifra con la que se contrasta, y el conteo es lo que no se puede redondear a que cuadre.
PARTICION_EDA = {"C": 49.99, "X": 21.28, "0": 27.47, "mora": 1.26}


@pytest.fixture(scope="module")
def bureau_balance_real():
    return load_table("bureau_balance")


@sin_bureau_balance
def test_la_puerta_del_3_1_sobre_el_dato_real(bureau_balance_real):
    limpio = limpiar_bureau_balance(bureau_balance_real)
    medido = {
        "filas": len(limpio),
        "columnas": bureau_balance_real.shape[1],
        "créditos": limpio.SK_ID_BUREAU.nunique(),
        "duplicados del par crédito-mes": int(
            limpio.duplicated(["SK_ID_BUREAU", "MONTHS_BALANCE"]).sum()
        ),
        "meses fuera de ventana": int((~limpio.MONTHS_BALANCE.between(*MESES_BB)).sum()),
        "códigos fuera de dominio": int((~limpio.STATUS.isin(list(STATUS_DPD))).sum()),
        "C": int(limpio.STATUS.eq("C").sum()),
        "X": int(limpio[COL_BB_IS_X].sum()),
        "0": int(limpio.STATUS.eq("0").sum()),
        "mora": int(limpio[COL_BB_IS_DPD].sum()),
    }
    assert medido == PUERTA_BB
    reparto = {k: round(100 * medido[k] / medido["filas"], 2) for k in PARTICION_EDA}
    assert reparto == PARTICION_EDA
    assert medido["C"] + medido["X"] + int(limpio[COL_BB_DPD].notna().sum()) == medido["filas"]
