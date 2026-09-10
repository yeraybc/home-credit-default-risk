"""Tests de la limpieza determinista de application_train (src/features/cleaning.py).

La limpieza es capa 1: corre fuera del split y sin estimar nada. Lo que estos tests protegen
es que siga siendo determinista (nada de percentiles colándose), que el recuento cuadre, y que
las tres cosas que el bloque edificio tiene que conservar no se vayan por delante con el resto.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.cleaning import (
    CAPS_BUREAU,
    COL_MONEDA_EXTRANJERA,
    COLUMNAS_FIRMES,
    COLUMNAS_PROVISIONALES,
    FILAS_POR_CATEGORIA,
    FILAS_POR_NULO,
    IMPORTES_BUREAU,
    IMPORTES_CON_FOTO,
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
    assert (bureau.AMT_CREDIT_SUM_DEBT < 0).sum() == 1, "falta el sobrepago, que tampoco se toca"


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

    Es la propiedad que 2.2 tiene que leer en vez de `notna()`. Con `notna()` sobre la tabla
    limpia, 432 filas y 341 clientes cambian de nivel en la tripartita de la cuota.
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
