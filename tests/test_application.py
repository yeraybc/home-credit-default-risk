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
from src.features.cleaning import limpiar_application
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


def test_la_bandera_del_buro_sale_de_las_columnas_del_prefijo(app):
    assert columnas_buro(app) == [
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    con = anadir_banderas_ausencia(app)
    assert list(con["HAS_BUREAU_INFO"]) == [1, 0, 1, 1, 1, 1]


def test_la_bandera_social_marca_el_bloque_ausente(app):
    con = anadir_banderas_ausencia(app)
    assert list(con["HAS_SOCIAL_INFO"]) == [1, 1, 0, 1, 1, 1]


def test_las_dos_banderas_de_bloque_exigen_el_bloque_entero_y_no_una_columna():
    """Con `any` en lugar de `all` las dos darían exactamente lo mismo, y no está cubierto.

    Sobre `application_train` el nulo es idéntico en las seis consultas al buró y en las dos
    del círculo social, así que los fixtures de arriba, que copian esa alineación, no
    distinguen una versión de la otra: las dos cuentan 41.516 y 1.021. Donde sí se separan es
    en lo que llegue a la API, que puede traer media consulta al buró, y ahí `any` marcaría
    como informado a quien no lo está. Hace falta un frame que rompa la alineación a propósito.
    """
    parcial = pd.DataFrame(
        {
            "AMT_REQ_CREDIT_BUREAU_DAY": [1.0, 1.0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [2.0, np.nan],
            "OBS_30_CNT_SOCIAL_CIRCLE": [3.0, 3.0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, np.nan],
        }
    )
    con = anadir_banderas_ausencia(parcial)

    assert list(con["HAS_BUREAU_INFO"]) == [1, 0]
    assert list(con["HAS_SOCIAL_INFO"]) == [1, 0]


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


def test_el_informe_conserva_sus_columnas_cuando_no_hay_features_nuevas(app):
    """El caso de la segunda pasada, donde la comprensión sale vacía.

    Sin las columnas declaradas el frame llega sin ninguna, y quien lo indexe después revienta
    por `KeyError` en un sitio que no tiene nada que ver con la causa.
    """
    from src.features.application import COLUMNAS_INFORME

    con = construir_features_capa1(app)

    assert list(informe_capa1(con, con).columns) == COLUMNAS_INFORME


# --- disciplina de capas ---------------------------------------------------------------------
# Los tres cortes que la capa 1 tiene derecho a pedir, todos de dominio. Se listan para que
# quitar uno también se note, no solo añadir uno de más.
PARAMETROS_DE_CAPA1 = {"dias_por_anio", "centinela_365243", "app_amt_req_bureau_day_max"}
MODULOS_DE_CAPA1 = ("cleaning.py", "application.py", "build_features.py")
MODULOS_QUE_PIDEN = {"cleaning.py", "application.py"}  # los dos que la sonda parchea


@pytest.fixture
def consumidos_por_capa1(app, monkeypatch):
    """Los parámetros que la capa 1 pide al correr, no los que aparecen escritos en el fuente.

    Esto se comprobaba escaneando el fuente con un regex y se le escapaban dos formas de
    llamada perfectamente normales: las comillas simples y el nombre en una variable
    (`n = "..."; valor(n)`), que es justo como se escribirá un bucle sobre varios cortes en la
    capa 2a. Sondear la ejecución no depende de cómo esté escrita la llamada.

    Para un reajustable se devuelve la referencia en vez de llamar a `valor()`, que hoy
    reventaría por su cuenta. No es un atajo: es lo que hace que el test siga sirviendo cuando
    el punto 1.3 empiece a llamar a `fijar_operativo()` y `valor()` deje de reventar. Ahí el
    respaldo en ejecución desaparece y esta comprobación se queda sola.
    """
    from src.features import application as mod_app
    from src.features import cleaning as mod_clean
    from src.features.params import REAJUSTABLES, parametro
    from src.features.params import valor as valor_real

    pedidos: list[str] = []

    def espia(nombre):
        pedidos.append(nombre)
        p = parametro(nombre)
        return p.valor_referencia if p.procedencia in REAJUSTABLES else valor_real(nombre)

    monkeypatch.setattr(mod_app, "valor", espia)
    monkeypatch.setattr(mod_clean, "valor", espia)
    construir_features_capa1(limpiar_application(app))
    return pedidos


def test_la_capa1_solo_consume_parametros_de_dominio(consumidos_por_capa1):
    """Un `estimado` o un `medido` dentro de la capa 1 sería un parámetro sin ajustar.

    La capa 1 corre sobre la tabla entera, antes del split, así que si consumiera un corte
    reajustable estaría usando la cifra del EDA medida sobre el conjunto completo.
    """
    from src.features.params import parametro

    assert consumidos_por_capa1, "la capa 1 no pidió ni un parámetro, revisar la sonda"
    for nombre in sorted(set(consumidos_por_capa1)):
        assert parametro(nombre).procedencia == "dominio", (
            f"la capa 1 consume {nombre!r}, que es "
            f"{parametro(nombre).procedencia!r} y no está ajustado sobre el split"
        )


def test_la_capa1_pide_exactamente_los_tres_cortes_declarados(consumidos_por_capa1):
    """Comprobar solo la procedencia deja pasar que una llamada desaparezca sin que nadie mire."""
    assert set(consumidos_por_capa1) == PARAMETROS_DE_CAPA1


def test_ningun_otro_modulo_de_capa1_pide_parametros():
    """La sonda parchea dos módulos; si un tercero importara `valor()`, no lo vería pasar."""
    from src.config import RAIZ

    con_valor = {
        m for m in MODULOS_DE_CAPA1 if "import valor" in (RAIZ / "src" / "features" / m).read_text()
    }
    assert not con_valor - MODULOS_QUE_PIDEN, f"la sonda no cubre {sorted(con_valor)}"
    assert con_valor == MODULOS_QUE_PIDEN


# --- contrato de esquema ---------------------------------------------------------------------
# Lo que impide que la misma feature signifique una cosa en entrenamiento y otra en serving.
# Las piezas del módulo son permisivas a propósito; el contrato lo exige preparar_application().


@pytest.fixture
def completo():
    """Un frame con todas las columnas de origen declaradas, que es lo que exige el contrato."""
    from src.features.application import ORIGEN_POR_FEATURE

    todas = sorted({c for cols in ORIGEN_POR_FEATURE.values() for c in cols})
    return pd.DataFrame({c: [1.0, 2.0] for c in todas})


def test_el_contrato_pasa_con_todas_las_columnas_declaradas(completo):
    from src.features.application import verificar_contrato_capa1

    verificar_contrato_capa1(completo)  # no levanta


def test_el_contrato_falla_si_falta_una_columna_de_origen(completo):
    """Sin dos del bloque, BUILDING_INFO_COUNT baja de escala sin cambiar de nombre."""
    from src.features.application import verificar_contrato_capa1

    with pytest.raises(ValueError, match="faltan columnas de origen"):
        verificar_contrato_capa1(completo.drop(columns=["APARTMENTS_AVG", "BASEMENTAREA_AVG"]))


def test_el_contrato_falla_si_sobra_una_columna_del_bloque(completo):
    """El caso simétrico: si la limpieza dejara de quitar MODE y MEDI, el conteo cambiaría."""
    from src.features.application import verificar_contrato_capa1

    con_extra = completo.assign(APARTMENTS_MODE=[0.1, 0.2], APARTMENTS_MEDI=[0.1, 0.2])
    with pytest.raises(ValueError, match="no coincide con el contrato"):
        verificar_contrato_capa1(con_extra)


def test_el_contrato_falla_si_sobra_una_consulta_al_buro(completo):
    from src.features.application import verificar_contrato_capa1

    with pytest.raises(ValueError, match="consultas al buró"):
        verificar_contrato_capa1(completo.assign(AMT_REQ_CREDIT_BUREAU_DECADE=[1.0, 2.0]))


def test_el_contrato_cubre_exactamente_las_features_que_se_construyen(completo):
    """Una feature nueva sin origen declarado se quedaría fuera del contrato sin que nadie mire."""
    from src.features.application import FEATURES_CAPA1, construir_features_capa1

    construidas = [c for c in construir_features_capa1(completo).columns if c not in completo]
    assert set(construidas) == set(FEATURES_CAPA1)


def test_ninguna_feature_de_capa1_depende_de_una_columna_provisional():
    """Si dependiera, la feature saldría de un dato que la capa 2b puede borrar.

    Hoy no pasa, pero por casualidad y no por diseño: nada lo impedía hasta este test.
    """
    from src.features.application import ORIGEN_POR_FEATURE
    from src.features.cleaning import COLUMNAS_PROVISIONALES

    for feature, origen in ORIGEN_POR_FEATURE.items():
        chocan = set(origen) & set(COLUMNAS_PROVISIONALES)
        assert not chocan, f"{feature} sale de {sorted(chocan)}, que la 2b puede eliminar"


def test_los_dos_juegos_de_sufijos_del_edificio_son_conceptos_distintos():
    """Compartían el nombre `SUFIJOS_EDIFICIO` con contenidos distintos en cada módulo."""
    from src.features.application import SUFIJOS_BLOQUE_EDIFICIO
    from src.features.cleaning import SUFIJOS_REDUNDANTES_EDIFICIO

    assert set(SUFIJOS_REDUNDANTES_EDIFICIO) < set(SUFIJOS_BLOQUE_EDIFICIO)
    assert "_AVG" not in SUFIJOS_REDUNDANTES_EDIFICIO, "la versión que se conserva no se elimina"


# --- el contrato contra un ancla independiente -----------------------------------------------
# Los tests de arriba no pueden cazar una declaración equivocada, porque el fixture `completo` se
# construye desde ORIGEN_POR_FEATURE, o sea desde lo que deberían estar comprobando: quitando
# TOTALAREA_MODE de COLUMNAS_EDIFICIO el fixture también lo pierde, declarado y derivado siguen
# coincidiendo y los cinco pasan. Lo único que lo cazaba era la integración contra el dato real,
# y ese fichero entero se salta en CI porque data/raw no viaja con el repo. Medido: con esa
# mutación puesta, el alcance que CI ejecuta daba 187 en verde igual que el árbol sano.
#
# El ancla tiene que ser independiente del contrato, así que aquí se declara el esquema del
# dataset **en otra forma**: los 14 conceptos del bloque en vez de las 15 columnas, y las 6
# ventanas del buró en vez de sus 6 nombres completos. Borrar algo del contrato no toca estas
# listas, y el contraste lo caza. No son cifras del EDA: son el esquema del csv, idéntico en
# train, en application_test y en lo que reciba la API.
CONCEPTOS_EDIFICIO = (
    "APARTMENTS",
    "BASEMENTAREA",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
    "YEARS_BEGINEXPLUATATION",
    "YEARS_BUILD",
)
VENTANAS_BURO = ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")
# el concepto 15, que no tiene pareja AVG ni MEDI y por eso sobrevive a la limpieza siendo _MODE
CONCEPTO_SUELTO = "TOTALAREA_MODE"
FEATURES_QUE_ENTREGA_EL_PUNTO = {
    "BUILDING_INFO_COUNT",
    "HAS_BUILDING_INFO",
    "HAS_BUREAU_INFO",
    "HAS_SOCIAL_INFO",
    "FLAG_EXT_SOURCE_1_NULL",
    "FLAG_EXT_SOURCE_3_NULL",
    "AGE_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
    "LTV",
}


@pytest.fixture
def esquema_crudo():
    """El bloque tal como llega del csv: las tres versiones por concepto y las categóricas."""
    from src.features.application import CATEGORICAS_EDIFICIO, SUFIJOS_BLOQUE_EDIFICIO

    cols = [f"{c}{s}" for c in CONCEPTOS_EDIFICIO for s in SUFIJOS_BLOQUE_EDIFICIO]
    cols += [CONCEPTO_SUELTO, *CATEGORICAS_EDIFICIO]
    cols += [f"AMT_REQ_CREDIT_BUREAU_{v}" for v in VENTANAS_BURO]
    return pd.DataFrame({c: [0.0] for c in cols})


def test_el_contrato_del_edificio_es_lo_que_deja_la_regla_de_limpieza(esquema_crudo):
    """Las 15 no son un número: son el bloque menos las versiones que la limpieza elimina.

    Pasa el esquema crudo por la regla real de `cleaning` y contrasta el resultado contra el
    contrato. Caza las dos direcciones a la vez: que el contrato se quede corto, y que la regla
    de limpieza deje de quitar lo que quitaba.
    """
    from src.features.application import COLUMNAS_EDIFICIO, columnas_edificio_numericas
    from src.features.cleaning import columnas_edificio_redundantes

    limpio = esquema_crudo.drop(columns=columnas_edificio_redundantes(esquema_crudo))
    assert set(COLUMNAS_EDIFICIO) == set(columnas_edificio_numericas(limpio))


def test_el_contrato_del_edificio_es_un_avg_por_concepto_mas_el_suelto():
    """La misma lista por el otro camino, sin pasar por la limpieza: dos derivaciones que cuadran.

    Si solo se comprobara contra la regla, un fallo en `columnas_edificio_redundantes` que se
    compensara con otro en el contrato pasaría desapercibido.
    """
    from src.features.application import COLUMNAS_EDIFICIO

    esperadas = {f"{c}_AVG" for c in CONCEPTOS_EDIFICIO} | {CONCEPTO_SUELTO}
    assert set(COLUMNAS_EDIFICIO) == esperadas


def test_el_contrato_del_buro_son_las_seis_ventanas(esquema_crudo):
    """Quitar una de las seis no rompe nada visible: la bandera se calcularía sobre cinco."""
    from src.features.application import COLUMNAS_BURO, columnas_buro

    assert set(COLUMNAS_BURO) == set(columnas_buro(esquema_crudo))
    assert set(COLUMNAS_BURO) == {f"AMT_REQ_CREDIT_BUREAU_{v}" for v in VENTANAS_BURO}


def test_la_capa1_entrega_exactamente_las_features_del_punto():
    """Perder una feature en silencio es tan malo como añadir una sin declarar.

    `test_el_contrato_cubre_exactamente_las_features_que_se_construyen` compara construidas
    contra declaradas y no ve el caso de que desaparezcan las dos a la vez.
    """
    from src.features.application import FEATURES_CAPA1

    assert set(FEATURES_CAPA1) == FEATURES_QUE_ENTREGA_EL_PUNTO


# --- la capa 1 no cruza filas ------------------------------------------------------------------
# Es la premisa que permite correr esta capa **antes** del split, y es lo primero que declara el
# docstring de build_features. Si una sola columna dependiera de las demás filas (una media, una
# mediana, un percentil, un rango), calcularla sobre la tabla entera metería en cada fila
# información de las otras, incluidas las de validación, y sería fuga con otro nombre.
#
# La comprobación es de código y no de datos: si hubiera una operación que cruza filas, la caza
# igual un frame sintético que 307.492 reales, siempre que tenga varias filas y valores
# variados. Por eso vive aquí y no en el fichero de integración, que se salta entero en CI.


@pytest.fixture
def con_bordes():
    """Seis clientes elegidos por sus casos límite, no al azar.

    Cada uno ejercita una rama distinta de la capa 1: el centinela, el cap de dominio, el
    denominador a cero y el nulo del LTV, y los dos extremos de la completitud del bloque.
    """
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, 7),
            "TARGET": [0, 1, 0, 1, 0, 0],
            # bloque edificio con sus tres versiones, para que la limpieza tenga qué eliminar
            "APARTMENTS_AVG": [0.1, np.nan, 0.3, np.nan, np.nan, 0.5],
            "APARTMENTS_MODE": [0.1, np.nan, 0.3, np.nan, np.nan, 0.5],
            "APARTMENTS_MEDI": [0.1, np.nan, 0.3, np.nan, np.nan, 0.5],
            "TOTALAREA_MODE": [0.2, np.nan, np.nan, np.nan, np.nan, 0.6],
            "HOUSETYPE_MODE": ["block", np.nan, "block", np.nan, np.nan, "block"],
            # el cliente 4 pasa del cap de dominio, que es 5
            "AMT_REQ_CREDIT_BUREAU_DAY": [0.0, np.nan, 1.0, 9.0, 0.0, 2.0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0, np.nan, 2.0, 3.0, 1.0, 4.0],
            "OBS_30_CNT_SOCIAL_CIRCLE": [2.0, 1.0, np.nan, 0.0, 3.0, 1.0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, 0.0, np.nan, 0.0, 1.0, 0.0],
            "EXT_SOURCE_1": [0.5, np.nan, 0.4, np.nan, 0.6, 0.3],
            "EXT_SOURCE_3": [0.5, 0.4, np.nan, 0.3, 0.6, 0.2],
            "DAYS_BIRTH": [-14610, -21915, -10958, -18263, -25568, -12775],
            # el cliente 2 trae el centinela crudo, que la limpieza convierte en nulo
            "DAYS_EMPLOYED": [-1000.0, 365243.0, -500.0, -2000.0, 0.0, -3000.0],
            "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            # el 4 sin precio declarado y el 5 con precio cero: los dos denominadores raros
            "AMT_GOODS_PRICE": [100.0, 100.0, 150.0, np.nan, 0.0, 250.0],
            # columnas que la limpieza elimina, para que ese camino también corra
            "FLAG_MOBIL": [1, 1, 1, 1, 1, 1],
            "FLAG_EMP_PHONE": [1, 0, 1, 1, 0, 1],
            "OBS_60_CNT_SOCIAL_CIRCLE": [2.0, 1.0, np.nan, 0.0, 3.0, 1.0],
        }
    )


def _capa1(frame):
    """La capa 1 entera: limpieza que no borra filas, más las features."""
    return construir_features_capa1(limpiar_application(frame))


def test_la_capa1_da_lo_mismo_fila_a_fila_que_sobre_la_tabla_entera(con_bordes):
    """Cada cliente sale idéntico calculado solo que acompañado de los demás.

    Es lo que sostiene el orden de las capas. Comprobado además contra el dato real fuera de
    los tests: 306 clientes, 300 al azar y 6 de borde dirigidos, ninguna de las diez columnas
    cambia.
    """
    juntas = _capa1(con_bordes)
    nuevas = [c for c in juntas.columns if c not in con_bordes.columns]
    assert nuevas, "la capa 1 no añadió ninguna columna, revisar el montaje"

    for i in con_bordes.index:
        sola = _capa1(con_bordes.loc[[i]])
        for c in nuevas:
            a, b = juntas.loc[i, c], sola.loc[i, c]
            iguales = a == b or (pd.isna(a) and pd.isna(b))
            assert iguales, f"{c} cambia en el cliente {i} al calcularlo solo: {a} frente a {b}"


def test_el_contraste_de_filas_cubre_los_casos_de_borde(con_bordes):
    """Si el frame no ejercitara las ramas raras, el test de arriba sería casi vacuo."""
    juntas = _capa1(con_bordes)

    assert juntas["FLAG_DAYS_EMPLOYED_ANOMALY"].sum() == 1, "falta el centinela"
    assert juntas["AMT_REQ_CREDIT_BUREAU_DAY"].max() == valor("app_amt_req_bureau_day_max")
    assert juntas["LTV"].isna().sum() == 2, "faltan el precio nulo y el precio a cero"
    assert juntas["BUILDING_INFO_COUNT"].eq(0).any(), "falta un cliente sin dato del edificio"
    assert juntas["BUILDING_INFO_COUNT"].max() == 2, "falta un cliente con el bloque completo"
