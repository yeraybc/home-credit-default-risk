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
# 92 y no 90: FLAG_CONT_MOBILE y DEF_60_CNT_SOCIAL_CIRCLE se descartaron comparando contra la
# tasa de default sobre train completo, así que la capa 1 ya no las elimina. Sobreviven para
# que la 2b las juzgue sobre solo_train, que es la evidencia que vale.
COLUMNAS_LIMPIAS = 92
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
# La tripartita del bloque edificio es la cifra que la fija. El notebook la computó sobre 15
# columnas, los 14 _AVG más TOTALAREA_MODE, aunque su prosa describa el bloque como 46; el
# detalle de los tres recuentos está en el comentario de cabecera de application.py. Aquí
# sobreviven esas mismas 15 tras la limpieza y la clasificación no cambia ni un grupo.
#
# Las tasas sí son las del EDA. Los n **no**: el notebook corrió sobre las 307.511 crudas y da
# 81.552, 77.794 y 148.165. Los de aquí son los de la población de modelado, esos mismos menos
# las 19 filas que la limpieza quita (4, 7 y 8 respectivamente). No es una diferencia a
# reconciliar, es que la población cambió.
COMPLETO, PARCIAL, TODO_NULO = 6.96, 7.03, 9.23
N_COMPLETO, N_PARCIAL, N_TODO_NULO = 81_548, 77_787, 148_157
N_CRUDOS_DEL_EDA = (81_552, 77_794, 148_165)
COLUMNAS_CON_FEATURES = 101  # las 92 limpias más las 9 features de capa 1
COLUMNAS_DE_MATRIZ = 99  # las mismas sin la etiqueta ni el identificador, que no son features
FEATURES_DE_CAPA1 = 9

GRUPOS = ("completo", "parcial", "todo nulo")


def _tripartita(conteo, target, top):
    """n y tasa de los tres grupos de completitud, para el denominador que se le pase."""
    grupo = np.select([conteo.eq(top), conteo.eq(0)], ["completo", "todo nulo"], default="parcial")
    tab = pd.DataFrame({"g": grupo, "t": target}).groupby("g")["t"].agg(["size", "mean"])
    return [(int(tab.loc[k, "size"]), round(tab.loc[k, "mean"] * 100, 2)) for k in GRUPOS]


def test_los_n_de_la_puerta_son_los_del_eda_menos_las_filas_limpiadas():
    """Que la diferencia sea exactamente las 19, y no un desajuste que nadie ha mirado."""
    nuestros = (N_COMPLETO, N_PARCIAL, N_TODO_NULO)
    assert [c - n for c, n in zip(N_CRUDOS_DEL_EDA, nuestros)] == [4, 7, 8]
    assert sum(N_CRUDOS_DEL_EDA) - sum(nuestros) == FILAS_NETAS
    assert sum(nuestros) == FILAS_LIMPIAS


def test_la_tripartita_del_bloque_edificio_reproduce_las_tasas_del_eda(base):
    n_cols = int(base["BUILDING_INFO_COUNT"].max())

    assert n_cols == 15, "el bloque edificio ya no tiene las 15 columnas que sobreviven"
    assert _tripartita(base["BUILDING_INFO_COUNT"], base["TARGET"], n_cols) == [
        (N_COMPLETO, COMPLETO),
        (N_PARCIAL, PARCIAL),
        (N_TODO_NULO, TODO_NULO),
    ]


# Cuánto pesa TOTALAREA_MODE dentro del denominador, que es lo que sostiene el comentario de
# cabecera de application.py. Los recuentos de 43 y 15 clasifican idéntico, y de ahí es fácil
# concluir de más y dar por bueno cualquier denominador mientras las categóricas se queden
# fuera. No: quitando esa columna quedan las 14 con pareja, que es el recuento de 42 del
# notebook una vez colapsan sus pares, y la tripartita se mueve.
TRIPARTITA_SIN_TOTALAREA = [(81_562, 6.96), (77_128, 7.05), (148_802, 9.22)]
SOLO_TIENEN_TOTALAREA = 645
COMPLETOS_QUE_LO_PIERDEN = 14


def test_totalarea_mode_mueve_la_tripartita_y_por_eso_esta_en_el_contrato(base):
    """Sin ella, 645 clientes con dato del edificio pasan a leerse como si no tuvieran ninguno."""
    from src.features.application import COLUMNAS_EDIFICIO

    con_pareja = [c for c in COLUMNAS_EDIFICIO if c != "TOTALAREA_MODE"]
    conteo = base[con_pareja].notna().sum(axis=1)

    completa = [(N_COMPLETO, COMPLETO), (N_PARCIAL, PARCIAL), (N_TODO_NULO, TODO_NULO)]
    assert len(con_pareja) == 14
    assert _tripartita(conteo, base["TARGET"], 14) == TRIPARTITA_SIN_TOTALAREA
    assert TRIPARTITA_SIN_TOTALAREA != completa, "si coincidieran, el test no diría nada"
    assert int(conteo.eq(0).sum() - base["BUILDING_INFO_COUNT"].eq(0).sum()) == SOLO_TIENEN_TOTALAREA
    assert int(conteo.eq(14).sum() - base["BUILDING_INFO_COUNT"].eq(15).sum()) == (
        COMPLETOS_QUE_LO_PIERDEN
    )


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
    assert base.shape[1] == limpia.shape[1] + FEATURES_DE_CAPA1


def test_el_informe_declara_el_recorrido_entero_de_columnas(cruda, limpia, base):
    """122, 92 y 101 salen del informe, para que el cierre no tenga que contarlas a mano."""
    inf = informe_base(cruda, limpia, base).set_index("medida")
    assert inf.loc["columnas", "cruda"] == COLUMNAS_CRUDAS
    assert inf.loc["columnas", "limpia"] == COLUMNAS_LIMPIAS
    assert inf.loc["columnas", "con features"] == COLUMNAS_CON_FEATURES
    assert inf.loc["features de capa 1", "con features"] == FEATURES_DE_CAPA1


# --- la ruta de inferencia, contra el dato real ----------------------------------------------

FILAS_TEST = 48_744
NULOS_ANNUITY_TEST = 24


def test_application_test_no_pierde_ni_un_cliente():
    """Con el borrado de filas dentro de la limpieza, estos 24 desaparecían en silencio.

    En una submission eso es una entrega inválida, y en la API un solicitante sin cuota
    declarada que se queda sin score en vez de recibir uno.
    """
    cruda = load_table("application_test", reduce_memory=False)
    limpia_test = cargar_y_limpiar("application_test")

    assert int(cruda["AMT_ANNUITY"].isna().sum()) == NULOS_ANNUITY_TEST
    assert len(cruda) == FILAS_TEST
    assert len(limpia_test) == FILAS_TEST, "la ruta de inferencia no puede perder clientes"
    assert "TARGET" not in limpia_test.columns


def test_las_provisionales_llegan_a_la_matriz(base):
    """Sobreviven a la capa 1 justo para que la 2b pueda decidirlas sobre el split."""
    from src.features.cleaning import COLUMNAS_PROVISIONALES

    for col in COLUMNAS_PROVISIONALES:
        assert col in base.columns


# --- puerta de salida del punto 1.3, los 3xp99 reestimados sobre el split --------------------
# Lo que el bloque pide es la comparación entre lo reestimado y la referencia del EDA, "para que
# la desviación quede medida y no supuesta". Sale cero en las diez, y por eso este test **no**
# es la guarda contra ajustar fuera del split: con desviación cero no distingue un ajuste sobre
# train de uno sobre la tabla entera. Esa guarda vive en tests/test_transformers.py, donde el
# fixture da percentiles distintos de la referencia a propósito.
N_TRAIN = 245_993
LIMITES_SOBRE_TRAIN = {
    "AMT_INCOME_TOTAL": (1_417_500, N_TRAIN),
    "DEF_30_CNT_SOCIAL_CIRCLE": (6, 245_162),
    "DEF_60_CNT_SOCIAL_CIRCLE": (6, 245_162),
    "OBS_30_CNT_SOCIAL_CIRCLE": (30, 245_162),
    "AMT_REQ_CREDIT_BUREAU_QRT": (6, 212_866),
    "AMT_REQ_CREDIT_BUREAU_MON": (12, 212_866),
    "AMT_REQ_CREDIT_BUREAU_WEEK": (3, 212_866),
    "CNT_CHILDREN": (9, N_TRAIN),
    "CNT_FAM_MEMBERS": (15, N_TRAIN),
    "OWN_CAR_AGE": (64, 83_745),
}


@pytest.fixture(scope="module")
def winsor(base):
    from src.features.build_features import matriz_de_features
    from src.features.split import solo_train
    from src.features.transformers import Winsorizador

    entrenamiento = matriz_de_features(solo_train(base))
    assert len(entrenamiento) == N_TRAIN
    return Winsorizador().fit(entrenamiento)


def test_los_3xp99_reestimados_sobre_train_reproducen_la_referencia_del_eda(winsor):
    from src.features.transformers import informe_winsorizacion

    assert winsor.limites_ == {c: float(v) for c, (v, _) in LIMITES_SOBRE_TRAIN.items()}
    assert winsor.n_ajuste_ == {c: n for c, (_, n) in LIMITES_SOBRE_TRAIN.items()}
    assert (informe_winsorizacion(winsor)["% desviación"] == 0).all()


def test_ajustar_capa2a_ajusta_sobre_train_y_registra_lo_reestimado(base):
    """El camino completo, que es lo que el test de arriba no cubre: fija sobre solo_train().

    Restaura `PARAMS` al salir porque `fijar_operativo` escribe en un dict de módulo, y este
    fichero corre antes que test_params.py, que comprueba que ningún reajustable arranca fijado.
    """
    from src.features import params as mod
    from src.features.build_features import ajustar_capa2a
    from src.features.params import parametro, valor

    copia = dict(mod.PARAMS)
    try:
        w, inf = ajustar_capa2a(base)
        assert w.limites_ == {c: float(v) for c, (v, _) in LIMITES_SOBRE_TRAIN.items()}
        assert (inf["% desviación"] == 0).all()
        assert valor("app_winsor_amt_income_total") == 1_417_500
        assert parametro("app_cap_p99_own_car_age").n_train_operativo == 83_745
    finally:
        mod.PARAMS.clear()
        mod.PARAMS.update(copia)


def test_el_contrato_de_nombres_es_el_de_la_matriz_y_no_el_de_la_base(base):
    """Las 101 de la base menos la etiqueta y el identificador, que no son features.

    Con la base entera el contrato prometía 101 columnas y `transform` sobre la X devolvía
    100, y el test de contrato no lo veía porque ajustaba y transformaba el mismo frame.
    """
    from src.features.build_features import matriz_de_features
    from src.features.transformers import Winsorizador

    X = matriz_de_features(base)
    assert base.shape[1] == COLUMNAS_CON_FEATURES
    assert X.shape[1] == COLUMNAS_DE_MATRIZ
    assert not {"TARGET", "SK_ID_CURR"} & set(X.columns)

    prometidas = list(Winsorizador().fit(X).get_feature_names_out())
    assert prometidas == list(X.columns)


def test_la_columna_que_la_limpieza_elimina_no_llega_a_winsorizarse(base, winsor):
    """OBS_60 es descarte firme y DEF_60 provisional: solo una de las dos se capa."""
    assert "OBS_60_CNT_SOCIAL_CIRCLE" not in base.columns
    assert "OBS_60_CNT_SOCIAL_CIRCLE" not in winsor.limites_
    assert "DEF_60_CNT_SOCIAL_CIRCLE" in winsor.limites_
