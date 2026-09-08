"""Tests de los transformers a medida de las capas 2 (src/features/transformers.py).

Corren sobre frames sintéticos, así que se ejecutan también en un clon limpio, que es donde
importan: la propiedad que protegen es del código y no del dato.

Lo que protegen por encima de todo es **que el límite salga del `fit` y no de la referencia del
EDA**. Sobre el dato real los diez cortes reestimados coinciden exactos con esa referencia, así
que un test que compare las dos cifras pasa igual esté el ajuste bien o mal. Aquí el fixture da
percentiles deliberadamente distintos de los del EDA, y por eso sí distingue los dos casos.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.features.params import valor
from src.features.transformers import (
    COLUMNAS_DETALLE,
    NULO,
    AgrupadorDeRaras,
    CORTES_WINSOR,
    FACTOR_POR_COLUMNA,
    RATIOS_POSTERIORES,
    RatiosPosteriores,
    Winsorizador,
    informe_agrupamiento,
    informe_winsorizacion,
    registrar_limites,
    residual_de,
)

# las que el EDA deja sin capar a propósito: su cola tiene señal real, en las dos direcciones
SIN_CAPAR = (
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
)


@pytest.fixture
def frame():
    """Cien clientes con percentiles que NO son los del EDA, que es lo que hace útil el test.

    Las primeras 99 filas llevan valores bajos y la última un extremo, así que el p99 cae en la
    frontera y el 3xp99 sale muy por debajo de la referencia publicada.
    """
    n = 100
    base = {
        "SK_ID_CURR": range(1, n + 1),
        "AMT_INCOME_TOTAL": [1_000.0] * (n - 1) + [117_000_000.0],
        "DEF_30_CNT_SOCIAL_CIRCLE": [0.0] * (n - 1) + [40.0],
        "DEF_60_CNT_SOCIAL_CIRCLE": [0.0] * (n - 1) + [30.0],
        "OBS_30_CNT_SOCIAL_CIRCLE": [1.0] * (n - 1) + [350.0],
        "AMT_REQ_CREDIT_BUREAU_QRT": [0.0] * (n - 1) + [261.0],
        "AMT_REQ_CREDIT_BUREAU_MON": [0.0] * (n - 1) + [27.0],
        "AMT_REQ_CREDIT_BUREAU_WEEK": [0.0] * (n - 1) + [8.0],
        "CNT_CHILDREN": [1.0] * (n - 1) + [19.0],
        "CNT_FAM_MEMBERS": [2.0] * (n - 1) + [20.0],
        # la mitad sin coche: el percentil tiene que salir de los que sí lo tienen
        "OWN_CAR_AGE": [np.nan] * (n // 2) + [5.0] * (n // 2 - 1) + [91.0],
    }
    base.update({c: np.linspace(1.0, 100.0, n) for c in SIN_CAPAR})
    return pd.DataFrame(base)


# --- la guarda que la desviación cero del dato real no puede dar -----------------------------


def test_el_limite_sale_del_percentil_del_fit_y_no_de_la_referencia_del_eda(frame):
    """El fallo que caza: leer `params` en vez de reestimar. Sobre el dato real no se vería."""
    w = Winsorizador().fit(frame)
    p99 = frame["AMT_INCOME_TOTAL"].quantile(0.99)
    assert w.limites_["AMT_INCOME_TOTAL"] == pytest.approx(3 * p99)
    assert w.limites_["AMT_INCOME_TOTAL"] != 1_417_500
    assert w.limites_["CNT_CHILDREN"] != 9
    assert w.limites_["OWN_CAR_AGE"] != 64


def test_el_fixture_da_percentiles_distintos_de_la_referencia_del_eda(frame):
    """Guardián del fixture: si alguien lo aplana, el test de arriba deja de distinguir nada."""
    from src.features.params import parametro

    w = Winsorizador().fit(frame)
    iguales = [
        c
        for c, limite in w.limites_.items()
        if limite == parametro(CORTES_WINSOR[c]).valor_referencia
    ]
    assert not iguales, f"el fixture reproduce la referencia del EDA en {iguales}"


def test_ajustar_sobre_dos_particiones_distintas_da_limites_distintos(frame):
    """Si el ajuste ignorase el frame recibido, las dos mitades darían lo mismo."""
    primera = Winsorizador().fit(frame.iloc[:50])
    segunda = Winsorizador().fit(frame.iloc[50:])
    assert primera.limites_["AMT_INCOME_TOTAL"] != segunda.limites_["AMT_INCOME_TOTAL"]


def test_transform_no_reajusta_los_limites(frame):
    """La validación se transforma, nunca se ajusta nada sobre ella."""
    w = Winsorizador().fit(frame.iloc[:50])
    antes = dict(w.limites_)
    w.transform(frame.iloc[50:])
    assert w.limites_ == antes


# --- qué hace con los valores ----------------------------------------------------------------


def test_capa_por_arriba_y_no_toca_por_abajo(frame):
    salida = Winsorizador().fit_transform(frame)
    limite = Winsorizador().fit(frame).limites_["AMT_INCOME_TOTAL"]
    assert salida["AMT_INCOME_TOTAL"].max() == pytest.approx(limite)
    assert salida["AMT_INCOME_TOTAL"].min() == frame["AMT_INCOME_TOTAL"].min()


def test_own_car_age_va_al_percentil_pelado_y_el_resto_al_multiplo(frame):
    w = Winsorizador().fit(frame)
    con_coche = frame["OWN_CAR_AGE"].dropna()
    assert w.limites_["OWN_CAR_AGE"] == pytest.approx(con_coche.quantile(0.99))
    assert FACTOR_POR_COLUMNA["OWN_CAR_AGE"] == 1.0
    # el resto sí lleva el múltiplo
    assert w.limites_["CNT_FAM_MEMBERS"] == pytest.approx(
        valor("app_winsor_factor") * frame["CNT_FAM_MEMBERS"].quantile(0.99)
    )


def test_el_percentil_de_own_car_age_sale_solo_de_los_clientes_con_coche(frame):
    """La mitad sin coche son NaN; contarlos como cero movería el percentil."""
    w = Winsorizador().fit(frame)
    assert w.n_ajuste_["OWN_CAR_AGE"] == int(frame["OWN_CAR_AGE"].notna().sum())
    assert w.n_ajuste_["OWN_CAR_AGE"] < len(frame)


def test_no_toca_las_columnas_con_senal_en_el_extremo(frame):
    """El EDA las deja sin capar a propósito; entrar aquí les borraría la señal."""
    salida = Winsorizador().fit_transform(frame)
    for columna in SIN_CAPAR:
        pd.testing.assert_series_equal(salida[columna], frame[columna])
    assert not set(SIN_CAPAR) & set(CORTES_WINSOR)


def test_es_idempotente(frame):
    w = Winsorizador().fit(frame)
    una = w.transform(frame)
    pd.testing.assert_frame_equal(w.transform(una), una)


def test_una_columna_toda_nula_no_se_ajusta(frame):
    frame["OWN_CAR_AGE"] = np.nan
    w = Winsorizador().fit(frame)
    assert "OWN_CAR_AGE" not in w.limites_


def test_el_n_de_ajuste_es_el_de_los_no_nulos(frame):
    w = Winsorizador().fit(frame)
    for columna, n in w.n_ajuste_.items():
        assert n == int(frame[columna].notna().sum()), columna


# --- las dos direcciones del contrato de columnas --------------------------------------------


def test_el_fit_es_permisivo_con_columnas_ausentes(frame):
    """A la API puede llegar un frame parcial; quien exige el esquema es la frontera."""
    parcial = frame.drop(columns=["OWN_CAR_AGE", "CNT_CHILDREN"])
    w = Winsorizador().fit(parcial)
    assert "OWN_CAR_AGE" not in w.limites_
    assert "AMT_INCOME_TOTAL" in w.limites_


def test_transform_revienta_si_falta_una_columna_ajustada(frame):
    """La dirección contraria: saltársela daría otra matriz que la del entrenamiento."""
    w = Winsorizador().fit(frame)
    with pytest.raises(ValueError, match="que el frame no trae"):
        w.transform(frame.drop(columns=["CNT_CHILDREN"]))


def test_transform_sin_fit_revienta(frame):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        Winsorizador().transform(frame)


def test_solo_capa_columnas_con_corte_declarado(frame):
    """Todo corte pasa por params.py, no como cifra suelta en el transformer.

    El `fit` recorre `CORTES_WINSOR`, así que la propiedad es estructural y no una guarda: no
    hay parámetro por el que colar una columna sin corte. Lo que este test fija es que el
    recorrido siga siendo ese y no las columnas del frame que llegue, que sí caparía de más.
    """
    assert set(Winsorizador().fit(frame).limites_) <= set(CORTES_WINSOR)
    assert "AMT_CREDIT" in frame.columns and "AMT_CREDIT" not in CORTES_WINSOR


# --- RatiosPosteriores, que existe por el orden y no por su contenido ------------------------


def test_los_dos_ratios_salen_de_la_columna_ya_winsorizada(frame):
    """El test que justifica que este transformer exista.

    Construir la carga antes del winsorizador dejaría los 117M de ingreso en el denominador, y
    el cliente con el error de captura saldría con una carga de casi cero, o sea leído como el
    de menor riesgo de la tabla en vez de como un dato roto.
    """
    frame["AMT_ANNUITY"] = 1_000.0
    tubo = Pipeline([("winsor", Winsorizador()), ("derivadas", RatiosPosteriores())])
    salida = tubo.fit_transform(frame)

    limite = tubo.named_steps["winsor"].limites_["AMT_INCOME_TOTAL"]
    roto = salida.iloc[-1]
    assert roto["ANNUITY_TO_INCOME_RATIO"] == pytest.approx(1_000.0 / limite)
    assert roto["ANNUITY_TO_INCOME_RATIO"] != pytest.approx(1_000.0 / 117_000_000.0)


def test_el_orden_inverso_da_un_resultado_distinto(frame):
    """Si diera lo mismo, el orden del Pipeline sería decorativo y no habría nada que proteger."""
    frame["AMT_ANNUITY"] = 1_000.0
    correcto = Pipeline([("winsor", Winsorizador()), ("derivadas", RatiosPosteriores())])
    invertido = Pipeline([("derivadas", RatiosPosteriores()), ("winsor", Winsorizador())])
    a = correcto.fit_transform(frame)["ANNUITY_TO_INCOME_RATIO"]
    b = invertido.fit_transform(frame)["ANNUITY_TO_INCOME_RATIO"]
    assert not a.equals(b)


def test_el_fixture_trae_el_error_de_captura_que_hace_visible_el_orden(frame):
    """Guardián: sin un extremo muy por encima del corte, los dos tests de arriba no dicen nada.

    El margen es de un orden de magnitud y no más porque el percentil interpola y el propio
    extremo tira de él hacia arriba: con 99 valores en 1.000 y uno en 117M, el corte sale en
    3.512.970 y el extremo queda 33 veces por encima, no 117.000.
    """
    limite = Winsorizador().fit(frame).limites_["AMT_INCOME_TOTAL"]
    assert frame["AMT_INCOME_TOTAL"].max() > 10 * limite


def test_el_ratio_protege_el_denominador_cero(frame):
    frame.loc[0, "CNT_FAM_MEMBERS"] = 0.0
    salida = RatiosPosteriores().fit_transform(frame)
    assert pd.isna(salida.loc[0, "CHILDREN_TO_FAM_RATIO"])
    assert not np.isinf(salida["CHILDREN_TO_FAM_RATIO"].dropna()).any()


def test_los_ratios_son_idempotentes(frame):
    t = RatiosPosteriores().fit(frame)
    una = t.transform(frame)
    pd.testing.assert_frame_equal(t.transform(una), una)


def test_solo_anuncia_los_ratios_que_puede_construir(frame):
    """Sin numerador no hay ratio, y get_feature_names_out no puede prometerlo."""
    parcial = frame.drop(columns=["CNT_CHILDREN"])
    t = RatiosPosteriores().fit(parcial)
    assert t.derivadas_ == ("ANNUITY_TO_INCOME_RATIO",)
    assert "CHILDREN_TO_FAM_RATIO" not in t.transform(parcial).columns
    assert list(t.get_feature_names_out()) == list(t.transform(parcial).columns)


def test_revienta_si_falta_en_transform_una_columna_que_estaba_en_fit(frame):
    t = RatiosPosteriores().fit(frame)
    with pytest.raises(ValueError, match="no trae"):
        t.transform(frame.drop(columns=["AMT_INCOME_TOTAL"]))


def test_los_ratios_no_tocan_ninguna_columna_de_entrada(frame):
    salida = RatiosPosteriores().fit_transform(frame)
    for columna in frame.columns:
        pd.testing.assert_series_equal(salida[columna], frame[columna])


def test_get_feature_names_out_de_los_ratios_casa_con_la_salida(frame):
    t = RatiosPosteriores().fit(frame)
    assert list(t.get_feature_names_out()) == list(t.transform(frame).columns)
    assert set(RATIOS_POSTERIORES) <= set(t.get_feature_names_out())


def test_ningun_ratio_posterior_se_construye_ya_en_la_capa_1():
    """Si la capa 1 se adelantara, el denominador llevaría el error de captura sin capar."""
    from src.features.application import FEATURES_CAPA1

    assert not set(RATIOS_POSTERIORES) & set(FEATURES_CAPA1)


def test_todo_denominador_posterior_es_una_columna_winsorizada():
    """Es lo que separa estos dos ratios de LTV, que sí va en la capa 1."""
    for _, (_, den) in RATIOS_POSTERIORES.items():
        assert den in CORTES_WINSOR


# --- los cortes salen del registro, no de literales -------------------------------------------
# Misma sonda que `consumidos_por_capa1` en test_application.py: se mira lo que el `fit` pide en
# ejecución y no cómo está escrita la llamada. Sin esto, sustituir `valor("app_winsor_percentil")`
# por un 0,99 a pelo deja los veinte tests en verde, que es justo el corte suelto que la regla de
# la fase prohíbe.
CORTES_QUE_PIDE_EL_FIT = {"app_winsor_percentil", "app_winsor_factor"}


@pytest.fixture
def pedidos_del_fit(frame, monkeypatch):
    from src.features import transformers as mod
    from src.features.params import valor as valor_real

    pedidos: list[str] = []

    def espia(nombre):
        pedidos.append(nombre)
        return valor_real(nombre)

    monkeypatch.setattr(mod, "valor", espia)
    Winsorizador().fit(frame)
    return pedidos


def test_el_percentil_y_el_factor_salen_del_registro_y_no_de_literales(pedidos_del_fit):
    assert set(pedidos_del_fit) == CORTES_QUE_PIDE_EL_FIT


def test_el_fit_no_consume_ningun_corte_reajustable(pedidos_del_fit):
    """Consumir uno sería usar la cifra del EDA: el `fit` los reestima, no los lee."""
    from src.features.params import parametro

    for nombre in set(pedidos_del_fit):
        assert parametro(nombre).procedencia == "dominio", nombre


# --- el registro se escribe fuera del fit -----------------------------------------------------
# Los cinco registran de verdad, y lo que deshace la escritura es la fixture autouse
# `restaurar_params` de conftest.py: sin ella dejarían los cortes fijados para todo lo que
# corra después, y el rojo saldría en test_params.py y no aquí.


def test_el_fit_no_escribe_en_el_registro(frame):
    """Es la razón de que registrar viva fuera: en el CV, cada fold reescribiría el global."""
    from src.features.params import valor as valor_real

    Winsorizador().fit(frame)
    with pytest.raises(ValueError, match="sin fijar"):
        valor_real("app_winsor_amt_income_total")


def test_registrar_deja_el_valor_del_fit_y_no_la_referencia_del_eda(frame):
    """La dirección contraria, y con la cifra que distingue los dos orígenes."""
    from src.features.params import valor as valor_real

    w = Winsorizador().fit(frame)
    registrar_limites(w)
    assert valor_real("app_winsor_amt_income_total") == w.limites_["AMT_INCOME_TOTAL"]
    assert valor_real("app_winsor_amt_income_total") != 1_417_500


def test_el_n_declarado_es_el_de_los_no_nulos_de_la_columna(frame):
    """No el de la partición: en OWN_CAR_AGE son los clientes con coche, tres veces menos."""
    from src.features.params import parametro

    w = Winsorizador().fit(frame)
    registrar_limites(w)
    p = parametro("app_cap_p99_own_car_age")
    assert p.n_train_operativo == w.n_ajuste_["OWN_CAR_AGE"] < len(frame)


def test_registrar_no_deja_pendiente_ningun_corte_del_winsorizador(frame):
    from src.features.params import operativos_pendientes

    registrar_limites(Winsorizador().fit(frame))
    assert not set(CORTES_WINSOR.values()) & set(operativos_pendientes())


def test_registrar_dos_veces_revienta_y_con_permiso_no(frame):
    """El segundo ajuste no puede pisar al primero sin pedirlo, y la guarda viaja hasta aquí."""
    w = Winsorizador().fit(frame)
    registrar_limites(w)
    with pytest.raises(ValueError, match="ya está fijado"):
        registrar_limites(Winsorizador().fit(frame.iloc[:50]))
    registrar_limites(Winsorizador().fit(frame.iloc[:50]), sobrescribir=True)


def test_todos_los_cortes_del_winsorizador_son_reajustables():
    """Uno de dominio haría reventar a `fijar_operativo`, que los rechaza por diseño."""
    from src.features.params import REAJUSTABLES, parametro

    for corte in CORTES_WINSOR.values():
        assert parametro(corte).procedencia in REAJUSTABLES, corte


# --- contrato de scikit-learn ----------------------------------------------------------------


def test_clone_mas_fit_reproduce_los_mismos_limites(frame):
    w = Winsorizador().fit(frame)
    copia = clone(w).fit(frame)
    assert copia.limites_ == w.limites_


def test_set_output_pandas_devuelve_dataframe_con_nombres(frame):
    tubo = Pipeline([("winsor", Winsorizador())]).set_output(transform="pandas")
    salida = tubo.fit_transform(frame)
    assert isinstance(salida, pd.DataFrame)
    assert list(salida.columns) == list(frame.columns)


def test_get_feature_names_out_casa_con_las_columnas_de_la_salida(frame):
    w = Winsorizador().fit(frame)
    assert list(w.get_feature_names_out()) == list(w.transform(frame).columns)


def test_ida_y_vuelta_por_joblib_da_la_misma_salida(frame, tmp_path):
    import joblib

    w = Winsorizador().fit(frame)
    destino = tmp_path / "winsor.joblib"
    joblib.dump(w, destino)
    pd.testing.assert_frame_equal(joblib.load(destino).transform(frame), w.transform(frame))


# --- el informe de la puerta -----------------------------------------------------------------


def test_el_informe_declara_la_desviacion_contra_la_referencia(frame):
    inf = informe_winsorizacion(Winsorizador().fit(frame))
    assert len(inf) == len(CORTES_WINSOR)
    assert set(inf.columns) >= {"columna", "n ajuste", "límite", "ref. EDA", "% desviación"}
    # el fixture se aparta de la referencia, así que ninguna desviación puede salir cero
    assert (inf["% desviación"].abs() > 0).all()


# --- AgrupadorDeRaras, capa 2a ---------------------------------------------------------------
# El mecanismo es uno solo, la puerta de rareza, y el fixture lo separa de lo único con lo que
# podría confundirse: la tasa. Las dos raras tienen tasas opuestas y las dos grandes también, así
# que cualquier reintroducción de una fusión por tasa sale del residual y se ve.

RARAS = ("rara_alta", "rara_baja")
GRANDES = ("grande_alta", "grande_baja")
RESIDUAL = residual_de("cat")


@pytest.fixture
def categorico():
    """Cuatro categorías, dos por encima del mínimo y dos por debajo, con tasas enfrentadas."""
    filas = (
        [("grande_baja", 0)] * 400
        + [("grande_alta", 1)] * 150
        + [("rara_alta", 1)] * 9
        + [("rara_baja", 0)] * 5
    )
    valores, objetivo = zip(*filas)
    return pd.DataFrame({"cat": list(valores)}), pd.Series(objetivo)


def test_solo_se_reetiqueta_lo_que_baja_del_minimo(categorico):
    """La puerta es de rareza y nada más: la categoría con n suficiente no se toca."""
    X, y = categorico
    a = AgrupadorDeRaras().fit(X, y)

    assert set(a.raras_["cat"]) == set(RARAS)
    assert set(a.transform(X)["cat"]) == {*GRANDES, RESIDUAL}


def test_las_dos_raras_caen_en_el_mismo_residual_pese_a_tener_tasas_opuestas(categorico):
    """El fallo que caza: reintroducir una fusión por tasa, que las separaría."""
    X, y = categorico
    salida = AgrupadorDeRaras().fit(X, y).transform(X)
    reetiquetadas = salida.loc[X["cat"].isin(RARAS), "cat"]

    assert set(reetiquetadas) == {RESIDUAL}
    assert not set(salida["cat"]) & set(RARAS), "quedó alguna rara sin absorber"


def test_el_fixture_da_tasas_opuestas_a_las_dos_raras(categorico):
    """Guardián: con las dos raras del mismo lado, el test de arriba no distinguiría nada."""
    X, y = categorico
    tasas = y.groupby(X["cat"]).mean()

    assert tasas["rara_alta"] == 1.0
    assert tasas["rara_baja"] == 0.0


def test_ninguna_rara_se_mezcla_con_una_categoria_real(categorico):
    """El residual es un nivel nuevo, no una categoría superviviente que absorba a las otras."""
    X, y = categorico
    salida = AgrupadorDeRaras().fit(X, y).transform(X)
    conteo = salida["cat"].value_counts()

    assert conteo[RESIDUAL] == 14
    for grande in GRANDES:
        assert conteo[grande] == X["cat"].value_counts()[grande], f"{grande} cambió de tamaño"


def test_el_residual_es_propio_de_cada_columna(categorico):
    """Dos columnas nunca comparten nivel, o el IV del bloque 5 no sabría de cuál viene."""
    X, y = categorico
    X = X.assign(otra=X["cat"])
    salida = AgrupadorDeRaras().fit(X, y).transform(X)

    assert residual_de("cat") != residual_de("otra")
    assert residual_de("cat") in set(salida["cat"])
    assert residual_de("otra") in set(salida["otra"])


def test_si_todas_son_raras_la_columna_queda_con_un_solo_nivel(categorico):
    """La misma regla sin caso especial: no hay superviviente y todas van al residual."""
    X, y = categorico
    raras = X["cat"].isin(RARAS)
    a = AgrupadorDeRaras().fit(X[raras], y[raras])

    assert set(a.raras_["cat"]) == set(RARAS)
    assert set(a.transform(X[raras])["cat"]) == {RESIDUAL}


def test_sin_ninguna_categoria_rara_no_se_anota_nada(categorico):
    """Es el caso de doce de las trece columnas del bucket: el mapa sale vacío."""
    X, y = categorico
    grandes = X["cat"].isin(GRANDES)
    a = AgrupadorDeRaras().fit(X[grandes], y[grandes])

    assert a.raras_ == {}
    assert list(a.detalle_.columns) == COLUMNAS_DETALLE, "el detalle vacío cambia de forma"
    pd.testing.assert_frame_equal(a.transform(X[grandes]), X[grandes])


def test_el_nulo_es_un_nivel_mas_y_va_al_residual_si_es_raro(categorico):
    """El nulo no se salta: cuenta como celda y se absorbe como cualquier otra."""
    X, y = categorico
    X = pd.concat([X, pd.DataFrame({"cat": [None] * 7})], ignore_index=True)
    y = pd.concat([y, pd.Series([1] * 7)], ignore_index=True)
    a = AgrupadorDeRaras().fit(X, y)

    assert NULO in a.raras_["cat"]
    assert not a.transform(X)["cat"].isna().any()
    assert a.transform(X)["cat"].value_counts()[RESIDUAL] == 21


def test_el_nulo_frecuente_no_se_toca(categorico):
    """La otra dirección: por encima del mínimo el nulo sigue siendo nulo."""
    X, y = categorico
    X = pd.concat([X, pd.DataFrame({"cat": [None] * 200})], ignore_index=True)
    y = pd.concat([y, pd.Series([1] * 200)], ignore_index=True)
    a = AgrupadorDeRaras().fit(X, y)

    assert NULO not in a.raras_["cat"]
    assert a.transform(X)["cat"].isna().sum() == 200


def test_la_categoria_no_vista_pasa_tal_cual_y_no_va_al_residual(categorico):
    """Este transformer solo sabe de las que contó; lo desconocido lo absorbe el OHE de detrás."""
    X, y = categorico
    a = AgrupadorDeRaras().fit(X, y)
    nueva = pd.DataFrame({"cat": ["jamas_vista"]})

    assert a.transform(nueva)["cat"].tolist() == ["jamas_vista"]


def test_una_categoria_llamada_como_el_residual_revienta(categorico):
    """Reetiquetar encima la mezclaría con las raras sin dar un solo error."""
    X, y = categorico
    X = pd.concat([X, pd.DataFrame({"cat": [RESIDUAL] * 3})], ignore_index=True)
    y = pd.concat([y, pd.Series([0] * 3)], ignore_index=True)
    with pytest.raises(ValueError, match="residual"):
        AgrupadorDeRaras().fit(X, y)


def test_ajustar_no_necesita_el_target(categorico):
    """Es capa 2a: el criterio es un recuento sobre la covariable, la etiqueta no entra."""
    X, y = categorico

    assert AgrupadorDeRaras().fit(X).raras_ == AgrupadorDeRaras().fit(X, y).raras_


def test_las_columnas_numericas_no_se_agrupan(categorico):
    """El bucket que le entrega el ColumnTransformer es categórico; lo demás se deja en paz."""
    X, y = categorico

    assert "numero" not in AgrupadorDeRaras().fit(X.assign(numero=1.0), y).raras_


def test_el_fit_es_permisivo_con_columnas_ausentes(categorico):
    X, y = categorico

    assert AgrupadorDeRaras().fit(X.drop(columns=["cat"]), y).raras_ == {}


def test_agrupador_transform_revienta_si_falta_una_columna_ajustada(categorico):
    """Estricto donde el fit es permisivo, igual que el winsorizador."""
    X, y = categorico
    a = AgrupadorDeRaras().fit(X, y)
    with pytest.raises(ValueError, match="columnas ajustadas"):
        a.transform(pd.DataFrame({"otra": [1]}))


def test_agrupar_es_idempotente(categorico):
    """Aplicarlo sobre su propia salida no puede mover nada más."""
    X, y = categorico
    a = AgrupadorDeRaras().fit(X, y)
    una = a.transform(X)

    pd.testing.assert_frame_equal(a.transform(una), una)


def test_el_agrupador_no_anade_ni_quita_columnas(categorico):
    X, y = categorico
    a = AgrupadorDeRaras().fit(X, y)

    assert list(a.get_feature_names_out()) == list(a.transform(X).columns)


def test_las_raras_salen_de_la_particion_con_la_que_se_ajusta(categorico):
    """La fuga que caza: contar sobre el conjunto entero en vez de sobre train.

    Las filas de validación suben `rara_alta` por encima del mínimo, así que deja de ser rara
    en cuanto se cuenta sobre las dos particiones juntas.
    """
    X, y = categorico
    entero_X = pd.concat([X, pd.DataFrame({"cat": ["rara_alta"] * 200})], ignore_index=True)
    entero_y = pd.concat([y, pd.Series([1] * 200)], ignore_index=True)

    assert set(AgrupadorDeRaras().fit(X, y).raras_["cat"]) == set(RARAS)
    assert set(AgrupadorDeRaras().fit(entero_X, entero_y).raras_["cat"]) == {"rara_baja"}


def test_el_informe_declara_el_n_de_cada_categoria_absorbida(categorico):
    """Es lo que el bloque 5 lee para saber de qué está hecho cada residual."""
    X, y = categorico
    inf = informe_agrupamiento(AgrupadorDeRaras().fit(X, y))

    assert list(inf.columns) == COLUMNAS_DETALLE
    assert len(inf) == len(RARAS)
    assert dict(zip(inf["categoría"], inf["n"])) == {"rara_alta": 9, "rara_baja": 5}
    assert set(inf["residual"]) == {RESIDUAL}
