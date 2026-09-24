"""Tests de `recomendar_codificacion()` (src/features/selection.py).

La función recoge lo que concluyó el EDA y quien codifica de verdad es `pipeline.py`. Las dos
coinciden salvo en cinco entradas, donde el punto 1.4 midió sobre el split y llegó a otra
conclusión. Lo que estos tests protegen es que esas cinco lo digan: sin la marca, quien lea la
función exportada se lleva la decisión superada sin nada que se lo advierta, que es el patrón de
las dos fuentes de verdad y en este proyecto ya ha mordido tres veces.

Sintéticos, así que corren también en un clon limpio.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features import selection as mod_selection
from src.features.params import valor
from src.features.pipeline import PRESENCIA_POR_COLUMNA, columnas_declaradas
from src.features.selection import (
    DECISIONES_IV,
    DECISIONES_REDUNDANCIA,
    PARES_DECLARADOS,
    TABLA_PRINCIPAL,
    DecisionIV,
    DecisionRedundancia,
    columnas_protegidas,
    descartes_fijos,
    estabilidad_banda,
    fuentes_de_presencia,
    informe_redundancia,
    pares_redundantes,
    protecciones_de_presencia,
    recomendar_codificacion,
    seleccion_final,
    tabla_de,
)
from src.features.transformers import SelectorIV

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


# Las 20 protegidas del 5.8, escritas a mano para que el test no las derive por el mismo camino
# que el código: las nueve del 5.7 que siguen (el término de la interacción dejó de estarlo en la
# auditoría del bloque 5) y las once fuentes de presencia, condicionadas desde entonces a que algo
# de lo que recuperan siga en la matriz del fold.
PROTEGIDAS_5_8 = {
    # control: true de las recetas
    "HAS_BUREAU_HISTORY",
    "HAS_PREV_APPLICATION",
    "PREV_HISTORIAL_RECORTADO",
    "PREV_DAYS_DECISION_MAX",
    # las tres HAS_*, los dos documentos y las dos raras
    "HAS_BUREAU_BALANCE",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_6",
    "BUREAU_NEGATIVE_LIMIT_FLAG",
    "PREV_REFUSED_LONG_TERM_FLAG",
    # fuentes de presencia
    "FLAG_DAYS_EMPLOYED_ANOMALY",
    "FLAG_EXT_SOURCE_1_NULL",
    "FLAG_EXT_SOURCE_3_NULL",
    "FLAG_OWN_CAR",
    "HAS_SOCIAL_INFO",
    "HAS_BUREAU_INFO",
    "HAS_BUILDING_INFO",
    "HAS_BUREAU_FINANCIAL_DETAIL",
    "BB_MONTHS_REPORTED",
    "BB_ANY_DPD_FLAG",
    "HAS_BUREAU_OVERDUE_HISTORY",
}


def test_las_protegidas_son_las_del_5_7_mas_las_fuentes_de_presencia():
    assert columnas_protegidas() == PROTEGIDAS_5_8


def test_las_funciones_de_presencia_leen_exactamente_estas_columnas():
    """El `_Lector` anota lo que piden las funciones de `PRESENCIA_POR_COLUMNA`, sin copiarlo."""
    lee = {
        fuente
        for fuente, recuperadas in fuentes_de_presencia().items()
        if recuperadas & set(PRESENCIA_POR_COLUMNA)
    }
    assert lee == {"HAS_BUREAU_BALANCE", "BB_MONTHS_REPORTED", "BB_ANY_DPD_FLAG"}
    assert fuentes_de_presencia()["BB_ANY_DPD_FLAG"] == {
        "BB_MONTHS_SINCE_LAST_DPD",
        "BB_MONTHS_SINCE_LAST_DPD_REL",
    }


def test_una_fuente_sin_nada_vivo_que_recuperar_no_se_protege(monkeypatch):
    """`HAS_BUREAU_OVERDUE_HISTORY` solo recupera `BUREAU_MAX_OVERDUE_EVER`, que ya no sale fija:
    es un perdedor del 5.7 cuya ganadora puede no quedarse. La fuente se protege, condicionada a
    esa columna. La otra dirección: con la columna entre los descartes fijos, la fuente deja de
    protegerse y sale por su receta."""
    assert protecciones_de_presencia()["HAS_BUREAU_OVERDUE_HISTORY"] == ("BUREAU_MAX_OVERDUE_EVER",)
    assert "HAS_BUREAU_OVERDUE_HISTORY" not in descartes_fijos()

    brutos = {**mod_selection._descartes_brutos(), "BUREAU_MAX_OVERDUE_EVER": "prueba"}
    monkeypatch.setattr(mod_selection, "_descartes_brutos", lambda: brutos)
    assert "HAS_BUREAU_OVERDUE_HISTORY" not in columnas_protegidas()
    assert "HAS_BUREAU_OVERDUE_HISTORY" in descartes_fijos()


# Las once fuentes protegidas solo por serlo, a mano: el `SelectorIV` retira su protección en el
# fold donde todo lo que recuperan sale. El resto de protegidas lo está por su papel, sin condición.
FUENTES_CONDICIONADAS = {
    "FLAG_DAYS_EMPLOYED_ANOMALY",
    "FLAG_EXT_SOURCE_1_NULL",
    "FLAG_EXT_SOURCE_3_NULL",
    "FLAG_OWN_CAR",
    "HAS_SOCIAL_INFO",
    "HAS_BUREAU_INFO",
    "HAS_BUILDING_INFO",
    "HAS_BUREAU_FINANCIAL_DETAIL",
    "HAS_BUREAU_OVERDUE_HISTORY",
    "BB_MONTHS_REPORTED",
    "BB_ANY_DPD_FLAG",
}


def test_las_fuentes_condicionadas_son_las_protegidas_solo_por_presencia():
    assert set(protecciones_de_presencia()) == FUENTES_CONDICIONADAS
    assert FUENTES_CONDICIONADAS <= columnas_protegidas()


def test_configurar_selector_pasa_las_fuentes_condicionadas_fuera_de_las_protegidas():
    """Una fuente condicionada no puede ir entre las protegidas del selector, o nunca caducaría. Si
    tiene motivo fijo, lo lleva en `descartes` para cuando caduque."""
    selector = mod_selection.configurar_selector()
    assert selector.fuentes == protecciones_de_presencia()
    assert not set(selector.fuentes) & set(selector.protegidas)
    brutos = mod_selection._descartes_brutos()
    assert selector.descartes["HAS_BUREAU_OVERDUE_HISTORY"] == brutos["HAS_BUREAU_OVERDUE_HISTORY"]
    assert selector.descartes["BB_ANY_DPD_FLAG"] == brutos["BB_ANY_DPD_FLAG"]


# Los cinco que pierden su par en el 5.7 sin otro motivo fijo de salida, con su ganadora, a mano.
# `BUREAU_HAS_ANY_OVERDUE`, `BUREAU_OVERDUE_UNION` y `BUREAU_DAYS_CREDIT_UPDATE_FLAG` también
# pierden su par, pero ya salen por su `degradada` del 5.6 y no esperan a la ganadora.
PERDEDORES_5_7 = {
    "BB_MANY_CREDITS_FLAG": ("BUREAU_COUNT_COLA",),
    "BUREAU_CLOSED_COUNT": ("BB_MONTHS_TOTAL",),
    "BUREAU_MAX_OVERDUE_EVER": ("BB_OVERDUE_UNION",),
    "BUREAU_ANNUITY_ACTIVE_RATIO": ("HAS_BUREAU_BALANCE",),
    "BUREAU_CREDITS_WITH_ANNUITY_COUNT": ("HAS_BUREAU_BALANCE",),
}


def test_los_perdedores_del_5_7_son_los_que_no_tienen_otro_motivo_fijo():
    assert mod_selection.perdedores_de_redundancia() == PERDEDORES_5_7


def test_ningun_perdedor_del_5_7_sale_fijo_solo_por_su_par():
    """La regla del 5.7 supone que la ganadora se queda: un perdedor sin otro motivo no puede salir
    fijo, porque el 5.6 o el IV pueden sacar también a la ganadora."""
    descartes = descartes_fijos()
    for perdedor in PERDEDORES_5_7:
        assert perdedor not in descartes, perdedor
    for perdedor in ("BUREAU_HAS_ANY_OVERDUE", "BUREAU_OVERDUE_UNION"):
        assert "degradada en el 5.6" in descartes[perdedor], perdedor
        assert "redundante con" in descartes[perdedor], perdedor


def test_el_termino_de_la_interaccion_es_candidata_y_no_protegida():
    """`PREV_ACTIVIDAD_12M_COLA` solo se protegía para acompañar a `PREV_RELACION_CORTA_ACTIVA`, y
    el `SelectorIV` saca esa interacción por IV: se juzga por el suyo, dentro de quien tiene
    previas."""
    selector = mod_selection.configurar_selector()
    assert "PREV_ACTIVIDAD_12M_COLA" not in columnas_protegidas()
    assert selector.candidatas["PREV_ACTIVIDAD_12M_COLA"] == "HAS_PREV_APPLICATION"


def test_configurar_selector_juzga_a_los_perdedores_como_candidatas_con_presencia():
    selector = mod_selection.configurar_selector()
    assert selector.redundantes == PERDEDORES_5_7
    for perdedor in PERDEDORES_5_7:
        assert selector.candidatas[perdedor] is not None, perdedor


def test_una_fuente_se_desprotege_si_todo_lo_que_recupera_sale(monkeypatch):
    """Con `EXT_SOURCE_3` entre los descartes, su bandera deja de tener qué proteger."""
    assert "FLAG_EXT_SOURCE_3_NULL" in columnas_protegidas()

    brutos = {**mod_selection._descartes_brutos(), "EXT_SOURCE_3": "prueba"}
    monkeypatch.setattr(mod_selection, "_descartes_brutos", lambda: brutos)
    assert "FLAG_EXT_SOURCE_3_NULL" not in columnas_protegidas()


def test_descartes_y_protegidas_no_se_cruzan_y_una_segunda_pasada_no_cambia_nada():
    """La protección se calcula contra los descartes brutos; contra los ya filtrados sale igual,
    así que no hace falta iterar hasta un punto fijo."""
    descartes, protegidas = descartes_fijos(), columnas_protegidas()
    assert not set(descartes) & protegidas
    segunda = {
        f for f, recuperadas in fuentes_de_presencia().items() if recuperadas - set(descartes)
    }
    assert segunda <= protegidas
    assert {f for f in fuentes_de_presencia() if f in protegidas} == segunda


def test_todo_descarte_fijo_es_una_columna_de_la_matriz_y_lleva_motivo():
    declaradas = set(columnas_declaradas())
    for columna, motivo in descartes_fijos().items():
        assert columna in declaradas, columna
        assert motivo.strip(), columna


def test_los_dos_pares_reabiertos_del_5_7_quedan_con_las_dos():
    for par in (
        ("FLAG_EXT_SOURCE_3_NULL", "HAS_BUREAU_HISTORY"),
        ("HAS_BUREAU_HISTORY", "HAS_BUREAU_INFO"),
    ):
        assert DECISIONES_REDUNDANCIA[par].queda == par


# --- informe_redundancia() del 5.7, el criterio con estratos cruzados ----------------------------


def _pares_de(a, b):
    """Un índice de un solo par, con el formato que espera `informe_redundancia()`."""
    return pd.DataFrame(index=pd.MultiIndex.from_tuples([(a, b)], names=["a", "b"]))


def test_senal_propia_en_las_dos_se_quedan_las_dos():
    """Dos continuas correlacionadas (0,55) que mueven el TARGET cada una con su propio peso: la
    correlación entre ellas no basta para que una explique a la otra. Con NaN cruzados, como pasa
    de verdad entre tablas (sin historial de una auxiliar), para probar también el filtro de
    población de `informe_redundancia`."""
    rng = np.random.default_rng(3)
    n = 60_000
    x = rng.normal(size=n)
    z = 0.55 * x + np.sqrt(1 - 0.55**2) * rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.6 + 0.65 * x + 0.65 * z)))
    y = (rng.random(n) < p).astype(int)
    train = pd.DataFrame({"a": x, "b": z, "TARGET": y})
    # una de cada cinco filas pierde una de las dos, como el cliente sin historial de una tabla
    sin_a = rng.random(n) < 0.1
    sin_b = rng.random(n) < 0.1
    train.loc[sin_a, "a"] = np.nan
    train.loc[sin_b, "b"] = np.nan

    inf = informe_redundancia(train, _pares_de("a", "b")).loc[("a", "b")]

    assert inf["n"] == int((train["a"].notna() & train["b"].notna()).sum())
    assert inf["inc_a"] > valor("min_iv")
    assert inf["inc_b"] > valor("min_iv")
    assert inf["queda"] == ("a", "b")


def test_copia_exacta_se_queda_solo_una():
    """`b` es literalmente `a`: el caso real de `BUREAU_COUNT_COLA` y `BB_MANY_CREDITS_FLAG`, que
    dan el mismo IV exacto porque son la misma partición con otro nombre."""
    rng = np.random.default_rng(5)
    n = 40_000
    a = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.5 + a)))
    y = (rng.random(n) < p).astype(int)
    train = pd.DataFrame({"a": a, "b": a.copy(), "TARGET": y})

    inf = informe_redundancia(train, _pares_de("a", "b")).loc[("a", "b")]

    assert inf["iv_a"] == pytest.approx(inf["iv_b"])
    assert inf["inc_a"] < valor("min_iv")
    assert inf["inc_b"] < valor("min_iv")
    assert len(inf["queda"]) == 1


def _informe_con_ivs_forzados(monkeypatch, a, b, ivs, incs):
    """Sustituye `calcular_iv` e `iv_condicionado` por valores fijados por nombre de columna, para
    probar solo la lógica de decisión y no la aritmética del IV, que ya prueba `test_iv.py`."""
    monkeypatch.setattr(mod_selection, "calcular_iv", lambda serie, y, alfa=None: ivs[serie.name])
    monkeypatch.setattr(
        mod_selection,
        "iv_condicionado",
        lambda serie, estrato, y, alfa=None: incs[(serie.name, estrato.name)],
    )
    train = pd.DataFrame({a: [0, 1] * 10, b: [1, 0] * 10, "TARGET": [0, 1] * 10})
    return informe_redundancia(train, _pares_de(a, b)).loc[(a, b)]


def test_protegida_se_queda_siempre_y_la_otra_solo_si_llega(monkeypatch):
    """`HAS_BUREAU_HISTORY` es real y protegida por su `control: true` en la receta de bureau;
    `X_CUALQUIERA` no lo es. Dos casos con el mismo par y solo el incremental de la no protegida
    cambiado, para que la diferencia la decida ese número y no otra cosa."""
    assert "HAS_BUREAU_HISTORY" in columnas_protegidas()
    assert "X_CUALQUIERA" not in columnas_protegidas()

    llega = _informe_con_ivs_forzados(
        monkeypatch,
        "X_CUALQUIERA",
        "HAS_BUREAU_HISTORY",
        ivs={"X_CUALQUIERA": 0.05, "HAS_BUREAU_HISTORY": 0.10},
        incs={
            ("X_CUALQUIERA", "HAS_BUREAU_HISTORY"): 0.03,
            ("HAS_BUREAU_HISTORY", "X_CUALQUIERA"): 0.01,
        },
    )
    assert llega["queda"] == ("X_CUALQUIERA", "HAS_BUREAU_HISTORY")

    no_llega = _informe_con_ivs_forzados(
        monkeypatch,
        "X_CUALQUIERA",
        "HAS_BUREAU_HISTORY",
        ivs={"X_CUALQUIERA": 0.05, "HAS_BUREAU_HISTORY": 0.10},
        incs={
            ("X_CUALQUIERA", "HAS_BUREAU_HISTORY"): 0.005,
            ("HAS_BUREAU_HISTORY", "X_CUALQUIERA"): 0.01,
        },
    )
    assert no_llega["queda"] == ("HAS_BUREAU_HISTORY",)


def test_ninguna_llega_queda_la_de_mas_iv(monkeypatch):
    """Ninguna de las dos aporta sobre la otra: sale la de mayor IV marginal, sin importar cuál
    de las dos entra primero al par. Las dos direcciones, para que un `ganadora = a` fijo no pase
    el test: aquí gana la que se pasa primero y también la que se pasa segunda."""
    gana_la_primera = _informe_con_ivs_forzados(
        monkeypatch,
        "z_segunda",
        "a_primera",
        ivs={"z_segunda": 0.09, "a_primera": 0.04},
        incs={("z_segunda", "a_primera"): 0.01, ("a_primera", "z_segunda"): 0.008},
    )
    assert gana_la_primera["queda"] == ("z_segunda",)

    gana_la_segunda = _informe_con_ivs_forzados(
        monkeypatch,
        "z_segunda",
        "a_primera",
        ivs={"z_segunda": 0.04, "a_primera": 0.09},
        incs={("z_segunda", "a_primera"): 0.01, ("a_primera", "z_segunda"): 0.008},
    )
    assert gana_la_segunda["queda"] == ("a_primera",)


def test_solo_una_llega_sin_proteger_queda_solo_esa(monkeypatch):
    """Ninguna de las dos es protegida: la rama del medio, distinta de la de arriba, donde solo
    una aporta y la otra sale aunque su IV marginal sea más alto."""
    gana_a = _informe_con_ivs_forzados(
        monkeypatch,
        "m",
        "n",
        ivs={"m": 0.03, "n": 0.20},
        incs={("m", "n"): 0.05, ("n", "m"): 0.01},
    )
    assert gana_a["queda"] == ("m",)

    gana_b = _informe_con_ivs_forzados(
        monkeypatch,
        "m",
        "n",
        ivs={"m": 0.20, "n": 0.03},
        incs={("m", "n"): 0.01, ("n", "m"): 0.05},
    )
    assert gana_b["queda"] == ("n",)


def test_las_dos_protegidas_se_quedan_las_dos(monkeypatch):
    """`HAS_BUREAU_HISTORY` y `HAS_PREV_APPLICATION` son las dos protegidas: no hay incremental
    que las saque, así que ni se calcula qué decide (los mocks revientan si algo las consulta)."""
    assert {"HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION"} <= columnas_protegidas()
    inf = _informe_con_ivs_forzados(
        monkeypatch,
        "HAS_BUREAU_HISTORY",
        "HAS_PREV_APPLICATION",
        ivs={"HAS_BUREAU_HISTORY": 0.01, "HAS_PREV_APPLICATION": 0.01},
        incs={
            ("HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION"): 0.0,
            ("HAS_PREV_APPLICATION", "HAS_BUREAU_HISTORY"): 0.0,
        },
    )
    assert inf["queda"] == ("HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION")


# --- estabilidad_banda() y seleccion_final() del 5.8 ---------------------------------------------


@pytest.fixture(scope="module")
def matriz_banda():
    """Una candidata calibrada aparte para caer dentro de la banda de revisión, en torno a
    `min_iv` (0,02): con este ruido y esta semilla sale en 0,0203. Es un valor de muestreo, no
    una propiedad del código, así que se fija con un margen (`assert` de guardián) y no a ciegas.
    `FUERTE` tiene una señal mucho más alta, deliberadamente fuera de la banda."""
    rng = np.random.default_rng(21)
    n = 50_000
    en_banda = rng.normal(size=n)
    p_banda = 1 / (1 + np.exp(-(-2.6 + 0.13 * en_banda)))
    y_banda = pd.Series((rng.random(n) < p_banda).astype(int))
    fuerte = rng.normal(size=n)
    p_fuerte = 1 / (1 + np.exp(-(-2.6 + 0.6 * fuerte)))
    y_fuerte = pd.Series((rng.random(n) < p_fuerte).astype(int))
    return pd.DataFrame({"EN_BANDA": en_banda, "FUERTE": fuerte}), y_banda, y_fuerte


def test_el_fixture_de_banda_cae_dentro_de_la_banda_de_revision(matriz_banda):
    """Guardián: si el sintético se desplaza, los tests de abajo no distinguen nada."""
    from src.features.iv import calcular_iv

    X, y_banda, y_fuerte = matriz_banda
    suelo, techo = valor("banda_revision_iv_suelo"), valor("banda_revision_iv_techo")
    assert suelo <= calcular_iv(X["EN_BANDA"], y_banda) < techo
    assert calcular_iv(X["FUERTE"], y_fuerte) >= techo


def test_estabilidad_banda_solo_cuenta_lo_que_cae_dentro_de_la_banda(matriz_banda):
    X, y_banda, _ = matriz_banda
    selector = SelectorIV(candidatas={"EN_BANDA": None, "FUERTE": None}).fit(X, y_banda)

    aciertos = estabilidad_banda(X, y_banda, selector)

    assert set(aciertos) == {"EN_BANDA"}, "FUERTE está fuera de la banda y no debe contarse"
    assert 0 <= aciertos["EN_BANDA"] <= 15


def test_estabilidad_banda_sin_nada_en_la_banda_da_vacio(matriz_banda):
    """La otra dirección: con las dos candidatas lejos de la banda, no hay nada que contar."""
    X, _, y_fuerte = matriz_banda
    selector = SelectorIV(candidatas={"FUERTE": None}).fit(X, y_fuerte)

    assert estabilidad_banda(X, y_fuerte, selector) == {}


def test_estabilidad_banda_no_cuenta_lo_que_el_umbral_no_decide(matriz_banda):
    """Con el IV dentro de la banda, un descarte y una candidata protegida no se cuentan: su IV no
    decide nada. La otra dirección la da `test_estabilidad_banda_solo_cuenta_lo_que_cae_dentro...`,
    con la misma columna como candidata."""
    X, y_banda, _ = matriz_banda
    descartada = SelectorIV(candidatas={"FUERTE": None}, descartes={"EN_BANDA": "m"})
    descartada.fit(X, y_banda)
    protegida = SelectorIV(candidatas={"EN_BANDA": None}, protegidas=("EN_BANDA",)).fit(X, y_banda)

    assert valor("banda_revision_iv_suelo") <= descartada.iv_["EN_BANDA"]
    assert estabilidad_banda(X, y_banda, descartada) == {}
    assert estabilidad_banda(X, y_banda, protegida) == {}


def test_estabilidad_banda_no_muta_el_selector_original(matriz_banda):
    X, y_banda, _ = matriz_banda
    selector = SelectorIV(candidatas={"EN_BANDA": None, "FUERTE": None}).fit(X, y_banda)
    iv_antes, quedan_antes = dict(selector.iv_), selector.quedan_

    estabilidad_banda(X, y_banda, selector)

    assert selector.iv_ == iv_antes
    assert selector.quedan_ == quedan_antes


# --- seleccion_final(), con las cuatro dependencias aisladas por monkeypatch --------------------


class _SelectorFalso:
    """Lo que `seleccion_final()` le pide a `named_steps['seleccion']`: `iv_`, lo que el umbral
    decide (`candidatas` y `protegidas`, por defecto todo lo que lleva IV y nada protegido), y para
    los perdedores del 5.7 y las fuentes de presencia, `redundantes`, `fuentes` y la decisión que
    dejó en `motivos_`."""

    def __init__(
        self, iv, redundantes=None, fuentes=None, motivos=None, candidatas=None, protegidas=()
    ):
        self.iv_ = iv
        self.candidatas = dict.fromkeys(iv) if candidatas is None else candidatas
        self.protegidas = protegidas
        self.redundantes = redundantes
        self.fuentes = fuentes
        self.motivos_ = motivos or {}


def _pipeline_falso(iv):
    return SimpleNamespace(named_steps={"seleccion": _SelectorFalso(iv)})


@pytest.fixture
def registro_aislado(monkeypatch):
    """Las seis columnas del contrato de capa 1 y las cuatro dependencias de `seleccion_final()`
    aisladas por monkeypatch: nada de lo que aquí se comprueba depende de la configuración real
    de producción, que ya prueban sus propios tests."""
    columnas = (
        "A_DESCARTE",
        "B_PROTEGIDA",
        "C_CANDIDATA_LLEGA",
        "D_CANDIDATA_NO_LLEGA",
        "E_SIN_DECISION",
        "F_EN_BANDA",
    )
    monkeypatch.setattr(mod_selection.pipeline, "columnas_declaradas", lambda: columnas)
    monkeypatch.setattr(
        mod_selection, "descartes_fijos", lambda: {"A_DESCARTE": "motivo del descarte"}
    )
    monkeypatch.setattr(mod_selection, "columnas_protegidas", lambda: frozenset({"B_PROTEGIDA"}))
    monkeypatch.setattr(
        mod_selection, "_nombres_por_tabla", lambda: {"bureau": {"C_CANDIDATA_LLEGA"}}
    )
    monkeypatch.setattr(
        mod_selection,
        "DECISIONES_REDUNDANCIA",
        {
            ("C_CANDIDATA_LLEGA", "Z"): DecisionRedundancia(
                ("C_CANDIDATA_LLEGA",), "motivo redundancia"
            )
        },
    )
    monkeypatch.setattr(mod_selection, "check_is_fitted", lambda _: None)
    min_iv = valor("min_iv")
    iv = {
        "B_PROTEGIDA": 0.0,
        "C_CANDIDATA_LLEGA": min_iv + 0.05,
        "D_CANDIDATA_NO_LLEGA": min_iv - 0.01,
        "F_EN_BANDA": (valor("banda_revision_iv_suelo") + valor("banda_revision_iv_techo")) / 2,
    }
    return columnas, seleccion_final(_pipeline_falso(iv))


def test_una_fila_por_columna_declarada(registro_aislado):
    columnas, reg = registro_aislado
    assert set(reg.index) == set(columnas)
    assert len(reg) == len(columnas)


def test_un_descarte_sale_con_su_motivo_sin_mirar_el_iv(registro_aislado):
    _, reg = registro_aislado
    fila = reg.loc["A_DESCARTE"]
    assert not fila["queda"]
    assert fila["motivo"] == "motivo del descarte"


def test_una_protegida_se_queda_aunque_su_iv_sea_cero(registro_aislado):
    _, reg = registro_aislado
    fila = reg.loc["B_PROTEGIDA"]
    assert fila["queda"]
    assert "protegida" in fila["motivo"]


def test_una_candidata_que_llega_se_queda_y_la_que_no_llega_sale(registro_aislado):
    _, reg = registro_aislado
    assert reg.loc["C_CANDIDATA_LLEGA", "queda"]
    assert not reg.loc["D_CANDIDATA_NO_LLEGA", "queda"]


def test_una_columna_sin_decision_se_queda_con_su_motivo_propio(registro_aislado):
    """Ni descarte, ni protegida, ni candidata: pasa intacta, como haría el `SelectorIV`."""
    _, reg = registro_aislado
    fila = reg.loc["E_SIN_DECISION"]
    assert fila["queda"]
    assert pd.isna(fila["iv"]), "pandas convierte el None de un dict.get() en NaN al montar la fila"
    assert "sin decisión" in fila["motivo"]


def test_la_banda_se_marca_sin_estabilidad_pasada(registro_aislado):
    _, reg = registro_aislado
    fila = reg.loc["F_EN_BANDA"]
    assert fila["en_banda"]
    assert fila["folds"] is None
    assert "en banda" in fila["motivo"]
    # las que no caen en la banda no la marcan
    assert not reg.loc["C_CANDIDATA_LLEGA", "en_banda"]


def test_la_banda_no_marca_lo_que_el_umbral_no_decide(monkeypatch):
    """Descarte, protegida y candidata protegida con el mismo IV dentro de la banda: solo la
    candidata libre la marca."""
    columnas = ("DESCARTE", "PROTEGIDA", "CANDIDATA_PROTEGIDA", "CANDIDATA")
    monkeypatch.setattr(mod_selection.pipeline, "columnas_declaradas", lambda: columnas)
    monkeypatch.setattr(mod_selection, "descartes_fijos", lambda: {"DESCARTE": "motivo"})
    monkeypatch.setattr(
        mod_selection,
        "columnas_protegidas",
        lambda: frozenset({"PROTEGIDA", "CANDIDATA_PROTEGIDA"}),
    )
    monkeypatch.setattr(mod_selection, "_nombres_por_tabla", lambda: {})
    monkeypatch.setattr(mod_selection, "DECISIONES_REDUNDANCIA", {})
    monkeypatch.setattr(mod_selection, "check_is_fitted", lambda _: None)
    en_banda = (valor("banda_revision_iv_suelo") + valor("banda_revision_iv_techo")) / 2
    selector = _SelectorFalso(
        dict.fromkeys(columnas, en_banda),
        candidatas=dict.fromkeys(("CANDIDATA_PROTEGIDA", "CANDIDATA")),
        protegidas=("PROTEGIDA", "CANDIDATA_PROTEGIDA"),
    )

    reg = seleccion_final(SimpleNamespace(named_steps={"seleccion": selector}))

    assert reg["en_banda"].to_dict() == {
        "DESCARTE": False,
        "PROTEGIDA": False,
        "CANDIDATA_PROTEGIDA": False,
        "CANDIDATA": True,
    }


def test_la_banda_lleva_los_folds_cuando_se_pasa_estabilidad(monkeypatch):
    columnas = ("F_EN_BANDA",)
    monkeypatch.setattr(mod_selection.pipeline, "columnas_declaradas", lambda: columnas)
    monkeypatch.setattr(mod_selection, "descartes_fijos", lambda: {})
    monkeypatch.setattr(mod_selection, "columnas_protegidas", lambda: frozenset())
    monkeypatch.setattr(mod_selection, "_nombres_por_tabla", lambda: {})
    monkeypatch.setattr(mod_selection, "DECISIONES_REDUNDANCIA", {})
    monkeypatch.setattr(mod_selection, "check_is_fitted", lambda _: None)
    iv = {"F_EN_BANDA": (valor("banda_revision_iv_suelo") + valor("banda_revision_iv_techo")) / 2}

    reg = seleccion_final(_pipeline_falso(iv), estabilidad={"F_EN_BANDA": 9})

    assert reg.loc["F_EN_BANDA", "folds"] == 9
    assert "9 de 15" in reg.loc["F_EN_BANDA", "motivo"]


def test_el_registro_lee_del_selector_la_decision_de_perdedores_y_fuentes(monkeypatch):
    """Perdedores del 5.7 y fuentes condicionadas: su `queda` no sale del IV sino de lo que decidió
    el selector en su `fit`. Las fuentes cuentan como protegidas en `columnas_protegidas()`, así que
    la que caduca prueba además que su rama va antes que la de protegida."""
    columnas = ("P_SALE", "P_QUEDA", "F_SALE", "F_QUEDA")
    monkeypatch.setattr(mod_selection.pipeline, "columnas_declaradas", lambda: columnas)
    monkeypatch.setattr(mod_selection, "descartes_fijos", lambda: {})
    monkeypatch.setattr(
        mod_selection, "columnas_protegidas", lambda: frozenset({"F_SALE", "F_QUEDA"})
    )
    monkeypatch.setattr(mod_selection, "_nombres_por_tabla", lambda: {})
    monkeypatch.setattr(mod_selection, "DECISIONES_REDUNDANCIA", {})
    monkeypatch.setattr(mod_selection, "check_is_fitted", lambda _: None)
    iv = dict.fromkeys(columnas, 0.001)
    iv["P_QUEDA"] = valor("min_iv") + 0.01
    selector = _SelectorFalso(
        iv,
        redundantes={"P_SALE": ("G",), "P_QUEDA": ("G",)},
        fuentes={"F_SALE": ("X",), "F_QUEDA": ("Y",)},
        motivos={
            "P_SALE": "redundante con G en el 5.7",
            "F_SALE": "motivo fijo; fuente de presencia sin nada que recuperar",
        },
    )

    reg = seleccion_final(SimpleNamespace(named_steps={"seleccion": selector}))

    assert not reg.loc["P_SALE", "queda"]
    assert reg.loc["P_SALE", "motivo"] == "redundante con G en el 5.7"
    assert reg.loc["P_QUEDA", "queda"]
    assert "no queda" in reg.loc["P_QUEDA", "motivo"]
    assert not reg.loc["F_SALE", "queda"]
    assert "sin nada que recuperar" in reg.loc["F_SALE", "motivo"]
    assert reg.loc["F_QUEDA", "queda"]
    assert "fuente de presencia de Y" in reg.loc["F_QUEDA", "motivo"]


def test_la_redundancia_aparece_solo_en_las_columnas_de_algun_par(registro_aislado):
    _, reg = registro_aislado
    assert reg.loc["C_CANDIDATA_LLEGA", "redundancia"] is not None
    assert reg.loc["D_CANDIDATA_NO_LLEGA", "redundancia"] is None


# --- configurar_selector(), la configuración de producción del SelectorIV (5.8) ------------------


def test_configurar_selector_no_declara_nada_fuera_del_contrato_de_capa_1():
    """Todo origen (candidata, descarte o protegida) es una de las 180 columnas declaradas, y no
    un nombre suelto que el `SelectorIV` resolvería por casualidad como grupo OHE."""
    selector = mod_selection.configurar_selector()
    declaradas = set(columnas_declaradas())
    todos = (
        set(selector.candidatas)
        | set(selector.descartes)
        | set(selector.protegidas)
        | set(selector.fuentes)
    )
    assert todos <= declaradas


def test_configurar_selector_descartes_y_protegidas_no_se_cruzan():
    selector = mod_selection.configurar_selector()
    assert not (set(selector.descartes) & set(selector.protegidas))


def test_configurar_selector_las_degradadas_de_receta_son_candidatas_con_presencia():
    selector = mod_selection.configurar_selector()
    for nombre in mod_selection.DEGRADADAS_DE_RECETA:
        assert nombre in selector.candidatas, nombre
        assert selector.candidatas[nombre] is not None, nombre


def test_configurar_selector_no_repite_building_info_count_como_candidata():
    """El descarte del 5.6 sale de las candidatas: `BUILDING_INFO_COUNT` no se juzga dos veces."""
    selector = mod_selection.configurar_selector()
    assert "BUILDING_INFO_COUNT" not in selector.candidatas
    assert "BUILDING_INFO_COUNT" in selector.descartes


def test_presencia_de_tabla_son_las_tres_has_de_pipeline_aux():
    from src.features.pipeline import PRESENCIA_AUX

    assert set(mod_selection.PRESENCIA_DE_TABLA.values()) == set(PRESENCIA_AUX)
