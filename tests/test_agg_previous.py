"""Tests de la agregación de previous_application por cliente (puntos 4.2 y 4.3).

Todos sobre un frame sintético, así que corren en CI sin los CSV, salvo la puerta contra el dato
real del final, que se salta sin `previous_application.csv` y el split y solo corre en local.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import RAIZ, ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_previous import (
    COLUMNAS_ORIGEN,
    CORTES,
    agregar_previous,
    unir_previous,
)
from src.features.params import CORTES_POR_FEATURE, fijar_operativo, parametro, valor
from src.features.split import NOMBRE_FICHERO, cargar_split

# Los cortes salen del registro y no de literales: dos copias del mismo corte dejan el fixture
# adaptándose en silencio al que cambie.
DIAS = valor("dias_por_anio")
VENTANA = valor("prev_ventana_reciente_dias")
SUELO_ANIOS = valor("suelo_anios_denominador")
# El plazo largo es `medido` y `valor()` lo bloquea hasta refijarlo sobre train: aquí se pasa con su
# referencia, porque solo se prueba el cómputo.
REFERENCIA = {
    n: parametro(n).valor_referencia for n in CORTES if parametro(n).procedencia == "medido"
}
LARGO = REFERENCIA["prev_plazo_largo_cuotas"]


def agregar(prev, cortes=None):
    return agregar_previous(prev, {**REFERENCIA, **(cortes or {})})


def solicitud(cliente, dias, tipo="New", **campos):
    return {
        "SK_ID_CURR": cliente,
        "DAYS_DECISION": dias,
        "NAME_CLIENT_TYPE": tipo,
        "NAME_CONTRACT_STATUS": "Approved",
        "CODE_REJECT_REASON": "XAP",
        "CNT_PAYMENT": 12.0,
        **campos,
    }


def rechazo(cliente, dias, tipo="New", motivo="HC", **campos):
    return solicitud(
        cliente,
        dias,
        tipo,
        NAME_CONTRACT_STATUS="Refused",
        CODE_REJECT_REASON=motivo,
        **campos,
    )


# Un cliente por caso de borde:
# 1  una sola solicitud, a menos de medio año: activa el suelo del ritmo
# 2  la más antigua es recurrente y dos entran en los doce meses: un `sum` no es un `max`
# 3  dos solicitudes el mismo día más antiguo, una recurrente y otra nueva: el empate discordante
# 4  la más antigua es XNA, que no delata recorte
# 5  una solicitud justo en -365 (fuera de la ventana) y otra en -364 (dentro)
# 6  recurrente, pero no en la más antigua: no es un historial recortado
# 365243  el código de ausencia del dataset como identificador: ningún mask sobre el frame lo toca
# Y para la familia del rechazo:
# 2  dos rechazos, uno por scoring externo y otro no: la bandera del motivo no es un conteo
# 3  dos rechazos por scoring externo: es un `max` y no un `sum`
# 4  un Canceled y un Approved con plazo largo: solo el no aprobado marca, y no hace falta Refused
# 5  un Refused con el plazo justo en el corte y otro con el plazo sin dato: ninguno marca
SOLICITUDES = [
    solicitud(1, -100),
    solicitud(2, -2000, "Repeater"),
    rechazo(2, -500, motivo="SCOFR"),
    rechazo(2, -200),
    solicitud(2, -30, "Repeater"),
    rechazo(3, -1500, "Repeater", motivo="SCOFR"),
    rechazo(3, -1500, "New", motivo="SCOFR"),
    solicitud(3, -10),
    solicitud(4, -800, "XNA", NAME_CONTRACT_STATUS="Canceled", CNT_PAYMENT=LARGO + 1),
    solicitud(4, -20, "Repeater", CNT_PAYMENT=LARGO + 1),
    rechazo(5, -900, CNT_PAYMENT=LARGO),
    rechazo(5, -365, CNT_PAYMENT=np.nan),
    solicitud(5, -364),
    solicitud(6, -1000, "New"),
    solicitud(6, -10, "Repeater"),
    solicitud(365243, -400),
]


@pytest.fixture
def prev():
    return pd.DataFrame(SOLICITUDES).astype({"DAYS_DECISION": "int16"})


def ritmo(n, dias_mas_antiguos):
    return n / max(dias_mas_antiguos / DIAS, SUELO_ANIOS)


# Lo esperado, a mano, por cliente
ESPERADO = {
    1: dict(
        n=1, minimo=-100, maximo=-100, c12=1, ritmo=ritmo(1, 100), recortado=0,
        rechazos=0, ratio=0.0, scofr=0, largo=0,
    ),
    2: dict(
        n=4, minimo=-2000, maximo=-30, c12=2, ritmo=ritmo(4, 2000), recortado=1,
        rechazos=2, ratio=0.5, scofr=1, largo=0,
    ),
    3: dict(
        n=3, minimo=-1500, maximo=-10, c12=1, ritmo=ritmo(3, 1500), recortado=1,
        rechazos=2, ratio=2 / 3, scofr=1, largo=0,
    ),
    4: dict(
        n=2, minimo=-800, maximo=-20, c12=1, ritmo=ritmo(2, 800), recortado=0,
        rechazos=0, ratio=0.0, scofr=0, largo=1,
    ),
    5: dict(
        n=3, minimo=-900, maximo=-364, c12=1, ritmo=ritmo(3, 900), recortado=0,
        rechazos=2, ratio=2 / 3, scofr=0, largo=0,
    ),
    6: dict(
        n=2, minimo=-1000, maximo=-10, c12=1, ritmo=ritmo(2, 1000), recortado=0,
        rechazos=0, ratio=0.0, scofr=0, largo=0,
    ),
    365243: dict(
        n=1, minimo=-400, maximo=-400, c12=0, ritmo=ritmo(1, 400), recortado=0,
        rechazos=0, ratio=0.0, scofr=0, largo=0,
    ),
}
COLUMNA = {
    "n": "PREV_APPLICATION_COUNT",
    "minimo": "PREV_DAYS_DECISION_MIN",
    "maximo": "PREV_DAYS_DECISION_MAX",
    "c12": "PREV_COUNT_12M",
    "ritmo": "PREV_APPLICATIONS_PER_YEAR",
    "recortado": "PREV_HISTORIAL_RECORTADO",
    "rechazos": "PREV_REFUSED_COUNT",
    "ratio": "PREV_REFUSED_RATIO",
    "scofr": "PREV_REFUSED_SCOFR_FLAG",
    "largo": "PREV_REFUSED_LONG_TERM_FLAG",
}


def test_el_fixture_ejercita_cada_rama(prev):
    """El guardián: si el fixture pierde un caso de borde, los tests de abajo pasan sin mirarlo."""
    por_cliente = prev.groupby("SK_ID_CURR")
    minimo = por_cliente.DAYS_DECISION.transform("min")
    primera = prev[prev.DAYS_DECISION.eq(minimo)]
    # el suelo: una relación de menos de medio año
    assert (-100 / DIAS) < SUELO_ANIOS
    # el empate discordante: el mismo día más antiguo con un tipo recurrente y otro que no
    empate = primera[primera.SK_ID_CURR == 3].NAME_CLIENT_TYPE
    assert len(empate) == 2 and empate.isin(["Repeater", "Refreshed"]).sum() == 1
    # la más antigua, XNA; y un recurrente que no es la más antigua
    assert primera[primera.SK_ID_CURR == 4].NAME_CLIENT_TYPE.tolist() == ["XNA"]
    assert "Repeater" not in primera[primera.SK_ID_CURR == 6].NAME_CLIENT_TYPE.tolist()
    # el borde de la ventana, a los dos lados
    assert {-VENTANA, -VENTANA + 1} <= set(prev[prev.SK_ID_CURR == 5].DAYS_DECISION)
    assert 365243 in set(prev.SK_ID_CURR)
    # ningún cliente sin ninguna solicitud reciente, para que el 0 real se vea
    assert ESPERADO[365243]["c12"] == 0
    # el motivo: un rechazo por scoring externo junto a otro que no, y dos que sí
    motivos = prev[prev.NAME_CONTRACT_STATUS.eq("Refused")].groupby("SK_ID_CURR").CODE_REJECT_REASON
    assert sorted(motivos.get_group(2)) == ["HC", "SCOFR"]
    assert motivos.get_group(3).tolist() == ["SCOFR", "SCOFR"]
    # el plazo largo: uno Canceled y uno Approved sobre el corte, y dos Refused que no marcan, uno
    # justo en el corte y otro sin plazo
    largo = prev[prev.CNT_PAYMENT.gt(LARGO)]
    assert sorted(largo.NAME_CONTRACT_STATUS) == ["Approved", "Canceled"]
    cinco = prev[prev.SK_ID_CURR == 5]
    assert LARGO in cinco.CNT_PAYMENT.tolist() and cinco.CNT_PAYMENT.isna().any()
    # sin ningún rechazo, con previas: el 0 de la bandera del motivo
    assert (prev[prev.SK_ID_CURR == 6].NAME_CONTRACT_STATUS != "Refused").all()


@pytest.mark.parametrize("cliente", sorted(ESPERADO))
def test_el_agregado_de_cada_cliente_es_el_calculado_a_mano(prev, cliente):
    fila = agregar(prev).loc[cliente]
    for clave, esperado in ESPERADO[cliente].items():
        assert fila[COLUMNA[clave]] == pytest.approx(esperado), f"{COLUMNA[clave]}: {fila}"


def test_la_ventana_de_doce_meses_deja_fuera_el_borde(prev):
    """Con -365 dentro contaría dos y con dias_por_anio (365,25) también: el borde es estricto."""
    assert agregar(prev).loc[5, "PREV_COUNT_12M"] == 1
    assert agregar(prev, {"prev_ventana_reciente_dias": 366}).loc[5, "PREV_COUNT_12M"] == 2


def test_el_plazo_largo_deja_fuera_el_borde_y_pide_no_aprobada(prev):
    """En las dos direcciones: con el corte una cuota por debajo el borde del 5 marca, y sin el
    filtro de estado el Approved del 4 marcaría igual que el Canceled."""
    assert agregar(prev).loc[[4, 5], "PREV_REFUSED_LONG_TERM_FLAG"].tolist() == [1, 0]
    baja = agregar(prev, {"prev_plazo_largo_cuotas": LARGO - 1})
    assert baja.loc[5, "PREV_REFUSED_LONG_TERM_FLAG"] == 1
    solo_aprobado = prev.assign(NAME_CONTRACT_STATUS=prev.NAME_CONTRACT_STATUS.where(
        prev.NAME_CONTRACT_STATUS.ne("Canceled"), "Approved"))
    assert agregar(solo_aprobado).loc[4, "PREV_REFUSED_LONG_TERM_FLAG"] == 0


def test_el_motivo_del_rechazo_es_una_bandera_y_solo_cuenta_si_hay_rechazo(prev):
    """El 3 tiene dos rechazos por scoring y vale 1, no 2; un motivo en una solicitud no rechazada
    no marca."""
    assert agregar(prev).loc[3, "PREV_REFUSED_SCOFR_FLAG"] == 1
    aprobada = prev.assign(NAME_CONTRACT_STATUS="Approved")
    assert (agregar(aprobada).PREV_REFUSED_SCOFR_FLAG == 0).all()


def test_sin_cortes_revienta_mientras_el_plazo_este_sin_fijar(prev):
    with pytest.raises(ValueError, match="sin fijar"):
        agregar_previous(prev)


def test_con_el_plazo_fijado_sobre_train_no_hace_falta_pasarlo(prev):
    for nombre, referencia in REFERENCIA.items():
        fijar_operativo(nombre, referencia, n_train=1)
    pd.testing.assert_frame_equal(agregar_previous(prev), agregar(prev))


def test_el_empate_discordante_no_depende_del_orden_de_las_filas(prev):
    """El `idxmin` del notebook se queda con la primera fila del empate, y aquí no puede.

    Se comprueba en las dos direcciones: el `idxmin` cambia de valor al invertir el frame, y la
    agregación no.
    """

    def con_idxmin(frame):
        primera = frame.loc[frame.groupby("SK_ID_CURR")["DAYS_DECISION"].idxmin()]
        return primera.set_index("SK_ID_CURR").NAME_CLIENT_TYPE.isin(["Repeater", "Refreshed"])

    invertido = prev.iloc[::-1]
    assert con_idxmin(prev)[3] != con_idxmin(invertido)[3]
    assert agregar(prev).loc[3, "PREV_HISTORIAL_RECORTADO"] == 1
    assert agregar(invertido).loc[3, "PREV_HISTORIAL_RECORTADO"] == 1
    pd.testing.assert_frame_equal(agregar(prev), agregar(invertido).sort_index())


def test_cada_cliente_agregado_solo_da_lo_mismo_que_acompanado(prev):
    """La premisa que deja a la agregación correr fuera del split: no cruza clientes.

    El 365243, de una sola solicitud lejana, caza un mínimo del lote, y el 1 un suelo global.
    """
    juntos = agregar(prev)
    for cliente in prev.SK_ID_CURR.unique():
        solo = agregar(prev[prev.SK_ID_CURR == cliente])
        pd.testing.assert_frame_equal(solo, juntos.loc[[cliente]])


# El tipo de cada columna, fijado en absoluto y no solo entre dos lotes: una bandera que pierde el
# int8 lo pierde en todos los lotes a la vez, y compararlos entre sí no lo ve.
ESQUEMA = {
    "PREV_APPLICATION_COUNT": "int64",
    "PREV_DAYS_DECISION_MIN": "float64",
    "PREV_DAYS_DECISION_MAX": "float64",
    "PREV_COUNT_12M": "int64",
    "PREV_HISTORIAL_RECORTADO": "int8",
    "PREV_REFUSED_COUNT": "int64",
    "PREV_REFUSED_RATIO": "float64",
    "PREV_REFUSED_SCOFR_FLAG": "int8",
    "PREV_REFUSED_LONG_TERM_FLAG": "int8",
    "PREV_APPLICATIONS_PER_YEAR": "float64",
}


def test_la_salida_tiene_el_esquema_declarado(prev):
    assert agregar(prev).dtypes.astype(str).to_dict() == ESQUEMA


@pytest.mark.parametrize("tipo", ["int16", "float32", "float64"])
def test_el_tipo_de_las_fechas_de_entrada_no_cambia_la_salida(prev, tipo):
    """`load_table` las carga en float32 y un cliente suelto de la API llega en float64."""
    convertido = agregar(prev.astype({"DAYS_DECISION": tipo}))
    pd.testing.assert_frame_equal(convertido, agregar(prev))


def test_la_salida_cumple_el_contrato_con_la_receta(prev):
    """Las columnas son una parte de las de la receta: las de los puntos siguientes aún no están."""
    receta = yaml.safe_load(
        (RAIZ / "config" / "previous_application_features.yaml").read_text()
    )["features"]
    nombres = {f["nombre"] for f in receta}
    agregado = agregar(prev)
    assert set(agregado.columns) <= nombres
    assert agregado.index.name == "SK_ID_CURR" and agregado.index.is_unique


def test_cada_columna_de_la_salida_varia_entre_clientes(prev):
    """Un agregado constante en el fixture no prueba nada de su cómputo (`patrones-de-fallo` §8)."""
    agregado = agregar(prev)
    constantes = [c for c in agregado.columns if agregado[c].nunique() < 2]
    assert not constantes, f"el fixture no las ejercita: {constantes}"


def test_los_cortes_que_lee_estan_en_el_registro_de_la_tabla():
    """Cada corte de la agregación lo declara alguna feature de la receta, y no queda huérfano."""
    cortes_de_la_tabla = CORTES_POR_FEATURE["previous_application"].values()
    declarados = {c for cortes in cortes_de_la_tabla for c in cortes}
    assert set(CORTES) <= declarados


def test_un_corte_que_la_agregacion_no_lee_revienta(prev):
    """Ni uno mal escrito ni uno del registro que aún no se consume se ignoran en silencio."""
    with pytest.raises(KeyError, match="prev_ventana"):
        agregar(prev, {"prev_ventana": 90})
    with pytest.raises(KeyError, match="prev_count_cola"):
        agregar(prev, {"prev_count_cola": 3})


def test_no_muta_el_frame_de_entrada(prev):
    antes = prev.copy()
    agregar(prev)
    pd.testing.assert_frame_equal(prev, antes)


def test_un_estado_fuera_de_dominio_revienta_tambien_desde_la_agregacion(prev):
    """La limpieza del 4.1 tiene llamante en producción: su guarda ya no es letra muerta."""
    roto = prev.assign(NAME_CONTRACT_STATUS="Approved")
    roto.loc[0, "NAME_CONTRACT_STATUS"] = "Cancelled"
    with pytest.raises(ValueError, match="NAME_CONTRACT_STATUS"):
        agregar(roto)


@pytest.mark.parametrize("sin_tipos", [False, True], ids=["recorte", "sin_tipos"])
def test_un_frame_vacio_da_un_agregado_vacio_con_el_mismo_esquema(prev, sin_tipos):
    """Es lo que se agrega para un cliente sin solicitudes: ni revienta ni cambia el esquema."""
    lleno = agregar(prev)
    vacio = agregar(pd.DataFrame([], columns=prev.columns) if sin_tipos else prev.iloc[:0])
    assert vacio.empty
    pd.testing.assert_series_equal(vacio.dtypes, lleno.dtypes)
    unido = unir_previous(pd.DataFrame({"SK_ID_CURR": [99]}), vacio)
    assert unido.HAS_PREV_APPLICATION.tolist() == [0]
    assert unido[lleno.columns].isna().all().all()


@pytest.mark.parametrize("columna", ["DAYS_DECISION", "CNT_PAYMENT"])
def test_una_numerica_que_no_es_numero_revienta_en_la_frontera(prev, columna):
    roto = prev.astype({columna: object})
    roto.loc[0, columna] = "ayer"
    with pytest.raises(ValueError, match="ayer"):
        agregar(roto)


@pytest.mark.parametrize("columna", COLUMNAS_ORIGEN)
def test_una_columna_de_origen_ausente_revienta_con_su_nombre(prev, columna):
    with pytest.raises(ValueError, match=columna):
        agregar(prev.drop(columns=columna))


# --- unir_previous ---------------------------------------------------------------------------


def test_unir_conserva_filas_orden_e_indice_y_no_rellena(prev):
    """El 99 no tiene solicitudes: NaN en todo, conteos incluidos, y la bandera a 0.

    El índice se conserva para alinear la predicción con la petición, y el huérfano del agregado
    (el 2, que no está en la lista) no añade ninguna fila.
    """
    agregado = agregar(prev)
    clientes = pd.DataFrame({"SK_ID_CURR": [99, 5, 1]}, index=[7, 8, 9])
    unido = unir_previous(clientes, agregado)
    assert unido.index.tolist() == [7, 8, 9]
    assert unido.SK_ID_CURR.tolist() == [99, 5, 1]
    assert unido.HAS_PREV_APPLICATION.tolist() == [0, 1, 1]
    assert unido.HAS_PREV_APPLICATION.dtype == "int8"
    assert unido.loc[7, agregado.columns].isna().all()
    assert unido.loc[[8, 9], "PREV_APPLICATION_COUNT"].tolist() == [3, 1]


def test_unir_revienta_si_el_agregado_trae_un_cliente_repetido(prev):
    agregado = agregar(prev)
    duplicado = pd.concat([agregado, agregado.loc[[5]]])
    with pytest.raises(pd.errors.MergeError):
        unir_previous(pd.DataFrame({"SK_ID_CURR": [5]}), duplicado)


# --- la puerta contra el dato real -----------------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["previous_application"]).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)

# La parte del 4.2 de la puerta del bloque 4, sobre sus dos poblaciones: la tabla cruda, que es la
# del EDA, y la de modelado, que es la del split. El 4.11 la completa con el resto de features.
PUERTA = {
    "crudo": {
        "con previas": 291_057,
        "sin previas": 16_454,
        "historial recortado": 53_934,
        "rechazo por scoring externo": 6_788,
        "plazo largo rechazado": 78,
    },
    "modelado": {
        "con previas": 291_041,
        "sin previas": 16_451,
        "historial recortado": 53_933,
        "rechazo por scoring externo": 6_787,
        "plazo largo rechazado": 78,
    },
}
# La tabla entera: 338.857 clientes, más que los de train porque incluye los de application_test
CLIENTES_TABLA = 338_857


@pytest.fixture(scope="module")
def dato_real():
    prev = load_table("previous_application")
    poblaciones = {
        "crudo": load_table("application_train", usecols=["SK_ID_CURR"]).SK_ID_CURR,
        "modelado": cargar_split().SK_ID_CURR,
    }
    return prev, agregar(prev), poblaciones


@sin_dato_real
@pytest.mark.parametrize("poblacion", sorted(PUERTA))
def test_la_puerta_del_bloque_sobre_el_dato_real(dato_real, poblacion):
    _, agregado, poblaciones = dato_real
    unido = unir_previous(pd.DataFrame({"SK_ID_CURR": poblaciones[poblacion]}), agregado)
    con = unido[unido.HAS_PREV_APPLICATION == 1]
    medido = {
        "con previas": len(con),
        "sin previas": len(unido) - len(con),
        "historial recortado": int(con.PREV_HISTORIAL_RECORTADO.sum()),
        "rechazo por scoring externo": int(con.PREV_REFUSED_SCOFR_FLAG.sum()),
        "plazo largo rechazado": int(con.PREV_REFUSED_LONG_TERM_FLAG.sum()),
    }
    assert con.PREV_REFUSED_RATIO.notna().all()
    assert medido == PUERTA[poblacion]
    assert len(unido) == len(poblaciones[poblacion])


@sin_dato_real
def test_sobre_el_dato_real_el_agregado_es_de_la_tabla_entera(dato_real):
    prev, agregado, _ = dato_real
    assert len(prev) == 1_670_214
    assert len(agregado) == CLIENTES_TABLA
    assert agregado.PREV_APPLICATION_COUNT.sum() == len(prev)


@sin_dato_real
def test_sobre_el_dato_real_cada_cliente_solo_da_lo_mismo_que_acompanado(dato_real):
    """El fila a fila sobre 300 al azar y los perfiles que el azar puede no traer.

    Son el del empate discordante, uno con XNA en la más antigua, el de más solicitudes, uno con
    una solicitud justo en -365, uno bajo el suelo del ritmo y el del identificador 365243. Del
    rechazo, uno con plazo largo solo en una solicitud Canceled, uno solo en Approved (que no
    marca), uno con todas sus solicitudes rechazadas, el de más rechazos y uno con scoring externo.
    """
    prev, agregado, _ = dato_real
    perfiles = [
        *[193_980, 307_630, 187_868, 261_733, 100_068, 365_243],
        *[101_379, 297_641, 109_699, 265_681, 282_125],
    ]
    muestra = [*np.random.default_rng(0).choice(agregado.index, 300, replace=False), *perfiles]
    trozo = prev[prev.SK_ID_CURR.isin(muestra)]
    assert set(perfiles) <= set(trozo.SK_ID_CURR)
    for cliente, filas in trozo.groupby("SK_ID_CURR"):
        pd.testing.assert_frame_equal(agregar(filas), agregado.loc[[cliente]])


@sin_dato_real
def test_sobre_el_dato_real_el_recorte_no_depende_del_orden(dato_real):
    """El único empate discordante de la tabla entera, con las filas en los dos órdenes."""
    prev, agregado, _ = dato_real
    filas = prev[prev.SK_ID_CURR == 193_980]
    assert agregar(filas.iloc[::-1]).loc[193_980, "PREV_HISTORIAL_RECORTADO"] == 1
    assert agregado.loc[193_980, "PREV_HISTORIAL_RECORTADO"] == 1
