"""Tests de la agregación de previous_application por cliente (puntos 4.2 a 4.7).

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
    COLUMNAS_SIN_RECETA,
    CORTES,
    DENOMINADOR,
    NUMERICAS_ORIGEN,
    agregar_previous,
    unir_previous,
)
from src.features.cleaning import limpiar_previous
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
SOBRE = REFERENCIA["prev_sobreconcesion_corte"]
ADELANTO = REFERENCIA["prev_adelanto_liquidacion_dias"]
URGENTES = REFERENCIA["prev_finalidades_urgentes"]
HORA = valor("prev_hora_temprana_max")
COLA = REFERENCIA["prev_count_cola"]
ACTIVIDAD = REFERENCIA["prev_actividad_12m_cola"]
LARGA = valor("prev_relacion_larga_anios")


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
        "AMT_APPLICATION": 100.0,
        "AMT_CREDIT": 100.0,
        "AMT_ANNUITY": 10.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "RATE_DOWN_PAYMENT": np.nan,
        "DAYS_LAST_DUE_1ST_VERSION": np.nan,
        "DAYS_LAST_DUE": np.nan,
        # por defecto: combinación definida y no de calle, mediodía, con acompañante y sin finalidad
        "PRODUCT_COMBINATION": "Cash X-Sell: low",
        "HOUR_APPR_PROCESS_START": 12.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_CASH_LOAN_PURPOSE": "XAP",
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
# Y para la relación entre cifras, cada uno con su denominador:
# 7  las cuatro ramas a la vez: una sobreconcedida, una justo en el corte, un plazo 0 y un Refused
#    sin crédito, con una entrada de consumo y otra de cash que no entra
# 8  con previas y ninguna fila en ningún denominador: solicitud 0 con crédito positivo (el cociente
#    sería infinito) y crédito 0 con solicitud positiva, ambos sin plazo
# 9  una entrada negativa de consumo, que la limpieza pasa a 0
# 10 una aprobada sin cuota y un Refused con cuota, plazo y crédito: el coste solo lee aprobadas y
#    se queda sin dato
# 11 dos aprobadas con plazo, una con crédito 0 y otra con cuota 0: el dato real no las tiene, y
#    sin ellas el `> 0` del coste puede pasar a `>= 0` y dar infinito o un coste 0
# Y para el ciclo de vida, con el fin previsto y el efectivo (NaN es "sin fecha"; el resto de
# clientes no trae ninguna y queda sin operación terminada):
# 12 una liquidada con más de un año de adelanto y el fin previsto por delante, y una viva con
#    otro fin por delante más cercano: la bandera marca y el máximo es de la ya cerrada
# 13 una liquidada justo en el corte, otra que cierra en la fecha prevista y otra con el fin
#    previsto en 0: ninguna marca, no cuenta como por vencer, y con terminadas la bandera vale 0
# 14 el centinela en el fin previsto de una operación con fin efectivo: sin limpiarlo marcaría
# 15 el centinela en el fin efectivo de una con fin previsto por delante: sin limpiarlo contaría
#    como terminada y la bandera saldría 0 en vez de NaN
# Y para la captación y la finalidad (el resto de clientes trae los valores por defecto: sin calle,
# sin hora temprana, con acompañante y sin finalidad declarada):
# 8  sus dos filas sin combinación de producto: fantasma, y sin ninguna fila en el denominador de la
#    calle ni en el de la finalidad
# 16 tres filas: la hora justo en el corte con calle, sin acompañante y finalidad urgente; una a la
#    hora siguiente con "Repairs", que no es "Car repairs"; y una sin combinación con XNA como
#    finalidad, que sale de los dos denominadores
# 17 todo fantasma y sin finalidad: la calle es NaN y no 0
# 18 una sola fila de calle con finalidad urgente, a las 23: los dos ratios valen 1
# 19 una finalidad informada y no urgente, a las 0: la proporción urgente vale 0 y no NaN
# Y para las colas y el término de la interacción (recientes son las de los últimos doce meses):
# 20 justo en la cola del conteo y en la de actividad, con la relación corta: las tres a 1
# 21 justo en la cola de actividad y con la más antigua justo en el corte de la relación larga: el
#    borde entra en la larga, así que el término vale 0
# 22 un conteo y una actividad por debajo de las dos colas, con la relación corta: las tres a 0
# 23 la más antigua un día por debajo del corte de la relación larga, con la actividad justo en su
#    cola: es corta con 365,25 días por año y sería larga con 365, así que el término vale 1
CONSUMO = "Consumer loans"
SENTINELA = 365243.0
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
    solicitud(7, -900, NAME_CONTRACT_TYPE=CONSUMO, AMT_CREDIT=130.0, AMT_ANNUITY=20.0,
              CNT_PAYMENT=10.0, RATE_DOWN_PAYMENT=0.2),
    solicitud(7, -600, NAME_CONTRACT_TYPE=CONSUMO, AMT_CREDIT=110.0, CNT_PAYMENT=0.0,
              RATE_DOWN_PAYMENT=0.0),
    solicitud(7, -300, AMT_CREDIT=90.0, AMT_ANNUITY=5.0, CNT_PAYMENT=6.0, RATE_DOWN_PAYMENT=0.5),
    rechazo(7, -100, AMT_CREDIT=0.0),
    rechazo(8, -700, AMT_APPLICATION=0.0, AMT_CREDIT=50.0, CNT_PAYMENT=np.nan,
            PRODUCT_COMBINATION=np.nan),
    solicitud(8, -400, NAME_CONTRACT_STATUS="Canceled", AMT_CREDIT=0.0, CNT_PAYMENT=0.0,
              PRODUCT_COMBINATION=np.nan),
    solicitud(9, -300, NAME_CONTRACT_TYPE=CONSUMO, RATE_DOWN_PAYMENT=-0.5),
    solicitud(9, -100, NAME_CONTRACT_TYPE=CONSUMO, RATE_DOWN_PAYMENT=0.3),
    solicitud(10, -250, AMT_ANNUITY=np.nan),
    rechazo(10, -200, AMT_ANNUITY=30.0),
    solicitud(11, -500, AMT_CREDIT=0.0),
    solicitud(11, -200, AMT_ANNUITY=0.0),
    solicitud(12, -900, DAYS_LAST_DUE_1ST_VERSION=100.0, DAYS_LAST_DUE=-300.0),
    solicitud(12, -400, DAYS_LAST_DUE_1ST_VERSION=50.0),
    solicitud(13, -900, DAYS_LAST_DUE_1ST_VERSION=-100.0, DAYS_LAST_DUE=-465.0),
    solicitud(13, -400, DAYS_LAST_DUE_1ST_VERSION=-50.0, DAYS_LAST_DUE=-50.0),
    solicitud(13, -200, DAYS_LAST_DUE_1ST_VERSION=0.0, DAYS_LAST_DUE=0.0),
    solicitud(14, -700, DAYS_LAST_DUE_1ST_VERSION=SENTINELA, DAYS_LAST_DUE=-200.0),
    solicitud(15, -300, DAYS_LAST_DUE_1ST_VERSION=300.0, DAYS_LAST_DUE=SENTINELA),
    solicitud(16, -300, HOUR_APPR_PROCESS_START=HORA, PRODUCT_COMBINATION="Cash Street: high",
              NAME_TYPE_SUITE=np.nan, NAME_CASH_LOAN_PURPOSE="Car repairs"),
    solicitud(16, -200, HOUR_APPR_PROCESS_START=HORA + 1, NAME_TYPE_SUITE="Family",
              NAME_CASH_LOAN_PURPOSE="Repairs"),
    solicitud(16, -100, PRODUCT_COMBINATION=np.nan, NAME_TYPE_SUITE=np.nan,
              NAME_CASH_LOAN_PURPOSE="XNA"),
    solicitud(17, -150, HOUR_APPR_PROCESS_START=HORA + 1, PRODUCT_COMBINATION=np.nan),
    solicitud(18, -120, HOUR_APPR_PROCESS_START=23.0, PRODUCT_COMBINATION="Card Street",
              NAME_CASH_LOAN_PURPOSE="Urgent needs"),
    solicitud(19, -110, HOUR_APPR_PROCESS_START=0.0, NAME_CASH_LOAN_PURPOSE="Medicine"),
    *[solicitud(20, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    *[solicitud(20, -400 - 60 * i) for i in range(COLA - ACTIVIDAD)],
    *[solicitud(21, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    solicitud(21, -round(LARGA * DIAS)),
    *[solicitud(22, -50 * (i + 1)) for i in range(ACTIVIDAD - 1)],
    *[solicitud(22, -400 - 60 * i) for i in range(COLA - ACTIVIDAD)],
    *[solicitud(23, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    solicitud(23, -round(LARGA * DIAS) + 1),
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
# La relación entre cifras, a mano y solo donde el cliente la ejercita: NaN es "sin fila en su
# denominador". El 7 lo calcula todo a la vez: cocientes 1,3, 1,1 y 0,9 (uno sobre el corte, uno
# justo en él y otro por debajo), plazos 10, 6 y 12 (el 0 no entra), coste de las dos aprobadas con
# cuota y plazo, y entrada de sus dos operaciones de consumo.
NAN = np.nan
ESPERADO_CIFRAS = {
    1: dict(concesion=1.0, sobre=0.0, plazo=12.0, coste=1.2, entrada=NAN, liquidada=NAN,
            por_vencer=NAN),
    4: dict(concesion=1.0, sobre=0.0, plazo=61.0, coste=6.1, entrada=NAN),
    5: dict(concesion=1.0, sobre=0.0, plazo=36.0, coste=1.2, entrada=NAN),
    7: dict(
        concesion=1.1, sobre=1 / 3, plazo=28 / 3,
        coste=(20 * 10 / 130 + 5 * 6 / 90) / 2, entrada=0.1,
    ),
    8: dict(concesion=NAN, sobre=NAN, plazo=NAN, coste=NAN, entrada=NAN, liquidada=NAN,
            por_vencer=NAN, calle=NAN, urgente=NAN),
    9: dict(concesion=1.0, sobre=0.0, plazo=12.0, coste=1.2, entrada=0.15),
    10: dict(concesion=1.0, sobre=0.0, plazo=12.0, coste=NAN, entrada=NAN),
    11: dict(concesion=1.0, sobre=0.0, plazo=12.0, coste=NAN, entrada=NAN),
    # el ciclo de vida: el 1 y el 8 no traen ninguna fecha y son NaN en las dos
    12: dict(liquidada=1.0, por_vencer=100.0),
    13: dict(liquidada=0.0, por_vencer=NAN),
    14: dict(liquidada=NAN, por_vencer=NAN),
    15: dict(liquidada=NAN, por_vencer=300.0),
}
COLUMNA_CIFRAS = {
    "concesion": "PREV_CREDIT_APPLICATION_RATIO",
    "sobre": "PREV_OVERGRANTED_RATIO",
    "plazo": "PREV_CNT_PAYMENT_MEAN",
    "coste": "PREV_IMPLIED_COST_MEAN",
    "entrada": "PREV_DOWN_PAYMENT_RATE_MEAN",
    "liquidada": "PREV_EARLY_SETTLED_FLAG",
    "por_vencer": "PREV_FUTURE_DUE_MAX",
    "calle": "PREV_STREET_RATIO",
    "urgente": "PREV_URGENT_PURPOSE_RATIO",
}
# La captación y la finalidad, a mano por cliente. Los clientes sin caso propio traen los valores
# por defecto: calle 0, hora temprana 0, sin acompañante 0 y urgente NaN, que no declaran finalidad.
ESPERADO_CAPTACION = {
    1: dict(calle=0.0, temprana=0.0, sin_acompanante=0.0, urgente=NAN, fantasma=0),
    8: dict(calle=NAN, temprana=0.0, sin_acompanante=0.0, urgente=NAN, fantasma=1),
    16: dict(calle=1 / 2, temprana=1 / 3, sin_acompanante=2 / 3, urgente=1 / 2, fantasma=1),
    17: dict(calle=NAN, temprana=0.0, sin_acompanante=0.0, urgente=NAN, fantasma=1),
    18: dict(calle=1.0, temprana=0.0, sin_acompanante=0.0, urgente=1.0, fantasma=0),
    19: dict(calle=0.0, temprana=1.0, sin_acompanante=0.0, urgente=0.0, fantasma=0),
}
COLUMNA_CAPTACION = {
    "calle": "PREV_STREET_RATIO",
    "temprana": "PREV_EARLY_HOUR_RATIO",
    "sin_acompanante": "PREV_NO_SUITE_RATIO",
    "urgente": "PREV_URGENT_PURPOSE_RATIO",
    "fantasma": "PREV_PHANTOM_FLAG",
}
# Las colas y el término, a mano. Los clientes sin caso propio traen las tres a 0: el 1, el 5 y el
# 365243 no llegan a ninguna cola
ESPERADO_COLAS = {
    1: dict(cola=0, actividad=0, corta_activa=0),
    5: dict(cola=0, actividad=0, corta_activa=0),
    20: dict(cola=1, actividad=1, corta_activa=1),
    21: dict(cola=0, actividad=1, corta_activa=0),
    22: dict(cola=0, actividad=0, corta_activa=0),
    23: dict(cola=0, actividad=1, corta_activa=1),
}
COLUMNA_COLAS = {
    "cola": "PREV_COUNT_COLA",
    "actividad": "PREV_ACTIVIDAD_12M_COLA",
    "corta_activa": "PREV_RELACION_CORTA_ACTIVA",
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
    # la relación entre cifras: las dos formas de quedar sin cociente y el borde del corte, con una
    # solicitud por encima y otra por debajo
    assert (prev.AMT_APPLICATION.eq(0) & prev.AMT_CREDIT.gt(0)).any()
    assert (prev.AMT_CREDIT.eq(0) & prev.AMT_APPLICATION.gt(0)).any()
    ratio = prev.AMT_CREDIT / prev.AMT_APPLICATION
    assert ratio.eq(SOBRE).any() and ratio.gt(SOBRE).any() and ratio.between(0.01, SOBRE).any()
    # un plazo 0 que sí está en el frame
    assert prev.CNT_PAYMENT.eq(0).any()
    # el 10: su única aprobada no tiene cuota y su Refused tiene las tres cifras, así que solo el
    # filtro de estado deja el coste sin dato
    diez = prev[prev.SK_ID_CURR == 10].set_index("NAME_CONTRACT_STATUS")
    assert diez.loc["Approved", "AMT_ANNUITY"] != diez.loc["Approved", "AMT_ANNUITY"]
    assert diez.loc["Refused", ["AMT_ANNUITY", "CNT_PAYMENT", "AMT_CREDIT"]].gt(0).all()
    # el 11: aprobadas con plazo, una con crédito 0 y cuota y otra con cuota 0 y crédito
    once = prev[prev.SK_ID_CURR == 11]
    assert once.NAME_CONTRACT_STATUS.eq("Approved").all() and once.CNT_PAYMENT.gt(0).all()
    assert (once.AMT_CREDIT.eq(0) & once.AMT_ANNUITY.gt(0)).any()
    assert (once.AMT_ANNUITY.eq(0) & once.AMT_CREDIT.gt(0)).any()
    # la entrada: consumo y cash con valor, y una negativa de consumo que la limpieza cambia
    entrada = prev[prev.RATE_DOWN_PAYMENT.notna()]
    assert set(entrada.NAME_CONTRACT_TYPE) == {CONSUMO, "Cash loans"}
    assert prev[prev.NAME_CONTRACT_TYPE.eq(CONSUMO)].RATE_DOWN_PAYMENT.lt(0).any()
    # el 8: con previas y todas sus filas fuera de todos los denominadores
    ocho = prev[prev.SK_ID_CURR == 8]
    assert len(ocho) == 2 and not (ocho.AMT_APPLICATION.gt(0) & ocho.AMT_CREDIT.gt(0)).any()
    assert not ocho.CNT_PAYMENT.gt(0).any() and not ocho.NAME_CONTRACT_STATUS.eq("Approved").any()
    # el ciclo de vida: adelanto por encima, justo en el corte y en la fecha prevista
    desfase = prev.DAYS_LAST_DUE_1ST_VERSION - prev.DAYS_LAST_DUE
    assert desfase.gt(ADELANTO).any() and desfase.eq(ADELANTO).any() and desfase.eq(0).any()
    # una liquidada con el fin previsto por delante, y una viva con otro fin por delante más cercano
    doce = prev[prev.SK_ID_CURR == 12]
    assert doce.DAYS_LAST_DUE.notna().sum() == 1 and doce.DAYS_LAST_DUE_1ST_VERSION.gt(0).all()
    assert doce.loc[doce.DAYS_LAST_DUE.notna(), "DAYS_LAST_DUE_1ST_VERSION"].item() > (
        doce.loc[doce.DAYS_LAST_DUE.isna(), "DAYS_LAST_DUE_1ST_VERSION"].item()
    )
    # el centinela en cada una de las dos fechas, cada uno en su cliente
    assert prev[prev.SK_ID_CURR == 14].DAYS_LAST_DUE_1ST_VERSION.eq(SENTINELA).all()
    assert prev[prev.SK_ID_CURR == 15].DAYS_LAST_DUE.eq(SENTINELA).all()
    # con previas y ninguna operación terminada, y con alguna terminada
    assert prev[prev.SK_ID_CURR == 1].DAYS_LAST_DUE.isna().all()
    # la captación: el 8 con todas sus filas sin combinación y sin finalidad, y el 17 igual
    assert prev[prev.SK_ID_CURR.isin([8, 17])].PRODUCT_COMBINATION.isna().all()
    assert prev[prev.SK_ID_CURR.isin([8, 17])].NAME_CASH_LOAN_PURPOSE.isin(["XAP", "XNA"]).all()
    # el 16: una fantasma junto a dos definidas, una con y otra sin calle, la hora en el corte y
    # una más arriba, un hueco de acompañante en dos filas, y la finalidad urgente, una casi
    # igual que no lo es y una XNA
    dieciseis = prev[prev.SK_ID_CURR == 16]
    assert dieciseis.PRODUCT_COMBINATION.isna().sum() == 1
    assert dieciseis.PRODUCT_COMBINATION.str.contains("Street", na=False).sum() == 1
    horas = dieciseis.HOUR_APPR_PROCESS_START
    assert HORA in set(horas) and (horas > HORA).any()
    acompanante = dieciseis.NAME_TYPE_SUITE
    assert acompanante.isna().sum() == 2 and acompanante.notna().sum() == 1
    assert "Car repairs" in URGENTES and "Repairs" not in URGENTES
    assert {"Car repairs", "Repairs", "XNA"} == set(dieciseis.NAME_CASH_LOAN_PURPOSE)
    # el 19 declara una finalidad que no es urgente y el 18 una que sí, así que el 0 y el 1 se ven
    assert prev[prev.SK_ID_CURR == 19].NAME_CASH_LOAN_PURPOSE.isin(URGENTES).sum() == 0
    assert prev[prev.SK_ID_CURR == 18].NAME_CASH_LOAN_PURPOSE.isin(URGENTES).all()
    # las colas: el conteo y la actividad justo en el corte y uno por debajo, y la más antigua justo
    # en el corte de la relación larga junto a otra más corta
    veinte, veintiuno, veintidos = (prev[prev.SK_ID_CURR == c] for c in (20, 21, 22))
    assert len(veinte) == COLA and len(veintidos) == COLA - 1
    for cliente in (veinte, veintiuno):
        assert cliente.DAYS_DECISION.gt(-VENTANA).sum() == ACTIVIDAD
    assert veintidos.DAYS_DECISION.gt(-VENTANA).sum() == ACTIVIDAD - 1
    assert -veintiuno.DAYS_DECISION.min() / DIAS == LARGA
    # el 23, un día por debajo del corte: corta con los días por año del registro y larga con 365
    veintitres = prev[prev.SK_ID_CURR == 23]
    assert -veintitres.DAYS_DECISION.min() / DIAS < LARGA <= -veintitres.DAYS_DECISION.min() / 365
    for corta in (veinte, veintidos):
        assert -corta.DAYS_DECISION.min() / DIAS < LARGA


@pytest.mark.parametrize("cliente", sorted(ESPERADO))
def test_el_agregado_de_cada_cliente_es_el_calculado_a_mano(prev, cliente):
    fila = agregar(prev).loc[cliente]
    for clave, esperado in ESPERADO[cliente].items():
        assert fila[COLUMNA[clave]] == pytest.approx(esperado), f"{COLUMNA[clave]}: {fila}"


@pytest.mark.parametrize("cliente", sorted(ESPERADO_CIFRAS))
def test_la_relacion_entre_cifras_de_cada_cliente_es_la_calculada_a_mano(prev, cliente):
    fila = agregar(prev).loc[cliente]
    for clave, esperado in ESPERADO_CIFRAS[cliente].items():
        columna = COLUMNA_CIFRAS[clave]
        assert fila[columna] == pytest.approx(esperado, nan_ok=True), f"{columna}: {fila}"


@pytest.mark.parametrize("cliente", sorted(ESPERADO_CAPTACION))
def test_la_captacion_y_la_finalidad_de_cada_cliente_son_las_calculadas_a_mano(prev, cliente):
    fila = agregar(prev).loc[cliente]
    for clave, esperado in ESPERADO_CAPTACION[cliente].items():
        columna = COLUMNA_CAPTACION[clave]
        assert fila[columna] == pytest.approx(esperado, nan_ok=True), f"{columna}: {fila}"


@pytest.mark.parametrize("cliente", sorted(ESPERADO_COLAS))
def test_las_colas_y_el_termino_de_cada_cliente_son_los_calculados_a_mano(prev, cliente):
    fila = agregar(prev).loc[cliente]
    for clave, esperado in ESPERADO_COLAS[cliente].items():
        assert fila[COLUMNA_COLAS[clave]] == esperado, f"{COLUMNA_COLAS[clave]}: {fila}"


def test_las_colas_y_el_termino_deja_dentro_sus_bordes(prev):
    """Cada corte, en las dos direcciones: uno más saca al cliente justo en el borde, y uno menos
    mete al que está justo por debajo."""
    def col(cortes, cliente, columna):
        return agregar(prev, cortes).loc[cliente, columna]

    assert col({"prev_count_cola": COLA + 1}, 20, "PREV_COUNT_COLA") == 0
    assert col({"prev_count_cola": COLA - 1}, 22, "PREV_COUNT_COLA") == 1
    assert col({"prev_actividad_12m_cola": ACTIVIDAD + 1}, 20, "PREV_ACTIVIDAD_12M_COLA") == 0
    assert col({"prev_actividad_12m_cola": ACTIVIDAD + 1}, 20, "PREV_RELACION_CORTA_ACTIVA") == 0
    assert col({"prev_actividad_12m_cola": ACTIVIDAD - 1}, 22, "PREV_ACTIVIDAD_12M_COLA") == 1
    assert col({"prev_actividad_12m_cola": ACTIVIDAD - 1}, 22, "PREV_RELACION_CORTA_ACTIVA") == 1
    # el 21 está justo en el corte de la relación larga: un poco más y pasa a corta
    assert col({}, 21, "PREV_RELACION_CORTA_ACTIVA") == 0
    assert col({"prev_relacion_larga_anios": LARGA + 0.01}, 21, "PREV_RELACION_CORTA_ACTIVA") == 1


def test_la_hora_temprana_deja_dentro_el_borde(prev):
    """En las dos direcciones: el 16 tiene una hora justo en el corte y otra una más arriba, y el
    corte una hora por debajo la saca, o una por encima mete la otra."""
    assert agregar(prev).loc[16, "PREV_EARLY_HOUR_RATIO"] == pytest.approx(1 / 3)
    assert agregar(prev, {"prev_hora_temprana_max": HORA - 1}).loc[16, "PREV_EARLY_HOUR_RATIO"] == 0
    subida = agregar(prev, {"prev_hora_temprana_max": HORA + 1})
    assert subida.loc[16, "PREV_EARLY_HOUR_RATIO"] == pytest.approx(2 / 3)


def test_la_finalidad_urgente_es_una_lista_exacta_y_no_un_contiene(prev):
    """"Repairs" no es "Car repairs": con la lista ampliada la proporción del 16 sube, con otra
    lista la del 19 pasa de 0 a 1, y una subcadena de una finalidad real no marca nada."""
    assert agregar(prev).loc[16, "PREV_URGENT_PURPOSE_RATIO"] == pytest.approx(1 / 2)
    ampliada = agregar(prev, {"prev_finalidades_urgentes": (*URGENTES, "Repairs")})
    assert ampliada.loc[16, "PREV_URGENT_PURPOSE_RATIO"] == 1
    otra = agregar(prev, {"prev_finalidades_urgentes": ("Medicine",)})
    assert otra.loc[[16, 19], "PREV_URGENT_PURPOSE_RATIO"].tolist() == [0.0, 1.0]
    subcadena = agregar(prev, {"prev_finalidades_urgentes": ("repairs",)})
    assert subcadena.loc[16, "PREV_URGENT_PURPOSE_RATIO"] == 0


def test_la_calle_y_la_finalidad_solo_cuentan_sobre_su_denominador(prev):
    """En las dos direcciones: con la combinación del 16 rellena su fila fantasma entra en el
    denominador de la calle (1/3), y con una finalidad real en la XNA entra en el de la urgente."""
    base = agregar(prev)
    assert base.loc[16, ["PREV_STREET_RATIO", "PREV_URGENT_PURPOSE_RATIO"]].tolist() == [0.5, 0.5]
    llena = prev.copy()
    fila = llena.index[(llena.SK_ID_CURR == 16) & llena.PRODUCT_COMBINATION.isna()]
    llena.loc[fila, "PRODUCT_COMBINATION"] = "Cash X-Sell: low"
    assert agregar(llena).loc[16, "PREV_STREET_RATIO"] == pytest.approx(1 / 3)
    llena.loc[fila, "NAME_CASH_LOAN_PURPOSE"] = "Repairs"
    assert agregar(llena).loc[16, "PREV_URGENT_PURPOSE_RATIO"] == pytest.approx(1 / 3)


def test_las_categoricas_como_category_dan_lo_mismo_que_como_texto(prev):
    """`load_table` puede traerlas como category, y un `str.contains` o un `isin` sobre ellas no
    tiene por qué dar lo mismo."""
    columnas = ["PRODUCT_COMBINATION", "NAME_TYPE_SUITE", "NAME_CASH_LOAN_PURPOSE"]
    categorica = prev.astype(dict.fromkeys(columnas, "category"))
    pd.testing.assert_frame_equal(agregar(categorica), agregar(prev))


def test_la_lista_de_finalidades_urgentes_es_medida_y_revienta_sin_fijar(prev):
    """Sale de ordenar las finalidades por tasa de default: no se consume sin refijarla."""
    assert parametro("prev_finalidades_urgentes").procedencia == "medido"
    sin_ella = {n: v for n, v in REFERENCIA.items() if n != "prev_finalidades_urgentes"}
    with pytest.raises(ValueError, match="prev_finalidades_urgentes"):
        agregar_previous(prev, sin_ella)


def test_la_sobreconcesion_deja_fuera_el_borde(prev):
    """En las dos direcciones: con el corte una centésima por debajo el 1,1 del 7 pasa a marcar."""
    assert agregar(prev).loc[7, "PREV_OVERGRANTED_RATIO"] == pytest.approx(1 / 3)
    baja = agregar(prev, {"prev_sobreconcesion_corte": SOBRE - 0.01})
    assert baja.loc[7, "PREV_OVERGRANTED_RATIO"] == pytest.approx(2 / 3)


def test_sin_dos_cifras_positivas_el_cociente_es_nan_y_no_infinito(prev):
    """El 8 tiene solicitud 0 con crédito positivo: sin la máscara sería infinito, y con solo la
    de la solicitud, su otra fila (crédito 0) daría 0 en vez de quedar fuera."""
    agregado = agregar(prev)
    assert not np.isinf(agregado.select_dtypes("float")).any().any()
    assert agregado.loc[8, ["PREV_CREDIT_APPLICATION_RATIO", "PREV_OVERGRANTED_RATIO"]].isna().all()


def test_cada_feature_con_denominador_propio_es_nan_sin_ninguna_fila_dentro(prev):
    """Con previas y ninguna fila en su denominador: NaN, y no 0 ni la media de todas las filas."""
    agregado = agregar(prev)
    columnas = list(DENOMINADOR)
    assert set(columnas) == set(COLUMNA_CIFRAS.values())
    assert agregado.loc[8, "PREV_APPLICATION_COUNT"] > 0
    assert agregado.loc[8, columnas].isna().all()
    # y cada una tiene dato en algún cliente: el 8 no es el fixture entero
    assert agregado[columnas].notna().any().all()


def test_la_entrada_solo_lee_consumo(prev):
    """El 7 tiene una entrada de cash de 0,5: si entrara, la media saldría 0,2333 y no 0,1."""
    assert agregar(prev).loc[7, "PREV_DOWN_PAYMENT_RATE_MEAN"] == pytest.approx(0.1)
    todo_consumo = prev.assign(NAME_CONTRACT_TYPE=CONSUMO)
    assert agregar(todo_consumo).loc[7, "PREV_DOWN_PAYMENT_RATE_MEAN"] == pytest.approx(0.7 / 3)


def test_la_liquidacion_deja_fuera_el_borde_y_es_nan_sin_operacion_terminada(prev):
    """En las dos direcciones: con el corte un día por debajo el borde del 13 marca, y quien no
    tiene ninguna operación terminada no es 0, es NaN (un 0 mezclaría "no evaluable" con "no
    liquidó pronto")."""
    agregado = agregar(prev)
    assert agregado.loc[[12, 13], "PREV_EARLY_SETTLED_FLAG"].tolist() == [1, 0]
    baja = agregar(prev, {"prev_adelanto_liquidacion_dias": ADELANTO - 1})
    assert baja.loc[13, "PREV_EARLY_SETTLED_FLAG"] == 1
    assert agregado.loc[1, "PREV_EARLY_SETTLED_FLAG"] != agregado.loc[1, "PREV_EARLY_SETTLED_FLAG"]


def test_lo_por_vencer_es_el_maximo_del_fin_previsto_aunque_la_operacion_este_cerrada(prev):
    """El 12 tiene una cerrada con el fin previsto en 100 y una viva en 50: el máximo es 100. Es la
    lectura literal del notebook, el plan pactado y no la deuda viva; con solo las vivas saldría
    50."""
    assert agregar(prev).loc[12, "PREV_FUTURE_DUE_MAX"] == 100
    solo_vivas = prev.assign(DAYS_LAST_DUE_1ST_VERSION=prev.DAYS_LAST_DUE_1ST_VERSION.where(
        prev.DAYS_LAST_DUE.isna()))
    assert agregar(solo_vivas).loc[12, "PREV_FUTURE_DUE_MAX"] == 50


def test_el_centinela_de_las_fechas_de_fin_no_cuenta_ni_como_terminada_ni_como_por_vencer(prev):
    """Sin limpiar, el 14 marcaría liquidación con un fin por vencer en 365243, y el 15 saldría
    con la bandera a 0 en vez de NaN: la limpieza tiene que ir antes."""
    agregado = agregar(prev)
    assert agregado.loc[[14, 15], "PREV_EARLY_SETTLED_FLAG"].isna().all()
    assert agregado.loc[14, "PREV_FUTURE_DUE_MAX"] != agregado.loc[14, "PREV_FUTURE_DUE_MAX"]
    assert agregado.loc[15, "PREV_FUTURE_DUE_MAX"] == 300
    crudo = prev[prev.SK_ID_CURR.isin([14, 15])]
    adelanto = crudo.DAYS_LAST_DUE_1ST_VERSION - crudo.DAYS_LAST_DUE
    assert adelanto.notna().all() and adelanto.iloc[0] > ADELANTO
    assert crudo.DAYS_LAST_DUE_1ST_VERSION.max() == SENTINELA


def test_el_coste_solo_lee_aprobadas(prev):
    """El 10 tiene una aprobada sin cuota y un Refused con las tres cifras: con todas las
    solicitudes el coste saldría 3,6 en vez de quedar sin dato."""
    assert np.isnan(agregar(prev).loc[10, "PREV_IMPLIED_COST_MEAN"])
    todas_aprobadas = agregar(prev.assign(NAME_CONTRACT_STATUS="Approved"))
    assert todas_aprobadas.loc[10, "PREV_IMPLIED_COST_MEAN"] == pytest.approx(3.6)


def test_agregar_el_crudo_y_el_limpio_da_lo_mismo(prev):
    """La entrada negativa del 9 es la primera cifra que lee un valor que la limpieza cambia: si la
    agregación dejara de limpiar por dentro, el crudo daría 0,-1 en vez de 0,15."""
    assert prev.RATE_DOWN_PAYMENT.lt(0).any()
    pd.testing.assert_frame_equal(agregar(limpiar_previous(prev)), agregar(prev))
    crudo = prev.RATE_DOWN_PAYMENT.where(prev.NAME_CONTRACT_TYPE.eq(CONSUMO))
    sin_limpiar = crudo[prev.SK_ID_CURR == 9].mean()
    assert sin_limpiar != pytest.approx(agregar(prev).loc[9, "PREV_DOWN_PAYMENT_RATE_MEAN"])


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
    "PREV_CREDIT_APPLICATION_RATIO": "float64",
    "PREV_OVERGRANTED_RATIO": "float64",
    "PREV_CNT_PAYMENT_MEAN": "float64",
    "PREV_IMPLIED_COST_MEAN": "float64",
    "PREV_DOWN_PAYMENT_RATE_MEAN": "float64",
    "PREV_EARLY_SETTLED_FLAG": "float64",
    "PREV_FUTURE_DUE_MAX": "float64",
    "PREV_STREET_RATIO": "float64",
    "PREV_EARLY_HOUR_RATIO": "float64",
    "PREV_NO_SUITE_RATIO": "float64",
    "PREV_URGENT_PURPOSE_RATIO": "float64",
    "PREV_PHANTOM_FLAG": "int8",
    "PREV_COUNT_COLA": "int8",
    "PREV_ACTIVIDAD_12M_COLA": "int8",
    "PREV_RELACION_CORTA_ACTIVA": "int8",
}


def test_la_salida_tiene_el_esquema_declarado(prev):
    assert agregar(prev).dtypes.astype(str).to_dict() == ESQUEMA


@pytest.mark.parametrize("tipo", ["int16", "float32", "float64"])
def test_el_tipo_de_las_fechas_de_entrada_no_cambia_la_salida(prev, tipo):
    """`load_table` las carga en float32 y un cliente suelto de la API llega en float64."""
    convertido = agregar(prev.astype({"DAYS_DECISION": tipo}))
    pd.testing.assert_frame_equal(convertido, agregar(prev))


def test_la_salida_cumple_el_contrato_con_la_receta(prev):
    """Las columnas son las de la receta menos los descartes firmes, más las declaradas sin receta.

    `HAS_PREV_APPLICATION` no sale de aquí sino de `unir_previous()`, que es donde existe el
    cliente sin solicitudes.
    """
    receta = yaml.safe_load(
        (RAIZ / "config" / "previous_application_features.yaml").read_text()
    )["features"]
    nombres = {f["nombre"] for f in receta}
    firmes = {f["nombre"] for f in receta if f["firmeza"] == "firme"}
    assert firmes == {
        "PREV_IMPLIED_COST_MAX",
        "PREV_EARLY_SETTLED_COUNT",
        "PREV_EARLY_SETTLED_RATIO",
    }
    esperadas = (nombres - firmes - {"HAS_PREV_APPLICATION"}) | set(COLUMNAS_SIN_RECETA)
    agregado = agregar(prev)
    assert set(agregado.columns) == esperadas
    assert agregado.index.name == "SK_ID_CURR" and agregado.index.is_unique
    assert not set(COLUMNAS_SIN_RECETA) & nombres, "una columna sin receta que sí está en ella"
    for motivo in COLUMNAS_SIN_RECETA.values():
        assert motivo.strip()


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
    with pytest.raises(KeyError, match="prev_ratio_rechazo_min_solicitudes"):
        agregar(prev, {"prev_ratio_rechazo_min_solicitudes": 2})


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


@pytest.mark.parametrize("columna", NUMERICAS_ORIGEN)
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

# La parte del 4.2 al 4.7 de la puerta del bloque 4, sobre sus dos poblaciones: la tabla cruda, que
# es la del EDA, y la de modelado, que es la del split. El 4.11 la completa con el resto.
PUERTA = {
    "crudo": {
        "con previas": 291_057,
        "sin previas": 16_454,
        "historial recortado": 53_934,
        "rechazo por scoring externo": 6_788,
        "plazo largo rechazado": 78,
        "cociente de concesión": 290_042,
        "sobreconcesión": 290_042,
        "plazo medio": 288_566,
        "coste implícito": 287_433,
        "entrada de consumo": 268_895,
        "con operación terminada": 267_623,
        "liquidación anticipada": 47_682,
        "algo por vencer": 147_954,
        "finalidad informada": 35_917,
        "registro fantasma": 284,
        "cola del conteo": 9_422,
        "cola de actividad": 45_771,
        "relación corta y activa": 21_913,
    },
    "modelado": {
        "con previas": 291_041,
        "sin previas": 16_451,
        "historial recortado": 53_933,
        "rechazo por scoring externo": 6_787,
        "plazo largo rechazado": 78,
        "cociente de concesión": 290_026,
        "sobreconcesión": 290_026,
        "plazo medio": 288_550,
        "coste implícito": 287_417,
        "entrada de consumo": 268_879,
        "con operación terminada": 267_608,
        "liquidación anticipada": 47_681,
        "algo por vencer": 147_944,
        "finalidad informada": 35_914,
        "registro fantasma": 284,
        "cola del conteo": 9_419,
        "cola de actividad": 45_764,
        "relación corta y activa": 21_912,
    },
}
COLAS = ("PREV_COUNT_COLA", "PREV_ACTIVIDAD_12M_COLA", "PREV_RELACION_CORTA_ACTIVA")
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
        "cociente de concesión": int(con.PREV_CREDIT_APPLICATION_RATIO.notna().sum()),
        "sobreconcesión": int(con.PREV_OVERGRANTED_RATIO.notna().sum()),
        "plazo medio": int(con.PREV_CNT_PAYMENT_MEAN.notna().sum()),
        "coste implícito": int(con.PREV_IMPLIED_COST_MEAN.notna().sum()),
        "entrada de consumo": int(con.PREV_DOWN_PAYMENT_RATE_MEAN.notna().sum()),
        "con operación terminada": int(con.PREV_EARLY_SETTLED_FLAG.notna().sum()),
        "liquidación anticipada": int(con.PREV_EARLY_SETTLED_FLAG.eq(1).sum()),
        "algo por vencer": int(con.PREV_FUTURE_DUE_MAX.notna().sum()),
        "finalidad informada": int(con.PREV_URGENT_PURPOSE_RATIO.notna().sum()),
        "registro fantasma": int(con.PREV_PHANTOM_FLAG.sum()),
        "cola del conteo": int(con.PREV_COUNT_COLA.sum()),
        "cola de actividad": int(con.PREV_ACTIVIDAD_12M_COLA.sum()),
        "relación corta y activa": int(con.PREV_RELACION_CORTA_ACTIVA.sum()),
    }
    # la calle y las otras tres proporciones tienen dato en todo el que tiene previas
    columnas_completas = ["PREV_REFUSED_RATIO", "PREV_STREET_RATIO", "PREV_EARLY_HOUR_RATIO",
                          "PREV_NO_SUITE_RATIO", "PREV_PHANTOM_FLAG"]
    assert con[columnas_completas].notna().all().all()
    assert medido == PUERTA[poblacion]
    assert len(unido) == len(poblaciones[poblacion])


@sin_dato_real
def test_sobre_el_dato_real_el_agregado_es_de_la_tabla_entera(dato_real):
    prev, agregado, _ = dato_real
    assert len(prev) == 1_670_214
    assert len(agregado) == CLIENTES_TABLA
    assert agregado.PREV_APPLICATION_COUNT.sum() == len(prev)
    # los no nulos de la relación entre cifras sobre la tabla entera, cada uno con su denominador
    assert agregado[list(DENOMINADOR)].notna().sum().to_dict() == {
        "PREV_CREDIT_APPLICATION_RATIO": 337_752,
        "PREV_OVERGRANTED_RATIO": 337_752,
        "PREV_CNT_PAYMENT_MEAN": 336_161,
        "PREV_IMPLIED_COST_MEAN": 334_894,
        "PREV_DOWN_PAYMENT_RATE_MEAN": 313_162,
        "PREV_EARLY_SETTLED_FLAG": 312_263,
        "PREV_FUTURE_DUE_MAX": 171_430,
        "PREV_STREET_RATIO": 338_857,
        "PREV_URGENT_PURPOSE_RATIO": 42_201,
    }
    assert agregado[list(COLAS)].sum().to_dict() == {
        "PREV_COUNT_COLA": 11_699,
        "PREV_ACTIVIDAD_12M_COLA": 52_113,
        "PREV_RELACION_CORTA_ACTIVA": 24_813,
    }
    # 346 filas sin combinación de producto, de 315 clientes, y ninguno con todas sus filas así
    assert prev.PRODUCT_COMBINATION.isna().sum() == 346
    assert agregado.PREV_PHANTOM_FLAG.sum() == 315


@sin_dato_real
def test_sobre_el_dato_real_cada_cliente_solo_da_lo_mismo_que_acompanado(dato_real):
    """El fila a fila sobre 300 al azar y los perfiles que el azar puede no traer.

    Son el del empate discordante, uno con XNA en la más antigua, el de más solicitudes, uno con
    una solicitud justo en -365, uno bajo el suelo del ritmo y el del identificador 365243. Del
    rechazo, uno con plazo largo solo en una solicitud Canceled, uno solo en Approved (que no
    marca), uno con todas sus solicitudes rechazadas, el de más rechazos y uno con scoring externo.
    De la relación entre cifras, del ciclo de vida, de la captación y de las colas, los del
    comentario de abajo.
    """
    prev, agregado, _ = dato_real
    perfiles = [
        *[193_980, 307_630, 187_868, 261_733, 100_068, 365_243],
        *[101_379, 297_641, 109_699, 265_681, 282_125],
        # de la relación entre cifras: solo con solicitud 0 y crédito positivo, la entrada negativa
        # de consumo (las dos de la tabla), el cociente justo en el corte, el único con consumo y
        # otro producto, y una aprobada con plazo 0 y cuota
        *[100_368, 133_068, 350_530, 277_619, 100_003, 100_006],
        # del ciclo de vida: el adelanto justo en el corte y uno un día por encima, el centinela en
        # el fin previsto y en el efectivo, un por vencer ya cerrado y uno sin operación terminada
        *[341_999, 163_755, 100_011, 100_014, 100_002, 100_022],
        # de la captación y la finalidad: un fantasma junto a solicitudes definidas, dos con
        # finalidades urgentes poco frecuentes, uno con urgente, no urgente y sin declarar, y uno
        # con horas a los dos lados del corte
        *[222_844, 417_884, 200_835, 297_922, 100_356, 100_035],
        # de las colas: exactamente 15 y exactamente 14 solicitudes, exactamente 4 y exactamente 3
        # en doce meses, la más antigua en -1461 y en -1460 con actividad alta (el borde de la
        # relación larga a los dos lados), y uno con las dos colas y el término
        *[100_082, 100_105, 100_025, 100_009, 104_221, 125_223, 100_302],
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


@sin_dato_real
def test_sobre_el_dato_real_la_relacion_corta_y_activa_es_superaditiva(dato_real):
    """Las cuatro esquinas de la celda 131 y sus +1,61pp, sobre la cruda y sobre la de modelado.

    Es un contraste y no una decisión: el corte de 4 años es de dominio y nada del objetivo entra
    en `src/`. La superaditividad es la de la esquina corta y activa frente a la suma de los otros
    tres efectos, que es lo que dice el notebook para conservar la cola de actividad.
    """
    _, agregado, _ = dato_real
    objetivos = {
        "crudo": load_table("application_train", usecols=["SK_ID_CURR", "TARGET"]),
        "modelado": cargar_split()[["SK_ID_CURR", "TARGET"]],
    }
    esquinas = {
        "crudo": {
            (True, 0): 125_505,
            (True, 1): 23_858,
            (False, 0): 119_781,
            (False, 1): 21_913,
        },
        "modelado": {
            (True, 0): 125_501,
            (True, 1): 23_852,
            (False, 0): 119_776,
            (False, 1): 21_912,
        },
    }
    for poblacion, objetivo in objetivos.items():
        unido = unir_previous(objetivo, agregado)
        con = unido[unido.HAS_PREV_APPLICATION == 1]
        larga = -con.PREV_DAYS_DECISION_MIN / DIAS >= LARGA
        grupo = con.groupby([larga, con.PREV_ACTIVIDAD_12M_COLA]).TARGET.agg(["size", "mean"])
        assert grupo["size"].to_dict() == esquinas[poblacion]
        tasa = (grupo["mean"] * 100).to_dict()
        aditivo = tasa[(False, 0)] + tasa[(True, 1)] - tasa[(True, 0)]
        assert tasa[(False, 1)] - aditivo == pytest.approx(1.61, abs=0.01)
        # la esquina es exactamente el término, no otra población parecida
        marcados = con[con.PREV_RELACION_CORTA_ACTIVA == 1]
        assert marcados.TARGET.mean() * 100 == pytest.approx(tasa[(False, 1)])
