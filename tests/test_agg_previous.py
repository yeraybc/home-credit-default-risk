"""Tests de la agregación de previous_application por cliente (puntos 4.2 a 4.7).

Todos sobre un frame sintético, así que corren en CI sin los CSV, salvo la puerta contra el dato
real del final, que se salta sin `previous_application.csv` y el split y solo corre en local.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import mannwhitneyu, norm

from src.config import RAIZ, ruta
from src.data.loader import TABLE_FILES, load_table
from src.features.agg_previous import (
    COLUMNAS_ORIGEN,
    COLUMNAS_SIN_RECETA,
    CORTES,
    DENOMINADOR,
    NUMERICAS_ORIGEN,
    POBLACIONES,
    agregar_previous,
    cociente_de_concesion,
    combinacion_definida,
    fin_de_ventana,
    finalidad_declarada,
    solicitudes_recientes,
    unir_previous,
)
from src.features.build_features import (
    ESTRATOS_RECENCIA_PREVIOUS,
    LECTURAS_RECENCIA_RELATIVA,
    REJILLA_SOBRECONCESION,
    ajustar_actividad_previous,
    ajustar_cola_previous,
    ajustar_finalidades_previous,
    ajustar_sobreconcesion_previous,
    informe_denominador_previous,
    informe_recencia_relativa_previous,
    lecturas_relativas_previous,
)
from src.features.cleaning import limpiar_previous
from src.features.eval import remedir_receta
from src.features.params import CORTES_POR_FEATURE, fijar_operativo, parametro, valor
from src.features.recipes import cargar_receta
from src.features.split import NOMBRE_FICHERO, cargar_split

# Los cortes salen del registro y no de literales: dos copias del mismo corte dejan el fixture
# adaptándose en silencio al que cambie.
DIAS = valor("dias_por_anio")
VENTANA = valor("prev_ventana_reciente_dias")
SUELO_ANIOS = valor("suelo_anios_denominador")
# Los `medido` los bloquea `valor()` hasta refijarlos sobre train: aquí se pasan con su referencia,
# porque solo se prueba el cómputo. El adelanto y el plazo largo son `dominio` desde el 4.8.
REFERENCIA = {
    n: parametro(n).valor_referencia for n in CORTES if parametro(n).procedencia == "medido"
}
LARGO = valor("prev_plazo_largo_cuotas")
SOBRE = REFERENCIA["prev_sobreconcesion_corte"]
ADELANTO = valor("prev_adelanto_liquidacion_dias")
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
# 18 dos filas de calle a las 23, una con finalidad urgente y otra sin finalidad (NaN, que la tabla
#    no tiene y solo puede llegar por la API): la que no la declara no entra en el denominador, así
#    que los dos ratios siguen valiendo 1
# 19 una finalidad informada y no urgente, a las 0: la proporción urgente vale 0 y no NaN
# Y para las colas y el término de la interacción (recientes son las de los últimos doce meses):
# 20 justo en la cola del conteo y en la de actividad, con la relación corta: las tres a 1
# 21 justo en la cola de actividad y con la más antigua justo en el corte de la relación larga: el
#    borde entra en la larga, así que el término vale 0
# 22 un conteo y una actividad por debajo de las dos colas, con la relación corta: las tres a 0
# 23 la más antigua un día por debajo del corte de la relación larga, con la actividad justo en su
#    cola: es corta con 365,25 días por año y sería larga con 365, así que el término vale 1
# Y para la recencia relativa del 4.10, cuyo origen es la última solicitud del cliente:
# 24 dos solicitudes el mismo día más reciente, una rechazada y otra aprobada: el empate del otro
#    extremo del historial, que un idxmax resolvería con la primera fila
# 25 una solicitud justo en el borde de la ventana relativa (a 365 días de la última, fuera) y otra
#    un día por dentro, y la de fuera es la única de calle, temprana y sin acompañante: los tres
#    ratios relativos valen 0 y los absolutos 1/3
# 26 su única solicitud reciente no tiene combinación de producto y la vieja sí: el denominador
#    relativo de la calle se queda vacío y el ratio es NaN donde el absoluto vale 0
# 27 dos rechazos el mismo día más reciente: la salida es un `max` y no un `sum`
CONSUMO = "Consumer loans"
CENTINELA = 365243.0
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
    solicitud(14, -700, DAYS_LAST_DUE_1ST_VERSION=CENTINELA, DAYS_LAST_DUE=-200.0),
    solicitud(15, -300, DAYS_LAST_DUE_1ST_VERSION=300.0, DAYS_LAST_DUE=CENTINELA),
    solicitud(16, -300, HOUR_APPR_PROCESS_START=HORA, PRODUCT_COMBINATION="Cash Street: high",
              NAME_TYPE_SUITE=np.nan, NAME_CASH_LOAN_PURPOSE="Car repairs"),
    solicitud(16, -200, HOUR_APPR_PROCESS_START=HORA + 1, NAME_TYPE_SUITE="Family",
              NAME_CASH_LOAN_PURPOSE="Repairs"),
    solicitud(16, -100, PRODUCT_COMBINATION=np.nan, NAME_TYPE_SUITE=np.nan,
              NAME_CASH_LOAN_PURPOSE="XNA"),
    solicitud(17, -150, HOUR_APPR_PROCESS_START=HORA + 1, PRODUCT_COMBINATION=np.nan),
    solicitud(18, -120, HOUR_APPR_PROCESS_START=23.0, PRODUCT_COMBINATION="Card Street",
              NAME_CASH_LOAN_PURPOSE="Urgent needs"),
    solicitud(18, -110, HOUR_APPR_PROCESS_START=23.0, PRODUCT_COMBINATION="Card Street",
              NAME_CASH_LOAN_PURPOSE=np.nan),
    solicitud(19, -110, HOUR_APPR_PROCESS_START=0.0, NAME_CASH_LOAN_PURPOSE="Medicine"),
    *[solicitud(20, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    *[solicitud(20, -400 - 60 * i) for i in range(COLA - ACTIVIDAD)],
    *[solicitud(21, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    solicitud(21, -round(LARGA * DIAS)),
    *[solicitud(22, -50 * (i + 1)) for i in range(ACTIVIDAD - 1)],
    *[solicitud(22, -400 - 60 * i) for i in range(COLA - ACTIVIDAD)],
    *[solicitud(23, -50 * (i + 1)) for i in range(ACTIVIDAD)],
    solicitud(23, -round(LARGA * DIAS) + 1),
    solicitud(24, -600),
    rechazo(24, -100),
    solicitud(24, -100),
    solicitud(25, -100),
    solicitud(25, -100 - VENTANA + 1),
    solicitud(25, -100 - VENTANA, PRODUCT_COMBINATION="Cash Street: low",
              HOUR_APPR_PROCESS_START=HORA, NAME_TYPE_SUITE=np.nan),
    solicitud(26, -50, PRODUCT_COMBINATION=np.nan),
    solicitud(26, -1000),
    solicitud(27, -900),
    rechazo(27, -50),
    rechazo(27, -50, motivo="SCOFR"),
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
            por_vencer=NAN),
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
    assert prev[prev.SK_ID_CURR == 14].DAYS_LAST_DUE_1ST_VERSION.eq(CENTINELA).all()
    assert prev[prev.SK_ID_CURR == 15].DAYS_LAST_DUE.eq(CENTINELA).all()
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
    # el 18: una urgente y otra sin finalidad, que es la única forma de ver el notna() del
    # denominador, porque la tabla real no trae ni un NaN en la finalidad
    finalidades = prev[prev.SK_ID_CURR == 18].NAME_CASH_LOAN_PURPOSE
    assert finalidades.isin(URGENTES).sum() == 1 and finalidades.isna().sum() == 1
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


def test_el_suelo_del_ritmo_se_reenvia(prev):
    """El 1 es el único por debajo del suelo, y es el único corte cuyo reenvío no se veía: con el
    valor del registro la salida es la misma lo lea de `cortes` o de `params.py`."""
    assert agregar(prev).loc[1, "PREV_APPLICATIONS_PER_YEAR"] == 2
    subido = agregar(prev, {"suelo_anios_denominador": 2})
    assert subido.loc[1, "PREV_APPLICATIONS_PER_YEAR"] == 0.5


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
    assert crudo.DAYS_LAST_DUE_1ST_VERSION.max() == CENTINELA


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
    with pytest.raises(KeyError, match="bureau_count_cola"):
        agregar(prev, {"bureau_count_cola": 18})


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


# Las dos listas van escritas a mano y no salen de la constante que prueban: derivadas de ella,
# quitar una columna del contrato **reduce** los tests en vez de romperlos, y la mutación pasa. Ya
# mordió dos veces, con la hora (4.6) y antes en el 4.3.
NUMERICAS_ESPERADAS = [
    "DAYS_DECISION",
    "CNT_PAYMENT",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "RATE_DOWN_PAYMENT",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "HOUR_APPR_PROCESS_START",
]
CLAVE_Y_CATEGORICAS = [
    "SK_ID_CURR",
    "NAME_CLIENT_TYPE",
    "NAME_CONTRACT_STATUS",
    "NAME_CONTRACT_TYPE",
    "CODE_REJECT_REASON",
    "PRODUCT_COMBINATION",
    "NAME_TYPE_SUITE",
    "NAME_CASH_LOAN_PURPOSE",
]


def test_el_contrato_de_columnas_de_origen_es_el_declarado():
    """Lo que hace que quitar una columna del contrato rompa y no encoja la parametrizada."""
    assert list(NUMERICAS_ORIGEN) == NUMERICAS_ESPERADAS
    assert list(COLUMNAS_ORIGEN) == CLAVE_Y_CATEGORICAS + NUMERICAS_ESPERADAS


@pytest.mark.parametrize("columna", NUMERICAS_ESPERADAS)
def test_una_numerica_que_no_es_numero_revienta_en_la_frontera(prev, columna):
    roto = prev.astype({columna: object})
    roto.loc[0, columna] = "ayer"
    with pytest.raises(ValueError, match="ayer"):
        agregar(roto)


@pytest.mark.parametrize("columna", CLAVE_Y_CATEGORICAS + NUMERICAS_ESPERADAS)
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


def test_las_poblaciones_son_exactamente_las_de_la_receta(prev):
    """Ninguna etiqueta de lo provisional sin máscara ni máscara sin usar, y cada población
    condicionada es exactamente la de las filas, contada aparte. En las banderas la n son los
    marcados y la puerta local no ve la población: esto es lo que la sostiene."""
    receta = cargar_receta("previous_application")["features"]
    medidas = {
        f["poblacion_medicion"]
        for f in receta
        if f["firmeza"] == "provisional" and f["tipo"] != "control"
    }
    assert medidas == set(POBLACIONES)
    # el bucle de abajo se salta las None: sin esto, una condicionada que pase a None no cae en CI
    sin_mascara = {e for e, m in POBLACIONES.items() if m is None}
    assert sin_mascara == {"global", "no nulos (auto-cond.)"}
    # el 99 no tiene solicitudes
    clientes = pd.DataFrame({"SK_ID_CURR": [99, *prev.SK_ID_CURR.unique()]})
    unido = unir_previous(clientes, agregar(prev)).set_index("SK_ID_CURR")
    # de las filas y no de las columnas que leen las propias máscaras; limpias, que el centinela
    # no es una fecha de fin
    p = limpiar_previous(prev)
    clientes_con = {
        "con previas": p,
        "con algún rechazo": p[p.NAME_CONTRACT_STATUS.eq("Refused")],
        "con operación terminada": p[p.DAYS_LAST_DUE_1ST_VERSION.notna() & p.DAYS_LAST_DUE.notna()],
        "con finalidad informada": p[finalidad_declarada(p)],
    }
    for etiqueta, filas in clientes_con.items():
        dentro = POBLACIONES[etiqueta](unido)
        assert dentro.dtype == bool, etiqueta
        assert set(dentro.index[dentro]) == set(filas.SK_ID_CURR), etiqueta
        if etiqueta != "con previas":
            assert (unido.HAS_PREV_APPLICATION.eq(1) & ~dentro).any(), etiqueta


# --- el 4.8, el refijado de las dos colas sobre train -----------------------------------------


def armar(grupos, filas_de):
    """Cada grupo es (dato, targets, parte[, extra]); `filas_de` da las filas de un cliente."""
    filas, clientes = [], []
    for dato, targets, parte, *extra in grupos:
        for target in targets:
            cliente = len(clientes) + 1
            clientes.append({"SK_ID_CURR": cliente, "TARGET": target, "split": parte})
            filas += filas_de(cliente, dato, *extra)
    return pd.DataFrame(filas), pd.DataFrame(clientes)


def escenario(*grupos):
    """Cada grupo es (solicitudes por cliente, targets, parte[, cuántas dentro de la ventana])."""

    def filas_de(cliente, total, dentro=0):
        return [solicitud(cliente, -30)] * dentro + [solicitud(cliente, -1000)] * (total - dentro)

    return armar(grupos, filas_de)


def refijar(ajustar, nombre, *grupos, sobrescribir=False):
    """El corte que sale de refijar sobre el escenario, y el barrido."""
    prev, clientes = escenario(*grupos)
    informe = ajustar(prev, clientes, clientes, sobrescribir)
    return valor(nombre), informe


# Con dos solicitudes o más la tasa baja (10,6% frente a 15%), con tres cruza (71% frente a 5%) y
# con cuatro separa todavía más (100% frente a 9%): el corte es 3, el primero que cruza, y ni el 2
# ni el de más delta. Todas fuera de la ventana, para que no mida las dos colas a la vez
GRUPOS_CONTEO = [
    (1, [1] * 3 + [0] * 17, "train"),
    (2, [0] * 40, "train"),
    (3, [1] * 3 + [0] * 2, "train"),
    (4, [1] * 2, "train"),
]
# diez clientes de dos solicitudes, todos impagados: si contaran, el corte bajaría a 2
GRUPOS_CONTEO_VALID = (2, [1] * 10, "valid")
# El mismo reparto medido sobre la ventana, con los veinte de una sola solicitud repetidos delante
# sin ninguna reciente: así el corte 1 no cruza (-3,06pp) y el primero que cruza sigue siendo el 3
GRUPOS_ACTIVIDAD = [GRUPOS_CONTEO[0], *[(n, t, p, n) for n, t, p in GRUPOS_CONTEO]]


def test_la_cola_del_conteo_es_el_primer_corte_que_cruza_y_no_el_de_mas_delta():
    corte, informe = refijar(ajustar_cola_previous, "prev_count_cola", *GRUPOS_CONTEO)
    assert corte == 3
    assert informe.loc[2, "delta_pp"] < valor("umbral_flags_pp") <= informe.loc[3, "delta_pp"]
    assert informe.delta_pp.idxmax() == 4
    assert informe.elegido.sum() == 1 and informe.loc[3, "elegido"]


def test_valid_no_mueve_la_cola_del_conteo():
    """Las dos direcciones: el grupo de valid no cuenta, y contado en train bajaría el corte a 2."""
    assert (
        refijar(ajustar_cola_previous, "prev_count_cola", *GRUPOS_CONTEO, GRUPOS_CONTEO_VALID)[0]
        == 3
    )
    en_train = (*GRUPOS_CONTEO_VALID[:2], "train")
    corte, _ = refijar(
        ajustar_cola_previous, "prev_count_cola", *GRUPOS_CONTEO, en_train, sobrescribir=True
    )
    assert corte == 2


def test_n_train_de_la_cola_del_conteo_son_los_clientes_de_train_con_previas():
    prev, clientes = escenario(*GRUPOS_CONTEO, GRUPOS_CONTEO_VALID)
    sin_previas = pd.DataFrame([{"SK_ID_CURR": 999, "TARGET": 1, "split": "train"}])
    ajustar_cola_previous(prev, pd.concat([clientes, sin_previas], ignore_index=True), clientes)
    assert parametro("prev_count_cola").n_train_operativo == 20 + 40 + 5 + 2


def test_una_cola_del_conteo_sin_senal_revienta():
    plana = [(1, [1] + [0] * 9, "train"), (3, [1] + [0] * 9, "train")]
    with pytest.raises(ValueError, match="ningún corte"):
        refijar(ajustar_cola_previous, "prev_count_cola", *plana)


def test_refijar_la_cola_del_conteo_otra_vez_exige_sobrescribir():
    refijar(ajustar_cola_previous, "prev_count_cola", *GRUPOS_CONTEO)
    with pytest.raises(ValueError, match="sobrescribir"):
        refijar(ajustar_cola_previous, "prev_count_cola", *GRUPOS_CONTEO)


def test_la_cola_de_actividad_solo_cuenta_lo_de_dentro_de_la_ventana():
    """Las dos direcciones: el mismo reparto elige 3 dentro de la ventana, y fuera no hay barrido.

    Es lo que la distingue de la cola del conteo: con las mismas solicitudes empujadas fuera de los
    doce meses, todos los clientes se quedan a cero y no queda ningún corte que medir.
    """
    corte, informe = refijar(
        ajustar_actividad_previous, "prev_actividad_12m_cola", *GRUPOS_ACTIVIDAD
    )
    assert corte == 3
    assert informe.loc[2, "delta_pp"] < valor("umbral_flags_pp") <= informe.loc[3, "delta_pp"]
    fuera = [(n, t, p) for n, t, p, *_ in GRUPOS_ACTIVIDAD]
    with pytest.raises(ValueError, match="ningún corte"):
        refijar(ajustar_actividad_previous, "prev_actividad_12m_cola", *fuera)


def test_la_cola_de_actividad_deja_fuera_el_borde_de_la_ventana():
    """La ventana usa `solicitudes_recientes()`, el mismo borde que `PREV_COUNT_12M` de la
    agregación: -365 días no cuenta. Con un `>=` a mano en vez de la función compartida el borde
    contaría, los dos grupos quedarían con la misma cuenta y no habría ningún corte que cruzar.
    """

    def filas_de(cliente, dia):
        return [solicitud(cliente, dia), solicitud(cliente, -1000)]

    en_el_borde = (-365, [0] * 30, "train")
    dentro = (-364, [1] * 8 + [0] * 2, "train")
    prev, clientes = armar([en_el_borde, dentro], filas_de)
    informe = ajustar_actividad_previous(prev, clientes, clientes)
    assert valor("prev_actividad_12m_cola") == 1
    assert informe.loc[1, "marcados"] == 10
    assert informe.loc[1, "delta_pp"] == pytest.approx(80.0)


def test_refijar_la_cola_de_actividad_otra_vez_exige_sobrescribir():
    refijar(ajustar_actividad_previous, "prev_actividad_12m_cola", *GRUPOS_ACTIVIDAD)
    with pytest.raises(ValueError, match="sobrescribir"):
        refijar(ajustar_actividad_previous, "prev_actividad_12m_cola", *GRUPOS_ACTIVIDAD)
    corte, _ = refijar(
        ajustar_actividad_previous, "prev_actividad_12m_cola", *GRUPOS_ACTIVIDAD, sobrescribir=True
    )
    assert corte == 3


def test_la_cola_de_actividad_puede_elegir_el_uno():
    """El guardián del barrido desde 1: aquí el cero es un nivel real y el 1 es el primer cruce.

    Empezando en 2, como las otras tres colas, este escenario se quedaría sin ningún corte que
    cruce: el 2 marca a cuatro clientes que no fallan y su delta es negativo.
    """
    uno = [(1, [0] * 20, "train"), (1, [1] * 4 + [0] * 6, "train", 1), (2, [0] * 4, "train", 2)]
    corte, informe = refijar(ajustar_actividad_previous, "prev_actividad_12m_cola", *uno)
    assert corte == 1
    assert informe.loc[1, "delta_pp"] > informe.loc[2, "delta_pp"]
    assert informe.index.min() == 1


def test_n_train_de_la_cola_de_actividad_cuenta_al_que_no_tiene_ninguna_reciente():
    """Los 20 sin actividad están en el denominador: son clientes con previas, no ausencias."""
    prev, clientes = escenario(*GRUPOS_ACTIVIDAD)
    ajustar_actividad_previous(prev, clientes, clientes)
    assert parametro("prev_actividad_12m_cola").n_train_operativo == 20 + 20 + 40 + 5 + 2


@pytest.mark.parametrize(
    "ajustar",
    [
        ajustar_cola_previous,
        ajustar_actividad_previous,
        ajustar_sobreconcesion_previous,
        ajustar_finalidades_previous,
        informe_denominador_previous,
    ],
    ids=["conteo", "actividad", "sobreconcesion", "finalidades", "denominador"],
)
def test_los_refijados_de_previous_limpian_por_dentro(ajustar):
    """Un estado fuera de dominio revienta desde el refijado, como desde la agregación.

    Ninguna de las lecturas usa una columna que la limpieza cambie, así que sin esta guarda
    quitar la llamada no movería ninguna cifra y nadie lo vería.
    """
    prev, clientes = escenario(*GRUPOS_CONTEO)
    prev.loc[0, "NAME_CONTRACT_STATUS"] = "Otro"
    with pytest.raises(ValueError, match="NAME_CONTRACT_STATUS"):
        ajustar(prev, clientes, clientes)


# --- el 4.8, el refijado de la sobreconcesion sobre train -------------------------------------


def escenario_concesion(*grupos):
    """Cada grupo es (cocientes de sus solicitudes, targets, parte); un número es una sola fila."""

    def filas_de(cliente, cocientes):
        return [
            solicitud(cliente, -100, AMT_APPLICATION=100.0, AMT_CREDIT=100.0 * cociente)
            for cociente in np.atleast_1d(cocientes)
        ]

    return armar(grupos, filas_de)


def refijar_concesion(*grupos, sobrescribir=False):
    """El corte que sale de refijar sobre el escenario, y el barrido."""
    prev, clientes = escenario_concesion(*grupos)
    informe = ajustar_sobreconcesion_previous(prev, clientes, clientes, sobrescribir)
    return valor("prev_sobreconcesion_corte"), informe


# Con una sola fila por cliente el r_rb de la proporción es la fracción de morosos marcados menos la
# de sanos: el 1,1 gana (0,50 frente a 0,41 del 1,05 y 0,12 del 1,2). El delta de la bandera es
# otra cosa y crece con el corte, hasta el 87pp del 1,2, que solo marca a los pocos del 1,6
GRUPOS_CONCESION = [
    (1.0, [1] * 20 + [0] * 300, "train"),
    (1.08, [1] * 5 + [0] * 60, "train"),
    (1.15, [1] * 30 + [0] * 20, "train"),
    (1.6, [1] * 8, "train"),
]
# cuatrocientos sanos con cociente 1,15, marcados por el 1,05 y el 1,1 y no por el 1,2: si contaran,
# el corte pasaría a 1,2
CONCESION_VALID = (1.15, [0] * 400, "valid")


def test_la_sobreconcesion_es_el_corte_de_mayor_rrb_y_no_el_de_mayor_delta():
    corte, informe = refijar_concesion(*GRUPOS_CONCESION)
    assert corte == 1.1
    assert informe.r_rb.idxmax() == 1.1
    assert informe.delta_bandera_pp.idxmax() != 1.1
    assert informe.elegido.sum() == 1 and informe.loc[1.1, "elegido"]
    assert informe.index.tolist() == list(REJILLA_SOBRECONCESION)


def test_valid_no_mueve_la_sobreconcesion():
    """Las dos direcciones: el grupo de valid no cuenta, y contado en train el corte pasa a 1,2."""
    assert refijar_concesion(*GRUPOS_CONCESION, CONCESION_VALID)[0] == 1.1
    en_train = (*CONCESION_VALID[:2], "train")
    corte, _ = refijar_concesion(*GRUPOS_CONCESION, en_train, sobrescribir=True)
    assert corte == 1.2


def test_n_train_de_la_sobreconcesion_son_los_clientes_de_train_con_cociente():
    """Ni el de solicitud 0, que no tiene cociente, ni el de valid, ni el que no tiene previas."""
    prev, clientes = escenario_concesion(*GRUPOS_CONCESION, CONCESION_VALID)
    sin_cociente = solicitud(999, -100, AMT_APPLICATION=0.0, AMT_CREDIT=100.0)
    sin_previas = {"SK_ID_CURR": 998, "TARGET": 1, "split": "train"}
    prev = pd.concat([prev, pd.DataFrame([sin_cociente])], ignore_index=True)
    clientes = pd.concat(
        [clientes, pd.DataFrame([{**sin_previas, "SK_ID_CURR": 999}, sin_previas])],
        ignore_index=True,
    )
    ajustar_sobreconcesion_previous(prev, clientes, clientes)
    assert parametro("prev_sobreconcesion_corte").n_train_operativo == 320 + 65 + 50 + 8


def test_la_sobreconcesion_mide_la_proporcion_del_cliente_y_no_la_bandera():
    """Cada morosa tiene alguna solicitud sobre el corte y cada sana una sola sobre el corte entre
    varias: con la proporción los morosos superan a los sanos (r_rb 1) y con un `max` empatarían."""
    dos_filas = [
        ([1.6, 1.0], [1], "train"),
        ([1.6, 1.6], [1], "train"),
        ([1.0], [0], "train"),
        ([1.6, 1.0, 1.0, 1.0], [0], "train"),
    ]
    _, informe = refijar_concesion(*dos_filas)
    assert informe.r_rb.round(6).eq(1.0).all()


def test_la_sobreconcesion_deja_fuera_el_borde_del_corte():
    """Un moroso justo en 1,1 no cuenta como sobreconcedido con el corte en 1,1 y sí con el 1,05.

    En el dato real son siete filas en ese borde. El crédito va escrito a mano: 100 * 1,1 daría
    110,00000000000001 y ya estaría por encima.
    """
    prev = pd.DataFrame(
        [solicitud(1, -100, AMT_CREDIT=110.0), solicitud(2, -100, AMT_CREDIT=100.0)]
    )
    clientes = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [1, 0], "split": "train"})
    informe = ajustar_sobreconcesion_previous(prev, clientes, clientes)
    assert informe.loc[1.05, "r_rb"] == pytest.approx(1.0)
    assert informe.loc[1.1, "r_rb"] == pytest.approx(0.0)


def test_refijar_la_sobreconcesion_otra_vez_exige_sobrescribir():
    refijar_concesion(*GRUPOS_CONCESION)
    with pytest.raises(ValueError, match="sobrescribir"):
        refijar_concesion(*GRUPOS_CONCESION)


def test_el_cociente_de_concesion_deja_fuera_las_dos_cifras_a_cero():
    """La solicitud a 0 daría infinito y el crédito a 0 daría 0: ninguno es un dato."""
    p = pd.DataFrame(
        {
            "AMT_APPLICATION": np.array([100.0, 0.0, 100.0, 0.0], dtype="float32"),
            "AMT_CREDIT": np.array([110.0, 100.0, 0.0, 0.0], dtype="float32"),
        }
    )
    cociente = cociente_de_concesion(p)
    assert cociente.notna().tolist() == [True, False, False, False]
    assert cociente.dtype == "float64"


# --- el 4.8, el refijado de las finalidades urgentes sobre train -------------------------------


def escenario_finalidad(*grupos):
    """Cada grupo es (finalidad, targets, parte[, solicitudes por cliente]), una por defecto."""

    def filas_de(cliente, finalidad, veces=1):
        return [solicitud(cliente, -100, NAME_CASH_LOAN_PURPOSE=finalidad)] * veces

    return armar(grupos, filas_de)


def refijar_finalidad(*grupos, sobrescribir=False):
    """La lista que sale de refijar sobre el escenario, y el barrido."""
    prev, clientes = escenario_finalidad(*grupos)
    informe = ajustar_finalidades_previous(prev, clientes, clientes, sobrescribir)
    return valor("prev_finalidades_urgentes"), informe


def con_tasa(n, morosos):
    return [1] * morosos + [0] * (n - morosos)


# La global de las declaradas es 22,08%. Alta (+7,9pp) y Beta (+11,9pp) la superan por mucho, y Beta
# tiene exactamente la n mínima. Justa la supera en +0,9pp y no cruza los 2pp. Rara tiene la tasa
# más alta y solo 20 solicitudes. Las tres de abajo no cuentan como declaradas: contadas, la global
# pasaría del 22,08% al 40% y entrarían XNA y XAP, con la n mínima y todas morosas
GRUPOS_FINALIDAD = [
    ("Repairs", con_tasa(300, 30), "train"),
    ("Alta", con_tasa(150, 45), "train"),
    ("Beta", con_tasa(100, 34), "train"),
    ("Justa", con_tasa(200, 46), "train"),
    ("Rara", con_tasa(20, 15), "train"),
    ("XNA", [1] * 100, "train"),
    ("XAP", [1] * 100, "train"),
    (np.nan, [1] * 30, "train"),
]
# cuatrocientos clientes de Justa, todos morosos: contados en train la llevarían sobre los 2pp
FINALIDAD_VALID = ("Justa", [1] * 400, "valid")


def test_las_finalidades_urgentes_son_las_que_superan_la_n_minima_y_los_2pp():
    """Beta en la n mínima exacta entra; Rara, con la mejor tasa y n de 20, no; Justa, encima de la
    global pero por debajo de los 2pp, tampoco."""
    assert valor("n_min_categoria") == 100, "el escenario pone a Beta en la n mínima"
    lista, informe = refijar_finalidad(*GRUPOS_FINALIDAD)
    assert lista == ("Alta", "Beta")
    assert informe.elegida.sum() == 2
    assert 0 < informe.loc["Justa", "delta_pp"] < valor("umbral_flags_pp")
    assert informe.loc["Rara", "n"] < valor("n_min_categoria")
    assert informe.tasa.idxmax() == "Rara"


def test_las_finalidades_sin_declarar_no_cuentan_ni_en_la_tasa_global():
    """XNA, XAP y el NaN no son finalidades y no entran en el barrido ni en la global."""
    lista, informe = refijar_finalidad(*GRUPOS_FINALIDAD)
    assert not {"XNA", "XAP"} & set(informe.index) and "XNA" not in lista
    declaradas = [g for g in GRUPOS_FINALIDAD if g[0] not in ("XNA", "XAP") and g[0] == g[0]]
    global_ = np.mean(sum((g[1] for g in declaradas), []))
    esperado = (informe.loc["Alta", "tasa"] / 100 - global_) * 100
    assert informe.loc["Alta", "delta_pp"] == pytest.approx(esperado)


def test_la_tasa_global_es_por_solicitud_y_no_por_cliente():
    """Repairs pesa cinco solicitudes por cliente: por solicitud la global es 10,5% y Alta, con 13%,
    la supera en 2,5pp; por cliente sería 11,5% y Alta se quedaría en 1,5pp. En el dato real leerla
    por cliente pasa la lista de tres finalidades a seis."""
    lista, informe = refijar_finalidad(
        ("Repairs", con_tasa(100, 10), "train", 5), ("Alta", con_tasa(100, 13), "train")
    )
    assert lista == ("Alta",)
    assert informe.loc["Repairs", "n"] == 500
    assert informe.loc["Alta", "delta_pp"] == pytest.approx(2.5)


def test_la_lista_sale_ordenada_y_como_tupla_aunque_la_categorica_no_lo_este():
    """`groupby` ordena las claves de una categórica por sus categorías, y `load_table` las deja en
    el orden que traiga el dato: con las categorías al revés, la lista saldría al revés."""
    prev, clientes = escenario_finalidad(*GRUPOS_FINALIDAD)
    niveles = sorted(prev.NAME_CASH_LOAN_PURPOSE.dropna().unique(), reverse=True)
    prev["NAME_CASH_LOAN_PURPOSE"] = pd.Categorical(
        prev.NAME_CASH_LOAN_PURPOSE, categories=niveles
    )
    ajustar_finalidades_previous(prev, clientes, clientes)
    lista = valor("prev_finalidades_urgentes")
    assert isinstance(lista, tuple) and len(lista) == 2 and list(lista) == sorted(lista)


def test_valid_no_mueve_las_finalidades():
    """Las dos direcciones: el grupo de valid no cuenta, y contado en train Justa entraría."""
    assert refijar_finalidad(*GRUPOS_FINALIDAD, FINALIDAD_VALID)[0] == ("Alta", "Beta")
    en_train = (*FINALIDAD_VALID[:2], "train")
    lista, _ = refijar_finalidad(*GRUPOS_FINALIDAD, en_train, sobrescribir=True)
    assert "Justa" in lista


def test_n_train_de_las_finalidades_son_los_clientes_de_train_con_finalidad_declarada():
    """Ni las sin declarar ni los de valid ni el cliente sin previas."""
    prev, clientes = escenario_finalidad(*GRUPOS_FINALIDAD, FINALIDAD_VALID)
    # el primer cliente con una segunda solicitud declarada: son clientes y no solicitudes
    prev = pd.concat([prev, prev.iloc[[0]]], ignore_index=True)
    sin_previas = pd.DataFrame([{"SK_ID_CURR": 9999, "TARGET": 1, "split": "train"}])
    clientes = pd.concat([clientes, sin_previas], ignore_index=True)
    ajustar_finalidades_previous(prev, clientes, clientes)
    assert parametro("prev_finalidades_urgentes").n_train_operativo == 300 + 150 + 100 + 200 + 20


def test_unas_finalidades_sin_senal_revientan():
    planas = [("A", con_tasa(200, 20), "train"), ("B", con_tasa(200, 20), "train")]
    with pytest.raises(ValueError, match="ninguna finalidad"):
        refijar_finalidad(*planas)


def test_refijar_las_finalidades_otra_vez_exige_sobrescribir():
    refijar_finalidad(*GRUPOS_FINALIDAD)
    with pytest.raises(ValueError, match="sobrescribir"):
        refijar_finalidad(*GRUPOS_FINALIDAD)


def test_la_finalidad_declarada_deja_fuera_el_nan_y_las_dos_etiquetas_de_no_declarada():
    p = pd.DataFrame({"NAME_CASH_LOAN_PURPOSE": ["Repairs", "XNA", "XAP", np.nan, "Urgent needs"]})
    assert finalidad_declarada(p).tolist() == [True, False, False, False, True]


# --- la puerta contra el dato real -----------------------------------------------------------

# --- el 4.9, el barrido del mínimo de denominador ----------------------------------------------

RECHAZADA = {"NAME_CONTRACT_STATUS": "Refused", "CODE_REJECT_REASON": "HC"}
SIN_COMBINACION = {"PRODUCT_COMBINATION": np.nan}


def escenario_denominador(*grupos):
    """Cada grupo es (solicitudes de cada cliente como campos a cambiar, targets, parte)."""

    def filas_de(cliente, solicitudes):
        return [solicitud(cliente, -100 - i, **campos) for i, campos in enumerate(solicitudes)]

    return armar(grupos, filas_de)


URGENTE = {"NAME_CASH_LOAN_PURPOSE": "Car repairs"}
NO_URGENTE = {"NAME_CASH_LOAN_PURPOSE": "Repairs"}

# Siete tipos de cliente, y tres de ellos separan los denominadores del conteo de solicitudes. El
# tercero tiene dos solicitudes pero solo una con combinación de producto, así que el rechazo cuenta
# dos y la calle una; el quinto tiene una con el crédito a 0, que no entra en la sobreconcesión, y
# el sexto declara finalidad solo en una. Los cuatro primeros no declaran ninguna finalidad.
GRUPOS_DENOMINADOR = [
    ([RECHAZADA], [1, 0, 0, 0], "train"),
    ([{}], [0] * 6, "train"),
    ([RECHAZADA, SIN_COMBINACION], [1, 0, 0], "train"),
    ([{}, {}], [1, 0, 0, 0, 0], "train"),
    ([{"AMT_CREDIT": 0.0}, {}], [1, 0, 0], "train"),
    ([URGENTE, {}], [1, 0, 0, 0], "train"),
    ([URGENTE, NO_URGENTE], [1, 0, 0], "train"),
]
CLIENTES_DENOMINADOR = sum(len(g[1]) for g in GRUPOS_DENOMINADOR)
# Los clientes con la proporción y los que tienen denominador de dos o más, escritos a mano por
# proporción: uno solo derivado de las máscaras del código no se rompería si se equivocan las dos.
N_ESPERADA = {
    "PREV_REFUSED_RATIO": (28, 18),
    "PREV_OVERGRANTED_RATIO": (28, 15),
    "PREV_STREET_RATIO": (28, 15),
    "PREV_NO_SUITE_RATIO": (28, 18),
    "PREV_EARLY_HOUR_RATIO": (28, 18),
    "PREV_URGENT_PURPOSE_RATIO": (7, 3),
}


def informe_de(*grupos):
    prev, clientes = escenario_denominador(*grupos)
    return informe_denominador_previous(prev, clientes, clientes, REFERENCIA)


def test_cada_proporcion_cuenta_sobre_su_propio_denominador():
    """Con el conteo de solicitudes las seis darían 18 con mínimo 2, y solo tres lo hacen."""
    assert CLIENTES_DENOMINADOR == 28, "el escenario cambió y la tabla de abajo no"
    informe = informe_de(*GRUPOS_DENOMINADOR)
    for feature, (con_valor, con_dos) in N_ESPERADA.items():
        assert informe.loc[(feature, 1), "n"] == con_valor, feature
        assert informe.loc[(feature, 2), "n"] == con_dos, feature


def test_lo_que_queda_y_lo_que_se_deja_fuera_suman_el_total_de_cada_proporcion():
    informe = informe_de(*GRUPOS_DENOMINADOR)
    for feature, filas in informe.groupby(level=0):
        total = filas.loc[(feature, 1), "n"]
        assert (filas["n"] + filas["n_fuera"] == total).all(), feature
        assert filas.loc[(feature, 1), "n_fuera"] == 0, "con mínimo 1 no queda nadie fuera"
        assert pd.isna(filas.loc[(feature, 1), "r_rb_fuera"]), feature


def test_el_grupo_que_deja_fuera_el_minimo_de_dos_se_lee_como_bandera():
    """Con denominador 1 la proporción vale 0 o 1: 25% de default con 1 frente al 0% con 0."""
    informe = informe_de(*GRUPOS_DENOMINADOR)
    assert informe.loc[("PREV_REFUSED_RATIO", 2), "delta_fuera_pp"] == pytest.approx(25.0)
    assert informe.loc[("PREV_REFUSED_RATIO", 2), "r_rb_fuera"] > 0
    # la calle vale 0 en todo el que se queda fuera, así que no hay delta que leer, y con mínimo 3
    # el grupo excluido ya tiene la mitad de un cliente y deja de ser una bandera
    assert np.isnan(informe.loc[("PREV_STREET_RATIO", 2), "delta_fuera_pp"])
    assert np.isnan(informe.loc[("PREV_STREET_RATIO", 2), "r_rb_fuera"]), "constante, no 0"
    assert np.isnan(informe.loc[("PREV_REFUSED_RATIO", 3), "delta_fuera_pp"])


def test_una_proporcion_sin_ninguna_solicitud_dentro_de_su_denominador_no_mide_nada():
    """Sin finalidad declarada en ningún cliente: ni n, ni r_rb ni delta, y no revienta."""
    sin_finalidad = GRUPOS_DENOMINADOR[:4]
    prev, _ = escenario_denominador(*sin_finalidad)
    assert not finalidad_declarada(prev).any(), "el escenario pierde el caso sin denominador"
    filas = informe_de(*sin_finalidad).loc["PREV_URGENT_PURPOSE_RATIO"]
    assert (filas["n"] == 0).all() and (filas["n_fuera"] == 0).all()
    assert filas[["r_rb", "r_rb_fuera", "delta_fuera_pp"]].isna().all().all()


def test_valid_no_entra_en_el_barrido_del_denominador():
    """Las dos direcciones: el grupo de valid no cuenta, y contado en train sumaría cinco."""
    valid = ([RECHAZADA], [1] * 5, "valid")
    pd.testing.assert_frame_equal(
        informe_de(*GRUPOS_DENOMINADOR, valid), informe_de(*GRUPOS_DENOMINADOR)
    )
    en_train = (*valid[:2], "train")
    assert informe_de(*GRUPOS_DENOMINADOR, en_train).loc[("PREV_REFUSED_RATIO", 1), "n"] == (
        CLIENTES_DENOMINADOR + 5
    )


def test_la_combinacion_definida_deja_fuera_solo_el_nan():
    p = pd.DataFrame({"PRODUCT_COMBINATION": ["Cash X-Sell: low", np.nan, "POS household: Street"]})
    assert combinacion_definida(p).tolist() == [True, False, True]


# --- el 4.10, la recencia relativa -------------------------------------------------------------

# Las seis lecturas a mano por cliente, con la ventana contada hacia atrás desde su última
# solicitud. `salida` es que alguna de las del día más reciente esté rechazada, `rechazo` la
# recencia del último rechazo contra ese mismo día, y las cuatro de abajo se cuentan solo sobre las
# solicitudes de la ventana relativa. Los clientes sin caso propio no entran aquí.
ESPERADO_RELATIVA = {
    # una sola solicitud: es su propio fin de ventana y entra en ella
    1: dict(salida=0.0, rechazo=NAN, c12=1.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # los dos rechazos del día más antiguo, a 1.490 días de la última
    3: dict(salida=0.0, rechazo=-1490.0, c12=1.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # el rechazo de -365 no es el último día, y con la ventana relativa entran dos y no una
    5: dict(salida=0.0, rechazo=-1.0, c12=2.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # su última solicitud es el rechazo: sale en rechazo y su recencia relativa es 0
    7: dict(salida=1.0, rechazo=0.0, c12=2.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # todo su historial cabe en la ventana relativa, así que los tres ratios son los absolutos
    16: dict(salida=0.0, rechazo=NAN, c12=3.0, calle=1 / 2, temprana=1 / 3, sin_acompanante=2 / 3),
    # el empate del día más reciente, una rechazada y otra no
    24: dict(salida=1.0, rechazo=0.0, c12=2.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # el borde: la de fuera es la única de calle, temprana y sin acompañante
    25: dict(salida=0.0, rechazo=NAN, c12=2.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
    # su única reciente no tiene combinación: el denominador relativo de la calle queda vacío
    26: dict(salida=0.0, rechazo=NAN, c12=1.0, calle=NAN, temprana=0.0, sin_acompanante=0.0),
    # dos rechazos el mismo día más reciente: la bandera vale 1 y no 2
    27: dict(salida=1.0, rechazo=0.0, c12=2.0, calle=0.0, temprana=0.0, sin_acompanante=0.0),
}
COLUMNA_RELATIVA = {
    "salida": "PREV_EXIT_REFUSED_FLAG",
    "rechazo": "PREV_DAYS_SINCE_REFUSED_REL",
    "c12": "PREV_COUNT_12M_REL",
    "calle": "PREV_STREET_RATIO_REL",
    "temprana": "PREV_EARLY_HOUR_RATIO_REL",
    "sin_acompanante": "PREV_NO_SUITE_RATIO_REL",
}
# La contraparte de cada una de las cuatro que la ventana decide, con la máscara que la reproduce.
# **No es la misma ventana en las dos familias:** el conteo se compara contra la de doce meses desde
# hoy, y los tres ratios contra el historial entero, que es sobre lo que los construye la matriz.
CONTRAPARTE_ABSOLUTA = {
    "PREV_COUNT_12M_REL": ("PREV_COUNT_12M", lambda f: solicitudes_recientes(f, VENTANA)),
    "PREV_STREET_RATIO_REL": ("PREV_STREET_RATIO", lambda f: pd.Series(True, index=f.index)),
    "PREV_EARLY_HOUR_RATIO_REL": ("PREV_EARLY_HOUR_RATIO", lambda f: pd.Series(True, f.index)),
    "PREV_NO_SUITE_RATIO_REL": ("PREV_NO_SUITE_RATIO", lambda f: pd.Series(True, f.index)),
}


def relativas(prev, ventana=None):
    """Las lecturas del 4.10 sobre el frame ya limpio.

    Sin `ventana` usa la relativa al fin de ventana del cliente, que es la del punto; con ella,
    la máscara que se le pase, para comprobar que reproduce lo que ya construye la agregación.
    """
    filas = limpiar_previous(prev)
    reciente = (
        solicitudes_recientes(filas, VENTANA, fin_de_ventana(filas))
        if ventana is None
        else ventana(filas)
    )
    return lecturas_relativas_previous(filas, reciente, HORA)


@pytest.mark.parametrize("cliente", sorted(ESPERADO_RELATIVA))
def test_cada_lectura_relativa_de_cada_cliente_es_la_calculada_a_mano(prev, cliente):
    fila = relativas(prev).loc[cliente]
    for clave, esperado in ESPERADO_RELATIVA[cliente].items():
        columna = COLUMNA_RELATIVA[clave]
        assert fila[columna] == pytest.approx(esperado, nan_ok=True), f"{columna}: {fila}"


def test_el_fixture_ejercita_cada_rama_de_la_recencia_relativa(prev):
    """El guardián del 4.10: sin uno de estos casos su test pasa sin mirar lo que dice mirar."""
    ultima = prev.DAYS_DECISION.eq(prev.groupby("SK_ID_CURR").DAYS_DECISION.transform("max"))
    estados = prev[ultima].groupby("SK_ID_CURR").NAME_CONTRACT_STATUS
    # el empate del día más reciente, con un estado de cada uno, y una salida en rechazo limpia
    assert sorted(estados.get_group(24)) == ["Approved", "Refused"]
    assert estados.get_group(7).tolist() == ["Refused"]
    # un rechazo que no es la última solicitud, y un cliente sin ninguno
    cinco = prev[prev.SK_ID_CURR == 5]
    assert cinco[cinco.NAME_CONTRACT_STATUS.eq("Refused")].DAYS_DECISION.max() < -364
    assert (prev[prev.SK_ID_CURR == 25].NAME_CONTRACT_STATUS != "Refused").all()
    # el borde de la ventana relativa a los dos lados, que no es el de la absoluta
    veinticinco = prev[prev.SK_ID_CURR == 25].DAYS_DECISION
    desde_el_fin = veinticinco - veinticinco.max()
    assert {-VENTANA, -VENTANA + 1} <= set(desde_el_fin)
    assert veinticinco.min() < -VENTANA, "sin esto el borde relativo sería el absoluto"
    # la de fuera de la ventana relativa es la única con calle, hora temprana y sin acompañante
    fuera = prev[prev.SK_ID_CURR.eq(25) & prev.DAYS_DECISION.eq(veinticinco.min())]
    assert fuera.PRODUCT_COMBINATION.str.contains("Street").all()
    assert fuera.HOUR_APPR_PROCESS_START.le(HORA).all() and fuera.NAME_TYPE_SUITE.isna().all()
    # el 26: la reciente sin combinación y la vieja con ella, que es el denominador que se vacía
    veintiseis = prev[prev.SK_ID_CURR == 26].set_index("DAYS_DECISION").PRODUCT_COMBINATION
    assert pd.isna(veintiseis.loc[-50]) and pd.notna(veintiseis.loc[-1000])
    # el 27: dos rechazos el mismo día más reciente, que es lo que separa un `max` de un `sum`
    veintisiete = prev[prev.SK_ID_CURR == 27]
    ultimos = veintisiete[veintisiete.DAYS_DECISION.eq(veintisiete.DAYS_DECISION.max())]
    assert len(ultimos) == 2 and ultimos.NAME_CONTRACT_STATUS.eq("Refused").all()


def test_con_la_ventana_absoluta_las_lecturas_dan_las_columnas_de_la_agregacion(prev):
    """La relativa y la absoluta son la misma cuenta con otra ventana, y eso no se supone.

    Es la guarda contra que las dos se separen por una diferencia de escritura: con la ventana de
    su contraparte, las cuatro tienen que salir clavadas a las columnas de la matriz.
    """
    agregado = agregar(prev)
    for lectura, (columna, ventana) in CONTRAPARTE_ABSOLUTA.items():
        pd.testing.assert_series_equal(
            relativas(prev, ventana)[lectura], agregado[columna].astype(float), check_names=False
        )


def test_el_borde_de_la_ventana_relativa_queda_fuera(prev):
    """Las dos direcciones: a 365 días de la última no entra y a 364 sí.

    Con `>=` el 25 contaría tres solicitudes y sus tres ratios pasarían de 0 a 1/3, que es
    justo lo que la ventana absoluta ya ve.
    """
    fila = relativas(prev).loc[25]
    assert fila["PREV_COUNT_12M_REL"] == 2
    filas = limpiar_previous(prev)
    reciente = solicitudes_recientes(filas, VENTANA, fin_de_ventana(filas))
    del_25 = filas[filas.SK_ID_CURR.eq(25)].assign(dentro=reciente)
    assert del_25.set_index("DAYS_DECISION").dentro.to_dict() == {
        -100: True,
        -100 - VENTANA + 1: True,
        -100 - VENTANA: False,
    }


def escenario_relativa(*grupos):
    """Cada grupo es (solicitudes como (días, si va rechazada[, campos]), targets, parte)."""

    def filas_de(cliente, solicitudes):
        return [
            (rechazo if rechazada else solicitud)(cliente, dias, **(campos[0] if campos else {}))
            for dias, rechazada, *campos in solicitudes
        ]

    return armar(grupos, filas_de)


# De calle, a hora temprana y sin acompañante a la vez: las tres proporciones de captación en una
# sola solicitud, para que las tres varíen entre clientes en vez de quedarse a cero.
CAPTADA = dict(
    PRODUCT_COMBINATION="Cash Street: low", HOUR_APPR_PROCESS_START=HORA, NAME_TYPE_SUITE=np.nan
)
# Un grupo por estrato de recencia, con la última solicitud dentro de cada uno, más un cliente sin
# ningún rechazo (la recencia del rechazo es NaN en los dos lados) y un grupo de valid. El segundo
# grupo tiene su solicitud captada fuera de la ventana relativa y el cuarto dentro, que es la
# diferencia que el informe tiene que ver.
GRUPOS_RELATIVA = [
    ([(-50, True, CAPTADA)], [1, 0, 0], "train"),
    ([(-50, False), (-500, True, CAPTADA)], [0] * 4, "train"),
    ([(-200, True, dict(motivo="SCOFR"))], [1, 0], "train"),
    ([(-400, True), (-410, False, CAPTADA)], [1, 0, 0], "train"),
    ([(-1000, True)], [0, 0], "train"),
    ([(-50, False)], [0, 0], "train"),
    ([(-50, True)], [1] * 5, "valid"),
]
CLIENTES_RELATIVA = sum(len(g[1]) for g in GRUPOS_RELATIVA if g[2] == "train")


def informe_relativa(*grupos):
    prev, clientes = escenario_relativa(*grupos)
    return informe_recencia_relativa_previous(prev, clientes, clientes, REFERENCIA)


def test_el_informe_mide_cada_lectura_contra_las_contrapartes_declaradas():
    informe = informe_relativa(*GRUPOS_RELATIVA)
    assert informe.index.names == ["lectura", "contraparte", "estrato"]
    emitidos = {(lectura, contraparte) for lectura, contraparte, _ in informe.index}
    assert emitidos == {
        (lectura, contraparte)
        for lectura, contrapartes in LECTURAS_RECENCIA_RELATIVA.items()
        for contraparte in contrapartes
    }
    assert informe.xs("todos", level="estrato")["tipo"].eq("flag").sum() == 2, "solo la salida"


def test_los_estratos_de_recencia_parten_la_poblacion_de_cada_lectura():
    """Los cuatro estratos suman el total, o sea que ninguno se solapa y ninguno se queda fuera."""
    informe = informe_relativa(*GRUPOS_RELATIVA)
    for clave, filas in informe.groupby(level=["lectura", "contraparte"]):
        total = filas.xs("todos", level="estrato")["n"].item()
        por_estrato = filas.drop("todos", level="estrato")["n"]
        assert por_estrato.sum() == total, clave
        assert len(por_estrato) == len(ESTRATOS_RECENCIA_PREVIOUS), clave


def test_el_informe_solo_mide_sobre_train():
    """Las dos direcciones: el grupo de valid no cuenta, y contado en train sumaría cinco."""
    informe = informe_relativa(*GRUPOS_RELATIVA)
    n = informe.loc[("PREV_COUNT_12M_REL", "PREV_COUNT_12M", "todos"), "n"]
    assert n == CLIENTES_RELATIVA
    en_train = (*GRUPOS_RELATIVA[-1][:2], "train")
    crecido = informe_relativa(*GRUPOS_RELATIVA[:-1], en_train)
    assert crecido.loc[("PREV_COUNT_12M_REL", "PREV_COUNT_12M", "todos"), "n"] == n + 5


def test_la_redundancia_va_solo_en_la_fila_global():
    """Por estrato es ruido, y una columna a medias sin declarar se leería como un 0."""
    informe = informe_relativa(*GRUPOS_RELATIVA)
    assert informe.xs("todos", level="estrato")["redundancia"].notna().all()
    assert informe.drop("todos", level="estrato")["redundancia"].isna().all()


def test_la_bandera_declara_sus_marcados_y_la_continua_no():
    """En las banderas la n son los marcados, la lección que el 2.3 dejó escrita."""
    informe = informe_relativa(*GRUPOS_RELATIVA).xs("todos", level="estrato")
    salida = informe.loc[("PREV_EXIT_REFUSED_FLAG", "PREV_REFUSED_RATIO > 0")]
    assert salida["n"] == CLIENTES_RELATIVA and salida["n_marcados"] == 3 + 2 + 3 + 2
    assert informe.loc[("PREV_COUNT_12M_REL", "PREV_COUNT_12M"), "n_marcados"] != (
        informe.loc[("PREV_COUNT_12M_REL", "PREV_COUNT_12M"), "n_marcados"]
    ), "una continua no tiene marcados"


def test_la_cobertura_que_pierde_la_relativa_va_contada(prev):
    """`n_solo_abs` son los clientes que la absoluta mide y la relativa no, como el 26."""
    clientes = pd.DataFrame(
        {"SK_ID_CURR": sorted(prev.SK_ID_CURR.unique()), "TARGET": 0, "split": "train"}
    )
    clientes.loc[clientes.SK_ID_CURR.eq(26), "TARGET"] = 1
    informe = informe_recencia_relativa_previous(prev, clientes, clientes, REFERENCIA)
    calle = informe.loc[("PREV_STREET_RATIO_REL", "PREV_STREET_RATIO", "todos")]
    assert calle["n_solo_abs"] == 1
    assert pd.isna(relativas(prev).loc[26, "PREV_STREET_RATIO_REL"])
    assert agregar(prev).loc[26, "PREV_STREET_RATIO"] == 0.0


def test_el_informe_no_cuenta_a_los_clientes_de_valid():
    """`n_solo_abs` es la cobertura que pierde train, así que valid no puede sumarle: 0 aquí.

    Sin `reindex` del agregado a los clientes de train, los cinco de valid contarían como "solo la
    absoluta los mide".
    """
    assert informe_relativa(*GRUPOS_RELATIVA)["n_solo_abs"].eq(0).all()


def test_una_contraparte_constante_da_redundancia_nan_sin_avisos():
    """Sin ningún rechazo por scoring externo esa bandera es constante y no hay Pearson que medir.

    `np.corrcoef` avisa de una división inválida con una columna constante, y los avisos se
    corrigen en origen. Es el único caso del escenario en el que la guarda tiene algo que guardar.
    """
    sin_scofr = [
        (([(-200, True)], *g[1:]) if g[0] == [(-200, True, {"motivo": "SCOFR"})] else g)
        for g in GRUPOS_RELATIVA
    ]
    assert sin_scofr != GRUPOS_RELATIVA
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        informe = informe_relativa(*sin_scofr).xs("todos", level="estrato")
    salida = informe.loc["PREV_EXIT_REFUSED_FLAG", "redundancia"]
    assert pd.isna(salida["PREV_REFUSED_SCOFR_FLAG"]) and salida["PREV_REFUSED_RATIO > 0"] > 0


def test_el_delta_de_las_banderas_es_el_calculado_a_mano():
    """Salen en rechazo 10 de 16: tres positivos, 30% frente al 0% de los otros seis.

    La contraparte de "algún rechazo" marca a 14, con los mismos tres positivos: 21,4286pp. Con
    la bandera leída como r_rb, o la contraparte como el ratio (que vale 0,5 en el cliente que
    se queda a medias y rompe el 0 y el 1), estas dos cifras no salen.
    """
    informe = informe_relativa(*GRUPOS_RELATIVA).xs("todos", level="estrato")
    salida = informe.loc["PREV_EXIT_REFUSED_FLAG"]
    assert salida.loc["PREV_REFUSED_RATIO > 0", "efecto"] == pytest.approx(30.0)
    assert salida.loc["PREV_REFUSED_RATIO > 0", "efecto_abs"] == pytest.approx(300 / 14)


def r_rb_a_mano(valores, target):
    """La r_rb en valor absoluto de `valores` contra `target`, sin pasar por el informe."""
    sanos, morosos = valores[target == 0], valores[target == 1]
    u, _ = mannwhitneyu(sanos, morosos)
    return abs(2 * u / (len(sanos) * len(morosos)) - 1)


def test_el_efecto_de_las_continuas_y_de_su_contraparte_es_el_calculado_a_mano():
    """La contraparte sale de `agregar_previous()` y la relativa de la lectura, cada una su r_rb.

    Con la contraparte igual a la propia lectura, las dos columnas serían la misma y el informe
    daría siempre un empate, que es la lectura de "no gana" sin haber medido nada.
    """
    prev, clientes = escenario_relativa(*GRUPOS_RELATIVA)
    train = clientes[clientes["split"] == "train"].set_index("SK_ID_CURR")["TARGET"]
    rel, agregado = relativas(prev).loc[train.index], agregar(prev).loc[train.index]
    ultimo_rechazo = (
        prev[prev.NAME_CONTRACT_STATUS.eq("Refused")].groupby("SK_ID_CURR").DAYS_DECISION.max()
    )
    contrapartes = {
        "PREV_COUNT_12M_REL": {
            "PREV_COUNT_12M": agregado["PREV_COUNT_12M"],
            "PREV_APPLICATIONS_PER_YEAR": agregado["PREV_APPLICATIONS_PER_YEAR"],
        },
        "PREV_STREET_RATIO_REL": {"PREV_STREET_RATIO": agregado["PREV_STREET_RATIO"]},
        "PREV_EARLY_HOUR_RATIO_REL": {"PREV_EARLY_HOUR_RATIO": agregado["PREV_EARLY_HOUR_RATIO"]},
        "PREV_NO_SUITE_RATIO_REL": {"PREV_NO_SUITE_RATIO": agregado["PREV_NO_SUITE_RATIO"]},
        "PREV_DAYS_SINCE_REFUSED_REL": {"PREV_DAYS_SINCE_REFUSED": ultimo_rechazo},
    }
    informe = informe_relativa(*GRUPOS_RELATIVA).xs("todos", level="estrato")
    medidos = 0
    for lectura, cs in contrapartes.items():
        for nombre, absoluta in cs.items():
            juntos = pd.DataFrame({"rel": rel[lectura], "abs": absoluta, "t": train}).dropna()
            fila = informe.loc[(lectura, nombre)]
            if juntos["rel"].nunique() < 2 or juntos["abs"].nunique() < 2:
                continue
            medidos += 1
            assert fila["efecto"] == pytest.approx(r_rb_a_mano(juntos["rel"], juntos["t"]))
            assert fila["efecto_abs"] == pytest.approx(r_rb_a_mano(juntos["abs"], juntos["t"]))
    assert medidos >= 4, "el escenario dejó casi todas las lecturas constantes"


# Un cliente a cada lado de cada borde de los estratos, todos con una solicitud muy antigua para
# que la recencia sea la de la última y no la de la primera. Los estratos son (desde, hasta]: el
# borde superior está dentro, así que -180 es de "6 a 12m" y -730 de "mas de 24m".
BORDES_ESTRATOS = {
    -179: "hasta 6m",
    -180: "6 a 12m",
    -364: "6 a 12m",
    -365: "12 a 24m",
    -729: "12 a 24m",
    -730: "mas de 24m",
}


def test_cada_borde_de_los_estratos_cae_en_su_lado():
    """La recencia es la de la última solicitud y el borde superior de cada estrato está dentro.

    Con `>=` en el inferior el -180 contaría en dos, y con `<` en el superior no contaría en
    ninguno; con la primera solicitud como recencia todos caerían en "mas de 24m".
    """
    grupos = [([(-2000, False), (dias, False)], [0, 1], "train") for dias in BORDES_ESTRATOS]
    informe = informe_relativa(*grupos)
    n = informe.xs(("PREV_COUNT_12M_REL", "PREV_COUNT_12M"), level=("lectura", "contraparte"))["n"]
    esperado = {estrato: 2 * sum(e == estrato for e in BORDES_ESTRATOS.values())
                for estrato in ESTRATOS_RECENCIA_PREVIOUS}
    assert n.drop("todos").to_dict() == esperado
    assert n["todos"] == 2 * len(BORDES_ESTRATOS)
    assert set(BORDES_ESTRATOS.values()) == set(ESTRATOS_RECENCIA_PREVIOUS), "un estrato sin borde"


sin_dato_real = pytest.mark.skipif(
    not (ruta("raw_data") / TABLE_FILES["previous_application"]).exists()
    or not (ruta("processed_data") / NOMBRE_FICHERO).exists(),
    reason="data/raw y el split no viajan con el repo",
)

# La puerta del bloque 4, sobre sus dos poblaciones: la tabla cruda, que es la del EDA, y la de
# modelado, que es la del split. Los efectos remedidos sobre train van en su propio test (4.11).
PUERTA = {
    "crudo": {
        "con previas": 291_057,
        "sin previas": 16_454,
        "historial recortado": 53_934,
        "con algún rechazo": 100_294,
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
        "con algún rechazo": 100_284,
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
        "con algún rechazo": int(POBLACIONES["con algún rechazo"](con).sum()),
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


# --- el 4.8 contra el dato real ----------------------------------------------------------------

# El barrido de la celda 153 sobre los 307.511: por corte, marcados, cobertura en % de los que
# tienen previas, y delta en pp. Los dos que el EDA registró son el 15 y el 4
BARRIDO_COLAS_EDA = {
    "PREV_APPLICATION_COUNT": {
        8: (55_694, 19.14, 1.21),
        11: (25_505, 8.76, 1.91),
        15: (9_422, 3.24, 2.97),
        20: (3_083, 1.06, 3.49),
    },
    "PREV_COUNT_12M": {
        2: (107_206, 36.83, 1.59),
        3: (68_888, 23.67, 2.01),
        4: (45_771, 15.73, 2.41),
        5: (30_983, 10.64, 2.82),
    },
}


@sin_dato_real
def test_las_dos_colas_reproducen_el_barrido_del_eda(dato_real):
    """Las dos rejillas de la celda 153, sobre la población del EDA y leídas del agregado.

    Es lo que separa un corte que se mueve porque el split es otro de un corte que se mueve porque
    el código mide otra cosa: aquí tiene que salir clavado lo que salió en el notebook.
    """
    _, agregado, _ = dato_real
    crudo = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    unido = unir_previous(crudo, agregado)
    con = unido[unido.HAS_PREV_APPLICATION == 1]
    for columna, barrido in BARRIDO_COLAS_EDA.items():
        for corte, esperado in barrido.items():
            cola = con[columna].ge(corte)
            delta = (con.TARGET[cola].mean() - con.TARGET[~cola].mean()) * 100
            medido = (int(cola.sum()), round(cola.mean() * 100, 2), round(delta, 2))
            assert medido == esperado, (columna, corte)


@sin_dato_real
@pytest.mark.parametrize(
    ("ajustar", "nombre", "esperado", "vecinos"),
    [
        (ajustar_cola_previous, "prev_count_cola", 11, {10: (26_356, 1.81), 11: (20_319, 2.04)}),
        (
            ajustar_actividad_previous,
            "prev_actividad_12m_cola",
            4,
            {3: (55_049, 1.99), 4: (36_575, 2.31)},
        ),
    ],
    ids=["conteo", "actividad"],
)
def test_sobre_el_split_las_dos_colas_se_refijan(dato_real, ajustar, nombre, esperado, vecinos):
    """El conteo baja de 15 a 11 y la actividad se queda en 4, sobre los 232.793 de train.

    Los dos vecinos van al lado porque los dos cortes se deciden por poco: el 10 se queda en
    +1,81pp y el 3 en +1,99pp, a 0,014pp del umbral.
    """
    split = cargar_split()
    informe = ajustar(dato_real[0], split, split)
    assert valor(nombre) == esperado
    assert parametro(nombre).n_train_operativo == 232_793
    cortes = sorted(vecinos)
    assert informe.loc[cortes, "marcados"].tolist() == [v[0] for v in vecinos.values()]
    assert informe.loc[cortes, "delta_pp"].round(2).tolist() == [v[1] for v in vecinos.values()]


# El barrido de la celda 160 sobre los 307.511, r_rb de la proporción por corte de la rejilla
BARRIDO_SOBRECONCESION_EDA = [0.0832, 0.1194, 0.0809, 0.0455, 0.0230, 0.0094]


@sin_dato_real
def test_el_barrido_de_la_sobreconcesion_reproduce_el_eda(dato_real):
    """Las seis r_rb de la celda 160 sobre los 290.042 con las dos cifras positivas.

    Va sobre la población del EDA y con el barrido de la función de ajuste, que sobre train solo
    cambia la población: aquí tiene que salir clavado lo que salió en el notebook.
    """
    prev, _, _ = dato_real
    crudo = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    informe = ajustar_sobreconcesion_previous(prev, crudo, crudo.assign(split="train"))
    assert informe.n.eq(290_042).all()
    assert informe.r_rb.round(4).tolist() == BARRIDO_SOBRECONCESION_EDA


@sin_dato_real
def test_sobre_el_split_la_sobreconcesion_se_queda_en_1_1(dato_real):
    """El 1,1 con r_rb 0,1208 sobre los 231.992 de train, con el 1,05 y el 1,2 en 0,0837 y 0,0820.

    La bandera daría otro corte: su delta crece con el corte hasta +9,28pp en 1,5.
    """
    split = cargar_split()
    informe = ajustar_sobreconcesion_previous(dato_real[0], split, split)
    assert valor("prev_sobreconcesion_corte") == 1.1
    assert parametro("prev_sobreconcesion_corte").n_train_operativo == 231_992
    assert informe.r_rb.round(4).tolist() == [0.0837, 0.1208, 0.0820, 0.0465, 0.0231, 0.0096]
    assert informe.delta_bandera_pp.idxmax() == 1.5


# La celda 92 sobre los 307.511, por solicitud: n y tasa de las finalidades con n de 100 o más que
# quedan por encima de la global de las declaradas (13,03% sobre 59.413), de mayor a menor tasa
FINALIDADES_EDA = {
    "Car repairs": (691, 18.38),
    "Gasification / water supply": (251, 17.93),
    "Payments on other loans": (1_573, 16.02),
    "Urgent needs": (7_236, 14.95),
    "Building a house or an annex": (2_344, 13.82),
    "Medicine": (1_871, 13.42),
    "Repairs": (20_117, 13.00),
}


@sin_dato_real
def test_las_finalidades_reproducen_la_celda_92_del_eda(dato_real):
    """Las cinco con n de 100 o más son las cinco primeras por tasa, con Medicine y Repairs justo
    detrás, y esas son las siete cifras del EDA. La lista que registró es la de las cinco más las
    dos de menos de 100 solicitudes."""
    prev, _, _ = dato_real
    crudo = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    informe = ajustar_finalidades_previous(prev, crudo, crudo.assign(split="train"))
    grandes = informe[informe.n >= valor("n_min_categoria")].head(7)
    assert grandes.index.tolist() == list(FINALIDADES_EDA)
    medido = dict(zip(grandes.index, zip(grandes.n, grandes.tasa.round(2))))
    assert medido == FINALIDADES_EDA
    assert informe.n.sum() == 59_413
    assert set(grandes.index[:5]) <= set(parametro("prev_finalidades_urgentes").valor_referencia)


@sin_dato_real
def test_sobre_el_split_las_finalidades_urgentes_son_tres(dato_real):
    """Car repairs, Gasification y Payments, sobre los 28.564 de train, con Urgent needs a 0,15pp.

    Con el crudo la regla da lo mismo: son las tres que pasan los 2pp, y la cuarta se queda en
    +1,93pp. El 4.11 remide el efecto con esta lista y no con la del EDA.
    """
    split = cargar_split()
    informe = ajustar_finalidades_previous(dato_real[0], split, split)
    assert valor("prev_finalidades_urgentes") == (
        "Car repairs",
        "Gasification / water supply",
        "Payments on other loans",
    )
    assert parametro("prev_finalidades_urgentes").n_train_operativo == 28_564
    urgentes = informe.loc[list(valor("prev_finalidades_urgentes"))]
    assert urgentes.n.tolist() == [565, 192, 1_277]
    assert urgentes.delta_pp.round(2).tolist() == [4.18, 5.76, 3.85]
    assert informe.loc["Urgent needs", "n"] == 5_803
    assert informe.loc["Urgent needs", "delta_pp"].round(2) == 1.85


def delta_en_train(prev, cortes, columna, poblacion=None):
    """Marcados y delta de la bandera en pp sobre los clientes de train, con el corte pasado.

    `poblacion` es la columna que ha de ser no nula para entrar: la bandera de liquidación solo se
    evalúa en quien tiene una operación terminada. Sin ella, todos los clientes con previas.
    """
    split = cargar_split()
    train = split.loc[split.split == "train", ["SK_ID_CURR", "TARGET"]]
    unido = unir_previous(train, agregar(prev, cortes))
    con = unido[unido.HAS_PREV_APPLICATION == 1]
    if poblacion is not None:
        con = con[con[poblacion].notna()]
    marcados = con[columna] == 1
    delta = (con.TARGET[marcados].mean() - con.TARGET[~marcados].mean()) * 100
    return len(con), int(marcados.sum()), round(delta, 2)


@sin_dato_real
def test_contraste_del_adelanto_de_liquidacion(dato_real):
    """La bandera cruza los 2pp con 365, 545, 730 y 1.095 días y no con 270: empieza en el año.

    Por eso el corte es de dominio y no medido: el delta sube casi continuo con el corte, sin pico,
    y el primer cruce dependería de la rejilla. Sobre los 214.134 clientes de train con operación
    terminada: +1,83pp con 270, +2,26pp con 365, +2,52pp con 545, +3,00pp con 730 y +2,87pp con
    1.095.
    """
    esperado = {
        270: (48_512, 1.83),
        365: (38_060, 2.26),
        545: (26_898, 2.52),
        730: (18_825, 3.00),
        1_095: (8_758, 2.87),
    }
    medido = {}
    for dias in esperado:
        n, marcados, delta = delta_en_train(
            dato_real[0],
            {"prev_adelanto_liquidacion_dias": dias},
            "PREV_EARLY_SETTLED_FLAG",
            "PREV_EARLY_SETTLED_FLAG",
        )
        assert n == 214_134, dias
        medido[dias] = (marcados, delta)
    assert medido == esperado
    umbral = valor("umbral_flags_pp")
    assert medido[270][1] < umbral
    assert all(delta >= umbral for dias, (_, delta) in medido.items() if dias >= 365)


@sin_dato_real
def test_contraste_del_plazo_largo(dato_real):
    """Con 60 y con 66 cuotas la bandera separa más de 10pp; con 48 y 54 mide la masa de 60 cuotas.

    Sobre los 232.793 clientes de train con previas: +14,62pp con 57 marcados en 60, +16,81pp con
    52 en 66, y +2,35pp y +2,27pp con 12.832 y 12.669 en 48 y 54. El plazo solo toma valores
    discretos, y el 60 es donde acaba la masa de solicitudes no aprobadas a exactamente 60 cuotas.
    """
    esperado = {
        48: (12_832, 2.35),
        54: (12_669, 2.27),
        60: (57, 14.62),
        66: (52, 16.81),
    }
    medido = {}
    for cuotas in esperado:
        n, marcados, delta = delta_en_train(
            dato_real[0], {"prev_plazo_largo_cuotas": cuotas}, "PREV_REFUSED_LONG_TERM_FLAG"
        )
        assert n == 232_793, cuotas
        medido[cuotas] = (marcados, delta)
    assert medido == esperado
    assert all(medido[c][1] > 10 for c in (60, 66))
    assert all(medido[c][1] < 5 for c in (48, 54))


def con_previas_en_train(prev, cortes):
    """Los clientes de train con previas, con la agregación construida con los cortes pasados."""
    split = cargar_split()
    train = split.loc[split.split == "train", ["SK_ID_CURR", "TARGET"]]
    unido = unir_previous(train, agregar(prev, cortes))
    return unido[unido.HAS_PREV_APPLICATION == 1]


def rrb_y_p(valores, target):
    """Rank-biserial con signo, positivo si los morosos toman valores más bajos, y su p."""
    sanos, morosos = valores[target == 0], valores[target == 1]
    u, p = mannwhitneyu(sanos, morosos)
    return 2 * u / (len(sanos) * len(morosos)) - 1, p


@sin_dato_real
def test_contraste_de_la_hora_temprana(dato_real):
    """El efecto se sostiene con la franja hasta las 7, las 8 y las 9, que por eso es dominio.

    Sobre los 232.793 clientes de train con previas, |r_rb| de 0,0387, 0,0522 y 0,0614 (0,0502 en
    el EDA con las 8), con p de 4e-54 o menos: un factor de 1,59 entre extremos.
    """
    esperado = {7: 0.0387, 8: 0.0522, 9: 0.0614}
    medido = {}
    for hora in esperado:
        con = con_previas_en_train(dato_real[0], {"prev_hora_temprana_max": hora})
        assert len(con) == 232_793, hora
        r, p = rrb_y_p(con.PREV_EARLY_HOUR_RATIO, con.TARGET)
        assert p < 0.05, hora
        medido[hora] = round(abs(r), 4)
    assert medido == esperado
    assert max(medido.values()) / min(medido.values()) <= valor("remedicion_factor_max")


@sin_dato_real
def test_contraste_de_la_ventana_reciente(dato_real):
    """El conteo separa el default con 6, 12 y 24 meses, pero la ventana no es un valor libre.

    Sobre los 232.793 de train con previas, |r_rb| del conteo de 0,0342 con 183 días, 0,0654 con
    365 y 0,0769 con 730: un factor de 2,25, fuera de `remedicion_factor_max`, porque una ventana
    más larga cuenta más solicitudes. La bandera con el corte en 4 da +1,81pp con 183 días, por
    debajo del umbral, y +2,31pp y +2,16pp con 365 y 730. Es una convención de un año y no un
    plateau, y queda declarado así.
    """
    esperado = {
        183: (0.0342, 10_672, 1.81),
        365: (0.0654, 36_575, 2.31),
        730: (0.0769, 63_762, 2.16),
    }
    medido = {}
    for dias in esperado:
        con = con_previas_en_train(dato_real[0], {"prev_ventana_reciente_dias": dias})
        assert len(con) == 232_793, dias
        r, p = rrb_y_p(con.PREV_COUNT_12M, con.TARGET)
        assert r < 0 and p < 0.05, dias
        marcados = con.PREV_ACTIVIDAD_12M_COLA == 1
        delta = (con.TARGET[marcados].mean() - con.TARGET[~marcados].mean()) * 100
        medido[dias] = (round(abs(r), 4), int(marcados.sum()), round(delta, 2))
    assert medido == esperado
    umbral = valor("umbral_flags_pp")
    assert medido[183][2] < umbral <= min(medido[365][2], medido[730][2])
    efectos = [medido[d][0] for d in esperado]
    assert efectos == sorted(efectos)
    assert max(efectos) / min(efectos) > valor("remedicion_factor_max")


@sin_dato_real
def test_contraste_de_la_relacion_larga(dato_real):
    """La relación corta con actividad alta es superaditiva con 3, 4 y 5 años: es dominio.

    Sobre los 232.793 de train con previas: +2,03pp con 3 años (13.653 en la esquina), +1,51pp con
    4 (17.503) y +1,68pp con 5 (21.566). El 4 es el más bajo de los tres, como ya lo era en el EDA.
    """
    esperado = {3: (13_653, 2.03), 4: (17_503, 1.51), 5: (21_566, 1.68)}
    medido = {}
    for anios in esperado:
        con = con_previas_en_train(dato_real[0], {"prev_relacion_larga_anios": anios})
        assert len(con) == 232_793, anios
        larga = -con.PREV_DAYS_DECISION_MIN / DIAS >= anios
        grupo = con.groupby([larga, con.PREV_ACTIVIDAD_12M_COLA]).TARGET.agg(["size", "mean"])
        tasa = (grupo["mean"] * 100).to_dict()
        aditivo = tasa[(False, 0)] + tasa[(True, 1)] - tasa[(True, 0)]
        superaditividad = tasa[(False, 1)] - aditivo
        assert int(con.PREV_RELACION_CORTA_ACTIVA.sum()) == grupo.loc[(False, 1), "size"], anios
        medido[anios] = (int(grupo.loc[(False, 1), "size"]), round(superaditividad, 2))
    assert medido == esperado
    assert all(s > 0 for _, s in medido.values())


# --- el 4.9, el barrido del mínimo de denominador sobre el dato real --------------------------


@sin_dato_real
def test_el_barrido_del_denominador_del_rechazo_reproduce_el_eda(dato_real):
    """Las cuatro r_rb de la celda 155 y la lectura de la 157, sobre los 291.057 de la tabla cruda.

    Es la puerta del punto: aquí tiene que salir clavado lo del notebook. Los 52.533 de una sola
    solicitud son el 18,05% de los que tienen previas, y su única rechazada da +1,63pp con p = 0,35.
    """
    _, agregado, _ = dato_real
    crudo = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    unido = unir_previous(crudo, agregado)
    con = unido[unido.HAS_PREV_APPLICATION == 1]
    esperado = {
        1: (0.1204, 291_057),
        2: (0.1558, 238_524),
        3: (0.1765, 192_725),
        5: (0.1993, 119_789),
    }
    for minimo, (r_esperada, n) in esperado.items():
        dentro = con[con.PREV_APPLICATION_COUNT >= minimo]
        sanos, morosos = (dentro.PREV_REFUSED_RATIO[dentro.TARGET == t] for t in (0, 1))
        u, _ = mannwhitneyu(sanos, morosos)
        assert len(dentro) == n, minimo
        assert round(abs(2 * u / (len(sanos) * len(morosos)) - 1), 4) == r_esperada, minimo
    una = con[con.PREV_APPLICATION_COUNT == 1]
    rechazada = una.TARGET[una.PREV_REFUSED_RATIO == 1]
    resto = una.TARGET[una.PREV_REFUSED_RATIO == 0]
    assert (len(una), round(len(una) / len(con) * 100, 2)) == (52_533, 18.05)
    assert (len(rechazada), round(rechazada.mean() * 100, 2)) == (250, 10.00)
    assert (len(resto), round(resto.mean() * 100, 2)) == (52_283, 8.37)
    pp = una.TARGET.mean()
    z = (rechazada.mean() - resto.mean()) / np.sqrt(
        pp * (1 - pp) * (1 / len(rechazada) + 1 / len(resto))
    )
    assert round((rechazada.mean() - resto.mean()) * 100, 2) == 1.63
    assert round(2 * norm.sf(abs(z)), 2) == 0.35


# Por proporción y mínimo: n, r_rb de los que quedan, n de los que se quedan fuera y su r_rb, sobre
# los 232.793 de train con previas y con los cuatro cortes refijados. El efecto sube siempre con el
# mínimo porque selecciona población, y lo que decide es lo de la última columna
BARRIDO_DENOMINADOR = {
    "PREV_REFUSED_RATIO": [
        (1, 232_793, 0.1221, 0, None),
        (2, 190_837, 0.1555, 41_956, 0.0022),
        (3, 154_115, 0.1762, 78_678, 0.0308),
        (5, 95_715, 0.2012, 137_078, 0.0665),
    ],
    "PREV_OVERGRANTED_RATIO": [
        (1, 231_992, 0.1208, 0, None),
        (2, 180_821, 0.1486, 51_171, 0.0569),
        (3, 133_717, 0.1690, 98_275, 0.0771),
        (5, 67_332, 0.1946, 164_660, 0.0968),
    ],
    "PREV_STREET_RATIO": [
        (1, 232_793, 0.0924, 0, None),
        (2, 190_824, 0.1114, 41_969, 0.0227),
        (3, 154_088, 0.1249, 78_705, 0.0352),
        (5, 95_697, 0.1413, 137_096, 0.0550),
    ],
    "PREV_NO_SUITE_RATIO": [
        (1, 232_793, 0.0714, 0, None),
        (2, 190_837, 0.0843, 41_956, 0.0400),
        (3, 154_115, 0.0935, 78_678, 0.0447),
        (5, 95_715, 0.1050, 137_078, 0.0544),
    ],
    "PREV_EARLY_HOUR_RATIO": [
        (1, 232_793, 0.0522, 0, None),
        (2, 190_837, 0.0584, 41_956, 0.0261),
        (3, 154_115, 0.0603, 78_678, 0.0341),
        (5, 95_715, 0.0652, 137_078, 0.0389),
    ],
    "PREV_URGENT_PURPOSE_RATIO": [
        (1, 28_564, 0.0216, 0, None),
        (2, 9_444, 0.0361, 19_120, 0.0078),
        (3, 4_173, 0.0424, 24_391, 0.0141),
        (5, 1_110, 0.0544, 27_454, 0.0187),
    ],
}


def redondeada(r_rb):
    """La r_rb a cuatro decimales, o None si no había nada que medir."""
    return None if pd.isna(r_rb) else round(r_rb, 4)


@sin_dato_real
def test_sobre_el_split_ninguna_proporcion_gana_con_un_minimo_de_denominador(dato_real):
    """Ninguna de las seis lleva mínimo: con 3 y 5 el grupo excluido separa en todas.

    Con 2 separa con significación en cuatro (calle, hora, acompañante y sobreconcesión) y en el
    rechazo y en la finalidad urgente no llega (p de 0,065 y 0,083), pero el mínimo dejaría fuera
    al 18% y al 67% de los clientes. La delta de bandera del grupo excluido con 2 es la de la celda
    157: +3,67pp con 193 únicas rechazadas en train, frente a +1,63pp con 250 en la tabla cruda.
    """
    prev = dato_real[0]
    split = cargar_split()
    for ajustar in (
        ajustar_cola_previous,
        ajustar_actividad_previous,
        ajustar_sobreconcesion_previous,
        ajustar_finalidades_previous,
    ):
        ajustar(prev, split, split)
    informe = informe_denominador_previous(prev, split, split, {n: valor(n) for n in CORTES})
    for feature, filas in BARRIDO_DENOMINADOR.items():
        medido = [
            (m, int(r.n), redondeada(r.r_rb), int(r.n_fuera), redondeada(r.r_rb_fuera))
            for m, r in informe.loc[feature].iterrows()
        ]
        assert medido == filas, feature
        assert (informe.loc[feature].loc[[3, 5], "p_fuera"] < 0.05).all(), feature
        assert informe.loc[feature, "r_rb"].is_monotonic_increasing, feature
    con_2 = informe.xs(2, level="minimo")
    assert (con_2["p_fuera"] < 0.05).sum() == 4
    assert con_2.loc["PREV_REFUSED_RATIO", "delta_fuera_pp"] == pytest.approx(3.6682, abs=5e-5)
    assert con_2.loc["PREV_URGENT_PURPOSE_RATIO", "delta_fuera_pp"] == pytest.approx(
        2.0062, abs=5e-5
    )


# --- el 4.10 contra el dato real --------------------------------------------------------------

# La fila global del informe sobre los 232.793 clientes de train con previas, con los cuatro cortes
# refijados: n, efecto de la lectura relativa (delta en pp la bandera, r_rb las continuas), efecto
# de su contraparte absoluta y redundancia entre las dos. Es la definición recomputable que el 2.3
# no dejó, y con ella la decisión: ninguna gana.
RECENCIA_RELATIVA_TRAIN = {
    ("PREV_EXIT_REFUSED_FLAG", "PREV_REFUSED_RATIO > 0"): (232_793, 4.3999, 3.3084, 0.5915),
    ("PREV_EXIT_REFUSED_FLAG", "PREV_REFUSED_SCOFR_FLAG"): (232_793, 4.3999, 11.5758, 0.1865),
    ("PREV_DAYS_SINCE_REFUSED_REL", "PREV_DAYS_SINCE_REFUSED"): (80_195, 0.1081, 0.1204, 0.8455),
    ("PREV_COUNT_12M_REL", "PREV_COUNT_12M"): (232_793, 0.0692, 0.0654, 0.8628),
    ("PREV_COUNT_12M_REL", "PREV_APPLICATIONS_PER_YEAR"): (232_793, 0.0692, 0.1151, 0.6823),
    ("PREV_STREET_RATIO_REL", "PREV_STREET_RATIO"): (232_785, 0.0912, 0.0925, 0.8315),
    ("PREV_EARLY_HOUR_RATIO_REL", "PREV_EARLY_HOUR_RATIO"): (232_793, 0.0467, 0.0522, 0.9024),
    ("PREV_NO_SUITE_RATIO_REL", "PREV_NO_SUITE_RATIO"): (232_793, 0.0329, 0.0714, 0.7962),
}


@sin_dato_real
def test_sobre_el_split_ninguna_lectura_de_recencia_relativa_gana(dato_real):
    """El pendiente 6 en `previous_application`: ninguna de las seis entra.

    Con el criterio escrito antes de medir. La salida en rechazo pasa el 0,50 de V de Cramér contra
    `PREV_REFUSED_RATIO > 0` (0,5915, está contenida en tener algún rechazo) y sube solo 1,09pp; la
    recencia del rechazo pierde contra la absoluta (0,1081 frente a 0,1204) y son la misma variable
    (0,8455); el conteo le gana 0,0038 al absoluto pero el ritmo, que ya está en la matriz, rinde
    0,1151; y ninguna de las tres de captación supera a la suya.
    """
    prev = dato_real[0]
    split = cargar_split()
    for ajustar in (
        ajustar_cola_previous,
        ajustar_actividad_previous,
        ajustar_sobreconcesion_previous,
        ajustar_finalidades_previous,
    ):
        ajustar(prev, split, split)
    informe = informe_recencia_relativa_previous(prev, split, split, {n: valor(n) for n in CORTES})
    global_ = informe.xs("todos", level="estrato")
    for clave, (n, efecto, efecto_abs, redundancia) in RECENCIA_RELATIVA_TRAIN.items():
        fila = global_.loc[clave]
        assert fila["n"] == n, clave
        assert fila["efecto"] == pytest.approx(efecto, abs=5e-5), clave
        assert fila["efecto_abs"] == pytest.approx(efecto_abs, abs=5e-5), clave
        assert fila["redundancia"] == pytest.approx(redundancia, abs=5e-5), clave
    salida = global_.loc[("PREV_EXIT_REFUSED_FLAG", "PREV_REFUSED_RATIO > 0")]
    assert salida["n_marcados"] == 36_159
    assert salida["redundancia"] > valor("redundancia_cramer")
    # solo el conteo supera a su contraparte, y pierde con el ritmo
    continuas = global_[global_["tipo"].eq("continua")]
    ganan = continuas[continuas["efecto"] > continuas["efecto_abs"]]
    assert list(ganan.index) == [("PREV_COUNT_12M_REL", "PREV_COUNT_12M")]


# --- el 4.11, la receta remedida sobre train ----------------------------------------------------


@sin_dato_real
def test_sobre_el_split_lo_provisional_de_previous_sigue_en_el_mismo_orden(dato_real):
    """Las 25 provisionales, todas significativas: 23 con cocientes de 0,95 a 1,19 y dos aparte.

    `PREV_COUNT_COLA` mide otra población, porque su corte bajó de 15 a 11 (2,16 veces la n de la
    receta): como `BB_MANY_CREDITS_FLAG`, el cociente no compara nada y lo sostiene `mismo_orden`.
    `PREV_URGENT_PURPOSE_RATIO` remide con la lista refijada de tres y no con la del notebook, que
    tenía siete: su población, "con finalidad informada", sigue siendo la de la finalidad
    declarada. Cae a 0,0216 frente al 0,0474 de la receta (el 4.9 midió 0,0470 con la lista del
    EDA sobre train), sigue significativa y `mismo_orden` sale en False.
    """
    prev = dato_real[0]
    split = cargar_split()
    for ajustar in (
        ajustar_cola_previous,
        ajustar_actividad_previous,
        ajustar_sobreconcesion_previous,
        ajustar_finalidades_previous,
    ):
        ajustar(prev, split, split)
    train = split.loc[split.split.eq("train"), ["SK_ID_CURR", "TARGET"]]
    unido = unir_previous(train, agregar_previous(prev))
    receta = cargar_receta("previous_application")
    tabla = remedir_receta(unido, unido.TARGET, receta, POBLACIONES).set_index(["tipo", "feature"])
    assert len(tabla) == 25
    assert tabla["p"].lt(receta["metodologia"]["alfa_bonferroni"]).all(), tabla["p"]

    cola = ("flag", "PREV_COUNT_COLA")
    urgente = ("continua", "PREV_URGENT_PURPOSE_RATIO")
    assert tabla.loc[cola, "n"] == 20_319 and tabla.loc[cola, "mismo_orden"] is True
    assert tabla.loc[urgente, "n"] == 28_564
    assert tabla.loc[urgente, "efecto"] == pytest.approx(0.0216, abs=5e-4)
    assert tabla.loc[urgente, "mismo_orden"] is False

    resto = tabla.drop(index=[cola, urgente])
    proporcion = resto.n / resto.n_receta
    assert proporcion.between(0.7, 0.9).all(), proporcion[~proporcion.between(0.7, 0.9)]
    # eq(True) y no all(): sobre object, un vacío cuenta como verdadero
    assert resto.mismo_orden.eq(True).all(), resto.index[~resto.mismo_orden.eq(True)].tolist()
    assert resto.cociente.between(0.95, 1.19).all(), resto.cociente
    assert tabla.loc[("flag", "HAS_PREV_APPLICATION"), "n"] == 232_793


def efecto_bandera(bandera, target):
    """Marcados, delta en pp y z del contraste de proporciones, el de `EvaluadorSenal`."""
    marcados = bandera == 1
    n1, n0 = int(marcados.sum()), int((~marcados).sum())
    p1, p0, pp = target[marcados].mean(), target[~marcados].mean(), target.mean()
    return n1, (p1 - p0) * 100, (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))


# Los tramos de la celda 20 del notebook: hasta 0,5, de 0,5 a 1, de 1 a 2, de 2 a 4 y más de 4
# solicitudes al año. El primero se queda sin las dos clases de la bandera, como en el notebook.
TRAMOS_RITMO = [0, 0.5, 1, 2, 4, 100]


def _delta_por_tramo(unido):
    """Delta en pp de PREV_ACTIVIDAD_12M_COLA dentro de cada tramo de PREV_APPLICATIONS_PER_YEAR."""
    con = unido[unido.HAS_PREV_APPLICATION == 1]
    tramo = pd.cut(con.PREV_APPLICATIONS_PER_YEAR, TRAMOS_RITMO)
    deltas = {}
    for k, g in con.groupby(tramo, observed=True):
        f = g.PREV_ACTIVIDAD_12M_COLA
        if f.nunique() < 2:
            continue
        n1, delta, _ = efecto_bandera(f, g.TARGET.to_numpy())
        deltas[str(k)] = delta
    return deltas


@sin_dato_real
def test_la_actividad_reciente_no_sobrevive_al_ritmo(dato_real):
    """El pendiente del 4.7: la bandera se conserva solo como término de la interacción.

    Sobre la cruda reproduce la celda 173 (+0,05, -0,51, -0,91 y +0,77pp, con el primer tramo sin
    las dos clases). Sobre train, con los cuatro cortes refijados, ningún tramo llega al umbral de
    2pp y los signos siguen sin coincidir: el efecto propio de la bandera no sobrevive tampoco
    aquí, así que se conserva por el otro motivo, ya cerrado en el 4.7.
    """
    prev, agregado, _ = dato_real
    crudo = load_table("application_train", usecols=["SK_ID_CURR", "TARGET"])
    deltas_crudo = _delta_por_tramo(unir_previous(crudo, agregado))
    assert [round(d, 2) for d in deltas_crudo.values()] == [0.05, -0.51, -0.91, 0.77]

    split = cargar_split()
    for ajustar in (
        ajustar_cola_previous,
        ajustar_actividad_previous,
        ajustar_sobreconcesion_previous,
        ajustar_finalidades_previous,
    ):
        ajustar(prev, split, split)
    train = split.loc[split.split.eq("train"), ["SK_ID_CURR", "TARGET"]]
    deltas_train = _delta_por_tramo(unir_previous(train, agregar_previous(prev)))
    umbral = valor("umbral_flags_pp")
    assert all(abs(d) < umbral for d in deltas_train.values()), deltas_train
    assert len({d > 0 for d in deltas_train.values()}) > 1, "los tramos no cambian de signo"
