"""Registro único de los cortes y umbrales del pipeline de features.

Cada parámetro declara de dónde sale, y esa procedencia es la que decide si hay que
reestimarlo sobre el split de entrenamiento o si es una constante que se queda como está.
"""

from __future__ import annotations

from dataclasses import dataclass

# `dominio`   constante fija de plausibilidad de negocio o convención metodológica del
#             proyecto. No se reestima: no sale de los datos.
# `estimado`  sale de la distribución de la covariable (percentiles, medianas). No toca el
#             TARGET, pero es un parámetro ajustado, así que se reestima en la capa 2a sobre
#             el 80% de entrenamiento.
# `medido`    se eligió comparando contra la tasa de default. Se refija en la capa 2b sobre
#             el split. Todo el EDA lo midió sobre train completo, así que ninguno vale tal cual.
PROCEDENCIAS = ("dominio", "estimado", "medido")

# las dos procedencias que obligan a reajustar antes de usar el valor
REAJUSTABLES = ("estimado", "medido")


@dataclass(frozen=True)
class Parametro:
    """Un corte del pipeline, con su valor de referencia y su procedencia."""

    valor: float | None
    procedencia: str
    descripcion: str
    fuente: str

    def __post_init__(self) -> None:
        if self.procedencia not in PROCEDENCIAS:
            raise ValueError(f"procedencia fuera del vocabulario cerrado: {self.procedencia!r}")
        if self.procedencia == "dominio" and self.valor is None:
            raise ValueError("un parámetro de dominio no puede estar sin valor")


PARAMS: dict[str, Parametro] = {
    # constantes transversales
    "dias_por_anio": Parametro(
        365.25, "dominio", "conversión de días a años en todas las tablas", "convención"
    ),
    "umbral_flags_pp": Parametro(
        2.0,
        "dominio",
        "delta mínimo en puntos porcentuales para que una bandera sea relevante",
        "metodologia-estadistica 8.4",
    ),
    "umbral_continuas_rb": Parametro(
        None,
        "medido",
        "equivalente de los 2pp para continuas, en rank-biserial; el EDA nunca lo declaró y "
        "la barra efectiva se movía por tabla (0,0119 iba a IV en bureau y 0,0058 se "
        "descartaba en bureau_balance)",
        "auditoría transversal, pendiente 4",
    ),
    "n_min_categoria": Parametro(
        100,
        "dominio",
        "observaciones por debajo de las cuales una categoría no va sola y se agrupa; con "
        "quién se agrupa es decisión por tasa y se refija aparte",
        "eda-bureau 5.3, agrupamiento de CREDIT_TYPE",
    ),
    "min_denominador_proporcion": Parametro(
        None,
        "medido",
        "denominador mínimo de toda proporción de cliente; hoy solo bureau_balance lo exige "
        "y previous_application lo mide sin fijarlo",
        "auditoría transversal, pendiente 1",
    ),
    "redundancia_pearson": Parametro(
        0.70,
        "dominio",
        "umbral de multicolinealidad entre magnitudes",
        "metodologia-estadistica 8.4",
    ),
    "redundancia_cramer": Parametro(
        0.50, "dominio", "umbral de redundancia entre banderas", "metodologia-estadistica 8.4"
    ),
    "banda_revision_pearson": Parametro(
        0.60,
        "dominio",
        "suelo de la banda que se revisa con estratos cruzados aunque no cruce el umbral: un "
        "corte deja pasar sin mirar lo que se le queda justo debajo",
        "auditoría de previous_application",
    ),
    "suelo_anios_denominador": Parametro(
        0.5,
        "medido",
        "suelo del denominador de los dos ritmos por año, para no dividir por casi cero; en "
        "bureau es convención heredada de previous_application y no un corte medido",
        "notebook 02 celda 88 y notebook 04 celda 20",
    ),
    # application_train: validez de dominio, constantes fijas
    "app_own_car_age_max": Parametro(
        64,
        "dominio",
        "cap de la antigüedad del coche; el criterio es que por encima no es un activo "
        "financiero real, aunque el número coincida con el p99",
        "eda-application-train 4.6",
    ),
    "app_amt_req_bureau_day_max": Parametro(
        5,
        "dominio",
        "cap fijo de consultas al buró en el día; su p99 es 0 y el 3xp99 no es aplicable",
        "eda-application-train 4.6",
    ),
    # application_train: winsorización, factor fijo y percentiles reestimados
    "app_winsor_factor": Parametro(
        3.0, "dominio", "múltiplo del p99 al que se winsoriza", "eda-application-train 4.6"
    ),
    "app_winsor_amt_income_total": Parametro(
        1_417_500,
        "estimado",
        "3xp99 del ingreso; corrige el error de captura de 117M",
        "eda-application-train 4.6",
    ),
    "app_winsor_def_30_cnt_social_circle": Parametro(
        6, "estimado", "3xp99 de impagos en el círculo social", "eda-application-train 4.6"
    ),
    "app_winsor_obs_30_cnt_social_circle": Parametro(
        30, "estimado", "3xp99 de observados en el círculo social", "eda-application-train 4.6"
    ),
    "app_winsor_amt_req_credit_bureau_qrt": Parametro(
        6, "estimado", "3xp99 de consultas al buró en el trimestre", "eda-application-train 4.6"
    ),
    "app_winsor_amt_req_credit_bureau_mon": Parametro(
        12, "estimado", "3xp99 de consultas al buró en el mes", "eda-application-train 4.6"
    ),
    "app_winsor_amt_req_credit_bureau_week": Parametro(
        3, "estimado", "3xp99 de consultas al buró en la semana", "eda-application-train 4.6"
    ),
    "app_winsor_cnt_children": Parametro(
        9, "estimado", "3xp99 del número de hijos", "eda-application-train 4.6"
    ),
    "app_winsor_cnt_fam_members": Parametro(
        15, "estimado", "3xp99 de miembros de la familia", "eda-application-train 4.6"
    ),
    # centinela del dataset, uno solo: es el mismo código en las dos tablas que lo usan, la
    # antigüedad laboral de application_train (55.374 obs) y las seis fechas del ciclo de vida
    # de previous_application. Declararlo dos veces es el patrón que ya dio 52.500 frente a
    # 52.497 en bureau, un mismo control con dos umbrales.
    "centinela_365243": Parametro(
        365243,
        "dominio",
        "código de ausencia del dataset: inactivo en DAYS_EMPLOYED y fecha no informada en el "
        "ciclo de vida de previous_application; pasa a NaN, con bandera en el primer caso",
        "eda-application-train 4.2 y notebook 04 celda 115",
    ),
    # bureau: validez de dominio a nivel fila, antes de agregar
    "bureau_importe_max": Parametro(
        50_000_000,
        "dominio",
        "por encima el importe es error de captura y pasa a NaN, aunque la cola tenga señal, "
        "porque entra crudo en las sumas de cliente",
        "metodologia-estadistica 8.2",
    ),
    "bureau_cuota_max": Parametro(
        10_000_000,
        "dominio",
        "cap de plausibilidad de la cuota reportada al buró",
        "eda-bureau 5.7",
    ),
    "bureau_ratio_deuda_credito_max": Parametro(
        3.0,
        "dominio",
        "por encima se capa la deuda al propio crédito; el exceso leve y el moderado son "
        "deuda real y se conservan",
        "eda-bureau 5.7",
    ),
    "bureau_enddate_max_anios": Parametro(
        20,
        "dominio",
        "vencimientos más lejanos son placeholder y quedan fuera del cómputo a término",
        "eda-bureau 5.6",
    ),
    "bureau_cierre_max_anios": Parametro(
        30, "dominio", "cierres más antiguos son error de captura", "eda-bureau 5.7"
    ),
    # bureau: cortes medidos
    "bureau_update_reciente_dias": Parametro(
        180,
        "medido",
        "ventana de actualización reciente del registro del buró",
        "notebook 02 celda 88",
    ),
    "bureau_enddate_tramo_min_anios": Parametro(
        2, "medido", "suelo del tramo de vencimiento que se cuenta aparte", "notebook 02 celda 88"
    ),
    "bureau_enddate_tramo_max_anios": Parametro(
        5, "medido", "techo del tramo de vencimiento que se cuenta aparte", "notebook 02 celda 88"
    ),
    # bureau_balance: cortes medidos
    "bb_min_meses_reportados": Parametro(
        6,
        "medido",
        "denominador mínimo del porcentaje de meses en mora",
        "notebook 03 celda 56",
    ),
    "bb_mora_reciente_meses": Parametro(
        6,
        "medido",
        "ventana de la mora reciente, cortada sobre la recencia relativa al fin de ventana "
        "de cada crédito y no sobre la absoluta",
        "notebook 03 celda 62",
    ),
    "bb_many_credits_corte": Parametro(
        22,
        "medido",
        "corte de la cola del conteo de créditos con histórico",
        "notebook 03 celda 59",
    ),
    "bb_min_meses_trayectoria": Parametro(
        6,
        "medido",
        "ventana mínima para partir el histórico en dos mitades y leer la trayectoria",
        "notebook 03 celda 37",
    ),
    # previous_application: cortes medidos
    "prev_count_cola": Parametro(
        15, "medido", "corte de la cola del conteo de solicitudes", "notebook 04 celda 153"
    ),
    "prev_actividad_12m_cola": Parametro(
        4,
        "medido",
        "corte de la actividad de los últimos doce meses; la bandera solo se conserva como "
        "término de la interacción con la longitud de relación",
        "notebook 04 celda 153",
    ),
    "prev_adelanto_liquidacion_dias": Parametro(
        365,
        "medido",
        "adelanto sobre la fecha de fin prevista que marca liquidación anticipada",
        "notebook 04 celda 125",
    ),
    "prev_plazo_largo_cuotas": Parametro(
        60,
        "medido",
        "plazo por encima del cual la petición rechazada marca; el mismo plazo concedido no marca",
        "notebook 04 celda 153",
    ),
    "prev_sobreconcesion_corte": Parametro(
        1.1,
        "medido",
        "exceso sobre lo solicitado; se barrió sobre la proporción, que es la codificación "
        "que entra, y no sobre la bandera, que daba un corte opuesto",
        "notebook 04 celda 160",
    ),
    "prev_ratio_rechazo_min_solicitudes": Parametro(
        None,
        "medido",
        "mínimo de solicitudes para que el ratio de rechazo tenga denominador; sin mínimo "
        "rinde 0,1204 con cobertura total y con dos rinde 0,1558 cubriendo el 81,95%",
        "notebook 04 celda 155",
    ),
}


def parametro(nombre: str) -> Parametro:
    """Devuelve el registro completo de un parámetro."""
    if nombre not in PARAMS:
        raise KeyError(f"parámetro no declarado: {nombre!r}")
    return PARAMS[nombre]


def valor(nombre: str):  # noqa: ANN201 - devuelve el tipo que declare el parámetro
    """Valor del parámetro. Falla si aún no se ha fijado, en vez de devolver None."""
    p = parametro(nombre)
    if p.valor is None:
        raise ValueError(
            f"{nombre!r} está sin fijar (procedencia {p.procedencia!r}): {p.descripcion}"
        )
    return p.valor


def por_procedencia(procedencia: str) -> dict[str, Parametro]:
    """Los parámetros de una procedencia concreta."""
    if procedencia not in PROCEDENCIAS:
        raise ValueError(f"procedencia desconocida: {procedencia!r}")
    return {k: v for k, v in PARAMS.items() if v.procedencia == procedencia}


def reajustables() -> dict[str, Parametro]:
    """Los que hay que reestimar o refijar sobre el split antes de cerrar la fase."""
    return {k: v for k, v in PARAMS.items() if v.procedencia in REAJUSTABLES}


def sin_fijar() -> dict[str, Parametro]:
    """Los que todavía no tienen valor y bloquean a quien los use."""
    return {k: v for k, v in PARAMS.items() if v.valor is None}
