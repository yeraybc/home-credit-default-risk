"""Registro único de los cortes y umbrales del pipeline de features.

Cada parámetro declara de dónde sale, y esa procedencia es la que decide si hay que
reestimarlo sobre el split de entrenamiento o si es una constante que se queda como está.

Frontera con config.yaml, declarada para que no vuelva a haber un mismo control con dos
valores en dos ficheros: **config.yaml lleva infraestructura** (rutas, semilla y proporción
de la partición, nombres de columna, servicio, monitorización y mlflow) y **este módulo lleva
los cortes de modelado**, que son los que tienen procedencia y los que pueden necesitar
reajuste. La prueba para decidir dónde va algo es si tiene una procedencia del EDA: si la
tiene, va aquí.

Tres claves se migraron desde config.yaml al aplicar esa frontera, y una se eliminó:

- `correlation_threshold` valía 0,95 mientras `redundancia_pearson` vale 0,70. Son el mismo
  control con dos valores, el patrón que en bureau dejó 52.500 en un sitio y 52.497 en otro.
  Gana 0,70, que es el umbral que usan las tres recetas y `metodologia-estadistica` 8.4; el
  0,95 era andamiaje del arranque del proyecto, anterior al EDA.
- `min_iv` y `n_bins_max` pasan aquí porque son cortes de modelado con procedencia.
- `missing_threshold` (0,60) se **elimina sin migrar**: el EDA refutó la regla de umbral único
  de nulos y decide columna a columna con IV. Con ella caerían `EXT_SOURCE_1` (56,4% de nulos
  y se conserva con su bandera), `FONDKAPREMONT_MODE` (68%, pendiente de IV) y buena parte del
  bloque edificio. Dejarla declarada invitaba a aplicar una regla que el EDA ya descartó.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    """Un corte del pipeline, con su valor de referencia y su procedencia.

    `valor_referencia` es la cifra que salió del EDA sobre el conjunto completo: queda como
    documentación y nunca la usa el pipeline para transformar datos. Lo que de verdad se
    consume es `valor_operativo`, que para un reajustable (`estimado`/`medido`) empieza vacío
    y solo lo rellena `fijar_operativo()` con el resultado de recalcular sobre `solo_train()`.
    Esto evita que la capa 2a/2b use por accidente la cifra medida sobre el conjunto entero.

    `contraste_pendiente` es para el caso raro de un corte de dominio que aun así hay que
    contrastar sobre el split: no se refija, pero el contraste queda declarado en vez de
    perderse al reclasificarlo.
    """

    valor_referencia: float | None
    procedencia: str
    descripcion: str
    fuente: str
    contraste_pendiente: str | None = None
    valor_operativo: float | None = None
    n_train_operativo: int | None = None

    def __post_init__(self) -> None:
        if self.procedencia not in PROCEDENCIAS:
            raise ValueError(f"procedencia fuera del vocabulario cerrado: {self.procedencia!r}")
        if self.procedencia == "dominio" and self.valor_referencia is None:
            raise ValueError("un parámetro de dominio no puede estar sin valor")
        if self.contraste_pendiente is not None and not self.contraste_pendiente.strip():
            raise ValueError("contraste_pendiente declarado y vacío")
        if self.procedencia == "dominio" and self.valor_operativo is not None:
            raise ValueError(
                "un parámetro de dominio no tiene valor operativo: usa valor_referencia"
            )
        if self.valor_operativo is not None and self.n_train_operativo is None:
            raise ValueError(
                "valor_operativo fijado sin n_train_operativo: no queda rastro de origen"
            )
        if self.n_train_operativo is not None and self.n_train_operativo <= 0:
            raise ValueError("n_train_operativo tiene que ser positivo")


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
    # Suavizado del WoE, convención y no corte medido. Son observaciones de prior, repartidas
    # entre buenos y malos **según la tasa global de train** y no a partes iguales: a partes
    # iguales el prior implícito es 50/50, que no describe una cartera del 8% de default, y al
    # nivel que ya está por encima de la media lo aleja de cero en vez de acercarlo. Medido sobre
    # train en `Industry: type 8`, que con n = 17 es el más pequeño de los 58: sin suavizar vale
    # +0,8920, sumarle los 20 del prior a cada uno de los dos recuentos lo empuja hasta +2,0415, y
    # repartirlos por la tasa global lo deja en +0,4840, que es la dirección que se busca.
    #
    # Con 20, un nivel necesita 20 observaciones propias para que su tasa pese tanto como el
    # prior, o sea que conserva n/(n+20) de su propia señal: el de 17 clientes se queda con la
    # mitad y el de 54.554 no se mueve. Barrido sobre train en 5, 10, 20, 50 y 100: por debajo
    # sigue fiándose demasiado de 17 observaciones, y por encima empieza a erosionar niveles con
    # señal real, como `Industry: type 13`, que baja del 77% al 58% de la suya con 56 clientes y
    # 7 impagos.
    #
    # Aparte de eso, impide el log de cero. Sobre train hoy ningún nivel de ORGANIZATION_TYPE se
    # queda sin positivos ni sin negativos, pero sí puede pasar en un fold del CV de la Fase 4 o
    # en lo que llegue a la API.
    "suavizado_woe": Parametro(
        20,
        "dominio",
        "observaciones de prior con las que se suaviza el WoE de cada nivel, repartidas entre "
        "buenos y malos según la tasa global de entrenamiento, para que un nivel con poca "
        "evidencia propia converja al comportamiento medio en vez de a su propio azar",
        "convención de suavizado bayesiano de WoE",
    ),
    "min_denominador_proporcion": Parametro(
        None,
        "medido",
        "denominador mínimo de toda proporción de cliente; hoy solo bureau_balance lo exige "
        "y previous_application lo mide sin fijarlo",
        "auditoría transversal, pendiente 1",
    ),
    "remedicion_factor_max": Parametro(
        2.0,
        "dominio",
        "veces que el efecto remedido sobre train puede alejarse del de la receta, por arriba o "
        "por abajo, para seguir en el mismo orden de magnitud; lo que sale se revisa, no se "
        "descarta, que la decisión es del IV",
        "puerta del bloque 2 en fase3-pipeline, criterio del punto 2.3",
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
        "dominio",
        "suelo del denominador de los dos ritmos por año, para no dividir por casi cero; no "
        "se eligió comparando contra el default, es convención heredada de "
        "previous_application, así que por el propio vocabulario es dominio y no medido",
        "notebook 02 celda 88 y notebook 04 celda 20",
        contraste_pendiente=(
            "la receta de bureau lo deja pendiente de refijar sobre el split. Al no ser un "
            "corte medido no se refija, pero el contraste se mantiene como control: medir "
            "BUREAU_CREDITS_PER_YEAR con varios suelos y comprobar que el gradiente monótono "
            "del 6,12% al 16,15% no depende de este valor. En bureau, ejecutado en el 2.3 y "
            "fijado en test_contraste_del_suelo_de_medio_anio: monótono con 0,25, 0,5 y 1 año"
        ),
    ),
    # migrados desde config.yaml al aplicar la frontera del docstring
    "min_iv": Parametro(
        0.02,
        "dominio",
        "IV por debajo del cual una feature no aporta capacidad predictiva; es la escala "
        "estándar de credit scoring, no un corte estimado sobre este dataset",
        "glosario-tecnico, escala de IV",
    ),
    "n_bins_max": Parametro(
        10,
        "dominio",
        "tramos máximos del binning con el que se calculan IV y WoE",
        "convención de binning del proyecto",
    ),
    # Las tres fronteras de la franja horaria de la solicitud. Son de dominio y no se reestiman:
    # salen de dónde empieza y acaba una jornada laboral, no de mirar la tasa de default. El EDA
    # las eligió con ese criterio ("pedir fuera de horario puede señalar un perfil distinto") y su
    # efecto medido es flojo, 8,490%, 7,720% y 8,026% sobre train, o sea 0,77pp de recorrido
    # frente a los 2pp del umbral de banderas. Quien decida si la franja se queda es el IV del
    # bloque 5, con su medida delante, no estos tres números.
    "app_hora_inicio_manana": Parametro(
        6, "dominio", "hora a la que empieza la franja de mañana", "eda-application-train 3.x"
    ),
    "app_hora_inicio_tarde": Parametro(
        12, "dominio", "hora a la que la mañana da paso a la tarde", "eda-application-train 3.x"
    ),
    "app_hora_fin_tarde": Parametro(
        18,
        "dominio",
        "hora a la que acaba la tarde y empieza el fuera de horario",
        "eda-application-train 3.x",
    ),
    # Suelo de varianza del filtro final. Cero significa que solo caen las constantes, que es lo
    # que se quiere: la selección de verdad la hace el IV del bloque 5. Se declara en vez de
    # dejarlo al defecto implícito de la librería para que subirlo sea una decisión visible, y
    # porque con el suelo a cero la exclusión de FLAG_DOCUMENT_3 y 6 que pide el plan es inocua.
    "app_umbral_varianza": Parametro(
        0.0,
        "dominio",
        "varianza por debajo de la cual una columna sale de la matriz; a cero solo elimina "
        "constantes, que es una red contra el fold que deja una bandera sin variación",
        "convención, la selección por señal es del bloque 5",
    ),
    # application_train: validez de dominio, constantes fijas
    # el EDA lo declaraba como criterio de dominio ("por encima de 64 años no es un activo
    # financiero real"), pero 64 es exactamente el p99 y el argumento de negocio justifica
    # capar, no capar en 64: un número de dominio sería redondo y no se movería con la
    # muestra. Recomputado sobre los 104.582 clientes con coche de la tabla cruda, el p99 es
    # 64,00 y por encima de 65 solo quedan 3 registros, así que el criterio no sostiene el
    # valor por sí solo y pasa a estimado, igual que el grupo app_winsor_*.
    # Las tres poblaciones del recuento, que no son la misma y conviene no confundir: 104.582
    # con coche en la tabla cruda, 104.576 en la limpia y 83.745 en el 80% de entrenamiento,
    # que es el que acaba declarando el ajuste y el que vale como n_train.
    "app_cap_p99_own_car_age": Parametro(
        64,
        "estimado",
        "p99 de la antigüedad del coche, no el 3xp99 del resto del grupo; el argumento de "
        "negocio justifica que haya cap, y el valor sale de la distribución",
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
    "app_winsor_percentil": Parametro(
        0.99,
        "dominio",
        "percentil sobre el que se aplica el múltiplo; va aquí y no como literal en el "
        "transformer por lo mismo que el factor, que los dos definen el mismo corte",
        "eda-application-train 4.6",
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
    # el EDA lo declara junto al de 30 días ("DEF_30/60 (6)") y le da el mismo valor. Está aquí
    # y no fuera porque la columna sobrevive a la limpieza: su descarte se decidió contra la
    # tasa de default, así que es provisional y la juzga la capa 2b. Sin este corte, la 2b la
    # juzgaría con la cola sin capar mientras su gemela DEF_30 sí la lleva capada, que es el
    # mismo control con dos tratamientos.
    "app_winsor_def_60_cnt_social_circle": Parametro(
        6,
        "estimado",
        "3xp99 de impagos a 60 días en el círculo social, la cola de mayor señal de la tabla",
        "eda-application-train 4.6",
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
    # Dominio y no medido: es la ventana de seis meses, y no hay pico que medir. Sobre los 210.875
    # clientes de train con historial la tasa fila a fila baja sin saltos del 9,35% del primer
    # medio año al 5,77% de más de cinco, y el delta de la bandera pasa de +2,62pp con 90 días a
    # +2,47pp con 180 y +1,91pp con 730. Un barrido por delta se iría a la ventana más corta.
    "bureau_update_reciente_dias": Parametro(
        180,
        "dominio",
        "ventana de actualización reciente del registro del buró, seis meses",
        "notebook 02 celda 88",
        contraste_pendiente=(
            "comprobar sobre el split que la tasa baja con la antigüedad de la última "
            "actualización y que el delta de BUREAU_DAYS_CREDIT_UPDATE_FLAG es positivo con "
            "cualquier ventana de 90 a 730 días, o sea que la señal no depende de este valor. "
            "Ejecutado en el 2.3 y fijado en test_contraste_de_la_ventana_de_actualizacion"
        ),
    ),
    # bureau: cortes medidos
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

# Qué corte lleva dentro cada feature de las tres recetas. Es lo que permite comprobar que
# ninguna feature cuya construcción depende de un umbral se quede sin ese umbral declarado, y
# lo que en la capa 2b dice qué hay que rebarrer al refijar cada corte sobre el split.
# Solo aparecen las features que llevan un corte: la mayoría de las provisionales lo son por
# tener su efecto medido sobre train completo, que se remide sin que haya ningún umbral que
# mover.
CORTES_POR_FEATURE: dict[str, dict[str, tuple[str, ...]]] = {
    "bureau": {
        "BUREAU_CREDITS_PER_YEAR": ("suelo_anios_denominador",),
        "BUREAU_DAYS_CREDIT_UPDATE_FLAG": ("bureau_update_reciente_dias",),
        "BUREAU_ENDDATE_2_5Y_COUNT": (
            "bureau_enddate_tramo_min_anios",
            "bureau_enddate_tramo_max_anios",
        ),
    },
    "bureau_balance": {
        "BB_PCT_MONTHS_DPD": ("bb_min_meses_reportados",),
        "BB_RECENT_DPD_FLAG_REL": ("bb_mora_reciente_meses",),
        "BB_RECENT_DPD_FLAG": ("bb_mora_reciente_meses",),
        "BB_MANY_CREDITS_FLAG": ("bb_many_credits_corte",),
        "BB_PERSISTENT_DPD_FLAG": ("bb_min_meses_trayectoria",),
        "BB_WORSENING_DPD_FLAG": ("bb_min_meses_trayectoria",),
        "BB_RECOVERED_DPD_FLAG": ("bb_min_meses_trayectoria",),
    },
    "previous_application": {
        "PREV_COUNT_COLA": ("prev_count_cola",),
        "PREV_ACTIVIDAD_12M_COLA": ("prev_actividad_12m_cola",),
        "PREV_EARLY_SETTLED_FLAG": ("prev_adelanto_liquidacion_dias",),
        "PREV_EARLY_SETTLED_COUNT": ("prev_adelanto_liquidacion_dias",),
        "PREV_EARLY_SETTLED_RATIO": ("prev_adelanto_liquidacion_dias",),
        "PREV_REFUSED_LONG_TERM_FLAG": ("prev_plazo_largo_cuotas",),
        "PREV_OVERGRANTED_RATIO": ("prev_sobreconcesion_corte",),
        "PREV_REFUSED_RATIO": ("prev_ratio_rechazo_min_solicitudes",),
        "PREV_APPLICATIONS_PER_YEAR": ("suelo_anios_denominador",),
    },
}


def parametro(nombre: str) -> Parametro:
    """Devuelve el registro completo de un parámetro."""
    if nombre not in PARAMS:
        raise KeyError(f"parámetro no declarado: {nombre!r}")
    return PARAMS[nombre]


def valor(nombre: str):  # noqa: ANN201 - devuelve el tipo que declare el parámetro
    """Valor que consume el pipeline. Falla si aún no se ha fijado, en vez de devolver None.

    Para un reajustable (`estimado`/`medido`) es el `valor_operativo`, que solo existe tras
    pasar por `fijar_operativo()`: la referencia del EDA sobre el conjunto completo nunca se
    devuelve aquí, para que no se cuele en una transformación sin haber pasado por el split.
    """
    p = parametro(nombre)
    if p.procedencia in REAJUSTABLES:
        if p.valor_operativo is None:
            raise ValueError(
                f"{nombre!r} está sin fijar (procedencia {p.procedencia!r}): {p.descripcion}. "
                "Hay que recalcularlo sobre solo_train() y fijarlo con fijar_operativo()."
            )
        return p.valor_operativo
    return p.valor_referencia


def fijar_operativo(
    nombre: str, valor_nuevo: float, n_train: int, sobrescribir: bool = False
) -> None:
    """Único punto de escritura del valor operativo de un reajustable.

    Exige `n_train`, el tamaño de la partición de entrenamiento usada para calcularlo: no
    basta con poner un número, tiene que declarar cuántas filas lo sostienen. No relee el
    split por su cuenta, quien llama ya lo filtró con `solo_train()` y aquí solo se registra
    el resultado.

    Refijar uno ya fijado exige `sobrescribir=True`, la misma guarda que `construir_split()`.
    Sin ella, escribir dos veces se resuelve por upsert y gana la última, que es el mecanismo
    del orden del registro de `patrones-de-fallo`: dos ajustes sobre poblaciones distintas
    dejan la segunda cifra con el n de la segunda y nadie se entera. Hoy solo hay un
    consumidor, `ajustar_capa2a()`; el riesgo aparece en cuanto haya el segundo.
    """
    p = parametro(nombre)
    if p.procedencia not in REAJUSTABLES:
        raise ValueError(
            f"{nombre!r} no es reajustable (procedencia {p.procedencia!r}); "
            "no lleva valor operativo"
        )
    if p.valor_operativo is not None and not sobrescribir:
        raise ValueError(
            f"{nombre!r} ya está fijado en {p.valor_operativo} con n_train {p.n_train_operativo} "
            f"y se intenta poner {valor_nuevo} con {n_train}. Refijarlo invalida todo "
            "lo ajustado con el valor anterior, así que hay que pedir sobrescribir=True"
        )
    PARAMS[nombre] = replace(p, valor_operativo=valor_nuevo, n_train_operativo=n_train)


def por_procedencia(procedencia: str) -> dict[str, Parametro]:
    """Los parámetros de una procedencia concreta."""
    if procedencia not in PROCEDENCIAS:
        raise ValueError(f"procedencia desconocida: {procedencia!r}")
    return {k: v for k, v in PARAMS.items() if v.procedencia == procedencia}


def reajustables() -> dict[str, Parametro]:
    """Los que hay que reestimar o refijar sobre el split antes de cerrar la fase."""
    return {k: v for k, v in PARAMS.items() if v.procedencia in REAJUSTABLES}


def sin_fijar() -> dict[str, Parametro]:
    """Los que el EDA nunca llegó a medir: sin referencia y, por tanto, sin operativo posible."""
    return {k: v for k, v in PARAMS.items() if v.valor_referencia is None}


def operativos_pendientes() -> dict[str, Parametro]:
    """Reajustables con referencia del EDA que todavía no se han refijado sobre solo_train().

    Es la lista de tareas de lo que queda de las capas 2a y 2b: al importar el módulo son todos,
    y `ajustar_capa2a()` descuenta los diez cortes del winsorizador en cuanto corre.
    """
    return {
        k: v
        for k, v in PARAMS.items()
        if v.procedencia in REAJUSTABLES and v.valor_operativo is None
    }


def con_contraste_pendiente() -> dict[str, Parametro]:
    """Los de dominio que aun así hay que contrastar sobre el split, como control."""
    return {k: v for k, v in PARAMS.items() if v.contraste_pendiente is not None}


def cortes_de(tabla: str, feature: str) -> tuple[str, ...]:
    """Cortes que lleva dentro una feature; tupla vacía si no depende de ninguno."""
    if tabla not in CORTES_POR_FEATURE:
        raise KeyError(f"tabla sin mapa de cortes: {tabla!r}")
    return CORTES_POR_FEATURE[tabla].get(feature, ())


def features_con_corte() -> dict[str, str]:
    """Mapa inverso: por cada corte, las features de receta que lo usan."""
    inverso: dict[str, list[str]] = {}
    for tabla, features in CORTES_POR_FEATURE.items():
        for feature, cortes in features.items():
            for corte in cortes:
                inverso.setdefault(corte, []).append(f"{tabla}.{feature}")
    return {k: ", ".join(sorted(v)) for k, v in sorted(inverso.items())}
