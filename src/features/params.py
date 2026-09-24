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

    # una tupla cuando el corte es una lista de categorías, como las finalidades urgentes
    valor_referencia: float | tuple[str, ...] | None
    procedencia: str
    descripcion: str
    fuente: str
    contraste_pendiente: str | None = None
    valor_operativo: float | tuple[str, ...] | None = None
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
    # Declarado en el 5.4 con los r_rb de train ya vistos: el 2.3, el 3.10 y el 4.11 los remidieron,
    # y la propuesta del 5.4 midió las 39 continuas provisionales de las tres recetas sobre los
    # 245.993 de train.
    # El 0,02 cae en su hueco, entre 0,0121 y 0,0216, y deja cuatro debajo (BB_STATUS_WORST,
    # BB_N_CREDITS_WBAL, BUREAU_LOAN_COUNT y BUREAU_CREDIT_TYPE_NUNIQUE). No descarta por sí solo:
    # es la lectura que va al lado del IV.
    "umbral_continuas_rb": Parametro(
        0.02,
        "dominio",
        "equivalente de los 2pp para continuas, en rank-biserial; el EDA nunca lo declaró y "
        "la barra efectiva se movía por tabla (0,0119 iba a IV en bureau y 0,0058 se "
        "descartaba en bureau_balance)",
        "auditoría transversal, pendiente 4",
    ),
    "alfa_familia": Parametro(
        0.05,
        "dominio",
        "error de tipo I de cada familia de contrastes, que Bonferroni reparte entre los "
        "contrastes emitidos con alfa_bonferroni()",
        "metodologia-estadistica 8.1 y auditoría transversal, pendiente 3",
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
    # Sin mínimo transversal de denominador, decidido en el 4.9 con el usuario: el pendiente 1 de
    # la auditoría transversal se cierra por tabla y no con un número. Los denominadores tienen
    # unidades distintas (meses reportados, créditos y solicitudes) y cada tabla decide el suyo:
    # bureau, sin mínimo (2.3); bureau_balance, `bb_min_meses_reportados`, de dominio; y
    # previous_application, sin mínimo en ninguna de sus seis proporciones de conteos, medido sobre
    # train con `informe_denominador_previous()`. Con el de dos el grupo que queda fuera separa con
    # significación en cuatro (r_rb de 0,0227 a 0,0569 en la calle, la hora, el acompañante y la
    # sobreconcesión); en el rechazo y en la finalidad urgente no llega (p de 0,065 y 0,083), pero
    # el mínimo mandaría a la mediana al 18% y al 67% de los clientes, y con 3 y 5 el grupo
    # excluido separa en las seis. El del rechazo, con su motivo, está al final de este registro.
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
    # la banda del IV es informativa (5.8): no mueve el corte de `min_iv`, marca lo que queda cerca
    # para que el registro de selección diga en cuántos folds se sostiene la decisión
    "banda_revision_iv_suelo": Parametro(
        0.015,
        "dominio",
        "suelo de la banda de revisión del IV: por debajo de min_iv y hasta aquí, la decisión se "
        "mira por folds antes de darla por firme",
        "auditoría del 5.4 y el 5.5, decidido con el usuario en el 5.8",
    ),
    "banda_revision_iv_techo": Parametro(
        0.025,
        "dominio",
        "techo de la banda de revisión del IV, simétrico al suelo alrededor de min_iv",
        "auditoría del 5.4 y el 5.5, decidido con el usuario en el 5.8",
    ),
    "n_bins_max": Parametro(
        10,
        "dominio",
        "tramos máximos del binning con el que se calculan IV y WoE",
        "convención de binning del proyecto",
    ),
    "solape_ext3_min": Parametro(
        0.20,
        "dominio",
        "Pearson mínimo contra EXT_SOURCE_3 para que una columna de bureau o bureau_balance "
        "tenga que demostrar IV incremental antes de conservarse (5.6). El EDA llamó solape a "
        "partir de 0,212 (BUREAU_DEBT_CREDIT_RATIO) y la mora quedó en 0,025 o menos: 0,20 "
        "separa las dos familias sin cortar ninguna a la mitad",
        "eda-bureau, historial-auditorias; eda-bureau-balance 5B.5",
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
    # Sin referencia porque el EDA de bureau no la midió, siendo la tabla con más cobertura. Se
    # refija con el criterio de prev_count_cola, el primer corte que cruza umbral_flags_pp; sobre
    # train sale 18, con +2,33pp y 4.327 marcados, y el p99 más uno de bb daría 21 con 1.943.
    "bureau_count_cola": Parametro(
        None,
        "medido",
        "créditos a partir de los cuales el cliente está en la cola del conteo: el primer corte "
        "cuyo delta cruza umbral_flags_pp",
        "auditoría transversal, pendiente 5",
    ),
    # bureau_balance. Los dos primeros son dominio desde el 3.8 y por la razón de los 180 días de
    # bureau: el barrido del EDA no tiene pico. El r_rb del porcentaje sube con el mínimo (0,0981
    # con 1 mes a 0,1129 con 12), así que un máximo se iría al corte más alto por selección de
    # población, y el 6 responde a que con uno o dos meses el ratio vale 0 o 1 por ruido de reporte.
    # El delta de la mora reciente baja con la ventana (+5,25pp con 3 a +4,31pp con 12) con z casi
    # plana, y el 6 quedó como compromiso.
    "bb_min_meses_reportados": Parametro(
        6,
        "dominio",
        "denominador mínimo del porcentaje de meses en mora",
        "notebook 03 celda 56",
        contraste_pendiente=(
            "comprobar sobre el split que BB_PCT_MONTHS_DPD separa el default con un mínimo de 1, "
            "3, 6 y 12 meses reportados, con efectos dentro de remedicion_factor_max entre sí, o "
            "sea que la señal no depende de este valor. Ejecutado en el 3.8 y fijado en "
            "test_contraste_del_denominador_minimo"
        ),
    ),
    "bb_mora_reciente_meses": Parametro(
        6,
        "dominio",
        "ventana de la mora reciente, cortada sobre la recencia relativa al fin de ventana "
        "de cada crédito y no sobre la absoluta",
        "notebook 03 celdas 61 y 62",
        contraste_pendiente=(
            "comprobar sobre el split que BB_RECENT_DPD_FLAG_REL cruza umbral_flags_pp con "
            "ventanas de 3, 6 y 12 meses, o sea que la señal no depende de este valor. Ejecutado "
            "en el 3.8 y fijado en test_contraste_de_la_ventana_de_mora_reciente"
        ),
    ),
    # bureau_balance: cortes medidos
    # El 22 es el p99 más uno del EDA sobre los 307.511. Desde el 3.9 se refija con el criterio de
    # bureau_count_cola: sobre los 73.767 clientes de train con histórico sale 18, con +2,51pp y
    # 1.653 marcados (17 se queda en +1,92pp), y el p99 más uno daría 22 con +3,39pp y 574.
    "bb_many_credits_corte": Parametro(
        22,
        "medido",
        "créditos con histórico a partir de los cuales el cliente está en la cola del conteo: el "
        "primer corte cuyo delta cruza umbral_flags_pp",
        "notebook 03 celdas 47 y 59",
    ),
    # Dominio y no medido, desde el 3.3, que es quien lo consume: el EDA no lo eligió contra la
    # tasa de default, es que con menos de seis meses cada mitad se queda en uno o dos y la
    # comparación entre ellas no dice nada de trayectoria.
    "bb_min_meses_trayectoria": Parametro(
        6,
        "dominio",
        "ventana mínima para partir el histórico en dos mitades y leer la trayectoria",
        "notebook 03 celda 37",
        contraste_pendiente=(
            "comprobar sobre el split que el orden de la trayectoria a nivel cliente (sin mora, "
            "mejora, empeora y estable) no depende de la ventana mínima, o sea que la señal no "
            "depende de este valor. Ejecutado en el 3.9 con 4, 6, 9 y 12 meses y fijado en "
            "test_contraste_de_la_ventana_minima_de_la_trayectoria"
        ),
    ),
    # previous_application. Dominio porque es la ventana de un año y no se eligió contra el default:
    # el notebook corta `DAYS_DECISION > -365` con el 365 literal y no con dias_por_anio, y no es
    # lo mismo: hay 1.855 solicitudes justo en -365 y con 365,25 entrarían, lo que mueve el
    # conteo de 1.201 clientes
    "prev_ventana_reciente_dias": Parametro(
        365,
        "dominio",
        "ventana de las solicitudes recientes, un año, con el borde fuera",
        "notebook 04 celda 121",
        contraste_pendiente=(
            "comprobar sobre el split que la actividad reciente separa el default con ventanas "
            "de 6, 12 y 24 meses. Ejecutado en el 4.8 y fijado en "
            "test_contraste_de_la_ventana_reciente: el conteo separa con las tres, pero su "
            "magnitud crece con la ventana y la cola de 4 solo cruza umbral_flags_pp desde los "
            "12 meses, así que la ventana es una convención y no un valor libre"
        ),
    ),
    # Dominio por lo mismo que las app_hora_*: hasta las 8 es la franja antes de que abra la
    # oficina a las 9, y el 82,44% de las decisiones de los clientes de train cae entre las 9 y las
    # 17 (82,43% sobre la tabla entera). El borde entra
    "prev_hora_temprana_max": Parametro(
        8,
        "dominio",
        "última hora de la franja temprana de la solicitud, con el borde dentro",
        "notebook 04 celda 153",
        contraste_pendiente=(
            "comprobar sobre el split que PREV_EARLY_HOUR_RATIO separa el default con la franja "
            "hasta las 7, las 8 y las 9, con efectos dentro de remedicion_factor_max entre sí, o "
            "sea que la señal no depende de este valor. Ejecutado en el 4.8 y fijado en "
            "test_contraste_de_la_hora_temprana"
        ),
    ),
    # Dominio y no medido, decidido en el 4.7: es un borde de la rejilla descriptiva de la celda
    # 121 (0, 1, 2, 4, 6 y 8 años), que parte la cartera casi por la mitad, y no salió de un
    # barrido. Sobre train completo la superaditividad con 2 a 6 años es +2,86, +1,99, +1,61,
    # +1,78 y +1,73pp: positiva con cualquiera, y el 4 es casi la más baja. El borde entra
    "prev_relacion_larga_anios": Parametro(
        4,
        "dominio",
        "años desde la solicitud más antigua a partir de los cuales la relación es larga, con el "
        "borde dentro",
        "notebook 04 celdas 121 y 131",
        contraste_pendiente=(
            "comprobar sobre el split que la relación corta con actividad alta sigue siendo "
            "superaditiva con 3, 4 y 5 años, o sea que PREV_RELACION_CORTA_ACTIVA no depende de "
            "este valor. Ejecutado en el 4.8 y fijado en test_contraste_de_la_relacion_larga"
        ),
    ),
    # previous_application: cortes medidos
    # Medido y no dominio, decidido en el 4.6 con auditoria-fuga-datos: la lista se escribe como
    # liquidez urgente, pero sus cinco finalidades con al menos 100 solicitudes son exactamente las
    # cinco primeras de la tabla ordenada por tasa de default (celda 92), y el corte cae justo
    # después: Medicine (13,42%) y Repairs (13,00%) van sexta y séptima y se quedan fuera. Las otras
    # dos no llegan a las 100 solicitudes de train (13 y 23; sobre la tabla entera, 15 y 25).
    # Refijada en el 4.8 con n de 100 o más y +2pp sobre la global de las declaradas: sale Car
    # repairs, Gasification y Payments on other loans sobre 28.564 clientes. Urgent needs se queda
    # fuera con +1,85pp en train (+1,93pp en el crudo), y con ella Building a house or an annex
    "prev_finalidades_urgentes": Parametro(
        (
            "Refusal to name the goal",
            "Car repairs",
            "Gasification / water supply",
            "Money for a third person",
            "Payments on other loans",
            "Urgent needs",
            "Building a house or an annex",
        ),
        "medido",
        "finalidades declaradas que cuentan como liquidez urgente",
        "notebook 04 celdas 92 y 153",
    ),
    # El 15 es el primero de la rejilla del EDA (8, 11, 15 y 20) que cruza umbral_flags_pp. Con el
    # barrido entero sobre train el corte baja a 11, con +2,04pp y 20.319 marcados: el 11 que el EDA
    # descartaba por +1,91pp cruza al medirlo sobre el 80%, y el 15 sube a +3,31pp
    "prev_count_cola": Parametro(
        15, "medido", "corte de la cola del conteo de solicitudes", "notebook 04 celda 153"
    ),
    # El EDA registró el 4 de su rejilla descriptiva sin declarar criterio, y su barrido ya cruzaba
    # en 3 (+2,01pp). Con el criterio de las otras tres colas el 4 se sostiene sobre train: el 3
    # baja a +1,99pp y el 4 da +2,31pp con 36.575 marcados
    "prev_actividad_12m_cola": Parametro(
        4,
        "medido",
        "corte de la actividad de los últimos doce meses; la bandera solo se conserva como "
        "término de la interacción con la longitud de relación",
        "notebook 04 celda 153",
    ),
    # Dominio y no medido, decidido en el 4.8 con el usuario: el año es la convención y el barrido
    # no tiene pico. Sobre los 214.134 clientes de train con operación terminada el delta de la
    # bandera sube casi continuo con el corte (+1,06pp con 180 días, +1,83pp con 270, +2,26pp con
    # 365, +3,00pp con 730) y baja a +2,87pp con 1.095. El primer cruce de los 2pp depende de la
    # rejilla, así que la rejilla y no el dato elegiría el corte, como en los 180 días de bureau
    "prev_adelanto_liquidacion_dias": Parametro(
        365,
        "dominio",
        "adelanto sobre la fecha de fin prevista que marca liquidación anticipada, con el borde "
        "fuera",
        "notebook 04 celda 125",
        contraste_pendiente=(
            "comprobar sobre el split que PREV_EARLY_SETTLED_FLAG cruza umbral_flags_pp con 365, "
            "545, 730 y 1.095 días y no con 270, o sea que la señal empieza en el año y no "
            "depende del valor por encima de él. Ejecutado en el 4.8 y fijado en "
            "test_contraste_del_adelanto_de_liquidacion"
        ),
    ),
    # Dominio y no medido, decidido en el 4.8 con el usuario: el plazo solo toma los valores 36, 42,
    # 48, 54, 60, 66, 72 y 84, y el 60 es el borde del plazo estándar, donde acaba la masa de 35.487
    # solicitudes no aprobadas a exactamente 60 cuotas. No hay pico que barrer: con cualquier corte
    # hasta el 54 la bandera marca de 12.669 a 43.387 clientes de train con +2,3pp a +3,9pp, y el
    # salto a +14,62pp con 57 marcados llega justo al pasar del 54 al 60
    "prev_plazo_largo_cuotas": Parametro(
        60,
        "dominio",
        "plazo por encima del cual la petición rechazada marca; el mismo plazo concedido no marca",
        "notebook 04 celda 153",
        contraste_pendiente=(
            "comprobar sobre el split que PREV_REFUSED_LONG_TERM_FLAG separa más de 10pp con 60 y "
            "con 66 cuotas, y que con 48 y 54 mide otra población, la masa de 60 cuotas, con "
            "efectos de otro orden. Ejecutado en el 4.8 y fijado en test_contraste_del_plazo_largo"
        ),
    ),
    # Refijado en el 4.8 con la rejilla del EDA: sobre train el 1,1 se queda en r_rb 0,1208 (0,1194
    # en el EDA) sobre 231.992 clientes, y el 1,05 y el 1,2 bajan a 0,0837 y 0,0820
    "prev_sobreconcesion_corte": Parametro(
        1.1,
        "medido",
        "exceso sobre lo solicitado; se barrió sobre la proporción, que es la codificación "
        "que entra, y no sobre la bandera, que daba un corte opuesto",
        "notebook 04 celda 160",
    ),
    # Sin corte de mínimo de solicitudes para PREV_REFUSED_RATIO, decidido en el 4.9 con el
    # usuario. Sobre train el único mínimo defendible es 2 (r_rb 0,1555 frente a 0,1221, con el
    # 81,98% de cobertura), porque con 3 y 5 el grupo que dejan fuera separa con significación
    # (0,0308 y 0,0665). Con 2 el grupo excluido apenas separa (0,0022), pero su lectura de
    # bandera se mueve entre +1,63pp con n=250 en la tabla cruda y +3,67pp con n=193 en train,
    # dentro del mismo error típico, y el mínimo mandaría a la mediana (0) a 41.956 clientes,
    # entre ellos las 193 únicas rechazadas.
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
        "BUREAU_COUNT_COLA": ("bureau_count_cola",),
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
        "PREV_COUNT_12M": ("prev_ventana_reciente_dias",),
        "PREV_ACTIVIDAD_12M_COLA": ("prev_actividad_12m_cola", "prev_ventana_reciente_dias"),
        "PREV_RELACION_CORTA_ACTIVA": (
            "prev_actividad_12m_cola",
            "prev_ventana_reciente_dias",
            "prev_relacion_larga_anios",
        ),
        "PREV_EARLY_SETTLED_FLAG": ("prev_adelanto_liquidacion_dias",),
        "PREV_EARLY_SETTLED_COUNT": ("prev_adelanto_liquidacion_dias",),
        "PREV_EARLY_SETTLED_RATIO": ("prev_adelanto_liquidacion_dias",),
        "PREV_REFUSED_LONG_TERM_FLAG": ("prev_plazo_largo_cuotas",),
        "PREV_OVERGRANTED_RATIO": ("prev_sobreconcesion_corte",),
        "PREV_APPLICATIONS_PER_YEAR": ("suelo_anios_denominador",),
        "PREV_EARLY_HOUR_RATIO": ("prev_hora_temprana_max",),
        "PREV_URGENT_PURPOSE_RATIO": ("prev_finalidades_urgentes",),
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
    nombre: str, valor_nuevo: float | tuple[str, ...], n_train: int, sobrescribir: bool = False
) -> None:
    """Único punto de escritura del valor operativo de un reajustable.

    Exige `n_train`, el tamaño de la partición de entrenamiento usada para calcularlo: no
    basta con poner un número, tiene que declarar cuántas filas lo sostienen. No relee el
    split por su cuenta, quien llama ya lo filtró con `solo_train()` y aquí solo se registra
    el resultado.

    Refijar uno ya fijado exige `sobrescribir=True`, la misma guarda que `construir_split()`.
    Sin ella, escribir dos veces se resuelve por upsert y gana la última, que es el mecanismo
    del orden del registro de `patrones-de-fallo`: dos ajustes sobre poblaciones distintas
    dejan la segunda cifra con el n de la segunda y nadie se entera. La guarda vale para todo el
    que refija, que hoy son `ajustar_capa2a()`, los siete refijados de bureau, bureau_balance y
    previous_application en `build_features.py`, y `cargar_cortes()`, que la hereda al llamar a
    esta misma función por cada corte que lee de `cortes.json`. Cada uno con su test de que
    refijar otra vez la exige (el de la capa 2a, sobre `registrar_limites()`).
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


def alfa_bonferroni(n_contrastes: int) -> float:
    """El alfa corregido de una familia, que son los contrastes emitidos en una medición.

    No las features ni las columnas: una columna cruzada contra varios estratos emite varios.
    """
    if n_contrastes < 1:
        raise ValueError(f"una familia sin contrastes no tiene alfa: {n_contrastes}")
    return float(valor("alfa_familia")) / n_contrastes


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
