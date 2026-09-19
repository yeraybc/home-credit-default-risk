"""Limpieza determinista de nivel fila, capa 1 del pipeline de features.

Sirve a cuatro tablas y en cada una hace una cosa distinta, así que las cuatro partes van
separadas: `limpiar_application()` con sus centinelas y sus columnas eliminadas,
`limpiar_bureau()` con la validez de dominio de los importes y las fechas,
`limpiar_bureau_balance()` con la decodificación de `STATUS` y `limpiar_previous()` con el
centinela de las fechas y el dominio del estado. Lo que comparten es el criterio de capa, no el
contenido.

Solo entra aquí lo que no estima ningún parámetro a partir de datos: centinelas, columnas que
se eliminan por motivo estructural, y caps con una constante fija de dominio. Todo lo que
salga de un percentil o de una mediana vive en la capa 2a, aunque parezca la misma clase de
operación.

Dos fronteras que este módulo mantiene, y las dos son de fuga:

1. **Ningún descarte decidido contra el TARGET se ejecuta aquí.** Un descarte estructural
   (varianza nula, redundancia entre columnas) se sostiene sobre cualquier submuestra y es
   firme. Uno que se eligió comparando contra la tasa de default se midió sobre train
   completo, que es exactamente lo que `params.py` obliga a refijar sobre `solo_train()`.
   Esos no se ejecutan: se declaran en `COLUMNAS_PROVISIONALES` y los decide la capa 2b, que
   es la que tiene el split. Si se eliminaran aquí, la columna ya no existiría cuando llegue
   el momento de juzgarla con la evidencia buena.
2. **Eliminar filas es de entrenamiento, nunca de inferencia.** Vive en
   `limpiar_application_entrenamiento()` y exige TARGET. En serving ningún cliente puede
   desaparecer: `application_test` trae 24 nulos de `AMT_ANNUITY` y la API recibirá los suyos.

Las decisiones son las del EDA y no se reabren aquí; lo que sí se declara es el motivo de
cada una, para que el informe pueda explicar el recuento línea a línea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.params import valor

# - filas -
# Las cinco son residuales y suman 19 filas, con 2 de solape. Las tres de nulos son el
# resultado matemático que el EDA marcó como MCAR: con 0% de default, su delta es exactamente
# la media global invertida, así que no hay nada que imputar ni señal que preservar.
FILAS_POR_CATEGORIA = {
    "CODE_GENDER": "los 4 registros XNA, que impiden binarizar el sexo",
    "NAME_FAMILY_STATUS": "los 2 registros Unknown",
}
FILAS_POR_NULO = {
    "CNT_FAM_MEMBERS": "2 nulos residuales",
    "DAYS_LAST_PHONE_CHANGE": "1 nulo residual",
    "AMT_ANNUITY": "12 nulos residuales",
}
# el valor que marca la fila fuera, declarado una sola vez: la máscara y el informe tienen que
# contar exactamente lo mismo
VALOR_CATEGORIA_FUERA = {"CODE_GENDER": "XNA", "NAME_FAMILY_STATUS": "Unknown"}

# - columnas -
# Firmes: el motivo es estructural y no depende del TARGET, así que aguanta igual sobre train,
# sobre validación y sobre lo que llegue a la API. Se ejecutan aquí.
COLUMNAS_FIRMES = {
    "FLAG_MOBIL": "varianza nula: un único registro a 0 en 307.511",
    "FLAG_EMP_PHONE": (
        "redundante con DAYS_EMPLOYED, r = -0,9998 y solo 12 discrepancias con el centinela; "
        "se conserva la continua, que es más granular"
    ),
    # El desempate de cuál de las dos gemelas sobrevive sí miró al TARGET, pero es inmaterial:
    # a r = 0,9985 las dos son intercambiables y lo que sostiene el descarte es que una de las
    # dos sobra, muy por encima del 0,70 de redundancia_pearson.
    "OBS_60_CNT_SOCIAL_CIRCLE": (
        "redundante con OBS_30, r = 0,9985, muy por encima del umbral de redundancia; sobra "
        "una de las dos con independencia de la tasa de default"
    ),
}


@dataclass(frozen=True)
class DescarteProvisional:
    """Un descarte que el EDA decidió contra el TARGET sobre train completo.

    No se ejecuta en capa 1. `remedir` es lo que la capa 2b tiene que reproducir sobre
    `solo_train()` antes de confirmarlo o revertirlo.
    """

    motivo: str
    remedir: str

    def __post_init__(self) -> None:
        if not self.motivo.strip() or not self.remedir.strip():
            raise ValueError("un descarte provisional declara motivo y qué hay que remedir")


# Provisionales: el descarte se decidió comparando contra la tasa de default sobre train
# completo. La columna sobrevive a la capa 1 para que la capa 2b pueda juzgarla con el split.
COLUMNAS_PROVISIONALES = {
    "FLAG_CONT_MOBILE": DescarteProvisional(
        "sin señal, r = 0,0004 con TARGET, pese a tener 574 registros a 0",
        "recalcular la correlación con el objetivo sobre solo_train; con 574 registros a 0 "
        "el estimador es frágil y el descarte no se sostiene en nada estructural",
    ),
    # La skill daba 1,00 de correlación y son 0,8605, así que el motivo declarado no es
    # redundancia numérica sino señal prestada, que es el caso de flags anidadas de la
    # metodología: con DEF_30 a cero no hay ni un caso de DEF_60 por encima de cero, dentro de
    # cada estrato de DEF_30 la de 60 días no aporta nada significativo (+0,94pp con p = 0,016,
    # +1,85pp con p = 0,13 y +3,86pp con p = 0,20) y al revés DEF_30 sí aporta (+1,76pp con
    # p = 4,4e-10). Todos esos contrastes son sobre train completo.
    "DEF_60_CNT_SOCIAL_CIRCLE": DescarteProvisional(
        "señal prestada de DEF_30, que la contiene; no aporta efecto propio dentro de sus "
        "estratos y r con TARGET es menor (0,0313 frente a 0,0322)",
        "repetir los estratos de DEF_30 sobre solo_train. Existe además el argumento "
        "estructural, r = 0,8605 con DEF_30 y por encima del 0,70 declarado, que si la 2b lo "
        "adopta convierte el descarte en firme sin necesidad de mirar al objetivo",
    ),
}

# Del bloque edificio se conserva solo la versión AVG de cada concepto numérico: MODE y MEDI
# son estadísticamente intercambiables. La lista se deriva del propio frame en vez de
# escribirse a mano, porque hay tres columnas que terminan en _MODE y no pertenecen al grupo:
# las cuatro categóricas del bloque, que no tienen AVG ni MEDI, y TOTALAREA_MODE, que es el
# concepto 15 y no tiene pareja.
# Solo los dos que se eliminan. `application.SUFIJOS_BLOQUE_EDIFICIO` son los tres que
# identifican el bloque entero: este es subconjunto de aquel, y son dos conceptos distintos que
# antes compartían el nombre `SUFIJOS_EDIFICIO` en los dos módulos.
SUFIJOS_REDUNDANTES_EDIFICIO = ("_MODE", "_MEDI")


def columnas_edificio_redundantes(app: pd.DataFrame) -> list[str]:
    """Las versiones MODE y MEDI de los conceptos del bloque edificio que sí tienen AVG."""
    con_avg = {c[: -len("_AVG")] for c in app.columns if c.endswith("_AVG")}
    return sorted(
        c
        for c in app.columns
        for suf in SUFIJOS_REDUNDANTES_EDIFICIO
        if c.endswith(suf) and c[: -len(suf)] in con_avg
    )


def columnas_a_eliminar(app: pd.DataFrame) -> dict[str, str]:
    """Columna y motivo de las que se eliminan en capa 1, todas por motivo estructural.

    Las provisionales no salen aquí a propósito: eliminarlas dejaría a la capa 2b sin la
    columna que tiene que juzgar.
    """
    motivos = {c: m for c, m in COLUMNAS_FIRMES.items() if c in app.columns}
    for c in columnas_edificio_redundantes(app):
        motivos[c] = "bloque edificio: se conserva solo la versión AVG del concepto"
    return motivos


def columnas_pendientes_de_decidir(app: pd.DataFrame) -> dict[str, DescarteProvisional]:
    """Las que el EDA descartó contra el TARGET y la capa 2b tiene que rejuzgar sobre el split.

    Sobreviven a la capa 1 y llegan a la matriz. Que se queden ahí sin decidir es el riesgo
    simétrico al de eliminarlas a ciegas, así que la 2b consume esta lista.
    """
    return {c: d for c, d in COLUMNAS_PROVISIONALES.items() if c in app.columns}


def filas_a_eliminar(app: pd.DataFrame) -> pd.Series:
    """Máscara de las filas residuales que se eliminan, con los solapes ya resueltos."""
    fuera = pd.Series(False, index=app.index)
    for col in FILAS_POR_CATEGORIA:
        if col in app.columns:
            fuera |= app[col].eq(VALOR_CATEGORIA_FUERA[col])
    for col in FILAS_POR_NULO:
        if col in app.columns:
            fuera |= app[col].isna()
    return fuera


COL_ANOMALIA = "FLAG_DAYS_EMPLOYED_ANOMALY"


def aplicar_centinela(app: pd.DataFrame) -> pd.DataFrame:
    """DAYS_EMPLOYED a NaN donde lleva el código de inactivo, con su bandera al lado.

    La bandera se calcula **antes** de sustituir: después el dato que la define ya no existe.
    Y no se sobrescribe si ya está, sino que se acumula con un o lógico: en una segunda pasada
    el centinela ya es NaN, así que recalcularla la pondría a cero y borraría en silencio los
    55.374 marcados de la primera.
    """
    centinela = valor("centinela_365243")
    anomalo = app["DAYS_EMPLOYED"].eq(centinela)
    app = app.copy()
    previa = app[COL_ANOMALIA].astype("int8") if COL_ANOMALIA in app.columns else 0
    app[COL_ANOMALIA] = (anomalo.astype("int8") | previa).astype("int8")
    app.loc[anomalo, "DAYS_EMPLOYED"] = pd.NA
    return app


# Qué columna lleva qué cap fijo, declarado y no escrito dentro de la función, por lo mismo que
# `CORTES_WINSOR` en la capa 2a: una columna que entra o que sale cambia la matriz sin cambiar
# ningún nombre.
#
# Las otras cinco ventanas del buró no llevan cap aquí, y cada una por su motivo, que conviene
# tener escrito porque son seis columnas gemelas y es fácil que una se caiga sin que se note:
# WEEK, MON y QRT se winsorizan al 3xp99 en la capa 2a; YEAR el EDA la deja sin capar a
# propósito, porque su cola tiene señal real y su máximo de 25 es plausible; y HOUR no lo
# necesita, que su p99 es 0 igual que el de DAY pero su máximo es 4 y no alcanza el cap de 5.
# O sea que HOUR está sin cap por medición, no por olvido.
CAPS_DE_DOMINIO: dict[str, str] = {"AMT_REQ_CREDIT_BUREAU_DAY": "app_amt_req_bureau_day_max"}


def aplicar_caps_de_dominio(app: pd.DataFrame) -> pd.DataFrame:
    """Caps con constante fija. Los que salen de un percentil van en la capa 2a, no aquí."""
    app = app.copy()
    for col, corte in CAPS_DE_DOMINIO.items():
        if col in app.columns:
            app[col] = app[col].clip(upper=valor(corte))
    return app


# - bureau -
# **Toda cifra de esta sección es de `bureau` completo, 1.716.428 filas y 305.811 clientes**, que
# es sobre lo que corre la limpieza: es capa 1 y el mismo frame lo recorren entrenamiento,
# validación, `application_test` y la API, así que acotarla a train dejaría a los demás con los
# 60M dentro. El EDA publicó las suyas sobre `bureau_t`, que es el cruce con `application_train`
# y son 1.465.325 filas, así que **no coinciden y las dos están bien**: 1.408 filas en moneda
# extranjera aquí y 1.231 allí, 1.110 clientes y 968, 27.644 con deuda mayor que crédito y
# 23.282, 62.604 vencimientos lejanos y 52.497. La proporción no se mueve (el exceso grosero es
# 4,38% aquí, 4,02% en bureau_t y 4,03% sobre el 80% de entrenamiento), porque los cortes son
# fijos y no se estiman de la muestra. Quien contraste las dos fuentes verá la diferencia, y es
# esperada.
#
# Los seis importes de la tabla. Son los que no se pueden sumar entre monedas distintas, y por
# eso la lista existe: el resto de columnas de una fila en moneda extranjera (estado, tipo y las
# cuatro fechas) sigue siendo válido y se conserva.
IMPORTES_BUREAU: tuple[str, ...] = (
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "AMT_ANNUITY",
)

MONEDA_NACIONAL = "currency 1"
COL_MONEDA = "CREDIT_CURRENCY"
# Nivel fila, y por eso no lleva el `HAS_`. El plan y `agregacion-multitabla` nombran
# `BUREAU_HAS_FOREIGN_CURRENCY`, que es la de nivel cliente y sale del `max` de esta al agregar:
# son dos columnas distintas, no dos nombres de la misma.
COL_MONEDA_EXTRANJERA = "BUREAU_FOREIGN_CURRENCY"

# `CREDIT_CURRENCY` **sobrevive a la limpieza** y no es olvido. El plan pide eliminarla como
# variable, y eso es decisión de la agregación: aquí sigue haciendo falta para poder recalcular la
# bandera en una segunda pasada, y tirarla ahora perdería de qué moneda se trata (2, 3 o 4) sin que
# nadie lo haya pedido. Lo que el plan prohíbe es que llegue a la matriz como predictor, y de eso
# responde `agg_bureau`, que decide qué columnas de nivel fila sobreviven al `groupby`.

# Qué importe pasa a NaN por encima de qué corte. Son las cuatro columnas que el EDA declaró, y
# las otras dos de `IMPORTES_BUREAU` quedan fuera por medición y no por olvido: el máximo de
# AMT_CREDIT_SUM_LIMIT es 4.705.600 y el de AMT_CREDIT_SUM_OVERDUE 3.756.681, o sea que ninguna
# se acerca a los 50M y un cap suyo no llegaría a dispararse nunca sobre esta tabla.
CAPS_BUREAU: dict[str, str] = {
    "AMT_CREDIT_SUM": "bureau_importe_max",
    "AMT_CREDIT_SUM_DEBT": "bureau_importe_max",
    "AMT_CREDIT_MAX_OVERDUE": "bureau_importe_max",
    "AMT_ANNUITY": "bureau_cuota_max",
}


def marcar_moneda_extranjera(bureau: pd.DataFrame) -> pd.DataFrame:
    """Los importes en moneda distinta de la nacional pasan a NaN, con su bandera al lado.

    Son 1.408 filas de 1.716.428, el 0,082%, y cargan el 3,83% de la cuota total de la tabla:
    la cifra no es grande, está en otra unidad, y sumarla con las nacionales infla el agregado
    sin que falle nada. La fila no se borra ni se vacía entera porque 1.072 de los 1.110
    clientes afectados tienen mezcla de monedas, y de los 38 que no, 29 tienen un único crédito
    y borrarlos los dejaría leyéndose como clientes sin historial, que es el grupo del 10,12%
    de default frente al 7,73%.

    Una moneda **no informada** cae del lado extranjero y pierde sus importes, que es la lectura
    prudente: sin saber la unidad no se puede sumar. Hoy no se dispara, porque `bureau.csv` no
    trae ni un nulo en la columna, pero la API sí puede mandarlos y conviene que la disposición
    esté escrita y no sea un efecto de que `ne()` diga verdad sobre un NaN.

    La bandera se recalcula en cada pasada y no se acumula con un o lógico, al revés que la del
    centinela: aquí el dato que la define es la moneda, que no se toca, así que la segunda
    pasada da lo mismo que la primera.
    """
    if COL_MONEDA not in bureau.columns:
        return bureau
    bureau = bureau.copy()
    extranjera = bureau[COL_MONEDA].ne(MONEDA_NACIONAL)
    bureau[COL_MONEDA_EXTRANJERA] = extranjera.astype("int8")
    presentes = [col for col in IMPORTES_BUREAU if col in bureau.columns]
    bureau[presentes] = bureau[presentes].mask(extranjera, axis=0)
    return bureau


# Los importes cuya **ausencia o signo** lee alguna feature de la receta, y por eso hay que
# fotografiarlos antes de anular nada. `AMT_CREDIT_SUM` es el único de `IMPORTES_BUREAU` sin foto,
# porque de él solo se lee la magnitud.
IMPORTES_CON_FOTO: dict[str, str] = {
    "AMT_ANNUITY": (
        "presencia y signo: BUREAU_CREDITS_WITH_ANNUITY_COUNT, _RATIO, HAS_BUREAU_ANNUITY y el "
        "nivel de la cuota"
    ),
    "AMT_CREDIT_SUM_DEBT": "presencia: HAS_BUREAU_FINANCIAL_DETAIL",
    "AMT_CREDIT_SUM_LIMIT": (
        "presencia y signo: HAS_BUREAU_FINANCIAL_DETAIL y BUREAU_NEGATIVE_LIMIT_FLAG"
    ),
    "AMT_CREDIT_MAX_OVERDUE": (
        "presencia y signo: HAS_BUREAU_OVERDUE_HISTORY, BUREAU_HAS_ANY_OVERDUE y "
        "BUREAU_OVERDUE_UNION"
    ),
    "AMT_CREDIT_SUM_OVERDUE": "signo: BUREAU_OVERDUE_UNION y BUREAU_HAS_CURRENT_OVERDUE",
}

SUFIJO_SIGNO = "_SIGNO"


def fotografiar_signo(bureau: pd.DataFrame) -> pd.DataFrame:
    """Guarda el signo de cada importe, **antes** de que la limpieza lo anule.

    La foto vale -1, 0 o +1, y NaN si el buró no reportó el dato, así que la presencia es su
    `notna()` y no hace falta otra columna que pueda contradecirla.

    Existe porque la corrección de magnitud y las banderas comparten columna y se pisan. Las
    features de cuota no leen el importe, leen si se reportó y si era cero:
    `BUREAU_CREDITS_WITH_ANNUITY_COUNT` entra con tres niveles (sin cuota reportada 7,50%,
    reportada a cero, reportada con valor) porque el nulo y el cero son grupos de riesgo
    contrario. La limpieza anula 432 cuotas de 341 clientes, y leída la presencia del importe
    limpio 27 clientes cambian de nivel, 13 de ellos al de sin cuota reportada. Y guardar solo la
    presencia se queda corto, porque de esas 432 cuotas 191 eran cero y 241 positivas: sin el
    signo, 20 clientes siguen cambiando de nivel. Lo mismo pasa con la mora:
    `BUREAU_HAS_ANY_OVERDUE` y `BUREAU_OVERDUE_UNION` leen `> 0` y perdían 60 clientes. El signo
    no depende de la moneda ni de que la magnitud sea un error de captura, así que se conserva
    donde el importe se anula.

    **La foto va antes y no se reconstruye después**, que era el primer intento y no funciona: la
    bandera de moneda marca las 1.408 filas extranjeras y solo 412 traían cuota, así que
    `notna() | bandera` no recupera las 412, inventa las otras 996. La información solo existe
    antes de la máscara.

    Idempotente por el mismo motivo que la bandera del centinela lleva su o lógico: en la segunda
    pasada el importe ya es NaN, así que refotografiar borraría la foto buena. Se escribe una vez
    y las siguientes pasadas la respetan.
    """
    bureau = bureau.copy()
    for col in IMPORTES_CON_FOTO:
        destino = f"{col}{SUFIJO_SIGNO}"
        if col in bureau.columns and destino not in bureau.columns:
            bureau[destino] = np.sign(bureau[col])
    return bureau


def aplicar_caps_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Importes por encima de su cap de plausibilidad a NaN, no capados al corte.

    A NaN y no al valor del corte porque no se sabe cuánto vale de verdad ese crédito, y dejarlo
    en 50M seguiría dominando la suma del cliente. La cola alta tiene señal, pero esto no es
    cola: es error de captura.
    """
    bureau = bureau.copy()
    for col, corte in CAPS_BUREAU.items():
        if col in bureau.columns:
            bureau[col] = bureau[col].mask(bureau[col] > valor(corte))
    return bureau


def capar_deuda_al_credito(bureau: pd.DataFrame) -> pd.DataFrame:
    """El exceso grosero de deuda sobre crédito se capa al propio crédito.

    Capa y no anula, que es la diferencia con el resto: el exceso leve y el moderado son deuda
    real (intereses y penalizaciones sobre el principal) y ahí vive la señal de
    sobreendeudamiento, así que solo se corta lo que pasa del ratio declarado. Sobre la tabla
    completa son 1.211 filas de las 27.644 con deuda mayor que crédito, el 4,38%, con un máximo
    de 139.396 veces el principal.

    Idempotente por construcción: después del cap el ratio de esas filas vale exactamente 1 y
    ya no vuelve a cruzar el corte.
    """
    if not {"AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"}.issubset(bureau.columns):
        return bureau
    bureau = bureau.copy()
    credito = bureau["AMT_CREDIT_SUM"]
    deuda = bureau["AMT_CREDIT_SUM_DEBT"]
    # el denominador solo cuenta donde es positivo: con crédito 0 o negativo el ratio no
    # significa nada y la fila se deja como está
    ratio = deuda / credito.where(credito > 0)
    grosero = ratio > valor("bureau_ratio_deuda_credito_max")
    bureau["AMT_CREDIT_SUM_DEBT"] = deuda.mask(grosero, credito)
    return bureau


# Qué fecha lleva qué acotación, declarado y no escrito dentro de la función, por lo mismo que
# `CAPS_BUREAU` y que las seis ventanas del buró de arriba: son columnas gemelas y es fácil que
# una se caiga de su lista sin que se note.
#
# El suelo de -30 años es el mismo para las tres y sale del mismo corte, porque son la misma
# clase de error y el inventario del EDA (notebook 02, celda 79) las enumera juntas. `DAYS_CREDIT`
# no lleva suelo y no es olvido: su mínimo es -2.922, o sea la ventana de ocho años del buró, y
# nunca se acerca al corte.
FECHAS_CON_SUELO: tuple[str, ...] = (
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
)
# Solo el vencimiento tiene además tope por arriba: es el único que mira al futuro.
FECHAS_CON_TOPE: dict[str, str] = {"DAYS_CREDIT_ENDDATE": "bureau_enddate_max_anios"}


def acotar_ventanas_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Las fechas fuera de ventana pasan a NaN, a nivel fila y antes de agregar.

    Son las cuatro colas del inventario de validez de dominio del EDA, y van las cuatro juntas
    porque el argumento es el mismo para todas: 62.604 vencimientos de más de 20 años, que son
    placeholder de revolving; y por el otro extremo 146 vencimientos, 95 actualizaciones y 1
    cierre de más de 30 años atrás, que son error de captura.

    El corte se aplica aquí y no dentro de la agregación, que es donde lo tenía el EDA: aplicado
    en un solo sitio alcanza también a `BUREAU_CLOSED_AFTER_ENDDATE`, que compara el cierre real
    contra esta misma fecha, y deja de haber un corte con dos criterios según quién lo mire.

    Las 146 del vencimiento pasado son el caso que más pesa, y no por su volumen: **95 de ellas
    encienden `BUREAU_CLOSED_AFTER_ENDDATE` siendo falsas**, porque cualquier cierre real es
    posterior a un vencimiento de hace 115 años. Sin acotarlas, 43 clientes llevan esa bandera de
    riesgo puesta por un dato roto.
    """
    bureau = bureau.copy()
    dias = valor("dias_por_anio")
    suelo = -valor("bureau_cierre_max_anios") * dias
    for col in FECHAS_CON_SUELO:
        if col in bureau.columns:
            bureau[col] = bureau[col].mask(bureau[col] < suelo)
    for col, corte in FECHAS_CON_TOPE.items():
        if col in bureau.columns:
            bureau[col] = bureau[col].mask(bureau[col] > valor(corte) * dias)
    return bureau


def limpiar_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Validez de dominio de bureau, a nivel fila y antes de cualquier agregación.

    Ninguna fila se borra: bureau no tiene TARGET propio y el mismo frame lo recorren
    entrenamiento, `application_test` y la API. Capar la deuda después de sumarla no arregla la
    suma, y por eso todo esto va aquí y no en `agregar_bureau()`.

    El orden importa en dos sitios. La foto va **la primera**, porque las tres reglas siguientes
    anulan importes y la presencia y el signo que las features leen se perderían. Y la
    moneda y los caps de importe van antes que el cap de deuda, porque un `AMT_CREDIT_SUM` que se
    ha ido a NaN deja el ratio sin denominador y su fila sin capar, que es lo correcto cuando el
    principal es el dato roto.
    """
    limpio = fotografiar_signo(bureau)
    limpio = marcar_moneda_extranjera(limpio)
    limpio = aplicar_caps_bureau(limpio)
    limpio = capar_deuda_al_credito(limpio)
    return acotar_ventanas_bureau(limpio)


# - bureau_balance -
# Es el caso opuesto a `bureau`: el EDA no encontró ni un valor fuera de dominio, así que esta
# limpieza no retira nada. Lo que aporta es la decodificación, que convierte una letra en la
# severidad y las dos banderas que el nivel crédito agrega, y el parar cuando el dominio se rompe.
#
# El dominio de `STATUS` y su traducción son la misma constante: declarar la lista de códigos
# válidos aparte del mapa sería el mismo dominio en dos sitios, que es lo que dejó 52.500 frente a
# 52.497 en `bureau`. `C` (saldado) y `X` (sin información) van a NaN y no a 0 porque están fuera
# de la escala de severidad: leer "sin dato" como "sin mora" es la codificación que diluye la
# señal rara, y es la razón de que el nivel crédito tenga que distinguir los créditos sin ningún
# estado numérico, que sobre esta tabla entera son **130.368, el 15,95%**. El EDA publica 67.116
# (12,82%) y no es otra cifra: es la misma medida sobre los 523.515 créditos enlazables a train,
# que es la población que él tenía. Las dos están bien, igual que `bureau_t` frente a `bureau`.
STATUS_DPD: dict[str, float] = {
    "C": np.nan,
    "X": np.nan,
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
}

# Categórica con los ocho códigos **siempre** y en este orden, estén o no en el lote. Un
# `astype("category")` a secas deduce los niveles del frame que tenga delante, así que el cliente
# suelto de la API saldría con un dtype distinto del de la tabla entera: es el patrón 12, el
# esquema que depende del lote.
DTYPE_STATUS = pd.CategoricalDtype(categories=list(STATUS_DPD))

# La severidad en ese mismo orden, para decodificar por el código de la categórica en vez de con
# un `map` sobre 27,3 millones de cadenas. Sale del mismo dict y en la línea de al lado, así que
# no puede desalinearse de las categorías, y un `map` sobre la categórica tampoco valdría: `C` y
# `X` colapsan al mismo NaN y pandas decide entonces si devuelve categórica o no.
SEVERIDAD_BB = np.array(list(STATUS_DPD.values()), dtype="float32")

COL_BB_DPD = "BB_DPD"
COL_BB_IS_X = "BB_IS_X"
COL_BB_IS_DPD = "BB_IS_DPD"

# La ventana del panel, cerrada por los dos lados. No sale de `params.py` y no es olvido: no es un
# corte de modelado con procedencia del EDA, es el rango que el buró reporta.
MESES_BB = (-96, 0)


def _normalizar_status(status: pd.Series) -> pd.Series:
    """`strip()` y `upper()` sobre los ocho niveles y no sobre los 27,3 millones de cadenas.

    `load_table` deja `STATUS` en `category`, y un `.str.strip().str.upper()` encima la
    devuelve a `object` reconstruyendo una cadena por fila. Renombrar las categorías toca ocho
    valores. El camino de `object` queda para el frame que llega sin tipos, como el de la API.
    """
    if isinstance(status.dtype, pd.CategoricalDtype):
        nuevas = {c: str(c).strip().upper() for c in status.cat.categories}
        # dos niveles que colapsan al mismo (" C" y "C") no se pueden renombrar en sitio
        if len(set(nuevas.values())) == len(nuevas):
            return status.cat.rename_categories(nuevas)
        status = status.astype(object)
    return status.where(status.isna(), status.astype(str).str.strip().str.upper())


def _exigir_dominio(valores: pd.Series, fuera: pd.Series, que: str, tabla: str) -> None:
    """Revienta nombrando qué encontró, no solo cuántas filas."""
    if not fuera.any():
        return
    vistos = sorted(pd.unique(valores[fuera]).tolist(), key=repr)[:10]
    raise ValueError(
        f"{que} fuera de dominio en {int(fuera.sum())} filas de {tabla}: {vistos}. "
        "La tabla no trae ninguna, así que es un dato que el pipeline no sabe leer y no se "
        "deja caer en silencio, que es el fallo abierto de CREDIT_ACTIVE en bureau"
    )


def limpiar_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """Decodifica `STATUS` y exige el dominio del panel, a nivel fila y antes de agregar.

    Deja tres columnas al lado de las tres de la tabla: `BB_DPD` con la severidad de 0 a 5 y NaN
    en `C` y `X`, y las dos banderas `BB_IS_X` y `BB_IS_DPD`. La escala fina se conserva aquí
    aunque el modelo vaya a usar la binaria, porque el nivel crédito necesita el peor estado.

    **`BB_IS_DPD` vale 0 en `C` y `X`, y es a propósito:** la pregunta que responde es si ese mes
    está en mora, y un mes cerrado o sin informar no lo está. Lo que no se puede hacer con ella es
    sumarla y dividir entre todos los meses, que sería leer el sin dato como sin mora: el
    denominador bueno son los meses reportados, o sea `BB_DPD.notna()`, y quien los cuenta es el
    nivel crédito. Las tres columnas son las dos lecturas en el mismo sitio a propósito, para que
    el de arriba no tenga que volver a mirar la letra.

    **Las dos validaciones revientan en vez de dejar caer la fila.** Un código de `STATUS` que no
    esté en `STATUS_DPD` saldría como NaN y se leería igual que un `C`, o sea como "sin mora"; y
    un `MONTHS_BALANCE` fuera de -96 a 0, o con decimales que el `int8` truncaría, es el eje del
    panel, del que el nivel crédito saca la ventana, así que anularlo la mediría mal sin que nada
    avise. Es la diferencia con `acotar_ventanas_bureau()`, donde la fecha rota es un dato
    entre otros y la fila sobrevive sin él.

    Ninguna fila se borra, por lo mismo que en `limpiar_bureau()`, y además porque la suma de
    `BB_MONTHS_TOTAL` contra las filas enlazadas es una puerta del nivel cliente.

    Idempotente sin necesitar guarda: las tres derivadas salen de `STATUS`, que esta función
    normaliza pero no destruye, así que la segunda pasada las recalcula idénticas. Es lo
    contrario de `fotografiar_signo()`, que sí lee una columna que sus vecinas anulan.

    **El coste del paso grande, medido y no supuesto**, que es lo que el nivel crédito necesita
    para saber si la doble agregación cabe entera: sobre las 27.299.925 filas la limpieza tarda
    0,2 segundos y deja el frame en 312 MB, frente a los 156 MB y 6,4 segundos de la carga. El
    pico de RSS con `load_table` es de 2,4 GB al cargar y 2,6 al limpiar, así que
    `reduce_mem_usage` basta y no hace falta tocar la carga. `BB_DPD` va en `float32` (104 MB) y
    no en `Int8` nullable (52 MB): los 52 MB de diferencia no se notan contra ese pico, el `max`
    por crédito tarda lo mismo con los dos, y `Int8` sería la única columna nullable del proyecto.

    Lo que **no** hace, para que no parezca olvido: no busca duplicados del par crédito-mes, que
    ya caza la guarda de ventana de `agregar_por_credito()`, y pagarlo dos veces sobre 27,3
    millones de filas no sale a cuenta; y no mira `SK_ID_BUREAU`, que es
    del puente, el único que tiene `bureau` delante para saber si la clave ajena vale. **Tampoco
    le fija el tipo**, así que sale en `uint32` desde `load_table` y en `int64` desde un
    `read_csv` a pelo: es el patrón 12 sobre la clave, y quien decide es el nivel crédito, que la
    agrupa, y el puente, que la cruza con `bureau`. Las cinco columnas que esta función sí fija no
    dependen del lote.

    **Su llamante es `agregar_por_credito()`**, que la pasa por dentro antes de agregar este frame
    por `SK_ID_BUREAU`.
    """
    bb = bb.copy()
    if "MONTHS_BALANCE" in bb.columns:
        meses = bb["MONTHS_BALANCE"]
        fuera = ~meses.between(*MESES_BB)
        # en una columna entera no cabe un decimal, y mirarlo ahí costaría 0,09 s sobre la tabla
        if not pd.api.types.is_integer_dtype(meses):
            fuera |= meses.mod(1).ne(0)
        _exigir_dominio(meses, fuera, "MONTHS_BALANCE", "bureau_balance")
        bb["MONTHS_BALANCE"] = meses.astype("int8")
    if "STATUS" in bb.columns:
        status = _normalizar_status(bb["STATUS"])
        _exigir_dominio(
            status, ~status.isin(DTYPE_STATUS.categories), "STATUS", "bureau_balance"
        )
        # El orden de los niveles se impone con el constructor y **no con un
        # `astype(DTYPE_STATUS)`**: pandas da por iguales dos categóricas no ordenadas con el
        # mismo conjunto de niveles, así que el astype devolvía la de entrada tal cual y el orden
        # se colaba desde el lote. `load_table` deja `C` y `X` al final, y leída la severidad por
        # el código con ese orden la puerta salía con el 71,56% de las filas en mora en vez del
        # 1,26%.
        bb["STATUS"] = pd.Categorical(status, categories=DTYPE_STATUS.categories)
        bb[COL_BB_DPD] = SEVERIDAD_BB[bb["STATUS"].cat.codes.to_numpy()]
        bb[COL_BB_IS_X] = bb["STATUS"].eq("X").astype("int8")
        bb[COL_BB_IS_DPD] = bb[COL_BB_DPD].gt(0).astype("int8")
    return bb


# - previous_application -
# **Cifras sobre la tabla entera, 1.670.214 filas y 338.857 clientes**, que es sobre lo que corre
# la limpieza. Aquí coinciden con las del EDA, que midió el nivel solicitud sobre la tabla entera y
# solo pasó a `prev_t` al cruzar con el objetivo.
#
# Las seis fechas del ciclo de vida. `DAYS_DECISION` no trae el centinela y va igual, para que la
# regla no dependa de una medición.
FECHAS_PREVIOUS: tuple[str, ...] = (
    "DAYS_DECISION",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)
# La entrada en importe y en tasa: hoy sus negativos son las mismas dos filas, pero cada una se
# corrige por su cuenta porque la API puede mandar una sin la otra.
ENTRADAS_PREVIOUS: tuple[str, ...] = ("AMT_DOWN_PAYMENT", "RATE_DOWN_PAYMENT")
# Sin normalizar a propósito: la tabla no trae ni una variante por espacios o caso, así que un
# estado mal escrito es un dato que el pipeline no sabe leer.
ESTADOS_CONTRATO: tuple[str, ...] = ("Approved", "Canceled", "Refused", "Unused offer")


def limpiar_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Validez de dominio de previous_application, a nivel fila y antes de agregar.

    Tres reglas y es el caso opuesto al buró: ningún importe se acerca a lo implausible, así que
    no hay caps. **El centinela 365243 pasa a NaN** en las fechas del ciclo de vida, 1.506.087
    celdas repartidas en cinco de las seis; sin esto el 4.5 contaría 365243 como fecha por vencer.
    **Las dos entradas negativas** (-0,90 y -0,45, redondeo sobre un importe que no puede serlo)
    pasan a 0. Y **un `NAME_CONTRACT_STATUS` fuera de sus cuatro estados revienta**, NaN
    incluido, en vez de caer en silencio, que es el fallo abierto de `CREDIT_ACTIVE` en bureau.

    **El centinela no lleva bandera, y está comprobado y no supuesto.** En `DAYS_TERMINATION`
    el EDA lo leyó como contrato vivo y no como ausencia, así que pasarlo a NaN lo junta con los
    673.065 del bloque sin formalizar. Pero todas las features de la receta se midieron sobre
    las fechas con el centinela ya a NaN, y la única que lo leía, el conteo de contratos vivos
    (0,0118), no llegó al ranking. Solo se tocan esas seis columnas: hay un cliente con
    `SK_ID_CURR` 365243.

    Lo que **no** hace, para que no parezca olvido: no borra filas, por lo mismo que las otras
    dos auxiliares; no toca `SELLERPLACE_AREA`, cuyo -1 es un código que no alimenta ninguna
    feature; y no fija el tipo de las fechas, que salen en `float32` desde `load_table` y
    cambian de tipo según lleve centinela el lote. Eso lo fija la agregación en su frontera.

    Idempotente sin guarda: en la segunda pasada el centinela ya es NaN y la entrada ya es 0.
    """
    prev = prev.copy()
    centinela = valor("centinela_365243")
    for col in FECHAS_PREVIOUS:
        if col in prev.columns:
            prev[col] = prev[col].mask(prev[col].eq(centinela))
    for col in ENTRADAS_PREVIOUS:
        if col in prev.columns:
            prev[col] = prev[col].clip(lower=0)
    if "NAME_CONTRACT_STATUS" in prev.columns:
        estado = prev["NAME_CONTRACT_STATUS"]
        _exigir_dominio(
            estado, ~estado.isin(ESTADOS_CONTRATO), "NAME_CONTRACT_STATUS", "previous_application"
        )
    return prev


def limpiar_application(app: pd.DataFrame) -> pd.DataFrame:
    """Limpieza válida en cualquier ruta: no elimina ni una fila.

    Es la que vale para `application_test` y para la API, donde un cliente no puede
    desaparecer. Conserva el índice de quien llama, para que la predicción se pueda alinear
    de vuelta con la petición.
    """
    limpio = app.drop(columns=list(columnas_a_eliminar(app)))
    limpio = aplicar_centinela(limpio)
    limpio = aplicar_caps_de_dominio(limpio)
    return limpio


def limpiar_application_entrenamiento(app: pd.DataFrame) -> pd.DataFrame:
    """La limpieza de arriba más el borrado de las 19 filas residuales.

    Exige TARGET, y no por capricho: eliminar filas solo es correcto construyendo la población
    de modelado. En inferencia la misma llamada perdería clientes en silencio, que es lo que
    pasaría con los 24 nulos de AMT_ANNUITY de application_test.
    """
    if "TARGET" not in app.columns:
        raise ValueError(
            "esta limpieza elimina filas y la tabla no trae TARGET: en inferencia ningún "
            "cliente puede desaparecer, usa limpiar_application()"
        )
    fuera = filas_a_eliminar(app)
    return limpiar_application(app.loc[~fuera]).reset_index(drop=True)


def informe_limpieza(app: pd.DataFrame) -> pd.DataFrame:
    """Recuento línea a línea de lo que la limpieza quita, lo que aplaza y por qué.

    Las guardas de columna ausente son las mismas que usa `filas_a_eliminar`: el informe no
    puede reventar con un frame que la limpieza sí sabe tratar.
    """
    filas = []
    for col, motivo in FILAS_POR_CATEGORIA.items():
        if col not in app.columns:
            continue
        v = VALOR_CATEGORIA_FUERA[col]
        filas.append(
            {"tipo": "fila", "objeto": col, "n": int(app[col].eq(v).sum()), "motivo": motivo}
        )
    for col, motivo in FILAS_POR_NULO.items():
        if col not in app.columns:
            continue
        filas.append(
            {"tipo": "fila", "objeto": col, "n": int(app[col].isna().sum()), "motivo": motivo}
        )
    filas.append(
        {
            "tipo": "fila",
            "objeto": "unión, con solapes resueltos",
            "n": int(filas_a_eliminar(app).sum()),
            "motivo": "es lo que de verdad se elimina, y solo en entrenamiento",
        }
    )
    for col, motivo in columnas_a_eliminar(app).items():
        filas.append({"tipo": "columna", "objeto": col, "n": 1, "motivo": motivo})
    for col, descarte in columnas_pendientes_de_decidir(app).items():
        filas.append(
            {
                "tipo": "columna aplazada",
                "objeto": col,
                "n": 1,
                "motivo": f"{descarte.motivo}. Pendiente en la capa 2b: {descarte.remedir}",
            }
        )
    return pd.DataFrame(filas)
