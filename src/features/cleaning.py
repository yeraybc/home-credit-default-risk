"""Limpieza determinista de nivel fila, capa 1 del pipeline de features.

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
# La limpieza de bureau (importes por encima de 50M, cuota por encima de 10M, deuda capada al
# crédito, moneda) entra en el bloque 2, con sus propias constantes.

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
