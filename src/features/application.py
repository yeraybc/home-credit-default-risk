"""Features de capa 1 de la tabla principal: banderas de ausencia y ratios sin parámetro.

Capa 1, o sea determinista y sin nada estimado: cada columna sale de una operación aritmética
o de un `notna()` sobre la propia fila. Los ratios que dependen de una columna winsorizada
(la carga de cuota sobre ingreso y el de hijos sobre miembros) **no están aquí**: se construyen
en la capa 2a, después del winsorizador, o llevarían dentro el error de captura de 117M de
ingreso que el cap corrige.

Se ejecuta después de `limpiar_application()` y antes del split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.params import valor

# Las cuatro categóricas del bloque edificio quedan fuera del cómputo de completitud. No es un
# detalle: la tripartita (completo 6,96%, parcial 7,03%, todo nulo 9,23%) sale de las columnas
# numéricas, y metiendo las cuatro categóricas da 6,96%, 7,04% y 9,26%, que son otras cifras.
#
# El bloque tiene tres recuentos distintos rondando y conviene saber cuál es cuál, porque quien
# venga del EDA va a leer un número que aquí no aparece:
#
#   46  es lo que dice la prosa del notebook 01, "42 columnas numéricas más 4 categóricas".
#       No cuenta TOTALAREA_MODE, que es el concepto 15 y no tiene pareja AVG ni MEDI.
#   15  es lo que el notebook de verdad computó, los 14 _AVG más TOTALAREA_MODE, y de ahí
#       salen las tasas publicadas. La prosa y su propio código no cuentan lo mismo.
#   43  es lo que esta función devuelve antes de limpiar: las 42 con sufijo más TOTALAREA_MODE.
#
# De los tres, 43 y 15 clasifican idéntico, y por una razón concreta: el nulo del bloque es por
# fila y no por columna, o sea que en los 14 conceptos por sus dos versiones redundantes, 28
# pares, no hay ni una fila donde _MODE o _MEDI esté presente con su _AVG nulo. Por eso 43
# colapsa a las 15 que sobreviven a la limpieza sin una sola discrepancia.
#
# El de 46 no clasifica igual, y no se arregla sacándole las cuatro categóricas: lo que queda
# son 42, que dan 6,96%, 7,05% y 9,22%. TOTALAREA_MODE es tan portante como ellas, porque 645
# clientes tienen ahí su único dato del edificio y sin esa columna pasan a leerse como todo
# nulo, que es el grupo que carga la señal, más 14 que perderían el completo. Las dos
# sensibilidades están fijadas en tests/test_build_features.py.
CATEGORICAS_EDIFICIO = (
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
)
# los tres sufijos del bloque, para identificarlo entero. No confundir con los dos de
# `cleaning.SUFIJOS_REDUNDANTES_EDIFICIO`, que son los que se eliminan: aquel es subconjunto de
# este y son dos conceptos distintos que antes compartían nombre en los dos módulos.
SUFIJOS_BLOQUE_EDIFICIO = ("_AVG", "_MODE", "_MEDI")

COLUMNAS_SOCIAL = ("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE")
PREFIJO_BURO = "AMT_REQ_CREDIT_BUREAU_"

# --- contrato de esquema ---------------------------------------------------------------------
# De qué columnas sale cada feature, declarado y no derivado. Derivarlo del frame que llegue es
# cómodo hasta que llega uno distinto: sin dos columnas del bloque edificio,
# BUILDING_INFO_COUNT pasa de escala 0 a 15 a escala 0 a 13 con el mismo nombre, y sin las seis
# del buró la bandera no se crea y la matriz sale con una columna menos, las dos cosas sin un
# solo error. El modelo se entrena con una escala y en serving recibiría otra.
#
# Las piezas de este módulo siguen siendo permisivas a propósito, porque a la API puede llegar
# un frame parcial. Quien exige el contrato es la frontera que construye la matriz,
# `preparar_application()`.
COLUMNAS_EDIFICIO = (
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "TOTALAREA_MODE",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
)
COLUMNAS_BURO = (
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)

ORIGEN_POR_FEATURE: dict[str, tuple[str, ...]] = {
    "BUILDING_INFO_COUNT": COLUMNAS_EDIFICIO,
    "HAS_BUILDING_INFO": COLUMNAS_EDIFICIO,
    "HAS_BUREAU_INFO": COLUMNAS_BURO,
    "HAS_SOCIAL_INFO": COLUMNAS_SOCIAL,
    "FLAG_EXT_SOURCE_1_NULL": ("EXT_SOURCE_1",),
    "FLAG_EXT_SOURCE_3_NULL": ("EXT_SOURCE_3",),
    "AGE_YEARS": ("DAYS_BIRTH",),
    "EMPLOYED_TO_AGE_RATIO": ("DAYS_BIRTH", "DAYS_EMPLOYED"),
    "LTV": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
}
FEATURES_CAPA1 = tuple(ORIGEN_POR_FEATURE)


def columnas_edificio_numericas(app: pd.DataFrame) -> list[str]:
    """Las del bloque edificio que entran en el cómputo de completitud."""
    return sorted(
        c
        for c in app.columns
        if c.endswith(SUFIJOS_BLOQUE_EDIFICIO) and c not in CATEGORICAS_EDIFICIO
    )


def columnas_buro(app: pd.DataFrame) -> list[str]:
    """Las seis consultas al buró, cuyo nulo es idéntico en las seis."""
    return sorted(c for c in app.columns if c.startswith(PREFIJO_BURO))


def columnas_de_origen_ausentes(app: pd.DataFrame) -> dict[str, list[str]]:
    """Por feature, qué columnas de las que declara necesitar no trae el frame."""
    ausentes = {
        f: [c for c in cols if c not in app.columns] for f, cols in ORIGEN_POR_FEATURE.items()
    }
    return {f: cols for f, cols in ausentes.items() if cols}


def verificar_contrato_capa1(app: pd.DataFrame) -> None:
    """Falla si el frame no trae exactamente las columnas de origen declaradas.

    Se comprueba en las dos direcciones porque las dos rompen igual y ninguna da error por su
    cuenta. Que **falte** una columna del bloque baja el denominador: BUILDING_INFO_COUNT pasa
    de escala 0 a 15 a escala 0 a 13 sin cambiar de nombre, y dos clientes cuyo único dato
    vivía ahí pasan a leerse como sin información. Que **sobre** es el caso simétrico: si la
    limpieza dejara de eliminar las versiones MODE y MEDI, el conteo se calcularía sobre 43
    columnas en vez de sobre 15.
    """
    ausentes = columnas_de_origen_ausentes(app)
    if ausentes:
        detalle = "; ".join(f"{f} necesita {cols}" for f, cols in sorted(ausentes.items()))
        raise ValueError(
            f"faltan columnas de origen declaradas en el contrato de capa 1: {detalle}"
        )

    for bloque, declarado, derivado in (
        ("bloque edificio", set(COLUMNAS_EDIFICIO), set(columnas_edificio_numericas(app))),
        ("consultas al buró", set(COLUMNAS_BURO), set(columnas_buro(app))),
    ):
        if declarado != derivado:
            raise ValueError(
                f"el {bloque} no coincide con el contrato: sobran "
                f"{sorted(derivado - declarado)}. La feature cambiaría de escala sin cambiar "
                "de nombre, así que nadie se enteraría"
            )


def anadir_banderas_ausencia(app: pd.DataFrame) -> pd.DataFrame:
    """Las banderas de los tres bloques de ausencia, más las dos de los scores externos.

    `HAS_BUILDING_INFO` es binaria a propósito, y no la tripartita completa: la señal está en
    no tener ningún dato del edificio (9,23% de default) y no en tenerlo incompleto, porque el
    grupo parcial (7,03%) y el completo (6,96%) no se distinguen entre sí. El conteo de
    completitud se construye al lado para que el IV decida si aporta algo por encima de la
    bandera; su denominador son las columnas del bloque que sobreviven a la limpieza, así que
    se declara en vez de suponerse.

    Los tres bloques van por separado porque son ejes independientes: su V de Cramér no llega
    a 0,10 en ninguna pareja.
    """
    app = app.copy()

    edificio = columnas_edificio_numericas(app)
    if edificio:
        presentes = app[edificio].notna().sum(axis=1)
        app["BUILDING_INFO_COUNT"] = presentes.astype("int16")
        app["HAS_BUILDING_INFO"] = presentes.gt(0).astype("int8")

    buro = columnas_buro(app)
    if buro:
        app["HAS_BUREAU_INFO"] = app[buro].notna().all(axis=1).astype("int8")

    social = [c for c in COLUMNAS_SOCIAL if c in app.columns]
    if social:
        app["HAS_SOCIAL_INFO"] = app[social].notna().all(axis=1).astype("int8")

    for n in (1, 3):
        col = f"EXT_SOURCE_{n}"
        if col in app.columns:
            app[f"FLAG_EXT_SOURCE_{n}_NULL"] = app[col].isna().astype("int8")

    return app


def anadir_ratios(app: pd.DataFrame) -> pd.DataFrame:
    """Los ratios de capa 1 y la edad en años.

    Solo los que no dependen de ninguna columna winsorizada. El denominador se protege del
    cero aunque hoy no haya ninguno, porque estas funciones también corren sobre
    application_test y sobre lo que llegue a la API.
    """
    app = app.copy()
    dias_anio = valor("dias_por_anio")

    if "DAYS_BIRTH" in app.columns:
        app["AGE_YEARS"] = -app["DAYS_BIRTH"] / dias_anio
        if "DAYS_EMPLOYED" in app.columns:
            # proporción de la vida que el cliente lleva empleado. Las dos columnas son
            # negativas, así que el cociente sale positivo, salvo en los 2 clientes con
            # DAYS_EMPLOYED a cero, que dan -0,0 y no 0,0. Da igual para cualquier
            # comparación o modelo, pero está medido y no supuesto.
            # Es NaN en los 55.374 del centinela, y eso es correcto: no llevan cero tiempo
            # empleados, es que no consta
            app["EMPLOYED_TO_AGE_RATIO"] = app["DAYS_EMPLOYED"] / app["DAYS_BIRTH"].replace(
                0, np.nan
            )

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"} <= set(app.columns):
        app["LTV"] = app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"].replace(0, np.nan)

    return app


def construir_features_capa1(app: pd.DataFrame) -> pd.DataFrame:
    """Banderas de ausencia y ratios, en un solo paso."""
    return anadir_ratios(anadir_banderas_ausencia(app))


def informe_capa1(app: pd.DataFrame, con_features: pd.DataFrame) -> pd.DataFrame:
    """Qué se ha añadido, sobre cuántas columnas y con cuánta cobertura."""
    nuevas = [c for c in con_features.columns if c not in app.columns]
    n_edificio = len(columnas_edificio_numericas(app))
    denominadores = {
        "BUILDING_INFO_COUNT": n_edificio,
        "HAS_BUILDING_INFO": n_edificio,
        "HAS_BUREAU_INFO": len(columnas_buro(app)),
        "HAS_SOCIAL_INFO": len([c for c in COLUMNAS_SOCIAL if c in app.columns]),
    }
    return pd.DataFrame(
        [
            {
                "feature": c,
                "columnas de origen": denominadores.get(c, 1),
                "no nulos": int(con_features[c].notna().sum()),
                "% cobertura": round(con_features[c].notna().mean() * 100, 2),
                "marcados o media": (
                    int(con_features[c].sum())
                    if c.startswith(("HAS_", "FLAG_"))
                    else round(float(con_features[c].mean()), 4)
                ),
            }
            for c in nuevas
        ]
    )
