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
# detalle: la tripartita del EDA (completo 6,96%, parcial 7,03%, todo nulo 9,23%) sale de las
# 43 columnas numéricas, y metiendo las cuatro categóricas da 6,96%, 7,04% y 9,26%, que son
# otras cifras. Tras la limpieza quedan 15 numéricas y clasifican idéntico, sin una sola
# discrepancia, porque el nulo del bloque es por fila y no por columna.
CATEGORICAS_EDIFICIO = (
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
)
SUFIJOS_EDIFICIO = ("_AVG", "_MODE", "_MEDI")

COLUMNAS_SOCIAL = ("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE")
PREFIJO_BURO = "AMT_REQ_CREDIT_BUREAU_"


def columnas_edificio_numericas(app: pd.DataFrame) -> list[str]:
    """Las del bloque edificio que entran en el cómputo de completitud."""
    return sorted(
        c for c in app.columns if c.endswith(SUFIJOS_EDIFICIO) and c not in CATEGORICAS_EDIFICIO
    )


def columnas_buro(app: pd.DataFrame) -> list[str]:
    """Las seis consultas al buró, cuyo nulo es idéntico en las seis."""
    return sorted(c for c in app.columns if c.startswith(PREFIJO_BURO))


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
            # las dos son negativas, así que el ratio sale positivo: proporción de la vida
            # que el cliente lleva empleado. Es NaN en los 55.374 del centinela, y eso es
            # correcto: no llevan cero tiempo empleados, es que no consta
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
    denominadores = {
        "BUILDING_INFO_COUNT": len(columnas_edificio_numericas(app)),
        "HAS_BUILDING_INFO": len(columnas_edificio_numericas(app)),
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
