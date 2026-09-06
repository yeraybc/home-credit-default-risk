"""Orquestador del pipeline de features: ensambla la matriz de modelado.

Orden de las capas, que no es libre y es lo que este módulo fija:

1. **Capa 1 sobre la tabla principal.** `limpiar_application()` corre **antes del split**, sobre
   la tabla entera. Es determinista, no estima ningún parámetro, no mira al TARGET y no cruza
   clientes, así que ejecutarla fuera de la partición no filtra nada. Es la misma razón por la
   que las agregaciones de las tres auxiliares también van fuera. Los descartes de columna que
   el EDA decidió contra la tasa de default **no** se aplican aquí: sobreviven declarados en
   `COLUMNAS_PROVISIONALES` y los juzga la capa 2b sobre el split.
2. **Capa 0, el split**, sobre el resultado ya limpio. Se hace en este orden y no al revés
   porque la partición tiene que cubrir exactamente la población de modelado: si se partiera
   la tabla cruda, el fichero de split declararía 19 clientes que después no existen en la
   matriz, y sus conteos de estratificación no serían los de la población real.
3. **Capas 2a y 2b**, que sí estiman parámetros y por eso se ajustan solo sobre la parte de
   entrenamiento. No entran todavía en este módulo.

Las 19 filas que la limpieza quita son todas TARGET a 0, así que los 24.825 positivos no se
mueven; lo que baja es el denominador, y la tasa pasa de 8,0729% a 8,0734%.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.loader import load_table
from src.features.application import construir_features_capa1
from src.features.cleaning import (
    filas_a_eliminar,
    limpiar_application,
    limpiar_application_entrenamiento,
)
from src.features.split import construir_split

logger = logging.getLogger(__name__)

# reduce_mem_usage no se aplica a la tabla principal: baja los float a float32 y el EDA ya
# documentó que eso convierte en desigualdad estricta 453 comparaciones de fechas que en
# realidad son iguales. Son 307.511 filas, la memoria no es el problema aquí.
REDUCIR_MEMORIA = False


def cargar_y_limpiar(nombre: str = "application_train") -> pd.DataFrame:
    """Carga la tabla principal y le aplica la limpieza determinista.

    Vale igual para `application_test`, y ahí está la diferencia que importa: la tabla que
    trae etiquetas es la de entrenamiento y es la única a la que se le pueden quitar filas.
    Sobre test se limpia sin borrar a nadie, que si no se perderían los 24 clientes sin cuota
    declarada.
    """
    app = load_table(nombre, reduce_memory=REDUCIR_MEMORIA)
    limpio = (
        limpiar_application_entrenamiento(app)
        if "TARGET" in app.columns
        else limpiar_application(app)
    )
    logger.info(
        "%s: %s filas x %s columnas -> %s x %s",
        nombre,
        f"{len(app):,}",
        app.shape[1],
        f"{len(limpio):,}",
        limpio.shape[1],
    )
    return limpio


def preparar_application(nombre: str = "application_train") -> pd.DataFrame:
    """La capa 1 entera sobre la tabla principal: limpieza más features sin parámetro.

    Las dos mitades se dejan invocables por separado porque cada una tiene su propia puerta
    de salida: la limpieza se mide en filas y columnas, y las features en la tripartita del
    bloque edificio.
    """
    return construir_features_capa1(cargar_y_limpiar(nombre))


def construir_base(sobrescribir: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La tabla principal con la capa 1 aplicada y su partición, en ese orden.

    Devuelve `(base, split)`. El split se persiste; si ya existe hay que pedir
    `sobrescribir=True`, porque rehacerlo invalida todo lo que se haya ajustado sobre él.
    """
    base = preparar_application()
    split = construir_split(app=base, sobrescribir=sobrescribir)
    return base, split


def informe_base(app_cruda: pd.DataFrame, limpio: pd.DataFrame) -> pd.DataFrame:
    """Puerta de salida de la capa 1: qué entra, qué sale y cuánto se mueve la tasa."""
    fuera = filas_a_eliminar(app_cruda)
    return pd.DataFrame(
        [
            {"medida": "filas", "cruda": len(app_cruda), "limpia": len(limpio)},
            {"medida": "columnas", "cruda": app_cruda.shape[1], "limpia": limpio.shape[1]},
            {
                "medida": "filas eliminadas (netas)",
                "cruda": int(fuera.sum()),
                "limpia": len(app_cruda) - len(limpio),
            },
            {
                "medida": "positivos",
                "cruda": int(app_cruda["TARGET"].sum()),
                "limpia": int(limpio["TARGET"].sum()),
            },
            {
                "medida": "% default",
                "cruda": round(app_cruda["TARGET"].mean() * 100, 4),
                "limpia": round(limpio["TARGET"].mean() * 100, 4),
            },
            {
                "medida": "centinela marcado",
                "cruda": int(app_cruda["DAYS_EMPLOYED"].eq(365243).sum()),
                "limpia": int(limpio["FLAG_DAYS_EMPLOYED_ANOMALY"].sum()),
            },
        ]
    )
