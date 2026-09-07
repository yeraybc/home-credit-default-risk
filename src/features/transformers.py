"""Transformers a medida de las capas 2, los que no existen en la librería.

Capa 2a, o sea con un parámetro estimado a partir de los datos: el límite de winsorización sale
de un percentil, y por eso **se ajusta solo sobre el 80% de entrenamiento**. No toca el TARGET,
pero calcularlo sobre la tabla entera dejaría que el 20% de validación participase en elegir el
umbral que después se le aplica a él mismo.

Los valores del EDA viven en `params.py` como referencia y nunca se consumen: `valor()` revienta
para cualquier reajustable sin fijar. Quien los reestima es el `fit` de aquí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.features.params import parametro, valor

# Qué columna lleva qué corte del registro. Declarado y no derivado del frame que llegue, por lo
# mismo que el contrato de esquema de la capa 1: una columna que se cuela o que falta cambia la
# matriz sin cambiar ningún nombre y sin dar un solo error.
#
# `OBS_60_CNT_SOCIAL_CIRCLE` no está y el plan de la fase lo listaba: la limpieza lo elimina como
# descarte firme por r = 0,9985 con `OBS_30`, así que nunca llega hasta aquí. `DEF_60` sí está,
# porque su descarte se decidió contra la tasa de default y sobrevive como provisional.
CORTES_WINSOR: dict[str, str] = {
    "AMT_INCOME_TOTAL": "app_winsor_amt_income_total",
    "DEF_30_CNT_SOCIAL_CIRCLE": "app_winsor_def_30_cnt_social_circle",
    "DEF_60_CNT_SOCIAL_CIRCLE": "app_winsor_def_60_cnt_social_circle",
    "OBS_30_CNT_SOCIAL_CIRCLE": "app_winsor_obs_30_cnt_social_circle",
    "AMT_REQ_CREDIT_BUREAU_QRT": "app_winsor_amt_req_credit_bureau_qrt",
    "AMT_REQ_CREDIT_BUREAU_MON": "app_winsor_amt_req_credit_bureau_mon",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "app_winsor_amt_req_credit_bureau_week",
    "CNT_CHILDREN": "app_winsor_cnt_children",
    "CNT_FAM_MEMBERS": "app_winsor_cnt_fam_members",
    "OWN_CAR_AGE": "app_cap_p99_own_car_age",
}

# La antigüedad del coche se capa al p99 pelado y no al 3xp99 del resto: por encima de 64 años
# no hay un activo financiero real, así que el múltiplo no tiene sentido aquí.
FACTOR_POR_COLUMNA: dict[str, float] = {"OWN_CAR_AGE": 1.0}


class Winsorizador(BaseEstimator, TransformerMixin):
    """Capa por arriba al múltiplo del percentil, reestimado en el `fit`.

    Solo por arriba: las colas que el EDA dejó sin capar tienen señal real en las dos
    direcciones, y este transformer no las toca en ninguna.
    """

    def __init__(self, columnas: tuple[str, ...] | None = None):
        self.columnas = columnas

    def _factor(self, columna: str) -> float:
        return FACTOR_POR_COLUMNA.get(columna, valor("app_winsor_factor"))

    def _declaradas(self) -> tuple[str, ...]:
        if self.columnas is None:
            return tuple(CORTES_WINSOR)
        desconocidas = [c for c in self.columnas if c not in CORTES_WINSOR]
        if desconocidas:
            raise ValueError(
                f"columnas sin corte declarado en CORTES_WINSOR: {sorted(desconocidas)}. "
                "Todo corte pasa por params.py con su procedencia, no como cifra suelta"
            )
        return tuple(self.columnas)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Winsorizador:
        """Reestima el límite de cada columna. Se llama sobre `solo_train()`, nunca sobre todo.

        Es permisivo con las columnas que no lleguen, igual que la capa 1: a la API puede
        llegar un frame parcial. Quien exige el contrato es la frontera que arma la matriz.
        `quantile` ignora los NaN por su cuenta, y eso es lo que hace que `OWN_CAR_AGE` se
        ajuste sobre los clientes con coche sin escribir ningún filtro.
        """
        percentil = valor("app_winsor_percentil")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.limites_ = {}
        self.n_ajuste_ = {}
        for columna in self._declaradas():
            if columna not in X.columns:
                continue
            serie = X[columna].dropna()
            if serie.empty:
                continue
            self.limites_[columna] = float(self._factor(columna) * serie.quantile(percentil))
            self.n_ajuste_[columna] = int(serie.size)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica los límites ajustados. Estricto donde el `fit` es permisivo.

        Si falta una columna que sí estaba al ajustar, revienta en vez de saltársela: el
        modelo se habría entrenado con esa columna capada y en serving recibiría otra cosa.
        """
        check_is_fitted(self)
        faltan = [c for c in self.limites_ if c not in X.columns]
        if faltan:
            raise ValueError(
                f"columnas ajustadas que el frame no trae: {sorted(faltan)}. "
                "Saltárselas daría una matriz distinta de la del entrenamiento"
            )
        X = X.copy()
        for columna, limite in self.limites_.items():
            X[columna] = X[columna].clip(upper=limite)
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Las mismas de entrada: no añade ni quita ninguna, solo recorta valores."""
        check_is_fitted(self)
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)


def informe_winsorizacion(winsorizador: Winsorizador) -> pd.DataFrame:
    """Puerta del punto: lo reestimado sobre train frente a la referencia del EDA.

    La desviación es el dato que se pide medido y no supuesto. Que salga cero no es una
    comprobación de que el ajuste fue sobre train: para eso está el test sintético, donde las
    dos particiones dan percentiles distintos.
    """
    check_is_fitted(winsorizador)
    filas = []
    for columna, limite in winsorizador.limites_.items():
        referencia = parametro(CORTES_WINSOR[columna]).valor_referencia
        factor = winsorizador._factor(columna)
        filas.append(
            {
                "columna": columna,
                "corte": CORTES_WINSOR[columna],
                "n ajuste": winsorizador.n_ajuste_[columna],
                "percentil": limite / factor,
                "factor": factor,
                "límite": limite,
                "ref. EDA": referencia,
                "% desviación": round((limite / referencia - 1) * 100, 4),
            }
        )
    return pd.DataFrame(filas)
