"""Transformers a medida de las capas 2, los que no existen en la librería.

Capa 2a, o sea con un parámetro estimado a partir de los datos: el límite de winsorización sale
de un percentil, y por eso **se ajusta solo sobre el 80% de entrenamiento**. No toca el TARGET,
pero calcularlo sobre la tabla entera dejaría que el 20% de validación participase en elegir el
umbral que después se le aplica a él mismo.

La puerta de rareza del agrupador es capa 2a por el mismo criterio: cuenta sobre la covariable y
no mira la etiqueta. Lo que sí será capa 2b es la tabla de WoE, donde el TARGET entra al cálculo.

Los valores del EDA viven en `params.py` como referencia y nunca se consumen: `valor()` revienta
para cualquier reajustable sin fijar. Quien los reestima es el `fit` de aquí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.features.params import fijar_operativo, parametro, valor

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

# Los dos ratios que llevan una columna winsorizada en el denominador, y que por eso no están en
# la capa 1 con LTV y los otros: construirlos antes del winsorizador dejaría el error de captura
# de 117M de ingreso dentro del divisor de la carga, que es justo lo que el cap corrige.
RATIOS_POSTERIORES: dict[str, tuple[str, str]] = {
    "ANNUITY_TO_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CHILDREN_TO_FAM_RATIO": ("CNT_CHILDREN", "CNT_FAM_MEMBERS"),
}


# El nulo es un nivel más para agrupar y para el WoE, así que necesita una clave con la que
# contarlo y agruparlo. Va como cadena y no como `object()` porque tiene que sobrevivir a un
# `value_counts` y a un `groupby`, y se comprueba al ajustar que ninguna categoría real la use.
NULO = "__NULO__"

# El registro de una celda movida. Va declarado porque el frame se arma por trozos y uno vacío
# tiene que traer las mismas columnas que uno lleno, o el informe cambia de forma según el dato.
COLUMNAS_DETALLE = ["columna", "categoría", "n", "residual"]

# El residual lleva el nombre de su columna a propósito: si todas compartieran un "Other" pelado,
# el ColumnTransformer sacaría un nivel con el mismo nombre desde varias columnas y el IV del
# bloque 5 no podría decir de cuál viene cada uno.
PREFIJO_RESIDUAL = "Other_"


def residual_de(columna: str) -> str:
    """El nivel al que van a parar las categorías raras de esa columna."""
    return f"{PREFIJO_RESIDUAL}{columna}"


SUFIJO_WOE = "_WOE"


def nombre_woe(columna: str) -> str:
    """El nombre que lleva la columna una vez deja de ser categoría y pasa a ser un número."""
    return f"{columna}{SUFIJO_WOE}"


def _clave(serie: pd.Series) -> pd.Series:
    """La columna con el nulo convertido en un nivel contable."""
    if (serie == NULO).any():
        raise ValueError(f"{serie.name} trae una categoría literal {NULO!r}, que es la del nulo")
    return serie.astype(object).where(serie.notna(), NULO)


def _nombre(categoria: object) -> object:
    """El nombre legible de un nivel, con la clave del nulo deshecha."""
    return "(nulo)" if categoria is NULO or categoria == NULO else categoria


class Winsorizador(BaseEstimator, TransformerMixin):
    """Capa por arriba al múltiplo del percentil, reestimado en el `fit`.

    Solo por arriba: las colas que el EDA dejó sin capar tienen señal real en las dos
    direcciones, y este transformer no las toca en ninguna.
    """

    def _factor(self, columna: str) -> float:
        return FACTOR_POR_COLUMNA.get(columna, valor("app_winsor_factor"))

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Winsorizador:
        """Reestima el límite de cada columna. Se llama sobre `solo_train()`, nunca sobre todo.

        Es permisivo con las columnas que no lleguen, igual que la capa 1: a la API puede
        llegar un frame parcial. Quien exige el contrato es la frontera que arma la matriz.
        `quantile` ignora los NaN por su cuenta, y eso es lo que hace que `OWN_CAR_AGE` se
        ajuste sobre los clientes con coche sin escribir ningún filtro.

        Recorre `CORTES_WINSOR` y no una lista que le pasen: así no hay forma de pedir una
        columna sin corte declarado, en vez de haberla y rechazarla.
        """
        percentil = valor("app_winsor_percentil")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.limites_ = {}
        self.n_ajuste_ = {}
        for columna in CORTES_WINSOR:
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


class RatiosPosteriores(BaseEstimator, TransformerMixin):
    """Los dos ratios que se construyen detrás del winsorizador.

    No aprende nada: el `fit` existe para el contrato y para dejar anotado qué derivadas va a
    producir, que es lo que tiene que devolver `get_feature_names_out`. Va detrás por el orden
    del `Pipeline`, no por su contenido.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RatiosPosteriores:
        """Anota qué ratios puede construir con las columnas que trae el frame.

        Permisivo igual que la capa 1, porque a la API puede llegar un frame parcial, y por eso
        se anota: sin esto, `get_feature_names_out` prometería columnas que `transform` no va a
        crear en cuanto le llegue un frame más pobre que el del ajuste.
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.derivadas_ = tuple(
            nombre
            for nombre, (num, den) in RATIOS_POSTERIORES.items()
            if num in X.columns and den in X.columns
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numerador entre denominador, con el cero del divisor a NaN y no a infinito.

        Hoy `AMT_INCOME_TOTAL` y `CNT_FAM_MEMBERS` no tienen ni un cero ni un nulo en la tabla,
        pero esto corre también sobre application_test y sobre lo que llegue a la API. La
        winsorización solo capa por arriba, así que tampoco puede fabricar un cero nuevo.
        """
        check_is_fitted(self)
        X = X.copy()
        for nombre in self.derivadas_:
            num, den = RATIOS_POSTERIORES[nombre]
            faltan = [c for c in (num, den) if c not in X.columns]
            if faltan:
                raise ValueError(
                    f"{nombre} se ajustó y el frame no trae {sorted(faltan)}. "
                    "Saltárselo daría una matriz distinta de la del entrenamiento"
                )
            X[nombre] = X[num] / X[den].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Las de entrada más las derivadas que de verdad se construyen."""
        check_is_fitted(self)
        entrada = list(self.feature_names_in_ if input_features is None else input_features)
        # las que ya vengan en el frame no se duplican: `transform` las reescribe en su sitio
        return np.asarray(
            entrada + [c for c in self.derivadas_ if c not in entrada], dtype=object
        )


class AgrupadorDeRaras(BaseEstimator, TransformerMixin):
    """Manda a un residual propio de cada columna las categorías que no llegan al mínimo.

    Un solo criterio, el de rareza, que sale de `n_min_categoria`. **No compara tasas ni elige
    vecino**, y esa ausencia está medida y no supuesta: las celdas que caen aquí tienen 3, 8, 12
    y 18 observaciones, así que su tasa observada es ruido de muestreo. El plan de la fase ya
    había medido este caso exacto con un Fisher entre los dos lados de `NAME_INCOME_TYPE` (p =
    0,0207, pero probabilidad 0,186 de ver cero positivos en 20 obs a la tasa base) y su
    conclusión era colapsar en un residual único si la separación no se sostenía. No se sostiene.

    El residual lleva el nombre de su columna, así que dos columnas nunca comparten nivel y el
    IV del bloque 5 puede juzgar cada uno por separado. Aquí no se decide nada más, igual que
    con las cuatro categóricas del bloque edificio.

    **Por qué va a medida, que no es lo que decía el plan de la fase.** Aquel lo justificaba con
    que `OneHotEncoder(min_frequency=...)` agrupa por frecuencia y la regla del proyecto era
    agrupar por tasa observada; esa regla se cayó al medir, así que la justificación se cayó con
    ella. Comparados los dos en los cinco casos borde (celdas raras sueltas, columna entera rara,
    binaria con un nivel raro, nulo raro y categoría no vista), **son equivalentes**. Lo que
    sostiene tenerlo propio es lo otro: el residual se llama `Other_<columna>` y no
    `<columna>_infrequent_sklearn`, que es el nombre que van a leer el scorecard y SHAP de la
    Fase 5; `informe_agrupamiento()` da el `n` de cada categoría absorbida, que
    `infrequent_categories_` no da y que el bloque 5 necesita para juzgar el residual; y el `fit`
    revienta si una categoría real se llama como el residual. Si algún día el scorecard renombra
    columnas por su cuenta, el primer argumento desaparece y esto sobra.

    **Es capa 2a y no 2b:** el `fit` no mira el TARGET. El parámetro que estima es un recuento
    sobre la covariable, así que se ajusta sobre train por la misma razón que el winsorizador.

    Trabaja sobre las columnas categóricas del frame que le llegue y no sobre una lista propia,
    porque quien lo usa es el `ColumnTransformer`, que ya le entrega exactamente su bucket.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> AgrupadorDeRaras:
        """Anota qué categorías de cada columna no llegan al mínimo. Se llama sobre `solo_train()`.

        Permisivo con la columna que no llegue, igual que la capa 1: a la API puede llegar un
        frame parcial. `y` se acepta y se ignora, que es lo que pide el contrato del `Pipeline`.
        """
        minimo = valor("n_min_categoria")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.raras_ = {}
        detalle = []
        for columna in X.select_dtypes(include=["object", "category"]).columns:
            conteo = _clave(X[columna]).value_counts()
            raras = conteo[conteo < minimo]
            if raras.empty:
                continue
            residual = residual_de(columna)
            if residual in conteo.index:
                raise ValueError(
                    f"{columna} ya trae una categoría llamada {residual!r}, que es la del "
                    "residual: reetiquetar encima la mezclaría con las raras"
                )
            self.raras_[columna] = tuple(raras.index)
            detalle += [
                {"columna": columna, "categoría": c, "n": int(n), "residual": residual}
                for c, n in raras.items()
            ]
        self.detalle_ = pd.DataFrame(detalle, columns=COLUMNAS_DETALLE)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reetiqueta las raras. Estricto donde el `fit` es permisivo.

        La categoría que no se vio al ajustar **pasa tal cual** y no va al residual: este
        transformer solo sabe de las que contó, y quien absorbe lo desconocido es el
        `handle_unknown="ignore"` del `OneHotEncoder` que va detrás.
        """
        check_is_fitted(self)
        faltan = [c for c in self.raras_ if c not in X.columns]
        if faltan:
            raise ValueError(
                f"columnas ajustadas que el frame no trae: {sorted(faltan)}. "
                "Saltárselas daría una matriz distinta de la del entrenamiento"
            )
        X = X.copy()
        for columna, raras in self.raras_.items():
            residual = residual_de(columna)
            serie = X[columna]
            if NULO in raras:
                serie = serie.fillna(residual)
            # el residual no es origen de nadie, así que el reemplazo no encadena
            X[columna] = serie.replace({c: residual for c in raras if c != NULO})
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Las mismas de entrada: no añade ni quita ninguna, solo reetiqueta valores."""
        check_is_fitted(self)
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)


def informe_agrupamiento(agrupador: AgrupadorDeRaras) -> pd.DataFrame:
    """Puerta del punto: qué categoría se absorbió en qué residual y con cuántas observaciones.

    El `n` va en el informe porque es lo que el bloque 5 necesita para saber de qué está hecho
    cada residual sin recalcularlo.
    """
    check_is_fitted(agrupador)
    informe = agrupador.detalle_.copy()
    informe["categoría"] = informe["categoría"].map(_nombre)
    return informe


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Sustituye cada nivel por su peso de la evidencia, ajustado sobre el 80% de entrenamiento.

    Es la codificación de las categóricas de cardinalidad alta, donde el OHE saca una columna por
    nivel y el ordinal no tiene orden que respetar. Va por nivel y sin binning: el binning es de
    continuas y vive en `iv.py`, que es del bloque 5.

    **Signo:** positivo significa que el nivel falla más que la media, o sea `log` de la razón
    entre la proporción de malos y la de buenos. Es la dirección que declara el glosario del
    proyecto ("log-odds de default") y la que comparten todos los efectos del EDA, donde el signo
    positivo es siempre más riesgo. La convención clásica de scorecard es la inversa, así que
    quien venga de ahí tiene que leer esto una vez: invertirla es cambiarle el signo al `log`.

    **Es capa 2b:** el TARGET entra en el cálculo, así que ajustar fuera de train sería fuga de
    etiqueta y no solo de segundo orden.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> WoEEncoder:
        """Calcula la tabla de cada columna. Se llama sobre `solo_train()`, nunca sobre todo."""
        if y is None:
            raise ValueError(
                "WoEEncoder es capa 2b y necesita el TARGET: sin él no hay malos ni buenos que "
                "contar y no hay peso de la evidencia que calcular"
            )
        alfa = valor("suavizado_woe")
        objetivo = pd.Series(np.asarray(y), index=X.index)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.tablas_ = {
            columna: self._tabla(_clave(X[columna]), objetivo, alfa)
            for columna in X.select_dtypes(include=["object", "category"]).columns
        }
        return self

    @staticmethod
    def _tabla(clave: pd.Series, objetivo: pd.Series, alfa: float) -> pd.Series:
        """El WoE de cada nivel, con `alfa` observaciones de prior repartidas por la tasa global.

        **El reparto no es a partes iguales y esa es la pieza que importa.** Sumar la misma
        constante a los dos recuentos arrima cada nivel a un prior implícito de 50/50, que no
        tiene nada que ver con una cartera del 8% de default: al nivel que ya está por encima de
        la media lo empuja más arriba todavía. Medido sobre train en `Industry: type 8`, que con
        n = 17 es el más pequeño de los 58: sin suavizar vale +0,8920 y sumarle `alfa` a cada uno
        de los dos recuentos lo lleva a +2,0415, o sea que el suavizado lo alejaba de cero justo
        donde menos evidencia propia hay. Repartiendo el prior según la tasa global sale +0,4840:
        un nivel sin evidencia propia converge al comportamiento medio, que es WoE cero, venga de
        la dirección que venga.

        Los denominadores son los totales pelados y no llevan el prior sumado. Con el reparto a
        partes iguales sí hacía falta corregirlos, porque el prior infla las dos partes en
        proporciones distintas (0,0584 frente a 0,0051 sobre train). Repartido por la tasa
        global los dos factores de inflado valen lo mismo, `alfa` por niveles entre el total, y
        se cancelan dentro del `log`: comprobado sobre train, la diferencia entre corregir y no
        corregir es de 2,8e-16. Escribirlo sería una línea que no cambia ningún resultado.
        """
        malos = objetivo.groupby(clave, observed=True).sum()
        buenos = objetivo.groupby(clave, observed=True).count() - malos
        prior_malos = alfa * malos.sum() / (malos.sum() + buenos.sum())
        prior_buenos = alfa - prior_malos
        parte_malos = (malos + prior_malos) / malos.sum()
        parte_buenos = (buenos + prior_buenos) / buenos.sum()
        return np.log(parte_malos / parte_buenos)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica las tablas ajustadas. Estricto donde el `fit` es permisivo.

        El nivel que no se vio al ajustar sale a 0, que es el WoE neutro: la API va a recibir
        organizaciones nuevas y no puede reventar ni inventarles un riesgo.
        """
        check_is_fitted(self)
        faltan = [c for c in self.tablas_ if c not in X.columns]
        if faltan:
            raise ValueError(
                f"columnas ajustadas que el frame no trae: {sorted(faltan)}. "
                "Saltárselas daría una matriz distinta de la del entrenamiento"
            )
        X = X.copy()
        for columna, tabla in self.tablas_.items():
            X[columna] = _clave(X[columna]).map(tabla).astype(float).fillna(0.0)
        return X.rename(columns={c: nombre_woe(c) for c in self.tablas_})

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Las de entrada con el sufijo puesto: la columna deja de ser la categoría original.

        Sin el sufijo, la matriz acabaría con una columna llamada `ORGANIZATION_TYPE` que ya no
        lleva una organización sino un número, y el `ColumnTransformer` va con
        `verbose_feature_names_out=False`, así que tampoco queda el nombre del bucket que lo diga.
        """
        check_is_fitted(self)
        entrada = self.feature_names_in_ if input_features is None else input_features
        nombres = [nombre_woe(c) if c in self.tablas_ else c for c in entrada]
        return np.asarray(nombres, dtype=object)


def informe_woe(encoder: WoEEncoder) -> pd.DataFrame:
    """Puerta del punto: la tabla de WoE nivel a nivel, con lo que la sostiene."""
    check_is_fitted(encoder)
    filas = [
        {"columna": columna, "nivel": _nombre(nivel), "woe": woe}
        for columna, tabla in encoder.tablas_.items()
        for nivel, woe in tabla.items()
    ]
    return pd.DataFrame(filas, columns=["columna", "nivel", "woe"])


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


def registrar_limites(winsorizador: Winsorizador, sobrescribir: bool = False) -> pd.DataFrame:
    """Deja en `params.py` lo reestimado, y devuelve el informe de la puerta.

    Va aquí y no dentro del `fit` a propósito. `fijar_operativo()` escribe en un diccionario a
    nivel de módulo, así que dentro del `fit` cada fold del CV de la Fase 4 lo reescribiría y
    ganaría el último, que es un estado global a merced de en qué orden corran los ajustes.
    Fuera, el `fit` es puro y el registro lo escribe una sola vez quien ajusta en serio.

    El `n_train` que se declara es el de los **no nulos de cada columna**, no el de la
    partición: es el que de verdad sostiene el percentil. En `OWN_CAR_AGE` son los clientes con
    coche y no todos los de entrenamiento, y la diferencia es de tres veces.

    `sobrescribir` viaja hasta `fijar_operativo()`, que rechaza refijar un corte ya fijado si
    no se le pide explícitamente.
    """
    informe = informe_winsorizacion(winsorizador)
    for columna, limite in winsorizador.limites_.items():
        fijar_operativo(
            CORTES_WINSOR[columna], limite, winsorizador.n_ajuste_[columna], sobrescribir
        )
    return informe
