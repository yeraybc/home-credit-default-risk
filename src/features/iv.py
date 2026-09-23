"""IV y WoE con binning, capa 2b: el TARGET entra al cálculo, así que todo se mide sobre train.

El método se escribió en el 5.4 antes de ver ningún IV, para que el 5.5 y el 5.6 no elijan tramos
ni criterios a medida del resultado. La lectura del IV es la escala de credit scoring que cierra
`min_iv`: por debajo de 0,02 la feature no separa morosos de no morosos ni sola.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.params import valor

# El nulo es un nivel más para agrupar y para el WoE, así que necesita una clave con la que
# contarlo y agruparlo. Va como cadena y no como `object()` porque tiene que sobrevivir a un
# `value_counts` y a un `groupby`, y se comprueba al ajustar que ninguna categoría real la use.
NULO = "__NULO__"


def _clave(serie: pd.Series) -> pd.Series:
    """La columna con el nulo convertido en un nivel contable."""
    if (serie == NULO).any():
        raise ValueError(f"{serie.name} trae una categoría literal {NULO!r}, que es la del nulo")
    return serie.astype(object).where(serie.notna(), NULO)


def tramos(serie: pd.Series) -> pd.Series:
    """El tramo de cada fila, con el binning que se fijó antes de medir.

    - Continuas: cuantiles sobre la propia serie, que es train, con `n_bins_max` como máximo.
      Los empates colapsan tramos (`duplicates="drop"`): un conteo con el 60% de ceros sale con
      menos de diez, y es lo correcto, que partir un empate sería inventar un orden.
    - Banderas, o cualquier numérica con dos valores o menos: un tramo por valor.
    - Categóricas: un tramo por nivel, sin agrupar; el suavizado de `tabla_woe()` se ocupa del
      nivel pequeño.
    - El NaN, siempre tramo propio: conserva el signo del grupo ausente, que en `bureau` es de
      más riesgo y en `previous_application` de menos.
    """
    numerica = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    if numerica and serie.nunique() > 2:
        corte = pd.qcut(serie, valor("n_bins_max"), duplicates="drop")
        # categórica y no object: los intervalos y la clave del nulo no se ordenan entre sí
        return corte.cat.add_categories(NULO).fillna(NULO)
    return _clave(serie).astype("category")


def tabla_woe(clave: pd.Series, objetivo: pd.Series, alfa: float | None = None) -> pd.DataFrame:
    """Por tramo: n, malos, buenos, las dos partes suavizadas, el WoE y su aporte al IV.

    **Signo:** WoE positivo es más riesgo que la media, la dirección del glosario y de todos los
    efectos del EDA. La convención clásica de scorecard es la inversa.

    **El suavizado reparte `alfa` observaciones de prior por la tasa global, no a partes
    iguales.** Sumar la misma constante a los dos recuentos arrima cada nivel a un prior
    implícito de 50/50, que no tiene nada que ver con una cartera del 8% de default: al nivel que
    ya está por encima de la media lo empuja más arriba todavía. Medido sobre train en
    `Industry: type 8`, que con n = 17 es el más pequeño de los 58 de `ORGANIZATION_TYPE`: sin
    suavizar vale +0,8920 y sumarle `alfa` a cada recuento lo lleva a +2,0415, o sea que el
    suavizado lo alejaba de cero justo donde menos evidencia propia hay. Repartiendo el prior por
    la tasa global sale +0,4840: un nivel sin evidencia propia converge a WoE cero, venga de la
    dirección que venga.

    Los denominadores son los totales pelados y no llevan el prior sumado. Con el reparto a
    partes iguales sí hacía falta corregirlos, porque el prior infla las dos partes en
    proporciones distintas (0,0584 frente a 0,0051 sobre train). Repartido por la tasa global los
    dos factores de inflado valen lo mismo, `alfa` por niveles entre el total, y se cancelan
    dentro del `log`: comprobado sobre train, la diferencia entre corregir y no es de 2,8e-16.

    El aporte al IV es `(parte_malos − parte_buenos) · woe`, que nunca es negativo: las dos
    diferencias llevan el mismo signo. Con el prior, además, nunca es infinito.
    """
    alfa = valor("suavizado_woe") if alfa is None else alfa
    agrupado = objetivo.groupby(clave, observed=True)
    malos = agrupado.sum()
    n = agrupado.count()
    buenos = n - malos
    prior_malos = alfa * malos.sum() / n.sum()
    parte_malos = (malos + prior_malos) / malos.sum()
    parte_buenos = (buenos + alfa - prior_malos) / buenos.sum()
    woe = np.log(parte_malos / parte_buenos)
    return pd.DataFrame(
        {
            "n": n,
            "malos": malos,
            "buenos": buenos,
            "parte_malos": parte_malos,
            "parte_buenos": parte_buenos,
            "woe": woe,
            "iv": (parte_malos - parte_buenos) * woe,
        }
    )


def calcular_iv(serie: pd.Series, objetivo: pd.Series, alfa: float | None = None) -> float:
    """El IV de una columna sobre sus tramos. Se llama sobre train, nunca sobre la matriz entera."""
    objetivo = pd.Series(np.asarray(objetivo), index=serie.index)
    return float(tabla_woe(tramos(serie), objetivo, alfa)["iv"].sum())


def iv_condicionado(
    serie: pd.Series, estrato: pd.Series, objetivo: pd.Series, alfa: float | None = None
) -> float:
    """El IV incremental sobre `estrato`, que en el 5.6 es `EXT_SOURCE_3`.

    Es el IV de `serie` dentro de cada tramo del estrato (sus deciles más el nulo), ponderado por
    el peso del tramo. La serie se tramea una sola vez sobre todo train, no dentro de cada tramo,
    para que sus cortes no cambien de un estrato a otro. Una copia disfrazada del estrato no
    separa nada dentro de sus tramos y sale cerca de cero.

    **Criterio, escrito antes de medir ninguna candidata:** la feature aporta sobre el estrato si
    su IV condicionado llega a `min_iv`, la misma barra que el marginal.

    Un tramo del estrato sin malos o sin buenos aporta cero: ahí no hay nada que separar, y su
    tabla dividiría por cero. También el del nulo cuando el estrato no trae ninguno, que existe
    vacío.
    """
    objetivo = pd.Series(np.asarray(objetivo), index=serie.index)
    clave, grupo = tramos(serie), tramos(estrato)
    total = 0.0
    for nivel in grupo.cat.categories:
        dentro = (grupo == nivel).to_numpy()
        y = objetivo[dentro]
        if y.nunique() < 2:
            continue
        total += dentro.sum() * tabla_woe(clave[dentro], y, alfa)["iv"].sum()
    return total / len(serie)
