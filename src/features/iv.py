"""IV y WoE con binning, capa 2b: el TARGET entra al cálculo, así que todo se mide sobre train.

El método se escribió en el 5.4 antes de ver ningún IV, para que el 5.5 y el 5.6 no elijan tramos
ni criterios a medida del resultado. La lectura del IV es la escala de credit scoring que cierra
`min_iv`: por debajo de 0,02 la feature no separa morosos de no morosos ni sola.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.agg_bureau import COLUMNAS_SIN_RECETA as _SIN_RECETA_BUREAU
from src.features.agg_bureau_balance import COLUMNAS_SIN_RECETA as _SIN_RECETA_BB
from src.features.agg_previous import lectura_solo_vivas
from src.features.params import valor
from src.features.recipes import cargar_receta

# El nulo es un nivel más para agrupar y para el WoE, así que necesita una clave con la que
# contarlo y agruparlo. Va como cadena y no como `object()` porque tiene que sobrevivir a un
# `value_counts` y a un `groupby`, y se comprueba al ajustar que ninguna categoría real la use.
NULO = "__NULO__"


def _clave(serie: pd.Series) -> pd.Series:
    """La columna con el nulo convertido en un nivel contable."""
    if (serie == NULO).any():
        raise ValueError(f"{serie.name} trae una categoría literal {NULO!r}, que es la del nulo")
    return serie.astype(object).where(serie.notna(), NULO)


def _numerica_no_booleana(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)


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
    if _numerica_no_booleana(serie) and serie.nunique() > 2:
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


@dataclass(frozen=True)
class Candidata:
    """Una columna que espera al IV, con de dónde sale y por qué.

    `presencia` es la bandera de su tabla, o `None` en las de la tabla principal. Con ella la
    lectura contra `min_iv` se hace sin la señal del grupo ausente, que el modelo ya recibe por la
    `HAS_*` y que el pipeline no le da a la columna (la bandera va a 0 y la continua a mediana).
    """

    fuente: str
    motivo: str
    presencia: str | None = None

    def __post_init__(self) -> None:
        if not self.fuente.strip() or not self.motivo.strip():
            raise ValueError("una candidata al IV declara fuente y motivo")


_BUREAU, _BB, _PREV = "HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE", "HAS_PREV_APPLICATION"
_GEO = "selection.py, pendiente de IV"
_EDIFICIO = "bloque edificio"

# La lista cerrada del 5.5, escrita antes de medir ningún IV. Las 38 salen de siete fuentes, y el
# motivo de las que vienen de receta es su `estado`.
CANDIDATAS_IV: dict[str, Candidata] = {
    # las 11 `decision: iv` de las recetas
    "HAS_BUREAU_FINANCIAL_DETAIL": Candidata("receta de bureau", "IV (banda débil)", _BUREAU),
    "HAS_BEEN_PROLONGED": Candidata("receta de bureau", "IV (banda débil)", _BUREAU),
    "BUREAU_DAYS_CREDIT_ENDDATE_MAX": Candidata("receta de bureau", "IV (secundaria)", _BUREAU),
    "BUREAU_ANNUITY_ACTIVE_RATIO": Candidata("receta de bureau", "IV (secundaria)", _BUREAU),
    "BUREAU_CREDIT_TYPE_NUNIQUE": Candidata("receta de bureau", "IV (secundaria)", _BUREAU),
    "BB_MANY_CREDITS_FLAG": Candidata(
        "receta de bureau_balance", "IV (no pasa Bonferroni); su corte bajó de 22 a 18", _BB
    ),
    "BB_MONTHS_TOTAL": Candidata(
        "receta de bureau_balance", "IV (descarte ya no estructural)", _BB
    ),
    "BB_DPD_MONTHS_COUNT": Candidata("receta de bureau_balance", "IV (secundaria)", _BB),
    "BB_CREDITS_WITH_DPD_COUNT": Candidata("receta de bureau_balance", "IV (secundaria)", _BB),
    "PREV_EARLY_HOUR_RATIO": Candidata("receta de previous_application", "IV (banda débil)", _PREV),
    "PREV_DAYS_DECISION_MAX": Candidata(
        "receta de previous_application", "IV (banda débil), además control", _PREV
    ),
    # las que selection.py deja al IV: tres de región y una de ciudad, y las tres de contacto
    "LIVE_CITY_NOT_WORK_CITY": Candidata(_GEO, "redundante con REG_CITY_NOT_WORK_CITY"),
    "REG_REGION_NOT_WORK_REGION": Candidata(_GEO, "las de región, un orden por debajo de ciudad"),
    "REG_REGION_NOT_LIVE_REGION": Candidata(_GEO, "las de región, un orden por debajo de ciudad"),
    "LIVE_REGION_NOT_WORK_REGION": Candidata(_GEO, "redundante con REG_REGION_NOT_WORK_REGION"),
    "FLAG_PHONE": Candidata("selection.py, contacto", "sin decisión propia en el EDA"),
    "FLAG_WORK_PHONE": Candidata("selection.py, contacto", "sin decisión propia en el EDA"),
    "FLAG_EMAIL": Candidata("selection.py, contacto", "la más clara del bloque a caerse"),
    "NAME_HOUSING_TYPE": Candidata(
        "selection.py, vivienda", "el pipeline no fusiona niveles: la puerta es de rareza"
    ),
    # el bloque edificio: el EDA dejó las categóricas entre binarizar y eliminar
    "FONDKAPREMONT_MODE": Candidata(_EDIFICIO, "binarizar o eliminar, 68% de nulos"),
    "HOUSETYPE_MODE": Candidata(_EDIFICIO, "binarizar o eliminar, 50% de nulos"),
    "WALLSMATERIAL_MODE": Candidata(_EDIFICIO, "OHE, binarizar o eliminar, 50% de nulos"),
    "EMERGENCYSTATE_MODE": Candidata(_EDIFICIO, "binarizar o eliminar, 47% de nulos"),
    "BUILDING_INFO_COUNT": Candidata(_EDIFICIO, "el 5.6 la decide frente a HAS_BUILDING_INFO"),
    # las seis de COLUMNAS_SIN_RECETA, sin receta con la que compararlas
    "BUREAU_HAS_FOREIGN_CURRENCY": Candidata(
        "COLUMNAS_SIN_RECETA", "efecto medido en el 2.3", _BUREAU
    ),
    "BUREAU_HAS_CURRENT_OVERDUE": Candidata(
        "COLUMNAS_SIN_RECETA", "la mora activa leída de la foto; el 5.6 elige lectura", _BUREAU
    ),
    "BUREAU_COUNT_COLA": Candidata(
        "COLUMNAS_SIN_RECETA", "la cola del conteo, pendiente 5", _BUREAU
    ),
    "BB_MONTHS_REPORTED": Candidata("COLUMNAS_SIN_RECETA", "denominador, no explicativa", _BB),
    "BB_TRAJECTORY": Candidata("COLUMNAS_SIN_RECETA", "el peor recorrido del cliente", _BB),
    "PREV_RELACION_CORTA_ACTIVA": Candidata(
        "COLUMNAS_SIN_RECETA", "la interacción del pendiente 7", _PREV
    ),
    # los cuatro efectos que en el 3.10 no se distinguen de cero. HAS_BUREAU_BALANCE se lee sobre
    # la presencia de bureau, que la contiene: sobre sí misma daría cero por construcción
    "BB_STATUS_WORST": Candidata(
        "efecto nulo del 3.10", "sin significación dentro de los morosos", _BB
    ),
    "BB_RECOVERED_DPD_FLAG": Candidata("efecto nulo del 3.10", "sin significación", _BB),
    "BB_N_CREDITS_WBAL": Candidata("efecto nulo del 3.10", "sin significación", _BB),
    "HAS_BUREAU_BALANCE": Candidata("efecto nulo del 3.10", "sin significación", _BUREAU),
    # los dos descartes provisionales de application_train, que se remiden además aparte
    "FLAG_CONT_MOBILE": Candidata("COLUMNAS_PROVISIONALES", "sin señal en el EDA"),
    "DEF_60_CNT_SOCIAL_CIRCLE": Candidata("COLUMNAS_PROVISIONALES", "señal prestada de DEF_30"),
    # las dos banderas raras, que llevan su n y no se descartan por un IV bajo
    "BUREAU_NEGATIVE_LIMIT_FLAG": Candidata("bandera rara", "conservar (severidad)", _BUREAU),
    "PREV_REFUSED_LONG_TERM_FLAG": Candidata("bandera rara", "conservar (severidad)", _PREV),
}

BANDERAS_RARAS: tuple[str, ...] = ("BUREAU_NEGATIVE_LIMIT_FLAG", "PREV_REFUSED_LONG_TERM_FLAG")


def _es_bandera(serie: pd.Series) -> bool:
    """El complemento de la rama continua de `tramos()`: numérica con dos valores o menos."""
    return _numerica_no_booleana(serie) and serie.nunique() <= 2


def informe_iv(train: pd.DataFrame, candidatas: dict[str, Candidata] | None = None) -> pd.DataFrame:
    """El IV de cada candidata sobre train, marginal y sin la presencia de su tabla.

    Revienta nombrando lo que falte, antes de medir nada: una candidata que no está en `train` (un
    nombre cambiado, por ejemplo) no puede desaparecer del informe en silencio.

    `iv_lectura` es la cifra que se compara contra `min_iv`: el sin presencia si la candidata la
    declara (`iv_condicionado()` reutilizado tal cual, con `presencia` como estrato de dos niveles),
    y si no el marginal. Reutiliza `calcular_iv()`, `tramos()` e `iv_condicionado()`; no hay código
    de IV nuevo aquí.
    """
    candidatas = CANDIDATAS_IV if candidatas is None else candidatas
    faltan = [c for c in candidatas if c not in train.columns]
    if faltan:
        raise KeyError(f"candidatas al IV ausentes del frame: {faltan}")
    if "TARGET" not in train.columns:
        raise KeyError("informe_iv() necesita TARGET en el frame")
    objetivo = train["TARGET"]
    min_iv = valor("min_iv")

    filas = []
    for nombre, candidata in candidatas.items():
        serie = train[nombre]
        bandera = _es_bandera(serie)
        iv = calcular_iv(serie, objetivo)
        iv_sin_presencia = np.nan
        if candidata.presencia is not None:
            iv_sin_presencia = iv_condicionado(serie, train[candidata.presencia], objetivo)
        lectura = iv if candidata.presencia is None else iv_sin_presencia
        filas.append(
            {
                "feature": nombre,
                "fuente": candidata.fuente,
                "n": int(serie.notna().sum()),
                "n_marcados": int(serie.sum()) if bandera else np.nan,
                # nunique() y no len(categories): la rama continua de tramos() añade el nulo
                # como categoría aunque la serie no traiga ninguno, y contarla infla n_tramos
                # en uno para toda continua sin NaN
                "n_tramos": int(tramos(serie).nunique()),
                "iv": iv,
                "iv_sin_presencia": iv_sin_presencia,
                "iv_lectura": lectura,
                "llega": bool(lectura >= min_iv),
                "rara": nombre in BANDERAS_RARAS,
            }
        )
    return pd.DataFrame(filas).set_index("feature")


# --- las decisiones del 5.6 ---------------------------------------------------------------------


def informe_mora_activa(train: pd.DataFrame) -> pd.DataFrame:
    """Las dos lecturas de la mora activa de `bureau`, sin la presencia de la tabla.

    `BUREAU_HAS_CURRENT_OVERDUE` lee el `> 0` de la foto y `BUREAU_CURRENT_OVERDUE_SUM > 0` el de la
    suma limpia, con su NaN fuera (la codificación de la receta), así que difieren en quien tiene
    mora solo en otra moneda o toda su deuda en otra moneda.

    **Criterio, escrito antes de medir:** entra la de mayor IV; si no difieren en la cuarta cifra
    decimal, decide la construcción y entra la foto, que no pierde la mora en moneda extranjera.
    """
    suma = train["BUREAU_CURRENT_OVERDUE_SUM"]
    lecturas = {
        "BUREAU_HAS_CURRENT_OVERDUE": train["BUREAU_HAS_CURRENT_OVERDUE"],
        "BUREAU_CURRENT_OVERDUE_SUM > 0": suma.gt(0).astype(float).where(suma.notna()),
    }
    presencia, objetivo = train["HAS_BUREAU_HISTORY"], train["TARGET"]
    return pd.DataFrame(
        {
            nombre: {
                "n": int(serie.notna().sum()),
                "n_marcados": int(serie.sum()),
                "iv_sin_presencia": iv_condicionado(serie, presencia, objetivo),
            }
            for nombre, serie in lecturas.items()
        }
    ).T


def tripartita_mora(train: pd.DataFrame) -> pd.Series:
    """Con mora (`BUREAU_OVERDUE_UNION` = 1), reportada a cero (unión a 0 con historial de mora
    reportado) o sin reportar (unión a 0 y sin historial), NaN sin `HAS_BUREAU_HISTORY`."""
    union, historial = train["BUREAU_OVERDUE_UNION"], train["HAS_BUREAU_OVERDUE_HISTORY"]
    niveles = np.select(
        [union.eq(1), historial.eq(1)], ["con_mora", "reportada_cero"], default="sin_reportar"
    )
    return pd.Series(niveles, index=train.index).where(train["HAS_BUREAU_HISTORY"].eq(1))


def informe_tripartita_mora(train: pd.DataFrame) -> pd.DataFrame:
    """El pendiente 2: si separar el cero del nulo dentro de `BUREAU_OVERDUE_UNION` aporta IV.

    **Criterio, escrito antes de medir:** gana la tripartita si su IV incremental sobre la unión
    (`iv_condicionado()`, dentro de los clientes con historial) llega a `min_iv`. Si no, sigue la
    unión y no se construye columna.

    Una fila por nivel, con su n y su tasa. Dos filas aparte, sin n ni tasa: `union` y
    `tripartita`, con el IV marginal de cada codificación sin la presencia de la tabla, para que
    quede constancia de que la tripartita tampoco gana sola; e `incremental_sobre_union`, con el
    IV que la tripartita añade sobre la unión y si llega a `min_iv`.
    """
    con_historial = train[train["HAS_BUREAU_HISTORY"].eq(1)]
    tripartita = tripartita_mora(con_historial)
    union, objetivo = con_historial["BUREAU_OVERDUE_UNION"], con_historial["TARGET"]
    presencia = con_historial["HAS_BUREAU_HISTORY"]
    incremental = iv_condicionado(tripartita, union, objetivo)

    tasas = con_historial.groupby(tripartita)["TARGET"].agg(n="size", tasa="mean")
    resumen = pd.DataFrame(
        {
            "n": [np.nan, np.nan, np.nan],
            "tasa": [np.nan, np.nan, np.nan],
            "iv": [
                iv_condicionado(union, presencia, objetivo),
                iv_condicionado(tripartita, presencia, objetivo),
                incremental,
            ],
            "llega": [np.nan, np.nan, bool(incremental >= valor("min_iv"))],
        },
        index=["union", "tripartita", "incremental_sobre_union"],
    )
    return pd.concat([tasas.assign(iv=np.nan, llega=np.nan), resumen])


def informe_building_info(train: pd.DataFrame) -> pd.DataFrame:
    """`BUILDING_INFO_COUNT` frente a `HAS_BUILDING_INFO`, la pregunta que su propio docstring
    en `application.py` dejó abierta.

    **Criterio, escrito antes de medir:** entra el conteo si su IV incremental sobre la bandera
    (`iv_condicionado()`, dentro de quien tiene algún dato del edificio) llega a `min_iv`. Si no,
    sale y se queda solo la bandera.
    """
    conteo, bandera, objetivo = (
        train["BUILDING_INFO_COUNT"],
        train["HAS_BUILDING_INFO"],
        train["TARGET"],
    )
    incremental = iv_condicionado(conteo, bandera, objetivo)
    return pd.DataFrame(
        {
            "n_con_dato": int(bandera.sum()),
            "iv_marginal": calcular_iv(conteo, objetivo),
            "incremental_sobre_bandera": incremental,
            "llega": bool(incremental >= valor("min_iv")),
        },
        index=["BUILDING_INFO_COUNT"],
    )


def informe_solo_vivas(train: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """`PREV_FUTURE_DUE_MAX` (el fin previsto, cuente o no la operación como terminada) frente a
    la lectura de solo vivas (`lectura_solo_vivas()`), el pendiente del 4.5.

    **Criterio, escrito antes de medir:** entra la de solo vivas como columna nueva si su IV
    incremental sobre la actual (`iv_condicionado()`, dentro de quien tiene previas) llega a
    `min_iv`. Si no, la actual se queda tal cual.
    """
    vivas = lectura_solo_vivas(prev).reindex(train["SK_ID_CURR"]).to_numpy()
    actual, presencia, objetivo = (
        train["PREV_FUTURE_DUE_MAX"],
        train["HAS_PREV_APPLICATION"],
        train["TARGET"],
    )
    vivas = pd.Series(vivas, index=train.index)
    incremental = iv_condicionado(vivas, actual, objetivo)
    return pd.DataFrame(
        {
            "n": [int(actual.notna().sum()), int(vivas.notna().sum())],
            "iv_sin_presencia": [
                iv_condicionado(actual, presencia, objetivo),
                iv_condicionado(vivas, presencia, objetivo),
            ],
        },
        index=["PREV_FUTURE_DUE_MAX", "PREV_FUTURE_DUE_VIVAS"],
    ).assign(
        incremental_vivas_sobre_actual=[np.nan, incremental],
        llega=[np.nan, bool(incremental >= valor("min_iv"))],
    )


# La decisión de las poblaciones que la receta conserva o deja sin ella, para el 5.6: las que
# entran al incremental sobre EXT_SOURCE_3 son las que un default (o mejor) razonable de bureau y
# bureau_balance podría seguir usando, con su tabla como estrato de presencia.
_DECISIONES_QUE_SIGUEN_VIVAS = ("conservar", "iv", "degradada")


def _columnas_vivas(tabla: str, columnas_sin_receta: dict[str, str]) -> list[str]:
    receta = cargar_receta(tabla)
    de_receta = [
        f["nombre"] for f in receta["features"] if f.get("decision") in _DECISIONES_QUE_SIGUEN_VIVAS
    ]
    return sorted(set(de_receta) | set(columnas_sin_receta))


def columnas_solapadas_ext3(train: pd.DataFrame) -> pd.Series:
    """Pearson contra `EXT_SOURCE_3`, dentro de con historial de su tabla, de toda columna de
    `bureau` y `bureau_balance` que la receta conserva (o que va en `COLUMNAS_SIN_RECETA`).

    No mira el TARGET, así que se fija antes de ver ningún IV. Solo las que llegan a
    `solape_ext3_min` en valor absoluto pasan al incremental del 5.6; `previous_application` no
    entra, por decisión del EDA (su máximo es 0,2652 y los scores son de buró, no ven la relación
    con el propio prestamista). Una columna no numérica (`BB_TRAJECTORY`, categórica) se salta:
    el Pearson no se define sobre categorías.
    """
    poblaciones = {
        "bureau": (
            _columnas_vivas("bureau", _SIN_RECETA_BUREAU),
            train["HAS_BUREAU_HISTORY"].eq(1),
        ),
        "bureau_balance": (
            _columnas_vivas("bureau_balance", _SIN_RECETA_BB),
            train["HAS_BUREAU_BALANCE"].eq(1),
        ),
    }
    r = {}
    for columnas, mascara in poblaciones.values():
        dentro = train[mascara]
        for c in columnas:
            if c not in dentro.columns or not pd.api.types.is_numeric_dtype(dentro[c]):
                continue
            valor_r = dentro[c].astype(float).corr(dentro["EXT_SOURCE_3"])
            if pd.notna(valor_r):
                r[c] = valor_r
    serie = pd.Series(r).sort_values(key=abs, ascending=False)
    minimo = valor("solape_ext3_min")
    return serie[serie.abs() >= minimo]


def informe_incremental_ext3(train: pd.DataFrame) -> pd.DataFrame:
    """El IV incremental sobre `EXT_SOURCE_3` de cada columna de `columnas_solapadas_ext3()`.

    **Criterio, escrito antes de medir:** dentro de la población con historial de su tabla, si el
    IV incremental sobre `EXT_SOURCE_3` no llega a `min_iv`, la columna queda `degradada` (sigue
    en la matriz, marcada) y no `descartar`: el 5.8 da el corte final con la redundancia y la
    banda de revisión delante.
    """
    presencia = {
        "BUREAU": ("HAS_BUREAU_HISTORY", tuple(_columnas_vivas("bureau", _SIN_RECETA_BUREAU))),
        "BB": ("HAS_BUREAU_BALANCE", tuple(_columnas_vivas("bureau_balance", _SIN_RECETA_BB))),
    }
    solapadas = columnas_solapadas_ext3(train)
    filas = []
    for nombre, r in solapadas.items():
        columna_presencia = next(
            col_presencia for col_presencia, columnas in presencia.values() if nombre in columnas
        )
        dentro = train[train[columna_presencia].eq(1)]
        marginal = calcular_iv(dentro[nombre], dentro["TARGET"])
        incremental = iv_condicionado(dentro[nombre], dentro["EXT_SOURCE_3"], dentro["TARGET"])
        filas.append(
            {
                "feature": nombre,
                "pearson_ext3": r,
                "n": int(dentro[nombre].notna().sum()),
                "iv_dentro_de_su_tabla": marginal,
                "incremental_sobre_ext3": incremental,
                "llega": bool(incremental >= valor("min_iv")),
            }
        )
    return pd.DataFrame(filas).set_index("feature")
