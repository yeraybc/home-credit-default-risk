"""Agregación de bureau_balance, capa 1 del pipeline de features.

La **doble agregación** de la tabla: de crédito-mes a crédito (`agregar_por_credito()`) y de
crédito a cliente (`agregar_bureau_balance()`), que cruza el primero con el puente `SK_ID_BUREAU`
a `SK_ID_CURR` que da `bureau` (`puente_credito_cliente()`). La unión a la lista de clientes
(`unir_bureau_balance()`) añade además `BB_OVERDUE_UNION`, la mora de las dos tablas.

Es capa 1 por lo mismo que `agg_bureau.py`: no estima nada, no mira al TARGET y **el agregado de
un crédito solo depende de sus propias filas**. Por eso se calcula sobre los 817.395 créditos de
la tabla entera y no sobre los 523.515 que el EDA podía enlazar a train, y por eso es función
pura y no transformer. Las cifras del EDA y las de aquí no coinciden y las dos están bien, igual
que `bureau_t` frente a `bureau`; quien recorta la población es el puente, un paso más arriba.

Traslada la celda de los ejes derivados del notebook 03 (el `groupby("SK_ID_BUREAU").agg(...)` de
la celda 30 cuando se escribió), con la ventana de la 34, la trayectoria por mitades de la 37 y el
estado final de la 39. Cinco diferencias con el notebook, que allí medía y aquí construye:

1. **`BB_DPD_MONTHS` va a NaN sin ningún mes reportado**, no a 0. Son 130.368 créditos sobre la
   tabla entera, todo `C` y `X`, y leerlos como "sin mora" es la codificación que diluye la señal
   rara. El nivel cliente la suma con `min_count=1`, así que el cliente con algún crédito
   reportado sigue saliendo con su cuenta y solo el que los tiene todos ciegos queda en NaN, que es
   lo correcto. `BB_WORST` también sale NaN, pero eso ya lo daba el `max` del notebook.
2. **Un hueco en la ventana revienta.** La contigüidad es lo que hace que `BB_MONTHS_OBS` sea la
   longitud de la ventana; sin ella la trayectoria del 3.3 parte por un punto medio que no lo es.
   Caza también el par crédito-mes duplicado, que `limpiar_bureau_balance()` no busca a propósito,
   aunque tape un hueco.
3. **El estado final se lee con `idxmax`** y no ordenando la tabla: el par crédito-mes es único,
   así que el mes máximo es el último. Da lo mismo (comprobado sobre los 817.395) y es unas cien
   veces más rápido.
4. **Las dos proporciones se guardan de 0 a 1** y el notebook las imprimía en porcentaje. No mueve
   ninguna decisión, porque los efectos de la receta son rank-biserial, pero un `BB_PCT_X` de 0,25
   es el 25% del notebook y no un desfase.
5. **La trayectoria no clasifica la mitad ciega.** El notebook leía la mitad sin ningún mes
   reportado como un 0 y la clasificaba igual, y la ceguera se concentra en sin mora y mejora
   (el 61,14% y el 65,78% en los enlazables a train), que son los créditos que se cierran antes.
   Aquí el peor estado de esa mitad es NaN y la trayectoria también, y no mueve la clase de ningún
   crédito con las dos mitades reportadas. Sobre la tabla entera clasifica 288.547 de los 751.038
   con ventana de 6 meses o más: sin mora 230.623, mejora 21.465, empeora 23.217 y estable 13.242.

**`SK_ID_BUREAU` sale en `int64`**, se cargue la tabla como se cargue (`uint32` desde
`load_table`, `int64` desde un `read_csv` a pelo). Es el índice de esta salida y la clave que el
3.4 cruza con `bureau`, así que dejarlo al gusto del cargador sería el patrón 12 sobre la clave.
Se castea el índice de 817.395 valores, nunca la columna de 27,3 millones.

**Coste medido sobre la tabla entera:** unos 3 segundos de punta a punta con la tabla ya cargada, la
mitad en la guarda del duplicado, que ordena los 27,3 millones de pares, y la otra mitad en la
agregación; la limpieza y el estado final no pasan de 0,2. Entre ejecuciones los absolutos se mueven
hasta el doble y el reparto no. El pico de RSS lo sube este paso a 2,9 GB, desde los 2,4 de la carga
con `load_table`. La trayectoria añade 0,4 segundos y medio GB de pico, hasta 3,5, medido en proceso
limpio contra la misma función sin ella (1,6 frente a 2,0 segundos esa vez).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.agg_bureau import POBLACIONES as POBLACIONES_BUREAU
from src.features.cleaning import (
    COL_BB_DPD,
    COL_BB_IS_DPD,
    COL_BB_IS_X,
    MESES_BB,
    limpiar_bureau_balance,
)
from src.features.params import CORTES_POR_FEATURE, valor

CLAVE = "SK_ID_BUREAU"

# El tipo de la clave, aquí y no en los dos sitios que la fijan: el 3.4 tiene que castear el
# `SK_ID_BUREAU` de `bureau` con este mismo valor antes de cruzar, y dos literales serían el mismo
# contrato en dos sitios.
DTYPE_CLAVE = "int64"

# Las tres columnas de la tabla. La agregación es la frontera que produce los ejes del crédito,
# así que exige su esquema: una ausente en silencio saldría como "sin dato" en todo lo que la lee.
COLUMNAS_ORIGEN: tuple[str, ...] = (CLAVE, "MONTHS_BALANCE", "STATUS")

# ordenada por prioridad del peor recorrido, que es lo que el nivel cliente toma con un max. Estable
# por encima de empeora es de dominio desde el 3.9, la mora en las dos mitades pesa más que el
# deterioro puntual: sobre train queda por encima con cualquier ventana mínima, sin significación
TRAYECTORIAS = pd.CategoricalDtype(["sin mora", "mejora", "empeora", "estable"], ordered=True)

# Los cortes que lleva dentro alguna feature de la tabla, del mismo registro que usa `params.py`, y
# cuáles consume cada nivel. Separarlos es lo que permite que el nivel crédito no resuelva la cola
# del conteo, que es `medido` y revienta en `valor()` hasta que `ajustar_cola_bb()` la refija.
CORTES = tuple(
    sorted({c for cortes in CORTES_POR_FEATURE["bureau_balance"].values() for c in cortes})
)
CORTES_CREDITO: tuple[str, ...] = ("bb_min_meses_trayectoria",)
CORTES_CLIENTE: tuple[str, ...] = (
    "bb_many_credits_corte",
    "bb_min_meses_reportados",
    "bb_mora_reciente_meses",
)

# Lo que el nivel cliente añade sin estar en la receta, con su motivo.
COLUMNAS_SIN_RECETA: dict[str, str] = {
    "BB_MONTHS_REPORTED": (
        "denominador de BB_PCT_MONTHS_DPD y de su mínimo, no explicativa, como BUREAU_LOAN_COUNT"
    ),
    "BB_TRAJECTORY": (
        "el peor recorrido del cliente, que la receta solo tiene por sus tres banderas"
    ),
}

# las tres banderas de la trayectoria de cliente y la clase que marca cada una
BANDERAS_TRAYECTORIA = {
    "BB_PERSISTENT_DPD_FLAG": "estable",
    "BB_WORSENING_DPD_FLAG": "empeora",
    "BB_RECOVERED_DPD_FLAG": "mejora",
}

# Las poblaciones sobre las que la receta midió cada feature, como máscara del frame que devuelve
# `unir_bureau_balance()` sobre la salida de `unir_bureau()`, como en `agg_bureau.py`. None es la
# población del propio evaluador.
POBLACIONES = {
    "global": None,
    "no nulos (auto-cond.)": None,
    "con histórico": lambda d: d["HAS_BUREAU_BALANCE"].eq(1),
    "con historial de bureau": POBLACIONES_BUREAU["con historial"],
    "trayectoria evaluable": lambda d: d["BB_TRAJECTORY"].notna(),
    # deja fuera al cliente con todos sus créditos ciegos, que tiene panel y ningún estado
    "con estado reportado": lambda d: d["BB_STATUS_WORST"].notna(),
    "clientes con mora": lambda d: d["BB_ANY_DPD_FLAG"].eq(1),
}


def _cortes(cortes: dict[str, float] | None, usa: tuple[str, ...]) -> dict[str, float]:
    """Los cortes que consume un nivel: lo que traiga `cortes` y el resto con `valor()`.

    La guarda mira contra los de la tabla entera y no contra los del nivel, para que
    `agregar_bureau_balance()` pueda reenviar el dict completo al nivel crédito; sin ella un
    nombre mal escrito dejaría el contraste corriendo con el valor de `params.py` sin avisar.
    """
    cortes = cortes or {}
    desconocidos = set(cortes) - set(CORTES)
    if desconocidos:
        raise KeyError(f"cortes que ninguna feature de bureau_balance usa: {sorted(desconocidos)}")
    return {n: cortes[n] if n in cortes else valor(n) for n in usa}


def agregar_por_credito(bb: pd.DataFrame, cortes: dict[str, float] | None = None) -> pd.DataFrame:
    """Una fila por crédito con histórico, indexada por `SK_ID_BUREAU`.

    Llama a `limpiar_bureau_balance()` antes de agregar: es idempotente, así que un frame ya
    limpio no cambia, y uno crudo no puede saltarse el dominio a nivel fila ni agregar una letra
    sin decodificar. Revienta con una clave nula, que el `groupby` tiraría en silencio.

    Las once columnas, con el nombre que tenían en el notebook al lado:

    - `BB_MONTHS_OBS` (`meses_obs`), la ventana, y `BB_MONTHS_REPORTED` (`n_rep`), los meses con
      estado numérico, que es el denominador bueno de la intensidad.
    - `BB_WORST` (`worst`) y `BB_DPD_MONTHS` (`n_dpd`), severidad e intensidad bruta, las dos a
      NaN sin ningún mes reportado.
    - `BB_PCT_X` (`pct_X`) y `BB_PCT_DPD` (`pct_dpd`), de 0 a 1.
    - `BB_LAST_DPD_MONTH` (`rec_dpd`), el mes del último impago, NaN sin ninguno.
    - `BB_WINDOW_INI`, `BB_WINDOW_END` y `BB_CENSORED` (`ini`, `fin`, `censurado`), la ventana y
      si deja de reportarse antes del mes de la solicitud.
    - `BB_LAST_STATUS` (`ult`), el estado del último mes observado, categórica con los ocho
      niveles siempre.

    El esquema no depende del lote: los tres float se fuerzan a float aunque el lote no traiga
    ningún NaN, y la categórica conserva sus ocho niveles con un frame vacío.

    `cortes` solo alcanza a `bb_min_meses_trayectoria`, que es el único que este nivel consume, y
    es lo que deja al 3.9 ejecutar su contraste sin tocar la agregación. Los cortes del nivel
    cliente no se resuelven aquí, así que este paso corre sin haber refijado la cola.
    """
    faltan = [c for c in COLUMNAS_ORIGEN if c not in bb.columns]
    if faltan:
        raise ValueError(f"a bureau_balance le faltan columnas para agregar: {faltan}")
    b = limpiar_bureau_balance(bb)
    sin_clave = b[CLAVE].isna()
    if sin_clave.any():
        raise ValueError(
            f"{int(sin_clave.sum())} filas de bureau_balance sin {CLAVE}: el groupby las tiraría y "
            "sus meses no llegarían a ningún crédito sin que nada avise"
        )
    # el índice de quien llama puede venir repetido (un concat, la API); `idxmax` devuelve
    # etiquetas, así que sin esto el estado final leería la fila equivocada
    b.index = pd.RangeIndex(len(b))
    g = b.groupby(CLAVE, sort=True)
    cred = g.agg(
        BB_MONTHS_OBS=("MONTHS_BALANCE", "size"),
        BB_MONTHS_REPORTED=(COL_BB_DPD, "count"),
        BB_WORST=(COL_BB_DPD, "max"),
        BB_PCT_X=(COL_BB_IS_X, "mean"),
        BB_WINDOW_INI=("MONTHS_BALANCE", "min"),
        BB_WINDOW_END=("MONTHS_BALANCE", "max"),
        _dpd_months=(COL_BB_IS_DPD, "sum"),
    )
    hueco = cred["BB_WINDOW_END"] - cred["BB_WINDOW_INI"] + 1 != cred["BB_MONTHS_OBS"]
    # el duplicado aparte, porque puede tapar un hueco y dejar la ventana con el largo justo. El par
    # va ordenado como un entero, único porque el mes ya está dentro de la ventana: la mitad de
    # tiempo que un `nunique` por crédito y sin los 1,7 GB de pico que ese añadía
    ancho = MESES_BB[1] - MESES_BB[0] + 1
    par = b[CLAVE].to_numpy("int64") * ancho
    par += b["MONTHS_BALANCE"].to_numpy("int64") - MESES_BB[0]
    par.sort()
    duplicados = par[1:][par[1:] == par[:-1]] // ancho
    del par  # 218 MB vivos durante la trayectoria: sin soltarlos el pico pasa de 3,5 a 3,8 GB
    rota = hueco | cred.index.isin(duplicados)
    if rota.any():
        raise ValueError(
            f"{int(rota.sum())} créditos con la ventana mensual rota: "
            f"{cred.index[rota][:10].tolist()}. La contigüidad es lo que hace que los meses "
            "observados sean la ventana, y la trayectoria parte por su punto medio, así que un "
            "hueco (o un par crédito-mes duplicado) la mediría mal sin que nada avise"
        )
    reportado = cred["BB_MONTHS_REPORTED"] > 0
    # en float siempre, y no solo cuando el lote trae algún ciego: `where` sobre un int8 lo deja
    # en int8 si no introduce ningún NaN, y el esquema de un crédito no puede depender de con
    # quién le agreguen
    cred["BB_DPD_MONTHS"] = cred.pop("_dpd_months").astype(float).where(reportado)
    # redundante hoy, porque el `max` de un crédito todo ciego ya es NaN, y a propósito: sostiene
    # la regla si la severidad deja de agregarse con `max`
    cred["BB_WORST"] = cred["BB_WORST"].where(reportado)
    # sin `replace` del cero: el numerador ya es NaN donde el denominador es 0, que son los mismos
    # créditos, y NaN entre 0 es NaN
    cred["BB_PCT_DPD"] = cred["BB_DPD_MONTHS"] / cred["BB_MONTHS_REPORTED"]
    ultima = g["MONTHS_BALANCE"].idxmax()
    cred["BB_LAST_STATUS"] = b["STATUS"].iloc[ultima.to_numpy()].set_axis(ultima.index)
    en_mora = b.loc[b[COL_BB_IS_DPD].eq(1)].groupby(CLAVE)["MONTHS_BALANCE"].max()
    cred["BB_LAST_DPD_MONTH"] = en_mora.reindex(cred.index).astype(float)
    cred["BB_CENSORED"] = (cred["BB_WINDOW_END"] < 0).astype("int8")
    cred = cred.join(trayectoria_por_credito(b, g, cortes))
    cred.index = cred.index.astype(DTYPE_CLAVE)
    return cred


def trayectoria_por_credito(
    b: pd.DataFrame, g: pd.api.typing.DataFrameGroupBy, cortes: dict[str, float] | None = None
) -> pd.DataFrame:
    """La trayectoria por mitades de cada crédito, sobre el frame limpio y la ventana ya guardada.

    Solo la llama `agregar_por_credito()`, después de la guarda: parte la ventana por su punto
    medio, y sin contigüidad ese punto no lo es. Tres columnas:

    - `BB_WORST_OLD_HALF` y `BB_WORST_RECENT_HALF` (`w_ant`, `w_rec`), el peor estado de cada
      mitad, NaN en la mitad sin ningún mes reportado.
    - `BB_CREDIT_TRAJECTORY` (`trayectoria`), sin mora, mejora, empeora o estable, NaN si la
      ventana no llega a `bb_min_meses_trayectoria` o si alguna mitad es ciega.
    """
    mes = b["MONTHS_BALANCE"].to_numpy()
    ini = g["MONTHS_BALANCE"].transform("min").to_numpy()
    fin = g["MONTHS_BALANCE"].transform("max").to_numpy()
    # `mes > (ini + fin) / 2` sin salir del int8: las dos distancias caben en 0 a 96. Con ventana
    # impar el mes central queda en la antigua, como en el notebook
    reciente = mes - ini > fin - mes
    dpd = b[COL_BB_DPD]
    mitades = pd.DataFrame(
        {"BB_WORST_OLD_HALF": dpd.where(~reciente), "BB_WORST_RECENT_HALF": dpd.where(reciente)}
    )
    mitades = mitades.groupby(b[CLAVE], sort=True).max()
    ant, rec = mitades["BB_WORST_OLD_HALF"], mitades["BB_WORST_RECENT_HALF"]
    minimo = _cortes(cortes, CORTES_CREDITO)["bb_min_meses_trayectoria"]
    evaluable = (g.size() >= minimo) & ant.notna() & rec.notna()
    # como el notebook, mejora es por severidad: un 3 que pasa a 1 mejora aunque siga en mora
    clase = np.select(
        [(ant == 0) & (rec == 0), rec > ant, rec < ant],
        ["sin mora", "empeora", "mejora"],
        "estable",
    )
    mitades["BB_CREDIT_TRAJECTORY"] = (
        pd.Series(clase, index=mitades.index).where(evaluable).astype(TRAYECTORIAS)
    )
    return mitades


def puente_credito_cliente(bureau: pd.DataFrame) -> pd.Series:
    """El `SK_ID_CURR` de cada crédito de `bureau`, indexado por `SK_ID_BUREAU` en `DTYPE_CLAVE`.

    Sobre `bureau` entero y sin limpiar: la limpieza no borra filas ni toca las claves. No sabe
    nada de los créditos huérfanos del panel, que tira el join del paso cliente.

    Revienta con una clave nula, con un `SK_ID_BUREAU` repetido (aunque sea con el mismo cliente,
    es PK hoy, y si dejara de serlo el join abriría créditos en silencio) y con una clave con
    decimales: un `1.5` truncado por el cast cruzaría con el crédito `1`, el mismo caso que el mes
    de `limpiar_bureau_balance()`. `SK_ID_CURR` se queda con el tipo del cargador, como en
    `agregar_bureau()`.
    """
    faltan = [c for c in (CLAVE, "SK_ID_CURR") if c not in bureau.columns]
    if faltan:
        raise ValueError(f"a bureau le faltan columnas para el puente: {faltan}")
    nulas = bureau[[CLAVE, "SK_ID_CURR"]].isna().sum()
    if nulas.any():
        raise ValueError(
            f"claves nulas en bureau: {nulas[nulas > 0].to_dict()}. Un crédito sin cliente no "
            "llega a ninguno, y el groupby del paso cliente lo tiraría sin avisar"
        )
    if not pd.api.types.is_integer_dtype(bureau[CLAVE]) and bureau[CLAVE].mod(1).ne(0).any():
        raise ValueError(
            f"{CLAVE} con decimales: el cast a {DTYPE_CLAVE} los truncaría y cruzaría créditos "
            "distintos sin que nada avise"
        )
    repetidos = bureau[CLAVE].duplicated()
    if repetidos.any():
        raise ValueError(
            f"{int(repetidos.sum())} {CLAVE} repetidos en bureau: "
            f"{bureau.loc[repetidos, CLAVE][:10].tolist()}. El join con el paso crédito "
            "duplicaría esos créditos sin que nada avise"
        )
    puente = bureau.set_index(CLAVE)["SK_ID_CURR"]
    puente.index = puente.index.astype(DTYPE_CLAVE)
    return puente


def agregar_bureau_balance(
    bb: pd.DataFrame, puente: pd.Series, cortes: dict[str, float] | None = None
) -> pd.DataFrame:
    """Una fila por cliente con histórico mensual, indexada por `SK_ID_CURR`.

    Segundo paso de la doble agregación. Agrega la tabla entera a nivel crédito y la cruza con el
    puente, así que los créditos huérfanos (43.041, sin padre en `bureau`) se caen en ese join y
    los clientes que quedan son los que `bureau` conoce. Es capa 1 por lo mismo que el paso
    crédito: el agregado de un cliente solo depende de sus propias filas.

    Las once columnas del `groupby("SK_ID_CURR")` del notebook 03 (la celda 51 cuando se escribió)
    más `BB_PCT_MONTHS_DPD`. **`BB_MONTHS_MAX` no se construye**, que es el único descarte
    `firmeza: firme` de la receta (Pearson -0,7451 con `BUREAU_DAYS_CREDIT_MIN`, redundancia
    estructural que vale en cualquier submuestra); los provisionales sí, incluidas las dos
    versiones absolutas de la recencia, que la capa 2b tiene que poder remedir.

    Cuatro diferencias con el notebook, que allí medía y aquí construye:

    1. **La severidad y los dos conteos de mora van a NaN** en el cliente cuyos créditos son todos
       ciegos (2.375 de los 92.231 clientes de train, el 2,58%, y 3.769 sobre la tabla entera).
       Sale del `max`, que ignora los NaN, y del `min_count=1`, no de un `.loc` posterior como en
       el notebook: es la misma cuenta con una fuente de verdad menos. El cliente con algún
       crédito reportado conserva la suya.
    2. **Las tres banderas se quedan en 0 en esos mismos clientes**, y es la única excepción
       declarada a "sin dato no es sin mora". Es lo que midió el EDA, que dejó el matiz escrito: la
       bandera genérica rinde +2,74pp con ellos dentro y +2,76pp sobre los que reportan, o sea que
       el efecto de la excepción es nulo. Qué se hace con esos clientes es de la capa 2.
    3. **`BB_PCT_MONTHS_DPD` se guarda de 0 a 1** y el notebook la imprimía en porcentaje, como las
       dos proporciones del paso crédito.
    4. **Las tres banderas de trayectoria van a NaN sin trayectoria evaluable**, y el notebook las
       dejaba en False. Se midieron sobre los 66.663 clientes evaluables de train, y fuera de ellos
       un 0 mezclaría "no evaluable" con "no persistente".

    Las derivadas de las celdas 58, 59, 61 y 62: `BB_MONTHS_SINCE_LAST_DPD_REL` (la recencia sobre
    el fin de ventana de cada crédito) y `BB_RECENT_DPD_FLAG_REL` cortada sobre ella,
    `BB_EXITS_IN_DPD_FLAG` y `BB_ALL_CLOSED_FLAG` (el último estado de cada crédito),
    `BB_TRAJECTORY` con sus tres banderas, y `BB_MANY_CREDITS_FLAG`. Los descartes provisionales
    se construyen, que la capa 2b los remide.

    `cortes` sustituye a los de `params.py` y se reenvía entero al paso crédito, que es el punto de
    entrada de `bb_min_meses_trayectoria` para el contraste del 3.9. De los tres que consume este
    nivel, `bb_many_credits_corte` es `medido` y revienta en `valor()` hasta que
    `ajustar_cola_bb()` lo refija sobre train; el denominador mínimo y la ventana de la mora
    reciente son `dominio` desde el 3.8.
    """
    c = _cortes(cortes, CORTES_CLIENTE)
    cred = agregar_por_credito(bb, cortes)
    # `1:1` y no contar filas: las dos claves son únicas, así que un puente con un `SK_ID_BUREAU`
    # repetido abriría créditos en silencio, que es el fallo del left join contra clave repetida
    cf = cred.join(puente, how="inner", validate="1:1")
    reportado = cf["BB_DPD_MONTHS"].notna()
    # False en el ciego, que es lo correcto para la bandera: no hay evidencia de mora. Para los
    # conteos hace falta la versión con NaN, o el cliente todo ciego saldría con mora cero
    mora = cf["BB_DPD_MONTHS"] > 0
    ultimo = cf["BB_LAST_STATUS"]
    f = cf.assign(
        _has_dpd=mora,
        _writeoff=cf["BB_WORST"].eq(5),
        # NaN >= corte ya da False: el crédito sin ningún impago no tiene mora reciente
        _recent_dpd=cf["BB_LAST_DPD_MONTH"] >= -c["bb_mora_reciente_meses"],
        _dpd_credits=mora.astype(float).where(reportado),
        # meses del último impago al fin de la ventana del crédito, no a la solicitud
        _rel=cf["BB_LAST_DPD_MONTH"] - cf["BB_WINDOW_END"],
        _exits=ultimo.isin(["1", "2", "3", "4", "5"]),
        _closed=ultimo.eq("C"),
    )
    g = f.groupby("SK_ID_CURR")
    agregado = g.agg(
        BB_N_CREDITS_WBAL=("BB_MONTHS_OBS", "size"),
        BB_MONTHS_TOTAL=("BB_MONTHS_OBS", "sum"),
        BB_MONTHS_REPORTED=("BB_MONTHS_REPORTED", "sum"),
        BB_STATUS_WORST=("BB_WORST", "max"),
        BB_ANY_DPD_FLAG=("_has_dpd", "max"),
        BB_RECENT_DPD_FLAG=("_recent_dpd", "max"),
        BB_WRITEOFF_FLAG=("_writeoff", "max"),
        BB_MONTHS_SINCE_LAST_DPD=("BB_LAST_DPD_MONTH", "max"),
        BB_CENSORED_RATIO=("BB_CENSORED", "mean"),
        BB_MONTHS_SINCE_LAST_DPD_REL=("_rel", "max"),
        BB_EXITS_IN_DPD_FLAG=("_exits", "max"),
        BB_ALL_CLOSED_FLAG=("_closed", "min"),
        # la categórica va ordenada por prioridad del peor recorrido, así que el max es el peor
        BB_TRAJECTORY=("BB_CREDIT_TRAJECTORY", "max"),
    )
    # las sumas aparte, porque la agregación nombrada no acepta min_count sin un lambda por cliente
    sumas = g[["_dpd_credits", "BB_DPD_MONTHS"]].sum(min_count=1)
    agregado["BB_CREDITS_WITH_DPD_COUNT"] = sumas["_dpd_credits"]
    agregado["BB_DPD_MONTHS_COUNT"] = sumas["BB_DPD_MONTHS"]
    # el denominador informativo: el extremo del ratio es ruido de reporte, y con un solo mes
    # reportado vale 1 sin querer decir nada. Sin `replace` del cero, que ya es NaN por el mínimo
    suficiente = agregado["BB_MONTHS_REPORTED"] >= c["bb_min_meses_reportados"]
    agregado["BB_PCT_MONTHS_DPD"] = agregado["BB_DPD_MONTHS_COUNT"] / agregado[
        "BB_MONTHS_REPORTED"
    ].where(suficiente)
    agregado["BB_RECENT_DPD_FLAG_REL"] = (
        agregado["BB_MONTHS_SINCE_LAST_DPD_REL"] >= -c["bb_mora_reciente_meses"]
    )
    agregado["BB_MANY_CREDITS_FLAG"] = agregado["BB_N_CREDITS_WBAL"] >= c["bb_many_credits_corte"]
    banderas = agregado.select_dtypes("bool").columns
    agregado[banderas] = agregado[banderas].astype("int8")
    trayectoria = agregado["BB_TRAJECTORY"]
    # NaN y no 0 sin trayectoria evaluable: no evaluable no es no persistente. En float siempre,
    # para que el esquema no dependa de si el lote trae algún cliente sin trayectoria
    for nombre, clase in BANDERAS_TRAYECTORIA.items():
        agregado[nombre] = trayectoria.eq(clase).astype(float).where(trayectoria.notna())
    return agregado


def unir_bureau_balance(clientes: pd.DataFrame, agregado: pd.DataFrame) -> pd.DataFrame:
    """Left join del agregado a una lista de clientes, con su presencia y la unión de mora.

    No rellena, igual que `unir_bureau()`: el cliente sin histórico mensual queda con NaN en todo.
    Su grupo no discrimina (8,14% frente a 8,04%, p=0,35), así que `HAS_BUREAU_BALANCE` es control
    de composición y no predictor, pero un 0 en los conteos lo mezclaría con quien tiene histórico
    y ninguna mora. Qué se hace con ese NaN es de la capa 2.

    `BB_OVERDUE_UNION` es la mora del cliente en cualquiera de las dos fuentes, y la única feature
    que cruza las dos tablas: por eso va aquí y no en el agregado, que no conoce al cliente con
    historial de `bureau` y sin panel, y ese toma el valor de `bureau`. Exige que `clientes` ya
    venga de `unir_bureau()`. Es NaN sin historial de `bureau`, donde el notebook ponía 0 y no
    medía; el cliente ciego aporta su `BB_ANY_DPD_FLAG` a 0, la excepción del paso cliente.
    """
    if "BUREAU_OVERDUE_UNION" not in clientes.columns:
        raise ValueError(
            "a clientes le falta BUREAU_OVERDUE_UNION: BB_OVERDUE_UNION la une con la mensual, así "
            "que unir_bureau() va antes, o la mora que solo ve bureau saldría sin marcar"
        )
    unido = clientes.join(agregado, on="SK_ID_CURR", validate="m:1")
    unido["HAS_BUREAU_BALANCE"] = unido["BB_N_CREDITS_WBAL"].notna().astype("int8")
    # todo cliente con panel tiene bureau si la lista sale del mismo bureau que dio el puente, así
    # que el NaN de la unión es el de bureau. En float siempre, para no depender del lote
    mora_bureau = unido["BUREAU_OVERDUE_UNION"]
    mora = mora_bureau.gt(0) | unido["BB_ANY_DPD_FLAG"].gt(0)
    unido["BB_OVERDUE_UNION"] = mora.astype(float).where(mora_bureau.notna())
    return unido
