"""Agregación de bureau_balance, capa 1 del pipeline de features.

Primer paso de la **doble agregación** de la tabla: de crédito-mes a crédito. El segundo, de
crédito a cliente, necesita el puente `SK_ID_BUREAU` a `SK_ID_CURR` que da `bureau` y llega en el
3.4 y el 3.5.

Es capa 1 por lo mismo que `agg_bureau.py`: no estima nada, no mira al TARGET y **el agregado de
un crédito solo depende de sus propias filas**. Por eso se calcula sobre los 817.395 créditos de
la tabla entera y no sobre los 523.515 que el EDA podía enlazar a train, y por eso es función
pura y no transformer. Las cifras del EDA y las de aquí no coinciden y las dos están bien, igual
que `bureau_t` frente a `bureau`; quien recorta la población es el puente, un paso más arriba.

Traslada la celda de los ejes derivados del notebook 03 (el `groupby("SK_ID_BUREAU").agg(...)` de
la celda 30 cuando se escribió), con la ventana de la 34 y el estado final de la 39. Cuatro
diferencias con el notebook, que allí medía y aquí construye:

1. **`BB_DPD_MONTHS` va a NaN sin ningún mes reportado**, no a 0. Son 130.368 créditos sobre la
   tabla entera, todo `C` y `X`, y leerlos como "sin mora" es la codificación que diluye la señal
   rara. El nivel cliente la suma con `min_count=1`, así que el cliente con algún crédito
   reportado sigue saliendo con su cuenta y solo el que los tiene todos ciegos queda en NaN, que es
   lo correcto. `BB_WORST` también sale NaN, pero eso ya lo daba el `max` del notebook.
2. **Un hueco en la ventana revienta.** La contigüidad es lo que hace que `BB_MONTHS_OBS` sea la
   longitud de la ventana; sin ella la trayectoria del 3.3 parte por un punto medio que no lo es.
   Cuenta los meses distintos, así que caza también el par crédito-mes duplicado, que
   `limpiar_bureau_balance()` no busca a propósito, aunque tape un hueco.
3. **El estado final se lee con `idxmax`** y no ordenando la tabla: el par crédito-mes es único,
   así que el mes máximo es el último. Da lo mismo (comprobado sobre los 817.395) y es unas cien
   veces más rápido.
4. **Las dos proporciones se guardan de 0 a 1** y el notebook las imprimía en porcentaje. No mueve
   ninguna decisión, porque los efectos de la receta son rank-biserial, pero un `BB_PCT_X` de 0,25
   es el 25% del notebook y no un desfase.

**`SK_ID_BUREAU` sale en `int64`**, se cargue la tabla como se cargue (`uint32` desde
`load_table`, `int64` desde un `read_csv` a pelo). Es el índice de esta salida y la clave que el
3.4 cruza con `bureau`, así que dejarlo al gusto del cargador sería el patrón 12 sobre la clave.
Se castea el índice de 817.395 valores, nunca la columna de 27,3 millones.

**Coste medido sobre la tabla entera:** unos 3 segundos de punta a punta con la tabla ya cargada, la
mitad en la guarda del duplicado, que ordena los 27,3 millones de pares, y la otra mitad en la
agregación; la limpieza y el estado final no pasan de 0,2. Entre ejecuciones los absolutos se mueven
hasta el doble y el reparto no. El pico de RSS lo sube este paso a 2,9 GB, desde los 2,4 de la carga
con `load_table`.
"""

from __future__ import annotations

import pandas as pd

from src.features.cleaning import (
    COL_BB_DPD,
    COL_BB_IS_DPD,
    COL_BB_IS_X,
    MESES_BB,
    limpiar_bureau_balance,
)

CLAVE = "SK_ID_BUREAU"

# El tipo de la clave, aquí y no en los dos sitios que la fijan: el 3.4 tiene que castear el
# `SK_ID_BUREAU` de `bureau` con este mismo valor antes de cruzar, y dos literales serían el mismo
# contrato en dos sitios.
DTYPE_CLAVE = "int64"

# Las tres columnas de la tabla. La agregación es la frontera que produce los ejes del crédito,
# así que exige su esquema: una ausente en silencio saldría como "sin dato" en todo lo que la lee.
COLUMNAS_ORIGEN: tuple[str, ...] = (CLAVE, "MONTHS_BALANCE", "STATUS")


def agregar_por_credito(bb: pd.DataFrame) -> pd.DataFrame:
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
    cred.index = cred.index.astype(DTYPE_CLAVE)
    return cred
