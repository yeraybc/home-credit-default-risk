"""Las tres propiedades de fuga que el plan de la fase pide, a nivel de pipeline entero.

Hasta aquí estaban probadas **por transformer y no por el pipeline**, así que nada impedía que
otro llamante hiciera `Winsorizador().fit(base)` con la tabla entera. Estas van sobre el objeto
que de verdad se ajusta y se serializa.

**Todo es sintético por construcción y eso no es una comodidad, es la condición.** Sobre la tabla
real los diez cortes de la capa 2a reestimados coinciden exactos con la referencia del EDA
(desviación cero, medida en el punto 1.3), así que una comparación de cifras contra el dato real
pasa igual esté el ajuste dentro o fuera del split: la cifra nunca va a delatar la fuga. Aquí las
dos particiones se separan a propósito, y por eso sí distinguen los dos casos.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import ajustar_capa2a, ajustar_pipeline, matriz_de_features
from src.features.params import PARAMS, parametro, reajustables, valor
from src.features.pipeline import (
    BINARIAS,
    CATEGORICAS_OHE,
    COL_DIA,
    COL_EDUCACION,
    COL_FRANJA,
    COL_HORA,
    COL_OCUPACION,
    COL_ORGANIZACION,
    JERARQUIA_EDUCACION,
    NUMERICAS,
    construir_pipeline,
)
from src.features.split import solo_train, solo_valid
from src.features.transformers import CORTES_WINSOR

N_TRAIN, N_VALID = 120, 60

# Los pasos que guardan estado ajustado, con el atributo por el que se les mira. Va declarado y
# no descubierto por reflexión: un paso nuevo que no se añada aquí tiene que romper el test de
# cobertura de abajo, no colarse sin que nadie compare su parámetro.
ESTADO_AJUSTADO = {
    "winsor": "limites_",
    "num": "statistics_",
    "agrupa": "raras_",
    "codifica": "categories_",
    "woe": "tablas_",
    "tgt": "encodings_",
    "varianza": "variances_",
}


def _bloque(n, semilla, *, ingreso, hora, dia, organizacion, ocupacion, rara):
    """Un trozo de tabla con los valores que separan a las dos particiones."""
    rng = np.random.default_rng(semilla)
    frame = pd.DataFrame({c: rng.uniform(1.0, 10.0, n) for c in NUMERICAS})
    frame["AMT_INCOME_TOTAL"] = ingreso
    frame[COL_HORA] = hora
    for c in BINARIAS:
        frame[c] = rng.integers(0, 2, n)
    for c in CATEGORICAS_OHE:
        if c not in (COL_FRANJA, COL_DIA):
            frame[c] = "comun"
    frame[COL_DIA] = dia
    frame.loc[frame.index[:2], "NAME_INCOME_TYPE"] = rara
    frame[COL_EDUCACION] = [JERARQUIA_EDUCACION[i % 5] for i in range(n)]
    frame[COL_ORGANIZACION] = organizacion
    # cuatro oficios intercalados y no uno solo: con un nivel unico el target encoding sale
    # constante, la varianza se lleva la columna y no hay codificacion cruzada que observar
    frame[COL_OCUPACION] = [f"{ocupacion}{i % 4}" for i in range(n)]
    return frame


@pytest.fixture
def base_y_split():
    """Dos particiones que dan parámetros distintos en **todos** los pasos que ajustan algo.

    Sin esa separación el test no distingue ajustar dentro del split de ajustar fuera, que es
    exactamente lo que pasa con el dato real.
    """
    train = _bloque(
        N_TRAIN,
        1,
        ingreso=1_000.0,
        hora=8,
        dia="MONDAY",
        organizacion="banca",
        ocupacion="oficio",
        rara="rarisima_train",
    )
    valid = _bloque(
        N_VALID,
        2,
        ingreso=1_000_000.0,
        hora=20,
        dia="SATURDAY",
        organizacion="obra",
        ocupacion="otro oficio",
        rara="rarisima_valid",
    )
    base = pd.concat([train, valid], ignore_index=True)
    base.insert(0, "SK_ID_CURR", range(1, len(base) + 1))
    base.insert(1, "TARGET", ([0] * 100 + [1] * 20) + ([1] * 50 + [0] * 10))
    split = pd.DataFrame(
        {
            "SK_ID_CURR": base["SK_ID_CURR"],
            "split": (["train"] * N_TRAIN) + (["valid"] * N_VALID),
        }
    )
    return base, split


def _pasos(pipeline):
    """Todos los pasos del pipeline por nombre, con los anidados del `ColumnTransformer` subidos.

    Lo comparten la comparación de parámetros y el guardián de cobertura, y esa es la gracia: el
    guardián existe para ver un paso nuevo que ajuste algo, y con su propia copia del aplanado
    dejaría de verlo justo el día que el pipeline anide uno más.
    """
    pasos = dict(pipeline.named_steps)
    ct = pasos["columnas"]
    pasos.update(ct.named_transformers_)
    pasos.update(dict(ct.named_transformers_["ohe"].named_steps))
    return pasos


def _ajustado(pipeline):
    """Los parámetros ajustados de cada paso, en una forma que se puede comparar."""
    pasos = _pasos(pipeline)
    return {
        nombre: repr(getattr(pasos[nombre], atributo))
        for nombre, atributo in ESTADO_AJUSTADO.items()
    }


def _fit(base, split, parte=None):
    frame = base if parte is None else parte(base, split)
    return construir_pipeline().fit(matriz_de_features(frame), frame["TARGET"])


def _transformar(pipeline, frame):
    """Transforma callando el aviso de categoría desconocida, que aquí es incidental.

    Viene de la librería y es la señal correcta: la validación trae a propósito categorías que
    el ajuste no vio, y eso ya lo afirma el guardián del fixture. Se acota aquí en vez de dejarlo
    suelto porque lo que estos tests miden es otra cosa.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unknown categories.*", category=UserWarning)
        return pipeline.transform(frame)


# --- 1. ningún paso se puede ajustar sobre el conjunto completo sin que se note ---------------


def test_todos_los_pasos_que_ajustan_dan_algo_distinto_dentro_y_fuera_del_split(base_y_split):
    """La fuga que caza: `fit` sobre la base entera en vez de sobre `solo_train()`.

    Se comprueba paso a paso y no sobre la salida entera: un solo parámetro que se ajustase
    fuera del split ya es fuga, y mirando solo la matriz final podría quedar tapado por el resto.
    """
    base, split = base_y_split
    en_train = _ajustado(_fit(base, split, solo_train))
    en_todo = _ajustado(_fit(base, split))

    iguales = [paso for paso in ESTADO_AJUSTADO if en_train[paso] == en_todo[paso]]
    assert not iguales, f"estos pasos no distinguen la partición: {iguales}"


def test_el_fixture_cubre_todos_los_pasos_que_guardan_estado(base_y_split):
    """Guardián: un paso que ajuste algo y no esté en la tabla se colaría sin comparar nada."""
    base, split = base_y_split
    pipeline = _fit(base, split, solo_train)
    pasos = _pasos(pipeline)
    con_estado = {
        nombre
        for nombre, paso in pasos.items()
        if any(a.endswith("_") and not a.startswith("_") for a in vars(paso))
        and nombre
        not in ("columnas", "ohe", "derivadas", "dominio", "contrato", "ord", "bin", "remainder")
    }

    sin_comparar = con_estado - set(ESTADO_AJUSTADO)
    assert not sin_comparar, f"pasos que ajustan algo y nadie compara: {sin_comparar}"


# --- 2. transformar la validación no reajusta nada --------------------------------------------


def test_transformar_la_validacion_no_mueve_ningun_parametro(base_y_split):
    """El 20% se transforma, nunca se ajusta nada sobre él, ni siquiera de refilón."""
    base, split = base_y_split
    pipeline = _fit(base, split, solo_train)
    antes = _ajustado(pipeline)

    _transformar(pipeline, matriz_de_features(solo_valid(base, split)))

    assert _ajustado(pipeline) == antes


def test_transformar_la_validacion_dos_veces_da_lo_mismo(base_y_split):
    """La otra cara: si algo se reajustase al transformar, la segunda pasada saldría distinta."""
    base, split = base_y_split
    pipeline = _fit(base, split, solo_train)
    valid = matriz_de_features(solo_valid(base, split))

    pd.testing.assert_frame_equal(_transformar(pipeline, valid), _transformar(pipeline, valid))


def test_la_validacion_de_verdad_trae_valores_que_el_ajuste_no_vio(base_y_split):
    """Guardián: con las dos particiones iguales, los dos tests de arriba no prueban nada."""
    base, split = base_y_split
    train, valid = solo_train(base, split), solo_valid(base, split)

    assert set(train[COL_ORGANIZACION]).isdisjoint(set(valid[COL_ORGANIZACION]))
    assert train["AMT_INCOME_TOTAL"].max() < valid["AMT_INCOME_TOTAL"].min()


# --- 3. ningún reajustable conserva el valor del EDA después del fit --------------------------


def test_los_diez_cortes_de_capa_2a_dejan_de_valer_lo_que_decia_el_eda(base_y_split):
    """El fallo que caza: leer `params` en vez de reestimar. Es la fuga de segundo orden.

    El fixture da percentiles deliberadamente distintos de los del EDA. Sobre el dato real los
    diez coinciden exactos con la referencia, así que allí esta comprobación no vale.
    """
    base, split = base_y_split
    ajustar_capa2a(base, split)

    heredados = [
        corte
        for corte in CORTES_WINSOR.values()
        if valor(corte) == parametro(corte).valor_referencia
    ]
    assert not heredados, f"estos cortes conservan el valor del EDA tras el fit: {heredados}"


def test_un_reajustable_sin_fijar_revienta_en_vez_de_devolver_la_referencia():
    """La barrera que impide que una cifra medida sobre el conjunto entero se cuele sin más."""
    pendientes = [n for n, p in reajustables().items() if p.valor_operativo is None]

    assert pendientes, "sin ningún pendiente este test no comprueba nada"
    with pytest.raises(ValueError, match="sin fijar"):
        valor(pendientes[0])


def test_los_quince_cortes_de_las_auxiliares_siguen_pendientes_tras_ajustar(base_y_split):
    """Este punto no puede fijar por accidente ninguno de los cortes de los bloques 2 a 4."""
    base, split = base_y_split
    antes = {n for n, p in reajustables().items() if p.valor_operativo is None}

    ajustar_capa2a(base, split)
    train = solo_train(base, split)
    construir_pipeline().fit(matriz_de_features(train), train["TARGET"])

    despues = {n for n, p in reajustables().items() if p.valor_operativo is None}
    assert despues == antes - set(CORTES_WINSOR.values())


def test_el_pipeline_no_escribe_en_params_al_ajustarse(base_y_split):
    """El `fit` es puro: quien escribe el registro es `registrar_limites()`, y a mano.

    Dentro del `fit` cada fold del CV de la Fase 4 reescribiría el diccionario de módulo y
    ganaría el último, que es estado global a merced del orden en que corran los ajustes.
    """
    base, split = base_y_split
    antes = {n: p.valor_operativo for n, p in PARAMS.items()}

    _fit(base, split, solo_train)

    assert {n: p.valor_operativo for n, p in PARAMS.items()} == antes


def test_el_pipeline_se_ajusta_sobre_la_particion_y_no_sobre_la_tabla_entera(base_y_split):
    """La fuga que caza: `solo_train(base, split)` sustituido por `base` en `ajustar_pipeline()`.

    Es la guarda del único camino sancionado para ajustar. `construir_pipeline()` devuelve el
    objeto sin ajustar, así que sin un punto de entrada propio no había nada que guardar y la
    propiedad solo se podía comprobar transformer a transformer.
    """
    base, split = base_y_split
    sobre_train = _ajustado(ajustar_pipeline(base, split)[0])
    sobre_todo = _ajustado(_fit(base, split))

    iguales = [paso for paso in ESTADO_AJUSTADO if sobre_train[paso] == sobre_todo[paso]]
    assert not iguales, f"el ajuste no distingue la partición en: {iguales}"


def test_el_pipeline_ajustado_declara_el_tamano_de_la_particion(base_y_split):
    """La otra dirección: que de verdad haya visto solo el 80% y no las 180 filas."""
    base, split = base_y_split

    pipeline, _ = ajustar_pipeline(base, split)

    assert pipeline.named_steps["winsor"].n_ajuste_["AMT_INCOME_TOTAL"] == N_TRAIN


def test_la_matriz_que_se_entrega_es_la_de_fuera_de_fold(base_y_split):
    """La fuga que caza: devolver `transform` del entrenamiento en vez de `fit_transform`.

    Por ese camino el `TargetEncoder` le da a cada fila la media de su propia categoría calculada
    con ella dentro, o sea la etiqueta de la propia fila metida en su feature. Devolver la matriz
    junto al pipeline es lo que hace que en la Fase 4 nadie tenga que acordarse de esto.
    """
    base, split = base_y_split
    pipeline, matriz = ajustar_pipeline(base, split)
    entrenamiento = matriz_de_features(solo_train(base, split))
    dentro_de_fold = pipeline.transform(entrenamiento)

    assert matriz[COL_OCUPACION].nunique() > dentro_de_fold[COL_OCUPACION].nunique()
    assert not np.allclose(matriz[COL_OCUPACION], dentro_de_fold[COL_OCUPACION])
    assert list(matriz.columns) == list(dentro_de_fold.columns)


def test_la_matriz_entregada_cubre_solo_el_entrenamiento(base_y_split):
    """Guardián: si viniera de la tabla entera, el test de arriba no vería la diferencia."""
    base, split = base_y_split
    _, matriz = ajustar_pipeline(base, split)

    assert len(matriz) == N_TRAIN
    assert len(matriz) < len(base)


def test_el_fixture_da_una_ocupacion_repartida_en_las_dos_particiones(base_y_split):
    """Guardián: con un solo oficio la columna sale constante y la varianza se la lleva."""
    base, split = base_y_split
    for parte in (solo_train, solo_valid):
        ocupaciones = parte(base, split)[COL_OCUPACION]
        assert ocupaciones.nunique() > 2, f"solo {ocupaciones.nunique()} oficios"
        tasas = parte(base, split).groupby(COL_OCUPACION, observed=True)["TARGET"].mean()
        assert ((tasas > 0) & (tasas < 1)).any(), "ningún oficio mezcla positivos y negativos"
