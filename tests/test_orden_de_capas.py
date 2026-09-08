"""El orden de las capas dentro del orquestador (src/features/build_features.py).

La capa 1 corre sobre la tabla entera y el split va **después**, sobre el resultado ya limpio,
para que la partición cubra exactamente la población de modelado. Al revés, el fichero de split
declararía clientes que después no existen en la matriz.

`construir_base()` es la única pieza donde ese orden vive, y no lo comprobaba nadie: el test de
integración contrasta el `split.parquet` ya escrito, o sea el artefacto y no el camino de código
que lo produce, así que invertir las dos líneas lo dejaba verde igual.

Lo mismo vale para la otra mitad del orden, que es **sobre qué frame ajusta la capa 2a**. Ahí el
único apoyo era una aserción del fichero de integración, así que sustituir `solo_train(base,
split)` por `base` dejaba los 182 tests del alcance de CI en verde: sobre la tabla real los diez
percentiles valen lo mismo con la partición y sin ella, o sea que la cifra no distingue los dos
casos y hace falta un frame donde sí los distinga.

Va sobre un frame sintético y no sobre el csv porque el orden es una propiedad del código: así
corre también en el clon limpio, que es donde el fichero de integración se salta entero. Y la
partición se espía o se pasa a mano en vez de ejecutarse, que construirla de verdad pisaría el
split persistido.
"""

import pandas as pd
import pytest

from src.features import build_features as mod
from src.features import params as params_mod
from src.features.build_features import ajustar_capa2a, matriz_de_features
from src.features.cleaning import filas_a_eliminar

FUERA_POR_SEXO, FUERA_POR_CUOTA = 3, 5
SUPERVIVIENTES = [1, 2, 4]


@pytest.fixture
def cruda():
    """Cinco clientes con las columnas que el contrato exige, dos de ellos a eliminar."""
    from src.features.application import ORIGEN_POR_FEATURE

    frame = pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4, 5], "TARGET": [0, 1, 0, 0, 1]})
    for columnas in ORIGEN_POR_FEATURE.values():
        for c in columnas:
            frame[c] = [-1000.0, -2000.0, -3000.0, -4000.0, -5000.0]
    # las dos versiones que la limpieza elimina, para que ese camino también corra
    frame["APARTMENTS_MODE"] = 1.0
    frame["APARTMENTS_MEDI"] = 1.0
    frame["CODE_GENDER"] = ["M", "F", "XNA", "F", "M"]
    frame["NAME_FAMILY_STATUS"] = "Married"
    frame["CNT_FAM_MEMBERS"] = 2.0
    frame["DAYS_LAST_PHONE_CHANGE"] = -100.0
    frame["AMT_ANNUITY"] = [1.0, 2.0, 3.0, 4.0, None]
    return frame


def test_el_split_se_construye_sobre_la_tabla_ya_limpia(cruda, monkeypatch):
    """Lo que llega a la partición es la salida de la capa 1, nunca la tabla cruda."""
    recibido = {}

    def espia_split(app, **kwargs):
        recibido["app"] = app
        return app[["SK_ID_CURR", "TARGET"]].assign(split="train")

    monkeypatch.setattr(mod, "load_table", lambda *a, **k: cruda)
    monkeypatch.setattr(mod, "construir_split", espia_split)

    base, _ = mod.construir_base()

    assert list(recibido["app"]["SK_ID_CURR"]) == SUPERVIVIENTES, (
        "a la partición le llegaron clientes que la limpieza elimina: se está construyendo "
        "sobre la tabla cruda, o sea antes de la capa 1"
    )
    assert recibido["app"] is base, "la partición y la matriz tienen que salir del mismo frame"
    assert "HAS_BUILDING_INFO" in base.columns, "el frame llegó sin las features de capa 1"


def test_la_tabla_sintetica_trae_de_verdad_filas_que_la_limpieza_quita(cruda):
    """Sin filas que eliminar, el test de arriba pasaría con el orden invertido."""
    fuera = filas_a_eliminar(cruda)

    assert int(fuera.sum()) == 2
    assert set(cruda.loc[fuera, "SK_ID_CURR"]) == {FUERA_POR_SEXO, FUERA_POR_CUOTA}


# --- la otra mitad del orden: sobre qué frame ajusta la capa 2a ------------------------------
# Los cinco de entrenamiento cobran mil y los cinco de validación un millón, así que el límite
# sale distinto según se ajuste sobre la partición o sobre la tabla entera. Es lo que el dato
# real no puede dar: allí los diez percentiles coinciden con y sin partición.
INGRESO_TRAIN, INGRESO_VALID = 1_000.0, 1_000_000.0
N_TRAIN_SINTETICO = 5


@pytest.fixture
def base_y_split():
    """Diez clientes, cinco por parte, con el ingreso separado entre las dos."""
    base = pd.DataFrame(
        {
            "SK_ID_CURR": range(1, 11),
            "TARGET": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "AMT_INCOME_TOTAL": [INGRESO_TRAIN] * 5 + [INGRESO_VALID] * 5,
        }
    )
    split = pd.DataFrame(
        {"SK_ID_CURR": range(1, 11), "split": ["train"] * 5 + ["valid"] * 5}
    )
    return base, split


@pytest.fixture(autouse=True)
def registro_limpio():
    """`ajustar_capa2a` escribe en el dict de módulo de params.py; se deja como estaba."""
    copia = dict(params_mod.PARAMS)
    yield
    params_mod.PARAMS.clear()
    params_mod.PARAMS.update(copia)


def test_la_capa2a_se_ajusta_sobre_la_particion_y_no_sobre_la_tabla_entera(base_y_split):
    """El fallo que caza: `solo_train(base, split)` sustituido por `base`."""
    from src.features.transformers import Winsorizador

    base, split = base_y_split
    winsorizador, _ = ajustar_capa2a(base, split)
    sobre_todo = Winsorizador().fit(base).limites_["AMT_INCOME_TOTAL"]

    assert winsorizador.limites_["AMT_INCOME_TOTAL"] != sobre_todo, (
        "el límite es el de la tabla entera: la capa 2a se está ajustando fuera del split y "
        "el 20% de validación participa en elegir el umbral que después se le aplica"
    )
    assert winsorizador.n_ajuste_["AMT_INCOME_TOTAL"] == N_TRAIN_SINTETICO


def test_la_capa2a_no_mete_la_etiqueta_ni_el_id_en_el_contrato_de_nombres(base_y_split):
    """Lo que promete `get_feature_names_out()` tiene que ser lo que `transform` entrega.

    Ajustando sobre la base entera, el contrato salía con `TARGET` y `SK_ID_CURR` dentro y
    prometía una columna más de las que la salida trae. Lo lee el `ColumnTransformer` del 1.4.
    """
    base, split = base_y_split
    winsorizador, _ = ajustar_capa2a(base, split)
    prometidas = list(winsorizador.get_feature_names_out())

    assert "TARGET" not in prometidas, "la etiqueta no puede viajar en el contrato de features"
    assert "SK_ID_CURR" not in prometidas
    assert prometidas == list(winsorizador.transform(matriz_de_features(base)).columns)


def test_el_frame_sintetico_separa_de_verdad_las_dos_particiones(base_y_split):
    """Guardián: con el mismo ingreso en las dos partes, el test de arriba no distingue nada."""
    from src.features.transformers import Winsorizador

    base, _ = base_y_split
    # las cinco primeras filas son las de train, y se cogen por posición a propósito: el
    # guardián comprueba el fixture, así que no puede apoyarse en la maquinaria que se audita
    entrenamiento = base.iloc[:N_TRAIN_SINTETICO]

    assert (
        Winsorizador().fit(entrenamiento).limites_["AMT_INCOME_TOTAL"]
        != Winsorizador().fit(base).limites_["AMT_INCOME_TOTAL"]
    )
