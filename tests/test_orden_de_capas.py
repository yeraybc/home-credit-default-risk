"""El orden de las capas dentro del orquestador (src/features/build_features.py).

La capa 1 corre sobre la tabla entera y el split va **después**, sobre el resultado ya limpio,
para que la partición cubra exactamente la población de modelado. Al revés, el fichero de split
declararía clientes que después no existen en la matriz.

`construir_base()` es la única pieza donde ese orden vive, y no lo comprobaba nadie: el test de
integración contrasta el `split.parquet` ya escrito, o sea el artefacto y no el camino de código
que lo produce, así que invertir las dos líneas lo dejaba verde igual.

Va sobre un frame sintético y no sobre el csv porque el orden es una propiedad del código: así
corre también en el clon limpio, que es donde el fichero de integración se salta entero. Y la
partición se espía en vez de ejecutarse, que construirla de verdad pisaría el split persistido.
"""

import pandas as pd
import pytest

from src.features import build_features as mod
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
