"""Tests del refijado conjunto y la persistencia de los cortes (punto 5.1).

Todos sobre un frame sintético y `PARAMS` manipulado a mano, así que corren en CI sin los CSV,
salvo la puerta contra el dato real del final, que se salta sin las tres tablas y el split y solo
corre en local. `guardar_cortes()` y `cargar_cortes()` no leen ni escriben más que `PARAMS` y un
json, así que no hace falta el dato real para probar sus guardas: fijar los ocho a mano con
`fijar_operativo()` reproduce exactamente lo que deja `refijar_cortes_auxiliares()`.
"""

import json
from dataclasses import replace

import pandas as pd
import pytest

from src.config import ruta
from src.data.loader import TABLE_FILES, load_table
from src.features import params as params_mod
from src.features.build_features import (
    CORTES_AUXILIARES,
    cargar_cortes,
    guardar_cortes,
    huella_split,
    refijar_cortes_auxiliares,
)
from src.features.params import fijar_operativo, parametro, valor
from src.features.split import cargar_split

# los ocho valores que refijar_cortes_auxiliares() deja sobre train, con un n_train cualquiera:
# el guardado y la carga no comprueban que el n_train sea plausible, solo que viaje intacto
VALORES_SINTETICOS = {
    "bureau_enddate_tramo_min_anios": (2.0, 210_875),
    "bureau_enddate_tramo_max_anios": (5.0, 210_875),
    "bureau_count_cola": (18, 210_875),
    "bb_many_credits_corte": (18, 73_767),
    "prev_count_cola": (11, 232_793),
    "prev_actividad_12m_cola": (4, 232_793),
    "prev_sobreconcesion_corte": (1.1, 231_992),
    "prev_finalidades_urgentes": (
        ("Car repairs", "Gasification / water supply", "Payments on other loans"),
        28_564,
    ),
}


def _fijar_los_ocho() -> None:
    for nombre, (v, n) in VALORES_SINTETICOS.items():
        fijar_operativo(nombre, v, n)


def _vaciar_los_ocho() -> None:
    """Simula un proceso nuevo: los ocho vuelven a sin fijar, como al importar el módulo."""
    for nombre in VALORES_SINTETICOS:
        p = parametro(nombre)
        params_mod.PARAMS[nombre] = replace(p, valor_operativo=None, n_train_operativo=None)


def test_cortes_auxiliares_son_exactamente_estos_ocho():
    """Guardián: si `CORTES_AUXILIARES` pierde o gana un nombre, esta lista escrita a mano lo
    caza. Sin este test, los de abajo iterarían sobre la lista encogida y seguirían en verde
    aunque `guardar_cortes()` dejara de persistir un corte de verdad."""
    assert set(CORTES_AUXILIARES) == set(VALORES_SINTETICOS)


@pytest.fixture
def split():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "split": ["train", "train", "valid", "valid"],
        }
    )


@pytest.fixture
def cortes_guardados(split, tmp_path):
    """Los ocho fijados y ya guardados en un `cortes.json` de prueba: `(split, destino)`."""
    _fijar_los_ocho()
    destino = tmp_path / "cortes.json"
    guardar_cortes(split, destino)
    return split, destino


# --- huella_split -------------------------------------------------------------------------------


def test_la_huella_no_depende_del_orden_de_filas(split):
    revuelto = split.sample(frac=1, random_state=0).reset_index(drop=True)
    assert huella_split(split) == huella_split(revuelto)


def test_la_huella_cambia_si_un_cliente_se_mueve_de_parte(split):
    movido = split.copy()
    movido.loc[movido["SK_ID_CURR"] == 3, "split"] = "train"
    assert huella_split(split) != huella_split(movido)


# --- guardar_cortes y cargar_cortes, la ida y la vuelta ------------------------------------------


def test_guardar_y_cargar_da_los_mismos_valores_y_tipos(cortes_guardados):
    """Pasa cuando debe pasar: un proceso nuevo recupera exactamente lo que se guardó."""
    split, destino = cortes_guardados
    esperados = {
        n: (valor(n), type(valor(n)), parametro(n).n_train_operativo) for n in VALORES_SINTETICOS
    }

    _vaciar_los_ocho()
    cargar_cortes(split, destino)

    for nombre, (v, tipo, n_train) in esperados.items():
        assert valor(nombre) == v
        assert type(valor(nombre)) is tipo, f"{nombre} cambió de tipo en la ida y vuelta"
        assert parametro(nombre).n_train_operativo == n_train, f"{nombre} perdió su n_train"


def test_cargar_con_la_huella_de_otro_split_revienta(cortes_guardados):
    """Falla cuando debe fallar: los cortes se midieron sobre otra partición."""
    split, destino = cortes_guardados
    _vaciar_los_ocho()

    otro = split.copy()
    otro.loc[otro["SK_ID_CURR"] == 1, "split"] = "valid"
    with pytest.raises(ValueError, match="huella"):
        cargar_cortes(otro, destino)


def test_guardar_dos_veces_sin_sobrescribir_revienta_y_con_el_no(cortes_guardados):
    split, destino = cortes_guardados
    with pytest.raises(FileExistsError, match="sobrescribir"):
        guardar_cortes(split, destino)
    # con sobrescribir=True no revienta
    guardar_cortes(split, destino, sobrescribir=True)


def test_guardar_con_un_corte_sin_fijar_revienta_sin_dejar_fichero(split, tmp_path):
    _fijar_los_ocho()
    p = parametro("prev_sobreconcesion_corte")
    params_mod.PARAMS["prev_sobreconcesion_corte"] = replace(
        p, valor_operativo=None, n_train_operativo=None
    )
    destino = tmp_path / "cortes.json"

    with pytest.raises(ValueError, match="prev_sobreconcesion_corte"):
        guardar_cortes(split, destino)
    assert not destino.exists()


def test_cargar_con_un_corte_de_menos_revienta(cortes_guardados):
    split, destino = cortes_guardados
    datos = json.loads(destino.read_text())
    del datos["cortes"]["prev_count_cola"]
    destino.write_text(json.dumps(datos))
    _vaciar_los_ocho()

    with pytest.raises(ValueError, match="faltan"):
        cargar_cortes(split, destino)


def test_cargar_con_un_corte_de_mas_revienta(cortes_guardados):
    split, destino = cortes_guardados
    datos = json.loads(destino.read_text())
    datos["cortes"]["un_corte_que_no_existe"] = {"valor": 1, "n_train": 1}
    destino.write_text(json.dumps(datos))
    _vaciar_los_ocho()

    with pytest.raises(ValueError, match="de más"):
        cargar_cortes(split, destino)


def test_cargar_sobre_uno_ya_fijado_exige_sobrescribir(cortes_guardados):
    """La misma guarda de fijar_operativo(), ejercitada a través de cargar_cortes()."""
    split, destino = cortes_guardados

    with pytest.raises(ValueError, match="sobrescribir"):
        cargar_cortes(split, destino)
    # con sobrescribir=True no revienta
    cargar_cortes(split, destino, sobrescribir=True)


def test_no_guardar_ninguno_de_los_diez_del_winsorizador(cortes_guardados):
    """Los `estimado` del winsorizador no viajan aquí: van dentro del joblib del Pipeline."""
    _, destino = cortes_guardados
    datos = json.loads(destino.read_text())
    assert set(datos["cortes"]) == set(VALORES_SINTETICOS)
    assert "app_winsor_amt_income_total" not in datos["cortes"]


# --- la puerta del 5.1 contra el dato real -------------------------------------------------------

sin_dato_real = pytest.mark.skipif(
    not all(
        (ruta("raw_data") / TABLE_FILES[t]).exists()
        for t in ("bureau", "bureau_balance", "previous_application", "application_train")
    )
    or not (ruta("processed_data") / "split.parquet").exists(),
    reason="data/raw o el split no viajan con el repo",
)

PUERTA = {
    "bureau_enddate_tramo_min_anios": 2.0,
    "bureau_enddate_tramo_max_anios": 5.0,
    "bureau_count_cola": 18,
    "bb_many_credits_corte": 18,
    "prev_count_cola": 11,
    "prev_actividad_12m_cola": 4,
    "prev_sobreconcesion_corte": 1.1,
    "prev_finalidades_urgentes": (
        "Car repairs",
        "Gasification / water supply",
        "Payments on other loans",
    ),
}
N_TRAIN = {
    "bureau_enddate_tramo_min_anios": 210_875,
    "bureau_enddate_tramo_max_anios": 210_875,
    "bureau_count_cola": 210_875,
    "bb_many_credits_corte": 73_767,
    "prev_count_cola": 232_793,
    "prev_actividad_12m_cola": 232_793,
    "prev_sobreconcesion_corte": 231_992,
    "prev_finalidades_urgentes": 28_564,
}


@sin_dato_real
def test_el_refijado_conjunto_reproduce_la_puerta_del_5_1():
    split = cargar_split()
    refijar_cortes_auxiliares(
        load_table("bureau"),
        load_table("bureau_balance"),
        load_table("previous_application"),
        split,
        split,
    )
    for nombre, esperado in PUERTA.items():
        p = parametro(nombre)
        assert p.valor_operativo == esperado, nombre
        assert p.n_train_operativo == N_TRAIN[nombre], nombre

    from src.features.params import operativos_pendientes

    medido_pendiente = [k for k, v in operativos_pendientes().items() if v.procedencia == "medido"]
    assert medido_pendiente == []


@sin_dato_real
def test_guardar_y_cargar_sobre_el_dato_real(tmp_path):
    """El mismo camino que la parada 2, pero fijado en test y no solo comprobado a mano."""
    split = cargar_split()
    refijar_cortes_auxiliares(
        load_table("bureau"),
        load_table("bureau_balance"),
        load_table("previous_application"),
        split,
        split,
    )
    # solo el valor y el n_train: el tipo exacto (int frente a float) ya lo fija el test
    # sintético de arriba. Aquí el operativo en caliente sale en np.float64 (numpy lo produce al
    # calcular) y el recargado en float nativo (lo que guarda json.dumps), y son el mismo número.
    esperados = {n: (valor(n), parametro(n).n_train_operativo) for n in PUERTA}
    destino = tmp_path / "cortes.json"
    guardar_cortes(split, destino)

    _vaciar_los_ocho()
    cargar_cortes(split, destino)

    for nombre, (v, n_train) in esperados.items():
        assert valor(nombre) == v
        assert parametro(nombre).n_train_operativo == n_train, f"{nombre} perdió su n_train"
