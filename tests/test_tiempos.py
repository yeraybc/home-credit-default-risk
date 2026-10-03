"""Tests del cronómetro y la línea base de tiempos (src/models/tiempos.py, punto 0.6).

Todos sobre un frame sintético con una fábrica de pipeline trivial, así que corren en CI. La línea
base real (minutos de cómputo) no entra en la suite: se reproduce con
`python -m src.models.tiempos`.
"""

import pickle

import lightgbm
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.models import tiempos
from src.models.folds import N_FOLDS
from src.models.metricas import ORDEN_SIMPLICIDAD
from src.models.tiempos import (
    DISPERSION_MAX,
    PRESUPUESTO_PROVISIONAL,
    _nan_canonico,
    escribir_linea_base,
    exigir_dispersion,
    homogeneidad_folds,
    medir_fold,
    modelos_ingenuos,
    resumir_tiempos,
    sanear_nombres,
)

N = 400


def fabrica_identidad():
    return Pipeline([("id", FunctionTransformer())])


_RESERVA = []


def fabrica_que_reserva():
    """Reserva 200 MB y los toca, para que cuenten en la memoria residente del hijo."""
    _RESERVA.append(np.ones(25_000_000))
    return fabrica_identidad()


@pytest.fixture
def poblacion():
    rng = np.random.default_rng(0)
    indice = pd.Index(np.arange(1000, 1000 + N), name="SK_ID_CURR")
    y = pd.Series((np.arange(N) % 25 == 0).astype(int), index=indice, name="TARGET")
    X = pd.DataFrame({"f1": y * 1.5 + rng.normal(size=N), "f2": rng.normal(size=N)}, index=indice)
    return X, y


def _medidas(**sobrescribir):
    """Medidas fabricadas: tres repeticiones del fold 0 por modelo."""
    filas = []
    for modelo, base in zip(ORDEN_SIMPLICIDAD, (10.0, 100.0, 30.0)):
        for i, factor in enumerate((0.95, 1.0, 1.05)):
            filas.append(
                {
                    "modelo": modelo,
                    "k": 0,
                    "pared_features": base * 0.9 * factor,
                    "pared_ajuste": base * 0.09 * factor,
                    "pared_prediccion": base * 0.01 * factor,
                    "cpu_features": base * 0.9 * factor,
                    "cpu_ajuste": base * 0.09 * factor * 4,
                    "cpu_prediccion": base * 0.01 * factor,
                    "mem_pico_mb": 1000.0 + i,
                    "auc": 0.76,
                }
            )
    return pd.DataFrame(filas).assign(**sobrescribir)


# --- los modelos ingenuos ---------------------------------------------------------------------


def test_los_modelos_ingenuos_son_los_tres_y_en_orden_de_simplicidad():
    assert tuple(modelos_ingenuos()) == ORDEN_SIMPLICIDAD


def test_los_modelos_ingenuos_solo_se_apartan_del_defecto_en_lo_declarado():
    modelos = modelos_ingenuos()
    # el bosque es el de por defecto salvo la semilla
    bosque = modelos["random_forest"].get_params()
    defecto = RandomForestClassifier().get_params()
    assert {k for k in bosque if bosque[k] != defecto[k]} == {"random_state"}
    # la logística sube max_iter a 1000 y fija la semilla
    logistica = modelos["logistica"].named_steps["modelo"].get_params()
    defecto = LogisticRegression().get_params()
    assert {k for k in logistica if logistica[k] != defecto[k]} == {"max_iter", "random_state"}
    # LightGBM fija la semilla y calla su log; ni early stopping ni n_jobs tocados
    lgbm = modelos["lightgbm"].named_steps["modelo"].get_params()
    defecto = LGBMClassifier().get_params()
    assert {k for k in lgbm if lgbm[k] != defecto.get(k)} == {"random_state", "verbose"}


def test_sanear_nombres_quita_la_coma_y_lightgbm_la_acepta_ya_saneada(poblacion):
    X, y = poblacion
    con_coma = X.rename(columns={"f1": "NAME_TYPE_SUITE_Spouse, partner"})

    with pytest.raises(lightgbm.basic.LightGBMError):
        LGBMClassifier(verbose=-1, n_estimators=3).fit(con_coma, y)
    saneado = sanear_nombres(con_coma)

    assert "NAME_TYPE_SUITE_Spouse_ partner" in saneado.columns
    LGBMClassifier(verbose=-1, n_estimators=3).fit(saneado, y)


def test_sanear_nombres_revienta_si_dos_nombres_chocan(poblacion):
    X, _ = poblacion
    choque = X.rename(columns={"f1": "a,b", "f2": "a:b"})

    with pytest.raises(ValueError, match="mismo nombre"):
        sanear_nombres(choque)


def test_el_modelo_de_lightgbm_ajusta_con_una_columna_con_coma(poblacion):
    X, y = poblacion
    con_coma = X.rename(columns={"f1": "WALLSMATERIAL_MODE_Stone, brick"})

    modelo = modelos_ingenuos()["lightgbm"].fit(con_coma, y)

    assert list(modelo.classes_) == [0, 1]


# --- el cronómetro ----------------------------------------------------------------------------


def test_medir_fold_cronometra_el_fold_pedido_y_devuelve_un_ajuste_real(poblacion):
    X, y = poblacion

    m = medir_fold(modelos_ingenuos()["logistica"], X, y, k=2, fabrica=fabrica_identidad)

    assert m["k"] == 2
    assert m["n_ajuste"] == int(N * 4 / 5)
    assert m["auc"] > 0.5
    for etapa in ("features", "ajuste", "prediccion"):
        assert m[f"pared_{etapa}"] > 0 and m[f"cpu_{etapa}"] >= 0


class _Invertido(LogisticRegression):
    """Devuelve las columnas de `predict_proba` al revés: el AUC sale a 1 menos el real."""

    def predict_proba(self, X):
        return super().predict_proba(X)[:, ::-1]


def test_medir_fold_revienta_si_el_auc_no_pasa_de_medio(poblacion):
    X, y = poblacion

    with pytest.raises(ValueError, match="no pasa de 0,5"):
        medir_fold(_Invertido(), X, y, k=0, fabrica=fabrica_identidad)


def test_la_memoria_pico_ve_lo_que_reserva_el_hijo_y_solo_eso(poblacion):
    X, y = poblacion

    sin = medir_fold(modelos_ingenuos()["logistica"], X, y, fabrica=fabrica_identidad)
    con = medir_fold(modelos_ingenuos()["logistica"], X, y, fabrica=fabrica_que_reserva)

    assert con["mem_pico_mb"] >= 190
    assert sin["mem_pico_mb"] < 100
    # cada medida en su proceso: la reserva de una no queda en la base de la otra
    assert not _RESERVA


def test_el_nan_de_las_columnas_object_vuelve_a_ser_un_unico_objeto_tras_el_pickle():
    X = pd.DataFrame({"c": pd.Series(["a", np.nan, "b", np.nan, np.nan], dtype=object)})
    viajado = pickle.loads(pickle.dumps(X, protocol=pickle.HIGHEST_PROTOCOL))
    valores = viajado["c"].to_numpy()
    assert len({id(v) for v in valores if v != v}) == 3

    canonico = _nan_canonico(viajado)

    assert len({id(v) for v in canonico["c"].to_numpy() if v != v}) == 1
    assert canonico.equals(X) and (canonico.dtypes == X.dtypes).all()
    # no toca el original
    assert len({id(v) for v in viajado["c"].to_numpy() if v != v}) == 3


# --- el resumen -------------------------------------------------------------------------------


def test_resumir_tiempos_da_mediana_dispersion_y_extrapolacion_exactas():
    r = resumir_tiempos(_medidas())

    assert list(r.index) == list(ORDEN_SIMPLICIDAD)
    # logística: folds de 9,5 / 10 / 10,5 s
    assert r.loc["logistica", "pared_fold"] == pytest.approx(10.0)
    assert r.loc["logistica", "dispersion"] == pytest.approx((10.5 - 9.5) / 10.0)
    assert r.loc["logistica", "trial_s"] == pytest.approx(N_FOLDS * 10.0)
    for modelo, trials in PRESUPUESTO_PROVISIONAL.items():
        assert r.loc[modelo, "total_s"] == pytest.approx(trials * r.loc[modelo, "trial_s"])
        assert r.loc[modelo, "trials"] == trials
    assert r["extrapolado"].all()
    assert r.loc["lightgbm", "mem_pico_mb"] == pytest.approx(1001.0)
    assert r.loc["logistica", "pared_features"] == pytest.approx(9.0)


def test_resumir_tiempos_mide_el_paralelismo_como_cpu_sobre_pared():
    r = resumir_tiempos(_medidas())

    # features 0,9 en un núcleo y ajuste 0,09 en cuatro: (0,9 + 0,36 + 0,01) / 1
    assert r.loc["logistica", "cpu_sobre_pared"] == pytest.approx(1.27)


def test_resumir_tiempos_ignora_las_pasadas_de_otros_folds():
    extra = _medidas().iloc[:1].assign(k=3, pared_features=999.0)

    r = resumir_tiempos(pd.concat([_medidas(), extra], ignore_index=True))

    assert r.loc["logistica", "pared_fold"] == pytest.approx(10.0)
    assert r.loc["logistica", "repeticiones"] == 3


def test_homogeneidad_marca_el_fold_que_se_sale_de_la_tolerancia():
    base = _medidas()
    pasadas = base.query("modelo == 'logistica'").iloc[:1].copy()
    ok = pasadas.assign(k=1, pared_features=9.0 * 1.05)
    mal = pasadas.assign(k=2, pared_features=9.0 * 1.5)

    h = homogeneidad_folds(pd.concat([base, ok, mal], ignore_index=True))

    assert list(h["k"]) == [1, 2]
    assert list(h["dentro"]) == [True, False]
    # fold de 9,45 + 0,855 + 0,095 = 10,4 s frente a la mediana de 10 s
    assert h.loc[0, "razon"] == pytest.approx(1.04)
    assert DISPERSION_MAX == 0.10


def test_linea_base_intercala_los_modelos_y_cubre_los_folds(monkeypatch, poblacion):
    X, y = poblacion

    def espia(modelo, X, y, k=0, fabrica=None):
        return {"k": k}

    monkeypatch.setattr(tiempos, "medir_fold", espia)

    medidas = tiempos.linea_base(X, y, repeticiones=2)

    assert list(medidas["modelo"][:6]) == list(ORDEN_SIMPLICIDAD) * 2
    assert list(medidas["k"][:6]) == [0] * 6
    otros = medidas[medidas["k"] != 0]
    assert sorted(otros["k"].unique()) == [1, 2, 3, 4]
    assert set(otros["modelo"]) == set(tiempos.MODELOS_TODOS_LOS_FOLDS)


# --- la guarda de la dispersión -------------------------------------------------------------


def _resumen(*dispersiones):
    return pd.DataFrame(
        {"dispersion": dispersiones}, index=list(ORDEN_SIMPLICIDAD)[: len(dispersiones)]
    )


def test_la_dispersion_en_el_limite_pasa_y_por_encima_revienta():
    exigir_dispersion(_resumen(0.02, DISPERSION_MAX))
    with pytest.raises(ValueError, match="random_forest"):
        exigir_dispersion(_resumen(0.02, DISPERSION_MAX + 0.01))


def test_la_linea_base_ruidosa_no_se_escribe(tmp_path):
    destino = tmp_path / "linea_base.json"
    medidas = _medidas()

    with pytest.raises(ValueError, match="dispersión"):
        escribir_linea_base(medidas, _resumen(0.02, 0.5, 0.02), destino)
    assert not destino.exists()

    escribir_linea_base(medidas, resumir_tiempos(medidas).assign(dispersion=0.05), destino)
    assert destino.exists()
