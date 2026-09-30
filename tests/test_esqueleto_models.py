"""Esqueleto de src/models (0.1): los siete módulos importan."""

import importlib

import pytest

MODULOS = ["folds", "metricas", "contrastes", "evaluar", "espacios", "tiempos", "train"]


@pytest.mark.parametrize("modulo", MODULOS)
def test_modulo_importa(modulo):
    assert importlib.import_module(f"src.models.{modulo}").__doc__


def test_config_sin_umbral_fijo_ni_nombre_de_ensemble():
    from src.config import cargar_config

    cfg = cargar_config()
    assert "decision_threshold" not in cfg["modeling"]
    assert "model_name" not in cfg["api"]
