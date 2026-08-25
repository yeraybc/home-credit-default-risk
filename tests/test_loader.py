"""Pruebas de src.data.loader.

reduce_mem_usage se aplica a las 7 tablas antes de cualquier análisis, así que un
downcast mal elegido no rompe nada de forma visible: cambia valores en silencio y
contamina todo lo que venga después. Estas pruebas fijan ese contrato.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import TABLE_FILES, data_audit, load_table, reduce_mem_usage


def test_reduce_mem_usage_no_altera_los_valores():
    """Bajar el dtype es una optimización de memoria, no una transformación del dato."""
    df = pd.DataFrame(
        {
            "entero_pequeno": [0, 1, 200],
            "entero_con_negativos": [-100, 0, 100],
            "entero_grande": [0, 100_000, 3_000_000],
            "flotante": [1.5, 2.25, 3.125],
        }
    )
    original = df.copy()
    reducido = reduce_mem_usage(df.copy(), verbose=False)

    for col in original.columns:
        assert np.allclose(reducido[col], original[col]), f"{col} cambió de valor al reducir"


def test_reduce_mem_usage_elige_el_dtype_minimo():
    """Los umbrales de la escalera de dtypes son un contrato: fijan los MB documentados."""
    df = pd.DataFrame(
        {
            "uint8": [0, 254],
            "uint16": [0, 65_534],
            "uint32": [0, 4_294_967_294],
            "int8": [-127, 126],
            "int16": [-32_767, 32_766],
            "int32": [-2_147_483_647, 2_147_483_646],
        }
    )
    reducido = reduce_mem_usage(df, verbose=False)

    for nombre_esperado in df.columns:
        assert reducido[nombre_esperado].dtype == np.dtype(nombre_esperado)


def test_reduce_mem_usage_nunca_usa_float16():
    """float16 da 3 dígitos significativos: inaceptable para importes de crédito."""
    df = pd.DataFrame({"importe": [1.5, 2.5, 3.5]})
    reducido = reduce_mem_usage(df, verbose=False)

    assert reducido["importe"].dtype == np.float32


def test_reduce_mem_usage_preserva_importes_grandes():
    """Regresión: AMT_CREDIT_SUM llega a 396M en bureau y debe sobrevivir al downcast."""
    df = pd.DataFrame({"AMT_CREDIT_SUM": [396_000_000.0, 117_000_000.0, 40_500.0]})
    reducido = reduce_mem_usage(df.copy(), verbose=False)

    # float32 tiene ~7 dígitos significativos: 396M cabe con holgura
    assert np.allclose(reducido["AMT_CREDIT_SUM"], df["AMT_CREDIT_SUM"], rtol=1e-6)


def test_reduce_mem_usage_reduce_de_verdad():
    df = pd.DataFrame({"a": range(10_000), "b": np.random.default_rng(0).random(10_000)})
    antes = df.memory_usage(deep=True).sum()
    despues = reduce_mem_usage(df, verbose=False).memory_usage(deep=True).sum()

    assert despues < antes


def test_reduce_mem_usage_categoriza_solo_baja_cardinalidad():
    """El umbral de 50 evita convertir un ID de texto en categoría, que gastaría más."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pocas": rng.choice(["Active", "Closed", "Sold"], 500),
            "muchas": [f"id_{i}" for i in range(500)],
        }
    )
    reducido = reduce_mem_usage(df, verbose=False)

    assert isinstance(reducido["pocas"].dtype, pd.CategoricalDtype)
    assert reducido["muchas"].dtype == object


def test_load_table_rechaza_nombre_desconocido():
    with pytest.raises(ValueError, match="tabla desconocida"):
        load_table("tabla_que_no_existe")


def test_table_files_cubre_las_ocho_tablas():
    """Las 7 del proyecto más application_test."""
    assert len(TABLE_FILES) == 8
    assert "application_train" in TABLE_FILES
    assert all(fichero.endswith(".csv") for fichero in TABLE_FILES.values())


def test_data_audit_resume_y_ordena_por_filas():
    dfs = {
        "chica": pd.DataFrame({"a": [1, 2]}),
        "grande": pd.DataFrame({"a": [1, None, 3, 4]}),
    }
    resumen = data_audit(dfs)

    assert list(resumen["table"]) == ["grande", "chica"]  # ordenado por filas, descendente
    assert resumen.loc[resumen["table"] == "grande", "missing_%"].iloc[0] == 25.0
    assert resumen.loc[resumen["table"] == "grande", "cols_w_missing"].iloc[0] == 1


def test_data_audit_no_divide_por_cero_con_tabla_vacia():
    resumen = data_audit({"vacia": pd.DataFrame()})

    assert resumen.loc[0, "missing_%"] == 0.0
