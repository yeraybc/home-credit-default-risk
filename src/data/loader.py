"""
src.data.loader: Carga de las 7 tablas de Home Credit.

Todo acceso a datos crudos pasa por aquí. Nadie más hace pd.read_csv().
Incluye reduce_mem_usage para bajar el consumo de RAM ~60%.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = _PROJECT_ROOT / "data" / "raw"

TABLE_FILES: dict[str, str] = {
    "application_train":    "application_train.csv",
    "application_test":     "application_test.csv",
    "bureau":               "bureau.csv",
    "bureau_balance":       "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash":             "POS_CASH_balance.csv",
    "credit_card":          "credit_card_balance.csv",
    "installments":         "installments_payments.csv",
}

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Baja los dtypes numéricos al mínimo sin perder rango, ahorra más del 60% RAM.

    float16 excluido: solo 3 dígitos significativos, inaceptable para montos.
    """
    start = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_integer_dtype(t):
            mn, mx = df[col].min(), df[col].max()
            if mn >= 0:
                if mx < 255: dtype = np.uint8
                elif mx < 65535: dtype = np.uint16
                elif mx < 4294967295: dtype = np.uint32
                else: dtype = np.uint64
            else:
                if mn > -128 and mx < 127: dtype = np.int8
                elif mn > -32768 and mx < 32767: dtype = np.int16
                elif mn > -2147483648 and mx < 2147483647: dtype = np.int32
                else: dtype = np.int64
            df[col] = df[col].astype(dtype)
        elif pd.api.types.is_float_dtype(t):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_object_dtype(t):
            if df[col].nunique() < 50:
                df[col] = df[col].astype("category")

    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info("memoria: %.1f MB → %.1f MB (%.0f%% reducción)", start, end, 100*(start-end)/start)
    return df

def load_table(
    name: str,
    data_dir: Path = RAW_DATA_DIR,
    reduce_memory: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Carga una tabla por nombre lógico. Opciones: ver TABLE_FILES."""
    if name not in TABLE_FILES:
        raise ValueError(f"tabla desconocida: '{name}'. válidas: {sorted(TABLE_FILES)}")

    filepath = data_dir / TABLE_FILES[name]
    logger.info("cargando %s...", TABLE_FILES[name])
    df = pd.read_csv(filepath, **kwargs)
    logger.info("  %s filas × %s cols", f"{len(df):,}", df.shape[1])

    if reduce_memory:
        df = reduce_mem_usage(df)
    return df


def load_all_tables(
    data_dir: Path = RAW_DATA_DIR,
    reduce_memory: bool = True,
    tables: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Carga todas las tablas o un subconjunto.

    Ejemplo rápido para prototipar:
        dfs = load_all_tables(tables=["application_train", "bureau"])
    """
    to_load = tables or list(TABLE_FILES)
    return {name: load_table(name, data_dir, reduce_memory) for name in to_load}


def data_audit(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resumen estructurado de todas las tablas cargadas.

    Filas, columnas, % nulos, columnas con nulos, RAM usada y tipos.
    """
    records = []
    for name, df in dfs.items():
        n_cells = df.size
        n_missing = df.isnull().sum().sum()
        records.append({
            "table":          name,
            "rows":           df.shape[0],
            "cols":           df.shape[1],
            "missing_%":      round(100 * n_missing / n_cells, 2) if n_cells else 0.0,
            "cols_w_missing": df.isnull().any().sum(),
            "memory_MB":      round(df.memory_usage(deep=True).sum() / 1024**2, 1),
            "int_cols":       df.select_dtypes(include="integer").shape[1],
            "float_cols":     df.select_dtypes(include="float").shape[1],
            "object_cols":    df.select_dtypes(include="object").shape[1],
        })
    return pd.DataFrame(records).sort_values("rows", ascending=False).reset_index(drop=True)


# Aplico una busqueda heurística de tipos basada en el nombre de las columnas y descriptions. 
# para escalar esta funcion implicaria hardcorear el schema mapeando vía dict o dtype explícito en config files.
def verify_dtypes(
    df_name: str,
    df: pd.DataFrame,
    description_path: Path = RAW_DATA_DIR / "HomeCredit_columns_description.csv",
) -> pd.DataFrame:
    """Verifica si los dtypes de un DataFrame corresponden a su intencionalidad lógica.

    Compara con las descripciones y metadatos de HomeCredit_columns_description.csv.
    """
    desc_df = pd.read_csv(description_path, encoding="latin1")

    # Mapeo de nombres lógicos a nombres de archivo en el CSV de descripción
    table_map = {
        "application_train": "application_{train|test}.csv",
        "application_test": "application_{train|test}.csv",
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "previous_application": "previous_application.csv",
        "pos_cash": "POS_CASH_balance.csv",
        "credit_card": "credit_card_balance.csv",
        "installments": "installments_payments.csv",
    }

    csv_table_name = table_map.get(df_name, df_name)
    table_desc = desc_df[desc_df["Table"] == csv_table_name]

    report = []

    for col in df.columns:
        row_desc = table_desc[table_desc["Row"] == col]
        desc = str(row_desc["Description"].values[0]).lower() if not row_desc.empty else ""
        special = str(row_desc["Special"].values[0]).lower() if not row_desc.empty else ""

        col_upper = col.upper()
        dtype = df[col].dtype
        num_nulls = df[col].isnull().sum()

        # Heurísticas de categorización
        if "ID" in col_upper or "SK_ID" in col_upper or "identifier" in desc:
            cat = "ID"
        elif "normalized" in special or col_upper.startswith("EXT_SOURCE_") or any(col_upper.endswith(s) for s in ["_AVG", "_MODE", "_MEDI"]):
            if any(s in col_upper for s in ["HOUSETYPE", "WALLSMATERIAL", "FONDKAPREMONT", "EMERGENCYSTATE"]):
                cat = "Categorical"
            else:
                cat = "Normalized/Float"
        elif col_upper.startswith("FLAG_") or col_upper.startswith("REG_") or col_upper.startswith("LIVE_") or "flag" in desc or "1 -" in desc or special in ["recoded", "grouped"]:
            cat = "Binary/Flag"
        elif col_upper.startswith("DAYS_") or "days" in desc or "time only relative" in special:
            cat = "Days/Time"
        elif col_upper.startswith("CNT_") or "number of" in desc or "count" in desc:
            cat = "Count"
        elif col_upper.startswith("AMT_") or "amount" in desc or "price" in desc or "annuity" in desc:
            cat = "Amount/Price"
        elif col_upper.startswith("NAME_") or col_upper.startswith("CODE_") or col_upper == "ORGANIZATION_TYPE" or pd.api.types.is_object_dtype(dtype):
            cat = "Categorical"
        else:
            cat = "Numerical" if pd.api.types.is_numeric_dtype(dtype) else "Categorical"

        status = "OK"
        detail = ""

        is_num = pd.api.types.is_numeric_dtype(dtype)
        is_float = pd.api.types.is_float_dtype(dtype)
        is_int = pd.api.types.is_integer_dtype(dtype)
        is_obj = pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype)

        if cat == "ID" and is_float:
            status = "ERROR"
            detail = "ID cargado como float (riesgo de pérdida de precisión)."
        elif cat == "Binary/Flag":
            if pd.api.types.is_bool_dtype(dtype) or str(dtype) == 'int8':
                status = "OK"
            elif is_obj:
                u = list(df[col].dropna().unique())
                if set(u).issubset({"Y", "N", "y", "n", "1", "0", 1, 0}):
                    status = "IMPROVEMENT"
                    detail = f"Texto Y/N ({u}). Convertir a boolean."
                else:
                    status = "WARNING"
                    detail = f"Flag no estándar: {u}"
            elif is_num and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                status = "IMPROVEMENT"
                detail = f"Flag numérico ({dtype}). Representar como boolean o int8 para ahorrar memoria."
        elif cat in ["Days/Time", "Count"]:
            if is_float:
                has_dec = not np.all(df[col].dropna() % 1 == 0) if num_nulls < len(df) else False
                if not has_dec:
                    # OK: se mantiene float debido a nulos para mayor compatibilidad con modelos
                    status = "OK"
                else:
                    status = "WARNING"
                    detail = "Días/conteo contiene decimales reales."
            elif is_obj:
                status = "ERROR"
                detail = "Días/conteo cargado como texto."
        elif cat in ["Amount/Price", "Normalized/Float"] and is_obj:
            status = "ERROR"
            detail = "Numérico cargado como texto."
        elif cat == "Categorical":
            if is_num:
                status = "WARNING"
                detail = "Categoría cargada como numérica."
            elif is_obj and not isinstance(dtype, pd.CategoricalDtype) and df[col].nunique() < 50:
                status = "IMPROVEMENT"
                detail = f"Baja cardinalidad ({df[col].nunique()} categorías). Usar tipo 'category'."

        report.append({
            "Variable": col,
            "Tipo Actual": str(dtype),
            "Categoría Esperada": cat,
            "Estado": status,
            "Detalle": detail
        })

    return pd.DataFrame(report)

