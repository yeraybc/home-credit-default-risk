"""Genera notebooks/01_eda.ipynb con el contenido del Día 1."""

import json
import uuid
from pathlib import Path


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source.strip(),
    }


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source.strip(),
    }


cells = []

cells.append(
    md(
        "# EDA — Día 1: overview del dataset\n\n**objetivo:** entender la estructura del dataset antes de tocar una sola feature."
    )
)

cells.append(
    code(
        """\
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

from src.data import load_all_tables, data_audit

print('setup ok')
"""
    )
)

cells.append(md("## 1. carga y audit"))

cells.append(
    code(
        """\
# carga completa — tarda 3-6 min la primera vez por bureau_balance (27M filas)
dfs = load_all_tables()
app = dfs['application_train']
"""
    )
)

cells.append(
    code(
        """\
# resumen de todas las tablas
audit = data_audit(dfs)
audit
"""
    )
)

cells.append(
    code(
        """\
total_gb = audit['memory_MB'].sum() / 1024
print(f"RAM total tras optimización: {total_gb:.2f} GB")
"""
    )
)

cells.append(
    md(
        "## 2. target — distribución de clases\n\nPrimero que se mira en cualquier problema de clasificación."
    )
)

cells.append(
    code(
        """\
target_counts = app['TARGET'].value_counts()
target_pct    = app['TARGET'].value_counts(normalize=True).mul(100)

print("TARGET    count        %")
for k in [0, 1]:
    print(f"  {k}       {target_counts[k]:>7,}    {target_pct[k]:.2f}%")
"""
    )
)

cells.append(
    code(
        """\
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# conteos
axes[0].bar(['no default', 'default'], target_counts.values,
            color=['#4C72B0', '#DD8452'], edgecolor='white', width=0.5)
axes[0].set_title('conteo por clase')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

# porcentaje
axes[1].pie(target_counts.values, labels=['no default\\n(92%)', 'default\\n(8%)'],
            colors=['#4C72B0', '#DD8452'], autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('proporción de clases')

plt.suptitle('TARGET — desbalanceo de clases', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

print("nota: un modelo que prediga siempre 0 tendría 92% accuracy. por eso usamos Gini/AUC-ROC.")
"""
    )
)

cells.append(md("## 3. tipos de variables"))

cells.append(
    code(
        """\
num_cols = app.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = app.select_dtypes(include='object').columns.tolist()

# quitamos el id y el target de los numéricos para análisis
feature_num = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]
feature_cat = cat_cols

print(f"numéricas  : {len(feature_num)}")
print(f"categóricas: {len(feature_cat)}")
print(f"target + id: 2")
print(f"total      : {app.shape[1]}")
"""
    )
)

cells.append(md("## 4. análisis de nulos"))

cells.append(
    code(
        """\
missing = (
    app.isnull().sum()
    .rename('n_missing')
    .to_frame()
    .assign(pct=lambda x: 100 * x['n_missing'] / len(app))
    .query('n_missing > 0')
    .sort_values('pct', ascending=False)
)

print(f"{len(missing)} columnas con nulos (de {app.shape[1]} totales)")
print(f"  > 60% nulos: {(missing['pct'] > 60).sum()}")
print(f"  > 30% nulos: {(missing['pct'] > 30).sum()}")
print(f"  > 0%  nulos: {len(missing)}")
"""
    )
)

cells.append(
    code(
        """\
# top 30 columnas con más nulos
missing.head(30).style.background_gradient(subset=['pct'], cmap='Reds')
"""
    )
)

cells.append(
    code(
        """\
# heatmap de patrón de nulos — columnas con > 40% missing
high_missing_cols = missing[missing['pct'] > 40].index.tolist()
sample = app[high_missing_cols].isnull().astype(int).sample(500, random_state=42)

fig, ax = plt.subplots(figsize=(14, max(4, len(high_missing_cols) * 0.35)))
sns.heatmap(sample.T, cbar=False, ax=ax, cmap='Blues_r', linewidths=0)
ax.set_title(f'patrón de nulos — {len(high_missing_cols)} columnas con >40% missing (muestra 500 filas)')
ax.set_xlabel('solicitudes')
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        """\
## 5. ¿son los nulos informativos?

Esta es la pregunta clave. Hay tres tipos de missings:
- **MCAR** (missing completely at random): el nulo no tiene relación con el target. Se puede imputar sin pérdida de información.
- **MAR** (missing at random): el nulo depende de otras variables observadas.
- **MNAR** (missing not at random): el nulo está relacionado con el valor del propio campo o con el target.

En crédito, muchos nulos son **MNAR**: si alguien no tiene historial en el bureau externo, *eso en sí mismo* es una señal de riesgo. El nulo *es* la información.

La forma de detectarlo: comparar la tasa de default cuando el campo es nulo vs cuando tiene valor."""
    )
)

cells.append(
    code(
        """\
# para cada columna con nulos, comparamos default rate: nulo vs presente
results = []
for col in missing.index[:30]:  # top 30 con más nulos
    mask = app[col].isnull()
    if mask.sum() == 0 or mask.all():
        continue
    default_null    = app.loc[mask,  'TARGET'].mean()
    default_present = app.loc[~mask, 'TARGET'].mean()
    results.append({
        'col':              col,
        'pct_missing':      round(missing.loc[col, 'pct'], 1),
        'default_null':     round(default_null, 4),
        'default_present':  round(default_present, 4),
        'diff':             round(default_null - default_present, 4)
    })

mcar_df = (
    pd.DataFrame(results)
    .sort_values('diff', key=abs, ascending=False)
)
mcar_df.style.background_gradient(subset=['diff'], cmap='RdYlGn_r', vmin=-0.1, vmax=0.1)
"""
    )
)

cells.append(md("## 6. distribuciones — variables numéricas clave"))

cells.append(
    code(
        """\
# variables financieras principales — las más relevantes para credit scoring
key_num = [
    'AMT_INCOME_TOTAL',   # ingresos del solicitante
    'AMT_CREDIT',         # importe del crédito solicitado
    'AMT_ANNUITY',        # cuota anual
    'AMT_GOODS_PRICE',    # precio del bien financiado
    'DAYS_BIRTH',         # edad (en días negativos desde hoy)
    'DAYS_EMPLOYED',      # antigüedad laboral (negativo = empleado, positivo = anomalía)
    'DAYS_REGISTRATION',  # días desde registro en Home Credit
    'EXT_SOURCE_1',       # score externo 1 (anonimizado)
    'EXT_SOURCE_2',       # score externo 2 — muy predictivo
    'EXT_SOURCE_3',       # score externo 3
]
key_num = [c for c in key_num if c in app.columns]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, col in enumerate(key_num):
    data_0 = app.loc[app['TARGET'] == 0, col].dropna()
    data_1 = app.loc[app['TARGET'] == 1, col].dropna()

    axes[i].hist(data_0, bins=50, alpha=0.6, color='#4C72B0', label='no default', density=True)
    axes[i].hist(data_1, bins=50, alpha=0.6, color='#DD8452', label='default',    density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_yticks([])
    if i == 0:
        axes[i].legend(fontsize=8)

plt.suptitle('distribución de variables clave por TARGET', y=1.01, fontsize=12)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(md("## 7. variables categóricas — tasa de default por categoría"))

cells.append(
    code(
        """\
# variables categóricas y su relación con el target
key_cat = [
    'NAME_CONTRACT_TYPE',    # tipo de contrato
    'CODE_GENDER',           # género
    'FLAG_OWN_CAR',          # tiene coche
    'FLAG_OWN_REALTY',       # tiene propiedad
    'NAME_INCOME_TYPE',      # tipo de ingresos
    'NAME_EDUCATION_TYPE',   # nivel educativo
    'NAME_FAMILY_STATUS',    # estado civil
    'NAME_HOUSING_TYPE',     # tipo de vivienda
    'OCCUPATION_TYPE',       # ocupación
    'ORGANIZATION_TYPE',     # tipo de organización empleadora
]
key_cat = [c for c in key_cat if c in app.columns]

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
axes = axes.flatten()

for i, col in enumerate(key_cat):
    default_rate = app.groupby(col)['TARGET'].mean().sort_values(ascending=False)
    counts       = app[col].value_counts()

    bars = axes[i].barh(range(len(default_rate)), default_rate.values, color='#DD8452', alpha=0.75)
    axes[i].set_yticks(range(len(default_rate)))
    axes[i].set_yticklabels([f"{k} ({counts.get(k,0)/1000:.1f}K)" for k in default_rate.index], fontsize=7)
    axes[i].set_title(col, fontsize=9)
    axes[i].axvline(app['TARGET'].mean(), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

plt.suptitle('tasa de default por categoría (línea = media global)', y=1.01, fontsize=12)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        "## 8. EXT_SOURCE — los features más predictivos\n\nLos tres scores externos son generalmente las variables con mayor poder predictivo en este dataset. Vale la pena entender bien su distribución."
    )
)

cells.append(
    code(
        """\
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
ext_cols = [c for c in ext_cols if c in app.columns]

fig, axes = plt.subplots(1, len(ext_cols), figsize=(14, 4))

for i, col in enumerate(ext_cols):
    for target_val, color, label in [(0, '#4C72B0', 'no default'), (1, '#DD8452', 'default')]:
        data = app.loc[app['TARGET'] == target_val, col].dropna()
        axes[i].hist(data, bins=40, alpha=0.6, color=color, label=label, density=True)

    axes[i].set_title(f'{col}\\nmissing: {100*app[col].isnull().mean():.1f}%')
    axes[i].set_xlabel('score externo (0-1)')
    axes[i].set_yticks([])
    axes[i].legend(fontsize=8)

plt.suptitle('scores externos — separación entre buenos y malos pagadores', fontsize=12)
plt.tight_layout()
plt.show()

# correlación con target
for col in ext_cols:
    r = app[col].corr(app['TARGET'])
    print(f"{col}: correlación con TARGET = {r:.4f}")
"""
    )
)

cells.append(
    md(
        """\
## 9. DAYS_BIRTH y DAYS_EMPLOYED — anomalías conocidas

`DAYS_BIRTH` son los días de vida del solicitante en negativo (p.ej. -15000 ≈ 41 años).
`DAYS_EMPLOYED` tiene una anomalía conocida: 365243 significa "desempleado/jubilado" — no es un valor real."""
    )
)

cells.append(
    code(
        """\
# convertimos DAYS_BIRTH a edad en años para que sea más legible
app['AGE_YEARS'] = -app['DAYS_BIRTH'] / 365

# DAYS_EMPLOYED: 365243 es un código para "no empleado"
days_emp_clean = app['DAYS_EMPLOYED'].replace(365243, np.nan)
app['YEARS_EMPLOYED'] = -days_emp_clean / 365

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for target_val, color, label in [(0, '#4C72B0', 'no default'), (1, '#DD8452', 'default')]:
    mask = app['TARGET'] == target_val
    axes[0].hist(app.loc[mask, 'AGE_YEARS'].dropna(), bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[1].hist(app.loc[mask, 'YEARS_EMPLOYED'].dropna(), bins=40, alpha=0.6, color=color, density=True)

axes[0].set_title('edad del solicitante (años)')
axes[1].set_title('antigüedad laboral (años, sin anomalía 365243)')
axes[0].legend()

for ax in axes:
    ax.set_yticks([])

plt.tight_layout()
plt.show()

anomaly_pct = 100 * (app['DAYS_EMPLOYED'] == 365243).mean()
print(f"DAYS_EMPLOYED == 365243: {anomaly_pct:.1f}% de los registros — se tratará como nulo")
"""
    )
)

cells.append(md("## 10. correlaciones entre variables numéricas"))

cells.append(
    code(
        """\
# matriz de correlación — top 15 features por correlación absoluta con target
top_corr_cols = (
    app[feature_num + ['TARGET']]
    .corr()['TARGET']
    .drop('TARGET')
    .abs()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)

corr_matrix = app[top_corr_cols + ['TARGET']].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('correlaciones — top 15 features más correladas con TARGET')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md("## resumen Día 1\n\nAnota aquí los hallazgos más importantes para la siguiente fase.")
)

cells.append(
    code(
        """\
print("checklist Día 1:")
print("  [ ] data audit completado — tamaños y memoria")
print("  [ ] target: ~8% default — dataset muy desbalanceado")
print("  [ ] X columnas con nulos > 60% identificadas")
print("  [ ] nulos MNAR detectados (nulo correlacionado con target)")
print("  [ ] EXT_SOURCE_1/2/3 — features más predictivos identificados")
print("  [ ] DAYS_EMPLOYED anomalía 365243 — tratar como nulo")
print("  [ ] features categóricos con mayor diferencia de default rate identificados")
"""
    )
)

# construcción del notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": ".venv"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent.parent / "notebooks" / "01_eda.ipynb"
out.write_text(json.dumps(notebook, indent=2, ensure_ascii=False))
print(f"notebook generado: {out}")
