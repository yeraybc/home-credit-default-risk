# Home Credit Default Risk: Credit Scoring End-to-End

Proyecto de portfolio: pipeline completo de credit scoring desde EDA multi-tabla hasta
API desplegada con monitorización de drift.

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
307K solicitudes de crédito, 7 tablas relacionales, target binario (default / no default).

Stack: Python 3.9, pandas, SciPy, matplotlib, seaborn.

## Estado actual del proyecto

```
Fase 1: Setup               - Completado
Fase 2: EDA                 - En curso (3 de 7 tablas cerradas)
Fase 3: Feature Engineering - Pendiente
Fase 4: Modelado            - Pendiente
Fase 5: Interpretabilidad   - Pendiente
Fase 6: API REST            - Pendiente
Fase 7: Docker + MLflow     - Pendiente
Fase 8: Monitoring          - Pendiente
```

## Estructura

```
Home-Credit-Default-Risk/
├── data/
│   ├── raw/        CSVs de Kaggle (no versionados)
│   ├── processed/  datos transformados (Fase 3)
│   └── external/   datos macro (fase futura)
├── notebooks/
│   ├── 01_eda_application_train.ipynb  EDA de la tabla principal
│   ├── 02_eda_bureau.ipynb             EDA de bureau (fases A y B)
│   └── 03_eda_bureau_balance.ipynb     EDA del panel mensual del buró
├── scripts/
│   └── make_notebook.py                generador de notebooks
├── src/
│   ├── data/loader.py                  carga y optimización de memoria de las 7 tablas
│   ├── features/                       evaluación de señal y selección de variables
│   ├── models/                         pendiente
│   └── monitoring/                     pendiente
├── config/
│   ├── config.yaml                     parámetros centralizados
│   ├── bureau_features.yaml            receta de features de bureau
│   └── bureau_balance_features.yaml    receta de features de bureau_balance
├── Makefile
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/yeraybc/home-credit-default-risk
cd home-credit-default-risk

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

Los datos de Kaggle van en `data/raw/` y no se versionan.
Descarga desde: https://www.kaggle.com/c/home-credit-default-risk/data

## Uso

```bash
# ejecutar el EDA
jupyter lab notebooks/

# comandos disponibles
make help
```

## Datos

| tabla | filas | descripción |
|---|---|---|
| application_train | 307K | solicitudes con TARGET |
| bureau | 1.7M | créditos en entidades externas |
| bureau_balance | 27M | saldos mensuales del bureau |
| previous_application | 1.7M | solicitudes previas en Home Credit |
| POS_CASH_balance | 10M | balances POS y cash anteriores |
| credit_card_balance | 3.8M | balances de tarjetas anteriores |
| installments_payments | 13.6M | historial de pagos de cuotas |
