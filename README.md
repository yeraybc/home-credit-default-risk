# Home Credit Default Risk

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/yeraybc/home-credit-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/yeraybc/home-credit-default-risk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Last commit](https://img.shields.io/github/last-commit/yeraybc/home-credit-default-risk)

Pipeline de credit scoring end-to-end sobre el dataset de Home Credit: 307.511 solicitudes de crédito, 7 tablas relacionales y un target binario de impago.

## Índice

- [Motivación detrás del proyecto](#motivación-detrás-del-proyecto)
- [Estado del proyecto](#estado-del-proyecto)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Fuente de datos](#fuente-de-datos)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Stack técnico](#stack-técnico)
- [Limitaciones reconocidas](#limitaciones-reconocidas)
- [Cómo ejecutar el proyecto en local](#cómo-ejecutar-el-proyecto-en-local)
- [Autor](#autor)
- [Licencia](#licencia)

## Motivación detrás del proyecto

Un EDA de Kaggle sobre este dataset suele durar un notebook: se ordenan los nulos, se imputan con la media, se capa todo en el percentil 99 y se pasa al modelo. El problema es que en credit risk esas tres decisiones destruyen información antes de que el modelo llegue a verla, y ninguna de las tres se puede tomar mirando la distribución de una variable en global.

Este proyecto va en la dirección contraria: cada decisión sobre un nulo, un valor extremo o una agregación se toma cruzando la variable contra la tasa de default, separando por grupo y validando con un contraste estadístico. No "¿cuántos valores faltan?" sino "¿por qué faltan y qué dice esa ausencia?". No "¿esto es un outlier?" sino "¿es un error de captura o es otra población?".

> **Hallazgo principal:** la variable más predictiva del historial de buró no existía como columna. `AMT_CREDIT_SUM_DEBT` (deuda) y `AMT_CREDIT_SUM` (crédito concedido) tienen cada una una correlación con el target por debajo de 0,011, o sea ruido. Su cociente, el ratio de sobreendeudamiento, da un rank-biserial de **0,1828**, la señal continua más fuerte de la tabla, con un gradiente monótono del **5,64 % al 18,61 %** de default. Deber 50.000 no significa nada; deber 50.000 sobre un crédito de 60.000 lo significa todo.

Y una segunda lección que atraviesa toda la fase de agregación: al comprimir 1,7 millones de créditos en una fila por cliente, el `fillna(0)` que hace cualquier pipeline por defecto mete a los 44.020 clientes **sin historial** en el grupo "0" de todas las features a la vez. Ese grupo defaultea al 10,12 % frente al 7,73 % de base, así que contamina la referencia de cada comparación. Corregirlo no cambia decimales: invierte el ranking de variables.

## Estado del proyecto

```
Fase 1: Setup               ✅ Completado
Fase 2: EDA                 🔄 En curso — 3 de 7 tablas cerradas
Fase 3: Feature Engineering ⏳ Pendiente
Fase 4: Modelado            ⏳ Pendiente
Fase 5: Interpretabilidad   ⏳ Pendiente
Fase 6: API REST            ⏳ Pendiente
Fase 7: Docker + MLflow     ⏳ Pendiente
Fase 8: Monitorización      ⏳ Pendiente
```

El EDA es la fase en curso y **no modifica los datos**. Todo lo que aquí aparece como "imputar", "cappear" o "crear feature" es una decisión razonada y justificada, no una transformación aplicada: la materialización real ocurre en el pipeline de la Fase 3, ajustado sobre el split de entrenamiento para no filtrar información del target.

## Estructura del proyecto

```
home-credit-default-risk/
├── notebooks/                            # La investigación, en orden de ejecución
│   ├── 01_eda_application_train.ipynb    # Tabla principal: 307K solicitudes, 122 variables
│   ├── 02_eda_bureau.ipynb               # Historial en otras entidades: 1,7M créditos
│   └── 03_eda_bureau_balance.ipynb       # Panel mensual del buró: 27,3M filas
├── src/
│   ├── data/loader.py                    # Carga de las 7 tablas y reducción de memoria
│   ├── features/
│   │   ├── eval.py                       # EvaluadorSenal: señal de features agregadas
│   │   ├── selection.py                  # Cardinalidad y recomendación de encoding
│   │   └── build_features.py             # Pendiente: Fase 3
│   ├── models/                           # Pendiente: Fase 4
│   └── monitoring/                       # Pendiente: Fase 8
├── config/
│   ├── config.yaml                       # Parámetros centralizados del proyecto
│   ├── bureau_features.yaml              # Receta de features de bureau
│   └── bureau_balance_features.yaml      # Receta de features de bureau_balance
├── data/
│   ├── raw/                              # CSVs de Kaggle (no versionados, 2,5 GB)
│   ├── processed/                        # Datos transformados (Fase 3)
│   └── external/                         # Enriquecimiento macro (fase futura)
├── scripts/make_notebook.py              # Generador de notebooks
├── .github/
│   ├── workflows/ci.yml                  # Verificación en cada push y PR
│   └── scripts/check.py                  # Sintaxis y dependencias, sin instalar el stack
├── Makefile
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

Las recetas de `config/` no son documentación: se exportan desde el ranking consolidado de cada notebook, con el nombre de cada feature, su agregación, la población sobre la que se midió, su efecto y la decisión tomada. El pipeline de Fase 3 las consume desde ahí, de modo que no puede desincronizarse de lo que concluyó el EDA.

## Fuente de datos

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). Siete tablas relacionales en dos ramas: la del buró de crédito externo y la del historial interno de Home Credit.

| Tabla | Filas | Relación | EDA |
|---|---|---|---|
| `application_train` | 307.511 | Principal, contiene el TARGET | ✅ |
| `bureau` | 1.716.428 | Muchos-a-uno con la principal | ✅ |
| `bureau_balance` | 27.299.925 | Muchos-a-uno con `bureau` (panel mensual) | ✅ |
| `previous_application` | 1,7M | Muchos-a-uno con la principal | ⏳ |
| `POS_CASH_balance` | 10M | Muchos-a-uno con `previous_application` | ⏳ |
| `credit_card_balance` | 3,8M | Muchos-a-uno con `previous_application` | ⏳ |
| `installments_payments` | 13,6M | Muchos-a-uno con `previous_application` | ⏳ |

El target está desbalanceado: **24.825 impagos sobre 307.511 solicitudes, un 8,07 %**. Las tablas auxiliares no tienen target propio, así que el análisis a nivel fila lo hereda vía merge y el análisis a nivel cliente lo recupera tras agregar.

## Metodología

Cada tabla auxiliar se recorre en dos fases, porque un modelo de scoring no predice créditos, predice clientes.

**Fase A — nivel fila**, con el target heredado: perfil e integridad por código, distribución de la variable marco, clasificación de variables, distribuciones, missings, correlaciones, análisis temporal y outliers.

**Fase B — nivel cliente**, agregando por `SK_ID_CURR` y cruzando contra el target real. `bureau_balance` necesita doble agregación (crédito-mes → crédito → cliente) y un enlace triple para llegar al target.

Los criterios que se aplican de forma sistemática:

1. **Hipótesis antes del resultado.** Cada hipótesis se plantea explícitamente antes de mirar los datos y se documenta cuando la evidencia la refuta. Varias clasificaciones cambiaron por esta vía.
2. **Doble criterio: estadístico y de negocio.** Ninguna decisión sale de un p-valor solo. Con muestras de millones de filas casi todo es significativo, así que el umbral vinculante es el tamaño del efecto (|delta| > 2 pp, rank-biserial) y Bonferroni cuando se comparan más de tres variables a la vez.
3. **Nunca en global.** Cada nulo y cada cola se cruzan contra la tasa de default separando por grupo. Un delta que existe en global puede desaparecer al controlar por una tercera variable, y al revés.
4. **Control de composición en toda feature agregada.** Cada variable se mide dos veces, global y condicionada al flag de presencia, y se reportan ambas.
5. **Redundancia medida sobre la feature agregada, no sobre la columna cruda.** Y con el estadístico que corresponde: Pearson si la variable entra como magnitud, V de Cramér si entra como flag. Confundirlos lleva a conclusiones opuestas.
6. **Validez de dominio separada de la señal.** Un importe de 396 millones es un error de captura y se revierte aunque su cola tenga señal, porque contamina las sumas agregadas. Un extremo plausible con señal se conserva.

Contrastes usados: Mann-Whitney U con rank-biserial para continuas, z-test de proporciones para flags, Chi² y V de Cramér para categóricas, Fisher exacto para eventos raros, y Bonferroni sobre la familia real de comparaciones.

## Resultados

Todas las cifras proceden de los outputs de los notebooks de este repositorio.

### La ausencia de un registro es una variable

| Qué se mide | Resultado |
|---|---|
| Clientes sin ningún crédito en el buró | 44.020 (**14,31 %**) |
| Su tasa de default frente a la de quienes sí tienen historial | **10,12 % vs 7,73 %** (+2,39 pp, z-test p = 2,33e-65) |
| Solapamiento con los nulos de consultas al buró de la tabla principal | **100 %** contenidos, 94,32 % de solapamiento |

No tener pasado crediticio no es un hueco a rellenar: es una de las señales más limpias de la tabla. Y como los 41.519 clientes con las consultas al buró en nulo están contenidos al 100 % dentro de estos 44.020, dos señales que parecían distintas eran la misma: un solo flag las resume y evita duplicar información en el modelo.

Lo que **no** se cumplió fue la relación simple entre número de créditos y riesgo. El patrón tiene forma de U: sin historial 10,12 %, mínimo de 7,40 % en la franja de 3-5 créditos, y repunte a 8,18 % a partir de 11. Ni la experiencia ni el sobreendeudamiento mandan solos.

### El grupo vacío contamina la medición

Al agregar, el `fillna(0)` mete a los clientes sin filas en el grupo 0 de cada feature. La distorsión no es uniforme: **depende del signo de la variable**.

| Feature | Medición global | Condicionada a tener historial |
|---|---|---|
| `BUREAU_CLOSED_COUNT` (protectora) | 0,1085 | **0,0948** — la señal estaba inflada |
| `BUREAU_ACTIVE_COUNT` (penalizadora) | 0,0587 | **0,1208** — la señal estaba enmascarada |
| `BUREAU_LOAN_COUNT` (suma de ambas) | 0,0465 | **0,0098** — ruido puro |

Más créditos cerrados protege, más créditos activos penaliza. Los sin-historial, metidos en el bin 0, inflaban el gradiente de la primera y aplanaban el de la segunda, hasta invertir cuál era el conteo primario. Y sumar las dos en una sola variable de "créditos totales" cancela las dos fuerzas: colapsa a 0,0098, ruido. Dos señales reales que, mal combinadas, se anulan.

El mismo control rescató una variable ya descartada: la flag de registro del buró actualizado en los últimos 6 meses daba +0,29 pp y no pasaba Bonferroni, porque el 50,1 % de su grupo "dormante" eran los sin-historial. Condicionada rinde **+2,55 pp**, y sobrevive en los cinco estratos de longitud de historial, así que la señal es propia y no un eco de la antigüedad.

### Una flag mide cómo la construiste, no lo que dice su nombre

`BUREAU_HAS_ANY_OVERDUE` se construyó como "máximo histórico de mora > 0". Con un 64,73 % de nulos en esa columna, el nulo evalúa a `False`, de modo que la variable no dice "tuvo mora" sino **"tuvo mora reportada"**: de los 3.334 clientes con mora **activa**, 1.834 (el 55 %) figuraban como limpios en su historial, lo cual es imposible por definición.

Partiendo el grupo correctamente, la señal aparece ordenada:

| Grupo | n | Default |
|---|---|---|
| Sin mora | 191.217 | **7,02 %** |
| Mora histórica ya saldada | 68.940 | **9,30 %** |
| Mora activa hoy | 3.334 | **16,20 %** |

La misma estructura tripartita nulo/cero/valor aparece en la cuota del buró, y ahí el orden **no** es monótono: sin cuota reportada 7,50 %, cuota real 8,88 %, y en medio los que reportan cuota **a cero** con un 6,32 %, el grupo más seguro de los tres. El cero no es ausencia de información, es un crédito saldado. Una flag `notna()` fusiona el grupo protector con el arriesgado y diluye la señal a +0,75 pp, frente a los +2,56 pp del contraste real: es una codificación estrictamente peor.

### El tamaño de un grupo no dice nada sobre su valor

De las 43 variables con cola analizadas en la tabla principal, 20 pasaron el filtro tras Bonferroni, y lo hicieron **en dos direcciones opuestas**: los outliers financieros y de scores tienen delta negativo (`EXT_SOURCE_2` −6,28 pp: son los mejores pagadores del dataset) y los de conteo familiar y círculo social lo tienen positivo (`DEF_60_CNT_SOCIAL_CIRCLE` +6,84 pp). Una regla única de capping habría borrado las dos.

El caso extremo está en `bureau`: el límite de crédito negativo en tarjeta afecta a **323 clientes, el 0,123 %** de los que tienen historial, y arrastra un delta de **+12,72 pp** (20,43 % de default), el más alto de la tabla. Severidad máxima e impacto de cartera casi nulo: cuál de las dos manda lo decide el Information Value en la Fase 3, no la severidad sola.

### El panel mensual aporta la dimensión que el snapshot no ve

`bureau_balance` cubre solo al 29,99 % de los clientes, y su mera presencia **no discrimina** (+0,10 pp, p = 0,35): la disponibilidad del panel la decide el buró, no el cliente. Lo que sí discrimina es **cuándo** ocurrió la mora.

| Recencia del último impago | Default |
|---|---|
| Hace más de 24 meses | 7,54 % |
| En los últimos 0-3 meses | **13,20 %** |

La flag de impago en los últimos 6 meses da **+4,89 pp**, por encima de la de fallido (+2,94 pp) y de la de cualquier impago (+2,74 pp). La mora reciente es un predictor más afilado que la mora grave o la histórica. Y complementa a `bureau` en vez de duplicarlo: V de Cramér 0,3945 entre ambas, y el panel **rescata 16.000 clientes** con impago mensual que la mora agregada del snapshot no marcaba.

En sentido contrario, la variable con el rank-biserial más alto del panel (0,1502) se descarta: correlaciona a **−0,7451** con la longitud de historial que ya aporta `bureau`. La señal más fuerte en crudo no siempre entra; hay que restarle lo que ya está en otra tabla.

## Stack técnico

- **Lenguaje:** Python 3.9
- **Análisis:** `pandas`, `numpy`
- **Estadística:** `scipy` (Mann-Whitney, Chi², Fisher, z-test de proporciones)
- **Visualización:** `matplotlib`, `seaborn`
- **Configuración:** `PyYAML`
- **CI:** GitHub Actions

Previsto para las fases siguientes: `scikit-learn`, `XGBoost`, `LightGBM`, `SHAP`, `FastAPI`, `Docker`, `MLflow` y `Evidently`.

## Limitaciones reconocidas

- **Las tasas de default están calculadas sobre `application_train` completo.** Sirven para explorar y fijar la receta, pero cualquier decisión basada en la asociación con el target debe reconfirmarse sobre el split de entrenamiento en la Fase 3. Los descartes por redundancia estructural (correlación entre features) son target-independientes y sí quedan firmes.
- **Pseudoreplicación en el análisis a nivel fila.** Los tests de la Fase A tratan 1,46 millones de créditos como observaciones independientes cuando están anidados en 263.491 clientes que comparten target. Los p-valores resultantes son anticonservadores, y por eso las decisiones se apoyan en el tamaño del efecto y en la revalidación a nivel cliente.
- **Causalidad.** La asociación entre edad y default (12,29 % en la franja 20-25 frente a 3,66 % en 65-70) es consistente con la estabilidad financiera acumulada, pero no descarta un sesgo de selección: quien impagó joven puede no seguir en el dataset de mayor.
- **Eventos raros.** Varias de las señales más severas viven en cientos de casos (323 clientes con límite negativo, 21 créditos en "bad debt"). Se reportan siempre con su n, y su estabilidad fuera de muestra está por demostrar.
- **Faltan cuatro tablas.** La rama de `previous_application` está sin analizar, así que las conclusiones cubren la solicitud y el buró externo, no el historial interno de Home Credit.

## Cómo ejecutar el proyecto en local

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/yeraybc/home-credit-default-risk.git
   cd home-credit-default-risk
   ```

2. **Crea el entorno e instala las dependencias:**
   ```bash
   make setup
   ```
   O a mano, si prefieres no usar el Makefile:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt && pip install -e .
   ```
   El `pip install -e .` es lo que permite que los notebooks hagan `from src.data.loader import ...` sin tocar `sys.path`.

3. **Descarga los datos.** No se versionan: son 2,5 GB de CSVs sujetos a los términos de la competición. Descárgalos desde [la página del dataset](https://www.kaggle.com/c/home-credit-default-risk/data) y descomprímelos en `data/raw/`, que viaja vacía con un `.gitkeep`.

4. **Ejecuta el análisis:**
   ```bash
   jupyter lab notebooks/
   ```
   Los notebooks 02 y 03 corren limpios de arriba abajo con Restart & Run All. `bureau_balance` son 27 millones de filas, así que conviene tener RAM disponible: `reduce_mem_usage` en [src/data/loader.py](src/data/loader.py) baja los dtypes al mínimo sin perder rango y ahorra más del 60 % de memoria.

5. **Comandos disponibles:**
   ```bash
   make help
   ```

## Autor

**Yeray Benito Calviño**
Data Science student, Universidad Complutense de Madrid
[LinkedIn](https://www.linkedin.com/in/yeraybenit0) · [GitHub](https://github.com/yeraybc)

## Licencia

Distribuido bajo licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
