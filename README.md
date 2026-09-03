# Home Credit Default Risk

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/yeraybc/home-credit-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/yeraybc/home-credit-default-risk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Last commit](https://img.shields.io/github/last-commit/yeraybc/home-credit-default-risk)

Pipeline de credit scoring end-to-end sobre el dataset de Home Credit: 307.511 solicitudes de crédito, un target binario de impago y cuatro tablas relacionales analizadas a fondo.

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

Un EDA de Kaggle sobre este dataset suele durar un notebook: se ordenan los nulos, se imputan con la media, se capa todo en el percentil 99 y se pasa al modelo. El problema es que en credit risk esas tres decisiones destruyen información antes de que el modelo llegue a verla, y ninguna se puede tomar mirando la distribución de una variable en global.

Este proyecto va en la dirección contraria: cada decisión sobre un nulo, un valor extremo o una agregación se toma cruzando la variable contra la tasa de default, separando por grupo y validando con un contraste estadístico. No "¿cuántos valores faltan?" sino "¿por qué faltan y qué dice esa ausencia?". No "¿esto es un outlier?" sino "¿es un error de captura o es otra población?".

> **Hallazgo principal:** las variables más predictivas no existían como columna. `AMT_CREDIT_SUM_DEBT` (deuda) y `AMT_CREDIT_SUM` (crédito concedido) correlacionan con el target por debajo de 0,011 cada una, o sea ruido. Su cociente da un rank-biserial de **0,1828**, con un gradiente monótono del **5,64% al 18,61%** de default. Deber 50.000 no significa nada; deber 50.000 sobre un crédito de 60.000 lo significa todo. El mismo mecanismo se repite en el historial interno, con el coste implícito del crédito (0,1334) y la sobreconcesión (0,1458).

Y una segunda lección que atraviesa la fase de agregación: al comprimir 1,7 millones de créditos en una fila por cliente, el `fillna(0)` que hace cualquier pipeline por defecto mete a los 44.020 clientes **sin historial** en el grupo "0" de todas las features a la vez. Ese grupo defaultea al 10,12% frente al 7,73% de base, así que contamina la referencia de cada comparación. Corregirlo no cambia decimales: invierte el ranking de variables.

## Estado del proyecto

```
Fase 1: Setup               ✅ Completado
Fase 2: EDA                 ✅ Cerrado, 4 de 4 tablas del alcance
Fase 3: Feature Engineering ⏳ Pendiente
Fase 4: Modelado            ⏳ Pendiente
Fase 5: Interpretabilidad   ⏳ Pendiente
Fase 6: API REST            ⏳ Pendiente
Fase 7: Docker + MLflow     ⏳ Pendiente
Fase 8: Monitorización      ⏳ Pendiente
```

Las cuatro tablas están analizadas, auditadas con recomputación desde el crudo y con su receta de features exportada. Queda la síntesis maestra del EDA antes de arrancar la Fase 3.

El EDA **no modifica los datos**. Todo lo que aquí aparece como "imputar", "cappear" o "crear feature" es una decisión razonada, no una transformación aplicada: la materialización ocurre en el pipeline de la Fase 3, ajustado sobre el split de entrenamiento para no filtrar información del target.

## Estructura del proyecto

```
home-credit-default-risk/
├── notebooks/                                 # La investigación, en orden de ejecución
│   ├── 01_eda_application_train.ipynb         # Tabla principal: 307K solicitudes, 122 variables
│   ├── 02_eda_bureau.ipynb                    # Historial en otras entidades: 1,7M créditos
│   ├── 03_eda_bureau_balance.ipynb            # Panel mensual del buró: 27,3M filas
│   └── 04_eda_previous_application.ipynb      # Historial interno: 1,7M solicitudes previas
├── src/
│   ├── data/loader.py                         # Carga de las tablas y reducción de memoria
│   ├── features/
│   │   ├── eval.py                            # EvaluadorSenal: señal de features agregadas
│   │   ├── selection.py                       # Cardinalidad y recomendación de encoding
│   │   ├── recipes.py                         # Exportación de la receta de cada tabla
│   │   └── build_features.py                  # Pendiente: Fase 3
│   ├── models/                                # Pendiente: Fase 4
│   └── monitoring/                            # Pendiente: Fase 8
├── config/
│   ├── config.yaml                            # Parámetros centralizados del proyecto
│   ├── bureau_features.yaml                   # Receta de features de bureau
│   ├── bureau_balance_features.yaml           # Receta de features de bureau_balance
│   └── previous_application_features.yaml     # Receta de features de previous_application
├── data/
│   ├── raw/                                   # CSVs de Kaggle (no versionados, 2,5 GB)
│   ├── processed/                             # Datos transformados (Fase 3)
│   └── external/                              # Enriquecimiento macro (fase futura)
├── tests/
│   ├── test_loader.py                         # Contrato de reduce_mem_usage y data_audit
│   ├── test_eval.py                           # Control de composición de EvaluadorSenal
│   └── test_recipes.py                        # Esquema y vocabulario de las tres recetas
├── .github/
│   ├── workflows/ci.yml                       # Verificación en cada push y PR
│   └── scripts/check.py                       # Sintaxis, dependencias, kernel y tipografía
├── Makefile
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

Las recetas de `config/` no son documentación: se exportan desde el ranking consolidado de cada notebook con `exportar_receta()`, así que no pueden desincronizarse de lo que concluyó el EDA, y el pipeline de Fase 3 las consume desde ahí. Las tres comparten esquema, con vocabulario cerrado en `decision` (un estado que no encaje rompe el export en vez de colarse), un campo `firmeza` que separa lo target-independiente de lo que hay que remedir sobre el split, y `control: true` en las features que se quedan pase lo que pase con su capacidad predictiva.

## Fuente de datos

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). Siete tablas relacionales en dos ramas: la del buró de crédito externo y la del historial interno de Home Credit.

| Tabla | Filas | Relación | EDA |
|---|---|---|---|
| `application_train` | 307.511 | Principal, contiene el TARGET | ✅ |
| `bureau` | 1.716.428 | Muchos-a-uno con la principal | ✅ |
| `bureau_balance` | 27.299.925 | Muchos-a-uno con `bureau` (panel mensual) | ✅ |
| `previous_application` | 1.670.214 | Muchos-a-uno con la principal | ✅ |
| `POS_CASH_balance` | 10.001.358 | Muchos-a-uno con `previous_application` | Fuera de alcance |
| `credit_card_balance` | 3.840.312 | Muchos-a-uno con `previous_application` | Fuera de alcance |
| `installments_payments` | 13.605.401 | Muchos-a-uno con `previous_application` | Fuera de alcance |

El alcance se fijó en cuatro tablas de forma deliberada: entre analizar las siete y terminar el pipeline hasta deployment, este proyecto prioriza lo segundo. Las cuatro cubren las tres fuentes de información distintas del problema (la solicitud que se evalúa, el buró externo y el historial interno del prestamista), mientras que las tres restantes son todas panel de pagos de créditos ya concedidos, así que aportan una cuarta dimensión y no tres.

El target está desbalanceado: **24.825 impagos sobre 307.511 solicitudes, un 8,07%**. Las tablas auxiliares no tienen target propio, así que el análisis a nivel fila lo hereda vía merge y el de nivel cliente lo recupera tras agregar.

## Metodología

Cada tabla auxiliar se recorre en dos fases, porque un modelo de scoring no predice créditos, predice clientes. La **Fase A** trabaja a nivel fila con el target heredado (integridad, distribuciones, missings, correlaciones, eje temporal y outliers) y la **Fase B** agrega por `SK_ID_CURR` y cruza contra el target real. `bureau_balance` necesita doble agregación y un enlace triple para llegar al target.

Los criterios que se aplican de forma sistemática:

1. **Hipótesis antes del resultado**, planteada explícitamente y documentada cuando la evidencia la refuta.
2. **Doble criterio, estadístico y de negocio.** Con millones de filas casi todo es significativo, así que el umbral vinculante es el tamaño del efecto (|delta| > 2pp, rank-biserial), con Bonferroni sobre la familia real de comparaciones emitidas.
3. **Nunca en global.** Un delta que existe en global puede desaparecer al controlar por una tercera variable, y al revés.
4. **Control de composición en toda feature agregada:** se mide global y condicionada al flag de presencia, y se reportan las dos.
5. **Redundancia medida sobre la feature agregada y no sobre la columna cruda**, con el estadístico que corresponde: Pearson si entra como magnitud, V de Cramér si entra como flag.
6. **Validez de dominio separada de la señal.** Un importe de 396 millones es un error de captura y se revierte aunque su cola tenga señal; un extremo plausible con señal se conserva.
7. **La medición que sostiene la decisión es la que se exporta a la receta**, no la última registrada ni la única que pasó por el acumulador. Las auditorías encontraron tres casos donde no coincidían.

Contrastes: Mann-Whitney U con rank-biserial para continuas, z-test de proporciones para flags, Chi² y V de Cramér para categóricas, y Fisher exacto para eventos raros.

## Resultados

Todas las cifras proceden de los outputs de los notebooks de este repositorio.

### La ausencia de un registro es una variable, y no significa lo mismo en cada fuente

| Qué se mide | Resultado |
|---|---|
| Clientes sin ningún crédito en el buró | 44.020 (14,31%), **10,12%** de default frente al 7,73% |
| Clientes sin ninguna solicitud previa al prestamista | 16.454 (5,35%), **5,96%** frente al 8,19% |
| Solapamiento entre las dos ausencias | V de Cramér **0,0047**, solo 2.470 clientes (0,80%) faltan en las dos |

Las dos ausencias tienen **signo opuesto**: la del buró es opacidad y penaliza, la del historial interno es un cliente que llega nuevo y protege. El cruce ordena el riesgo de forma aditiva, del 5,72% de quien tiene buró y no previas al 10,29% de quien tiene previas y no buró, así que las dos banderas se conservan porque miden fenómenos distintos. Lo que sí es redundante es otra cosa: los 41.519 clientes con las consultas al buró en nulo están contenidos al 100% dentro de los 44.020 sin historial.

Lo que **no** se cumplió fue la relación simple entre número de créditos y riesgo: el patrón tiene forma de U en las dos tablas, del 10,12% sin historial al mínimo de 7,40% entre 3 y 5 créditos y repunte al 8,18% a partir de 11.

### El grupo vacío contamina la medición, y el conteo tenía la ventana escondida

| Feature | Medición global | Condicionada a tener historial |
|---|---|---|
| `BUREAU_CLOSED_COUNT` (protectora) | 0,1085 | **0,0948**, la señal estaba inflada |
| `BUREAU_ACTIVE_COUNT` (penalizadora) | 0,0587 | **0,1208**, la señal estaba enmascarada |
| `BUREAU_LOAN_COUNT` (suma de ambas) | 0,0465 | **0,0098**, ruido puro |

La distorsión del `fillna(0)` **depende del signo de la variable**: los sin-historial, metidos en el bin 0, inflaban el gradiente de la protectora y aplanaban el de la penalizadora, hasta invertir cuál era el conteo primario. Y sumar las dos en una sola variable de créditos totales cancela las dos fuerzas.

Ese 0,0098 tampoco era ruido, era **el conteo sin su ventana**: el buró observa 8,0000 años exactos, así que censura por la izquierda a quien lleva más tiempo pidiendo. Normalizado por los años observados rinde **0,1569**, dieciséis veces más, con gradiente monótono del 6,12% al 16,15%. El mismo mecanismo aparece en el historial interno, donde el conteo pasa de 0,0210 a **0,1149**.

### Una flag mide cómo la construiste, no lo que dice su nombre

`BUREAU_HAS_ANY_OVERDUE` se construyó como "máximo histórico de mora > 0". Con un 64,73% de nulos en esa columna el nulo evalúa a `False`, así que la variable no dice "tuvo mora" sino **"tuvo mora reportada"**: de los 3.334 clientes con mora **activa**, 1.834 figuraban como limpios, lo cual es imposible por definición. Partiendo el grupo bien, la señal aparece ordenada: sin mora 7,02%, mora histórica ya saldada 9,30% y mora activa hoy **16,20%**.

La misma estructura tripartita de nulo, cero y valor aparece en la cuota del buró, y ahí el orden **no** es monótono: sin cuota reportada 7,50%, cuota real 8,88%, y en medio los que reportan cuota **a cero** con un 6,32%, el grupo más seguro de los tres. El cero no es ausencia de información, es un crédito saldado, así que una flag `notna()` fusiona el grupo protector con el arriesgado y diluye la señal a +0,75pp frente a los +2,56pp del contraste real.

### El tamaño de un grupo no dice nada sobre su valor

De las 43 variables con cola analizadas en la tabla principal, 20 pasaron el filtro tras Bonferroni, y lo hicieron **en dos direcciones opuestas**: los outliers financieros y de scores tienen delta negativo (`EXT_SOURCE_2` -6,28pp, son los mejores pagadores del dataset) y los de conteo familiar y círculo social lo tienen positivo (`DEF_60_CNT_SOCIAL_CIRCLE` +6,84pp). Una regla única de capping habría borrado las dos.

El caso extremo está en `bureau`: el límite de crédito negativo en tarjeta afecta a **323 clientes, el 0,123%** de los que tienen historial, con un delta de **+12,72pp**, el más alto de la tabla. Severidad máxima e impacto de cartera casi nulo: cuál de las dos manda lo decide el Information Value en la Fase 3.

### El panel mensual aporta la dimensión que el snapshot no ve

`bureau_balance` cubre solo al 29,99% de los clientes, y su mera presencia **no discrimina** (+0,10pp, p = 0,35): la disponibilidad del panel la decide el buró, no el cliente. Lo que sí discrimina es **cuándo** ocurrió la mora, del 7,54% de quien la tuvo hace más de 24 meses al **13,20%** de quien la tuvo en los últimos 3.

La flag de impago reciente, medida sobre la recencia relativa a la ventana observada de cada crédito, da **+4,92pp sobre 14.288 clientes**, por encima de la de fallido (+2,94pp) y de la de cualquier impago (+2,74pp), y complementa a `bureau` en vez de duplicarlo: V de Cramér 0,3945 entre ambas, y el panel **rescata 16.000 clientes** que la mora agregada del snapshot no marcaba. En sentido contrario, la variable con el rank-biserial más alto del panel (0,1502) se descarta por correlacionar a -0,7451 con la longitud de historial que ya aporta `bureau`: la señal más fuerte en crudo no siempre entra.

### El historial interno responde preguntas que la solicitud no puede

`previous_application` es la tabla de mayor cobertura (94,65% de los clientes) y sus tres señales de primer orden no están en ninguna columna cruda:

| Feature | r_rb | Qué mide |
|---|---|---|
| `PREV_CREDIT_APPLICATION_RATIO` | **0,1458** | Lo concedido frente a lo solicitado |
| `PREV_IMPLIED_COST_MEAN` | **0,1334** | Coste implícito: cuota × plazo / concedido |
| `PREV_REFUSED_RATIO` | **0,1204** | Proporción de solicitudes rechazadas |

El coste implícito sale de la propia multicolinealidad: si cuota, plazo e importe correlacionan entre 0,72 y 0,82, la información nueva está en cómo se relacionan. A nivel solicitud rinde 0,1111 frente al 0,0532 de la cuota, el 0,0526 del plazo y el 0,0287 del importe. Y **no se puede calcular en la solicitud que se evalúa**, porque `application_train` no trae plazo pactado: es un argumento que justifica esta tabla por sí solo.

La sobreconcesión va en contra de la intuición: recibir **menos** de lo pedido protege (6,87%), recibir la cifra exacta sale peor (8,80%) y el exceso grande dispara (**14,12%** por encima de 1,3 veces lo solicitado). Y el rechazo no es un evento homogéneo: dentro de las rechazadas el motivo va del **6,25%** de default del rechazo administrativo al **20,93%** del rechazo por scoring externo, así que contar rechazos sin distinguir motivo mezcla cosas opuestas.

## Stack técnico

- **Lenguaje:** Python 3.9
- **Análisis:** `pandas`, `numpy`
- **Estadística:** `scipy` (Mann-Whitney, Chi², Fisher, z-test de proporciones)
- **Visualización:** `matplotlib`, `seaborn`
- **Configuración:** `PyYAML`
- **CI:** GitHub Actions

Previsto para las fases siguientes: `scikit-learn`, `XGBoost`, `LightGBM`, `SHAP`, `FastAPI`, `Docker`, `MLflow` y `Evidently`.

## Limitaciones reconocidas

- **Las tasas de default están calculadas sobre `application_train` completo.** Sirven para explorar y fijar la receta, pero toda decisión basada en la asociación con el target se reconfirma sobre el split de entrenamiento en la Fase 3. Los descartes por redundancia estructural son target-independientes y quedan firmes: el campo `firmeza` de cada receta marca cuáles son.
- **Pseudoreplicación en el análisis a nivel fila.** Los tests de la Fase A tratan millones de créditos como observaciones independientes cuando están anidados en clientes que comparten target. Los p-valores son anticonservadores, y por eso las decisiones se apoyan en el tamaño del efecto y en la revalidación a nivel cliente.
- **Causalidad.** La asociación entre edad y default (12,29% en la franja de 20 a 25 años frente al 3,66% en la de 65 a 70) es consistente con la estabilidad financiera acumulada, pero no descarta un sesgo de selección: quien impagó joven puede no seguir en el dataset de mayor.
- **Eventos raros.** Varias de las señales más severas viven en cientos de casos (323 clientes con límite de crédito negativo, 284 con registro fantasma). Se reportan siempre con su n, y su estabilidad fuera de muestra está por demostrar.
- **El comportamiento de pago mes a mes queda fuera.** Las cuatro tablas cubren la solicitud, el buró y el historial interno a nivel de solicitud, pero no cómo pagó el cliente los créditos que sí se le concedieron, que es lo que vive en las tres tablas de panel fuera de alcance.

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
   Los notebooks corren limpios de arriba abajo con Restart & Run All. `bureau_balance` son 27 millones de filas, así que conviene tener RAM disponible: `reduce_mem_usage` en [src/data/loader.py](src/data/loader.py) baja los dtypes al mínimo sin perder rango y ahorra más del 60% de memoria.

5. **Pasa los tests** (no necesitan los datos de Kaggle):
   ```bash
   make test    # 53 pruebas sobre el loader, EvaluadorSenal y el esquema de las recetas
   make lint    # ruff sobre src/
   ```
   La verificación de sintaxis, kernel, dependencias y tipografía va aparte y necesita un intérprete 3.10 o superior, no el del venv:
   ```bash
   python3.13 .github/scripts/check.py
   ```

6. **Comandos disponibles:**
   ```bash
   make help
   ```

## Autor

**Yeray Benito Calviño**
Data Science student, Universidad Complutense de Madrid
[LinkedIn](https://www.linkedin.com/in/yeraybenit0) · [GitHub](https://github.com/yeraybc)

## Licencia

Distribuido bajo licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
