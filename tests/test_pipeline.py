"""Tests del ensamblado de las capas 2 (src/features/pipeline.py).

Sobre frame sintético, que es donde CI los ejecuta: lo que protegen es el reparto de columnas y
las recodificaciones de dominio, que son propiedades del código y no del dato.

Lo que más importa aquí es **el contrato de columnas**. El `ColumnTransformer` va con
`remainder="drop"`, así que una columna que no esté declarada en ningún bucket se cae de la
matriz sin dar un error y sin que cambie ningún nombre: la matriz sale más pobre y todo lo demás
sigue pareciendo correcto.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    BINARIAS,
    COLUMNAS_PROTEGIDAS_DE_VARIANZA,
    IMPUTACION_SIN_RASTRO,
    CATEGORICAS_OHE,
    COL_DIA,
    COL_EDUCACION,
    COL_FRANJA,
    COL_HORA,
    COL_OCUPACION,
    COL_ORGANIZACION,
    DIA_FIN_DE_SEMANA,
    DIA_LABORABLE,
    CODIGO_EDUCACION_DESCONOCIDA,
    FRANJA_FUERA,
    FRANJA_MANANA,
    FRANJA_TARDE,
    JERARQUIA_EDUCACION,
    NIVEL_SIN_OCUPACION,
    NUMERICAS,
    PRESENCIA_CASI_EXACTA,
    PRESENCIA_POR_BANDERA,
    PRESENCIA_POR_BLOQUE,
    aplicar_dominio,
    columnas_declaradas,
    construir_pipeline,
    franja_horaria,
    informe_buckets,
    verificar_contrato_columnas,
)

# Seiscientas y no sesenta: con sesenta, **ninguna celda categórica llega a las 100 que exige
# `n_min_categoria`**, así que el AgrupadorDeRaras colapsaba las catorce columnas del bucket a un
# nivel único y el OHE trabajaba sobre constantes que después se llevaba el VarianceThreshold.
# Los tests pasaban igual, midiendo una matriz degenerada. Lo vigila `test_el_fixture_no_colapsa`.
N = 600
COLUMNAS_ENTRADA = 101
N_POSITIVOS = 100
N_NULOS_OCUPACION = 100
N_FIN_DE_SEMANA = 200
N_ORGANIZACION_MINORITARIA = 150


@pytest.fixture
def entrada():
    """Las 101 columnas que ve el pipeline, con valores plausibles y variedad en cada bucket.

    Se arma desde las listas declaradas y no a mano: si mañana entra una columna nueva al
    contrato, el fixture la trae sola y no hay dos sitios que sincronizar.
    """
    frame = pd.DataFrame({c: np.linspace(1.0, 100.0, N) for c in NUMERICAS})
    # los cuatro numeradores y denominadores de los dos ratios posteriores, separados: con el
    # mismo linspace en todas las columnas los dos ratios salían constantes a 1,0 y el
    # VarianceThreshold se los llevaba, así que no llegaban a la matriz
    frame["AMT_INCOME_TOTAL"] = np.linspace(1_000.0, 50_000.0, N)
    frame["AMT_ANNUITY"] = np.linspace(100.0, 5_000.0, N)[::-1]
    frame["CNT_CHILDREN"] = np.tile([0.0, 1.0, 2.0, 3.0], N // 4)
    frame["CNT_FAM_MEMBERS"] = frame["CNT_CHILDREN"] + 2.0
    frame[COL_HORA] = list(range(24)) * (N // 24)
    for c in BINARIAS:
        frame[c] = ([0] * (N - 50)) + ([1] * 50)
    for c in CATEGORICAS_OHE:
        if c == COL_FRANJA:
            continue
        frame[c] = ["uno", "dos"] * (N // 2)
    # una con más de dos niveles: es la única forma de distinguir `drop="if_binary"` de
    # `drop="first"`, porque sobre una binaria las dos sacan una columna igual
    frame["NAME_TYPE_SUITE"] = ["uno", "dos", "tres", "cuatro"] * (N // 4)
    frame[COL_DIA] = (["MONDAY"] * (N - N_FIN_DE_SEMANA)) + (["SATURDAY"] * N_FIN_DE_SEMANA)
    frame[COL_EDUCACION] = [JERARQUIA_EDUCACION[i % len(JERARQUIA_EDUCACION)] for i in range(N)]
    # con riesgo diferencial: alternándolas, las dos salían con la misma tasa, el WoE era
    # constante y el VarianceThreshold se llevaba la columna
    frame[COL_ORGANIZACION] = (["banca"] * (N - N_ORGANIZACION_MINORITARIA)) + (
        ["obra"] * N_ORGANIZACION_MINORITARIA
    )
    # cinco oficios repartidos y los nulos intercalados, no en bloque: con un solo oficio el
    # target encoding es binario y sus medias por fold coinciden, así que la codificación cruzada
    # no deja huella y no se puede distinguir de una media directa
    oficios = [f"oficio{i % 5}" for i in range(N)]
    frame[COL_OCUPACION] = [None if i % (N // N_NULOS_OCUPACION) == 0 else o
                            for i, o in enumerate(oficios)]
    return frame


@pytest.fixture
def objetivo():
    return pd.Series(([0] * (N - N_POSITIVOS)) + ([1] * N_POSITIVOS))


# --- el contrato de columnas -----------------------------------------------------------------


def test_el_fixture_trae_exactamente_las_columnas_del_contrato(entrada):
    """Guardián: si el fixture se desalinea, todos los tests de abajo miden otra cosa."""
    assert entrada.shape[1] == COLUMNAS_ENTRADA
    verificar_contrato_columnas(aplicar_dominio(entrada))


def test_los_buckets_no_se_solapan_ni_repiten(entrada):
    """Una columna en dos buckets entraría dos veces en la matriz, con dos codificaciones."""
    declaradas = columnas_declaradas()

    assert len(declaradas) == len(set(declaradas))
    assert len(declaradas) == COLUMNAS_ENTRADA


def test_una_columna_de_mas_revienta_en_vez_de_caerse_en_silencio(entrada):
    """El fallo que caza: `remainder='drop'` llevándose una columna sin avisar."""
    con_extra = aplicar_dominio(entrada).assign(COLUMNA_NUEVA=1.0)
    with pytest.raises(ValueError, match="sobran"):
        verificar_contrato_columnas(con_extra)


def test_una_columna_de_menos_tambien_revienta(entrada):
    """La otra dirección: el bucket pediría una columna que el frame no trae."""
    sin_una = aplicar_dominio(entrada).drop(columns=[NUMERICAS[0]])
    with pytest.raises(ValueError, match="faltan"):
        verificar_contrato_columnas(sin_una)


def test_el_informe_de_buckets_cuadra_con_el_frame(entrada):
    inf = informe_buckets(entrada)

    assert inf["entran"].sum() == COLUMNAS_ENTRADA
    assert (inf["entran"] == inf["presentes"]).all()


# --- el paso de dominio ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hora", "esperada"),
    [(0, FRANJA_FUERA), (5, FRANJA_FUERA), (6, FRANJA_MANANA), (11, FRANJA_MANANA),
     (12, FRANJA_TARDE), (17, FRANJA_TARDE), (18, FRANJA_FUERA), (23, FRANJA_FUERA)],
)
def test_las_fronteras_de_la_franja_horaria(hora, esperada):
    """Los bordes son donde se equivoca un rango, así que van uno a uno."""
    assert franja_horaria(pd.Series([hora]))[0] == esperada


def test_la_hora_nula_no_se_inventa_franja():
    assert pd.isna(franja_horaria(pd.Series([np.nan]))[0])


def test_el_paso_de_dominio_sustituye_y_no_anade(entrada):
    """Lo que deja el conteo cuadrado en 101: la hora se va y la franja ocupa su sitio."""
    salida = aplicar_dominio(entrada)

    assert salida.shape[1] == entrada.shape[1]
    assert COL_HORA not in salida.columns
    assert list(salida.columns).index(COL_FRANJA) == list(entrada.columns).index(COL_HORA)


def test_el_dia_de_la_semana_se_reescribe_en_su_sitio_y_sigue_categorico(entrada):
    """No se convierte en binaria 0/1 ni se va al passthrough: se queda en el bucket de OHE."""
    salida = aplicar_dominio(entrada)

    assert set(salida[COL_DIA]) == {DIA_LABORABLE, DIA_FIN_DE_SEMANA}
    assert salida[COL_DIA].dtype == object
    assert COL_DIA in CATEGORICAS_OHE and COL_DIA not in BINARIAS


def test_el_nulo_de_la_ocupacion_pasa_a_ser_un_nivel(entrada):
    salida = aplicar_dominio(entrada)

    assert not salida[COL_OCUPACION].isna().any()
    assert (salida[COL_OCUPACION] == NIVEL_SIN_OCUPACION).sum() == N_NULOS_OCUPACION


def test_el_paso_de_dominio_es_idempotente(entrada):
    """Aplicarlo sobre su propia salida no puede mover nada más."""
    una = aplicar_dominio(entrada)

    pd.testing.assert_frame_equal(aplicar_dominio(una), una)


def test_el_paso_de_dominio_es_permisivo_con_columnas_ausentes(entrada):
    """A la API puede llegar un frame parcial; quien exige el contrato es la verificación."""
    parcial = entrada[[COL_DIA]]

    assert list(aplicar_dominio(parcial).columns) == [COL_DIA]


# --- el pipeline entero ----------------------------------------------------------------------


def test_el_pipeline_devuelve_un_frame_con_nombres(entrada, objetivo):
    """Sin `set_output` el ColumnTransformer devuelve un array y se pierden los nombres."""
    salida = construir_pipeline().fit_transform(entrada, objetivo)

    assert isinstance(salida, pd.DataFrame)
    assert len(salida) == len(entrada)
    assert not salida.isna().any().any(), "la matriz no puede salir con nulos"


def test_get_feature_names_out_casa_con_las_columnas_de_la_salida(entrada, objetivo):
    """El fallo que ya mordió en el 1.3: prometer una columna más de las que se entregan."""
    p = construir_pipeline().fit(entrada, objetivo)

    assert list(p.get_feature_names_out()) == list(p.transform(entrada).columns)


def test_la_educacion_sale_ordinal_en_el_orden_de_la_jerarquia(entrada, objetivo):
    """La jerarquía es de dominio, así que el orden tiene que ser el declarado y no el alfabético."""
    p = construir_pipeline().fit(entrada, objetivo)
    salida = p.transform(entrada)
    codigos = pd.Series(salida[COL_EDUCACION].to_numpy(), index=entrada[COL_EDUCACION])

    assert [codigos[n].iloc[0] for n in JERARQUIA_EDUCACION] == list(range(len(JERARQUIA_EDUCACION)))


def test_la_organizacion_sale_con_el_sufijo_del_woe(entrada, objetivo):
    p = construir_pipeline().fit(entrada, objetivo)
    columnas = list(p.get_feature_names_out())

    assert f"{COL_ORGANIZACION}_WOE" in columnas
    assert COL_ORGANIZACION not in columnas


def test_el_fixture_da_riesgo_diferencial_a_la_organizacion(entrada, objetivo):
    """Guardián: con las dos ramas a la misma tasa el WoE sale constante y la varianza lo tira."""
    tasas = objetivo.groupby(entrada[COL_ORGANIZACION]).mean()

    assert tasas.nunique() == len(tasas), "las organizaciones tienen que separarse en riesgo"


def test_la_varianza_se_lleva_la_columna_constante(entrada, objetivo):
    """La red que justifica el paso: hoy no elimina nada, y con una constante sí tiene que hacerlo."""
    con_constante = entrada.assign(**{BINARIAS[0]: 0})
    salida = construir_pipeline().fit_transform(con_constante, objetivo)

    assert BINARIAS[0] not in salida.columns
    assert BINARIAS[0] in construir_pipeline().fit_transform(entrada, objetivo).columns


def test_el_dia_recodificado_sobrevive_a_una_segunda_pasada(entrada):
    """El fallo que caza: mandar a entresemana todo lo que no sea SATURDAY ni SUNDAY.

    Con esa forma, "fin de semana" no es un día de la semana en la segunda pasada y cae al otro
    lado. Es la no idempotencia que borra, la misma de la bandera del centinela.
    """
    una = aplicar_dominio(entrada)
    dos = aplicar_dominio(una)

    assert (una[COL_DIA] == DIA_FIN_DE_SEMANA).sum() == N_FIN_DE_SEMANA
    assert (dos[COL_DIA] == DIA_FIN_DE_SEMANA).sum() == N_FIN_DE_SEMANA


# --- el contrato de la maquinaria de scikit-learn ---------------------------------------------
# Las cinco propiedades que exige `sklearn.md`. Dos ya están arriba (que la salida sea un frame
# con nombres y que `get_feature_names_out` case con ella); estas son las tres que faltan, y las
# tres son de la Fase 4 y de la 6: `clone` es lo que reajusta el pipeline por fold en el CV,
# `joblib` es como llega a la API, y la categoría no vista es lo que la API va a recibir.

NUEVAS = {
    COL_ORGANIZACION: "cooperativa jamas vista",
    COL_OCUPACION: "oficio jamas visto",
    COL_EDUCACION: "titulo jamas visto",
    "NAME_TYPE_SUITE": "acompanante jamas visto",
}


def test_clonar_y_reajustar_reproduce_los_mismos_parametros(entrada, objetivo):
    """`clone` es lo que usa el CV de la Fase 4 para reajustar por fold: tiene que dar lo mismo."""
    from sklearn.base import clone

    original = construir_pipeline().fit(entrada, objetivo)
    clonado = clone(original).fit(entrada, objetivo)

    assert clonado.named_steps["winsor"].limites_ == original.named_steps["winsor"].limites_
    ct_o = original.named_steps["columnas"].named_transformers_
    ct_c = clonado.named_steps["columnas"].named_transformers_
    pd.testing.assert_series_equal(
        ct_c["woe"].tablas_[COL_ORGANIZACION], ct_o["woe"].tablas_[COL_ORGANIZACION]
    )
    pd.testing.assert_frame_equal(clonado.transform(entrada), original.transform(entrada))


def test_el_clon_sale_sin_ajustar(entrada, objetivo):
    """Guardián: si `clone` arrastrase el ajuste, el test de arriba no probaría el reajuste."""
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    clonado = clone(construir_pipeline().fit(entrada, objetivo))
    with pytest.raises(NotFittedError):
        clonado.transform(entrada)


def test_ida_y_vuelta_por_joblib_da_la_misma_salida(entrada, objetivo, tmp_path):
    """Así es como el pipeline ajustado llega a la API en la Fase 6."""
    import joblib

    p = construir_pipeline().fit(entrada, objetivo)
    destino = tmp_path / "pipeline.joblib"
    joblib.dump(p, destino)
    recargado = joblib.load(destino)

    pd.testing.assert_frame_equal(recargado.transform(entrada), p.transform(entrada))
    assert list(recargado.get_feature_names_out()) == list(p.get_feature_names_out())


def test_sobrevive_a_categorias_no_vistas_en_transform(entrada, objetivo):
    """La API va a recibir categorías nuevas en las cuatro codificaciones y no puede reventar."""
    p = construir_pipeline().fit(entrada, objetivo)
    desconocido = entrada.copy()
    for columna, valor_nuevo in NUEVAS.items():
        desconocido[columna] = valor_nuevo
    # el aviso del OneHotEncoder se afirma en vez de silenciarse: es la señal correcta y viene de
    # la librería, así que lo que interesa es dejar escrito que se espera. En serving lo emite en
    # cada petición con una categoría nueva, que es ruido a tener en cuenta en la Fase 6
    with pytest.warns(UserWarning, match="unknown categories"):
        salida = p.transform(desconocido)

    assert list(salida.columns) == list(p.transform(entrada).columns), "cambió el juego de columnas"
    assert not salida.isna().any().any(), "una categoría nueva no puede meter nulos en la matriz"


def test_cada_codificacion_trata_lo_no_visto_como_toca(entrada, objetivo):
    """El detalle de la anterior: cada bucket tiene su propia respuesta declarada."""
    p = construir_pipeline().fit(entrada, objetivo)
    desconocido = entrada.copy()
    for columna, valor_nuevo in NUEVAS.items():
        desconocido[columna] = valor_nuevo
    with pytest.warns(UserWarning, match="unknown categories"):
        salida = p.transform(desconocido)
    fila = salida.iloc[0]

    assert fila[f"{COL_ORGANIZACION}_WOE"] == 0.0, "el WoE neutro es cero"
    assert fila[COL_EDUCACION] == CODIGO_EDUCACION_DESCONOCIDA, "el ordinal sale de la escala"
    columnas_suite = [c for c in p.get_feature_names_out() if c.startswith("NAME_TYPE_SUITE")]
    assert salida[columnas_suite].to_numpy().sum() == 0, (
        "el OHE con handle_unknown='ignore' tiene que dejar la fila a ceros"
    )


def test_el_fixture_usa_categorias_que_de_verdad_no_estaban(entrada):
    """Guardián: con una categoría ya vista, los dos tests de arriba no prueban nada."""
    for columna, valor_nuevo in NUEVAS.items():
        assert valor_nuevo not in set(entrada[columna]), f"{columna} ya traía ese valor"


# --- el orden de los dos primeros pasos, que es fuga estructural -----------------------------


@pytest.fixture
def entrada_con_extremo(entrada):
    """Un cliente con el error de captura de 117M en el ingreso, que es donde el orden se nota.

    Sin un valor por encima del cap, el winsorizador no recorta nada y los dos órdenes dan lo
    mismo: el fixture normal no distingue los dos casos.

    Con pocas filas esto no funcionaría: con un solo extremo en 60, el p99 cae al 41% del camino
    hacia él y el triple se queda por encima, así que no recortaría nada. Con las 600 del fixture
    el p99 se apoya en la masa, que es la situación de las 245.993 de verdad.
    """
    frame = entrada.copy()
    frame["AMT_INCOME_TOTAL"] = 1_000.0
    frame.loc[frame.index[-1], "AMT_INCOME_TOTAL"] = 117_000_000.0
    frame["AMT_ANNUITY"] = 10_000.0
    return frame


def test_la_carga_de_cuota_se_calcula_sobre_el_ingreso_ya_recortado(entrada_con_extremo, objetivo):
    """La fuga estructural que caza: `derivadas` colocado delante de `winsor`.

    Con ese orden el error de captura de 117M se queda dentro del denominador de la carga, que es
    justo lo que el cap corrige. Sobre el dato real son 71 clientes de entrenamiento los que
    cambian su ratio, y 9 los que cambian el de hijos sobre miembros: la matriz sale distinta y
    ni el conteo de columnas ni ninguna otra cifra lo delatan.
    """
    p = construir_pipeline().fit(entrada_con_extremo, objetivo)
    salida = p.transform(entrada_con_extremo)
    limite = p.named_steps["winsor"].limites_["AMT_INCOME_TOTAL"]
    cuota = entrada_con_extremo["AMT_ANNUITY"].iloc[-1]
    ingreso_crudo = entrada_con_extremo["AMT_INCOME_TOTAL"].iloc[-1]

    assert salida["ANNUITY_TO_INCOME_RATIO"].iloc[-1] == pytest.approx(cuota / limite)
    assert salida["ANNUITY_TO_INCOME_RATIO"].iloc[-1] != pytest.approx(cuota / ingreso_crudo)


def test_el_fixture_del_extremo_dispara_de_verdad_el_cap(entrada_con_extremo, objetivo):
    """Guardián: sin un ingreso por encima del límite, el test de arriba no distingue nada."""
    p = construir_pipeline().fit(entrada_con_extremo, objetivo)
    limite = p.named_steps["winsor"].limites_["AMT_INCOME_TOTAL"]
    ingreso_crudo = entrada_con_extremo["AMT_INCOME_TOTAL"].iloc[-1]

    assert ingreso_crudo > limite, "el ingreso extremo no llega a recortarse"
    assert ingreso_crudo / limite > 100, "la diferencia tiene que ser de órdenes de magnitud"


# --- el ancla del conteo de columnas de salida -----------------------------------------------


def _niveles_por_columna(entrada):
    """Los niveles que ve el OHE de cada categórica, ya pasada por el paso de dominio."""
    tras_dominio = aplicar_dominio(entrada)
    return {c: tras_dominio[c].nunique(dropna=False) for c in CATEGORICAS_OHE}


def test_cada_categorica_saca_una_columna_por_nivel_salvo_las_binarias(entrada, objetivo):
    """El fallo que caza: `drop="first"`, que le quita un nivel a todas y no solo a las binarias.

    Sobre el dato real baja el OHE de 52 columnas a 43 sin que nada mas se mueva, y con un
    modelo sin regularizar eso es informacion perdida en las categoricas de varios niveles.
    """
    p = construir_pipeline().fit(entrada, objetivo)
    codificador = p.named_steps["columnas"].named_transformers_["ohe"].named_steps["codifica"]
    salen = dict(zip(CATEGORICAS_OHE, (len(c) for c in codificador.categories_)))
    niveles = _niveles_por_columna(entrada)

    for columna, k in niveles.items():
        esperadas = 1 if k == 2 else k
        real = len(
            [c for c in codificador.get_feature_names_out() if c.startswith(f"{columna}_")]
        )
        assert real == esperadas, f"{columna} tiene {k} niveles y saca {real} columnas"
        assert salen[columna] == k


def test_el_fixture_trae_una_categorica_de_mas_de_dos_niveles(entrada):
    """Guardián: solo con binarias, `if_binary` y `first` sacan lo mismo y el test es vacuo."""
    niveles = _niveles_por_columna(entrada)

    assert max(niveles.values()) > 2, f"todas las categóricas son binarias: {niveles}"


def test_el_total_de_columnas_de_salida_cuadra_bucket_a_bucket(entrada, objetivo):
    """La aritmética completa, calculada aparte y no leída del pipeline ya ajustado."""
    niveles = _niveles_por_columna(entrada)
    esperado = (
        len(NUMERICAS)
        + sum(1 if k == 2 else k for k in niveles.values())
        + 3  # ordinal, WoE y target encoding, una columna cada uno
        + len(BINARIAS)
    )
    salida = construir_pipeline().fit_transform(entrada, objetivo)

    assert salida.shape[1] == esperado


def test_el_fixture_no_colapsa_las_categoricas_en_su_residual(entrada, objetivo):
    """Guardián del fixture, y el que destapó que estaba aguado.

    Con 60 filas ninguna celda llegaba a `n_min_categoria`, así que el agrupador mandaba las
    catorce columnas enteras a su residual y el OHE codificaba constantes que el VarianceThreshold
    se llevaba después. Los tests de esta sección pasaban midiendo una matriz degenerada.
    """
    p = construir_pipeline().fit(entrada, objetivo)
    agrupador = p.named_steps["columnas"].named_transformers_["ohe"].named_steps["agrupa"]
    codificador = p.named_steps["columnas"].named_transformers_["ohe"].named_steps["codifica"]

    assert agrupador.raras_ == {}, f"el fixture tiene celdas raras: {agrupador.raras_}"
    assert min(len(c) for c in codificador.categories_) >= 2, "alguna columna quedó con un nivel"


def test_el_fixture_da_ratios_posteriores_que_varian(entrada, objetivo):
    """Guardián: con el mismo linspace en todas, los dos ratios salen constantes y se caen.

    Los eliminaba el VarianceThreshold antes de llegar a la matriz, así que ningún test sobre
    ellos medía nada. Es el mismo aguado que colapsaba las categóricas, en otra columna.
    """
    salida = construir_pipeline().fit_transform(entrada, objetivo)

    for ratio in ("ANNUITY_TO_INCOME_RATIO", "CHILDREN_TO_FAM_RATIO"):
        assert ratio in salida.columns, f"{ratio} no llegó a la matriz"
        assert salida[ratio].nunique() > 1, f"{ratio} es constante y no mide nada"


# --- la codificación cruzada del target encoding ---------------------------------------------


def test_la_ocupacion_se_codifica_fuera_de_fold_al_ajustar(entrada, objetivo):
    """El `TargetEncoder` de la librería hace codificación cruzada, y esto lo fija.

    Es la propiedad que separa un target encoding honesto de una media por categoría: al ajustar,
    cada fila recibe la media de los folds en los que **no** estaba. Sustituirlo por una media
    directa dejaría las dos salidas idénticas, que es el fallo que caza.
    """
    fuera_de_fold = construir_pipeline().fit_transform(entrada, objetivo)[COL_OCUPACION]
    dentro = construir_pipeline().fit(entrada, objetivo).transform(entrada)[COL_OCUPACION]

    assert fuera_de_fold.nunique() > dentro.nunique()
    assert not np.allclose(fuera_de_fold, dentro)


def test_solo_la_ocupacion_cambia_entre_ajustar_y_transformar(entrada, objetivo):
    """El resto de la matriz es idéntico: la codificación cruzada solo la hace el paso `tgt`."""
    ajustando = construir_pipeline().fit_transform(entrada, objetivo)
    transformando = construir_pipeline().fit(entrada, objetivo).transform(entrada)
    distintas = [c for c in ajustando.columns if not np.allclose(ajustando[c], transformando[c])]

    assert distintas == [COL_OCUPACION]


def test_el_fixture_da_una_ocupacion_con_varios_niveles_y_repartida(entrada, objetivo):
    """Guardián: con un solo oficio, las medias por fold coinciden y no hay huella que ver.

    Y en bloque tampoco: los niveles tienen que cruzar la frontera del objetivo para que cada
    fold vea una mezcla distinta.
    """
    ocupacion = entrada[COL_OCUPACION]

    assert ocupacion.nunique(dropna=False) > 2, "la ocupación no tiene niveles suficientes"
    tasas = objetivo.groupby(ocupacion.fillna("(nulo)")).mean()
    assert ((tasas > 0) & (tasas < 1)).any(), "ningún nivel mezcla positivos y negativos"


def test_las_fronteras_de_la_franja_salen_de_params_y_no_del_codigo():
    """Todo corte pasa por `params.py`, y la forma de comprobarlo es moverlo y ver el efecto.

    Escrito a mano en el módulo, cambiar la declaración no cambiaría nada y el registro estaría
    diciendo una cosa mientras el código hace otra. `conftest` restaura PARAMS al acabar.
    """
    from src.features.params import PARAMS, Parametro

    assert franja_horaria(pd.Series([6]))[0] == FRANJA_MANANA
    viejo = PARAMS["app_hora_inicio_manana"]
    PARAMS["app_hora_inicio_manana"] = Parametro(9, "dominio", viejo.descripcion, viejo.fuente)

    assert franja_horaria(pd.Series([6]))[0] == FRANJA_FUERA
    assert franja_horaria(pd.Series([9]))[0] == FRANJA_MANANA


def test_el_umbral_de_varianza_sale_de_params(entrada, objetivo):
    """Igual con el suelo del filtro final: subirlo tiene que llevarse columnas."""
    from src.features.params import PARAMS, Parametro

    antes = construir_pipeline().fit_transform(entrada, objetivo).shape[1]
    viejo = PARAMS["app_umbral_varianza"]
    PARAMS["app_umbral_varianza"] = Parametro(0.2, "dominio", viejo.descripcion, viejo.fuente)

    assert construir_pipeline().fit_transform(entrada, objetivo).shape[1] < antes


def test_el_nivel_educativo_no_visto_queda_por_debajo_de_la_escala(entrada, objetivo):
    """La dirección, que es lo que importa y lo que nadie fijaba.

    El test de arriba compara contra `CODIGO_EDUCACION_DESCONOCIDA`, así que sigue a la constante
    y pasa con cualquier valor: subirla a 5 dejaba los 375 en verde y un título desconocido se
    leía como el nivel más alto de la jerarquía, que es la lectura opuesta. Aquí se compara
    contra el código del nivel más bajo, que es la propiedad de verdad.
    """
    p = construir_pipeline().fit(entrada, objetivo)
    desconocido = entrada.assign(**{COL_EDUCACION: "titulo jamas visto"})
    mas_bajo = entrada.assign(**{COL_EDUCACION: JERARQUIA_EDUCACION[0]})

    # sin `pytest.warns`: quien avisa de la categoría nueva es el OneHotEncoder, y aquí la única
    # columna con un valor no visto es la del ordinal, que lo resuelve con `use_encoded_value`
    assert (
        p.transform(desconocido)[COL_EDUCACION].iloc[0]
        < p.transform(mas_bajo)[COL_EDUCACION].iloc[0]
    )


COLUMNA_ASIMETRICA = "EXT_SOURCE_1"
N_ALTOS, N_NULOS_ASIMETRICA = 50, 50
VALOR_BAJO, VALOR_ALTO = 1.0, 100.0


@pytest.fixture
def entrada_asimetrica(entrada):
    """Una numérica con nulos donde la mediana y la media se separan diez veces.

    Con el `linspace` del fixture normal las dos coinciden, así que cambiar la estrategia del
    imputador no movía un solo valor y la mutación sobrevivía.
    """
    columna = ([VALOR_BAJO] * (N - N_ALTOS - N_NULOS_ASIMETRICA)) + ([VALOR_ALTO] * N_ALTOS)
    return entrada.assign(**{COLUMNA_ASIMETRICA: columna + ([np.nan] * N_NULOS_ASIMETRICA)})


def test_la_imputacion_es_la_mediana_y_no_la_media(entrada_asimetrica, objetivo):
    """El plan pide mediana, y con la media la cola arrastra el relleno de las 32 con nulos."""
    salida = construir_pipeline().fit_transform(entrada_asimetrica, objetivo)
    rellenados = salida[COLUMNA_ASIMETRICA][entrada_asimetrica[COLUMNA_ASIMETRICA].isna()]
    con_dato = entrada_asimetrica[COLUMNA_ASIMETRICA].dropna()

    assert (rellenados == con_dato.median()).all()
    assert not (rellenados == con_dato.mean()).any()


def test_el_fixture_asimetrico_separa_la_mediana_de_la_media(entrada_asimetrica):
    """Guardián: con las dos iguales el test de arriba no distingue las dos estrategias."""
    con_dato = entrada_asimetrica[COLUMNA_ASIMETRICA].dropna()

    assert con_dato.isna().sum() == 0
    assert entrada_asimetrica[COLUMNA_ASIMETRICA].isna().sum() == N_NULOS_ASIMETRICA
    assert con_dato.mean() > con_dato.median() * 5


# --- de dónde se recupera lo que la mediana rellena -------------------------------------------


def test_los_cuatro_grupos_de_presencia_no_se_solapan():
    """Una numérica en dos grupos diría dos cosas distintas sobre la misma imputación."""
    grupos = [
        set(PRESENCIA_POR_BANDERA),
        set(PRESENCIA_CASI_EXACTA),
        set(PRESENCIA_POR_BLOQUE),
        set(IMPUTACION_SIN_RASTRO),
    ]
    union = set().union(*grupos)

    assert sum(len(g) for g in grupos) == len(union)
    assert union <= set(NUMERICAS), f"declaradas fuera del bucket numérico: {union - set(NUMERICAS)}"


def test_las_banderas_de_presencia_estan_de_verdad_en_la_matriz():
    """De poco sirve declarar que una bandera recupera la ausencia si no llega a la matriz."""
    for numerica, bandera in {**PRESENCIA_POR_BANDERA, **PRESENCIA_CASI_EXACTA}.items():
        assert bandera in BINARIAS or bandera in CATEGORICAS_OHE, f"{numerica} apunta a {bandera}"


def test_el_bloque_edificio_se_recupera_por_su_conteo_y_su_bandera():
    """No hay bandera por columna, así que lo que la recupera en agregado sí tiene que estar."""
    assert "HAS_BUILDING_INFO" in BINARIAS
    assert "BUILDING_INFO_COUNT" in NUMERICAS
    assert "BUILDING_INFO_COUNT" not in PRESENCIA_POR_BLOQUE, "el conteo no es del bloque"


# --- las dos banderas que el EDA conserva pase lo que pase ------------------------------------


def test_las_dos_banderas_protegidas_llegan_a_la_matriz(entrada, objetivo):
    """La alarma que sustituye a la exclusión que el plan pide y que no se implementa.

    Con el suelo de varianza a cero no hay de qué excluirlas, porque ninguna de las dos puede
    quedarse constante. Si alguien sube el suelo, esto se pone rojo y obliga a decidir en vez de
    que las dos desaparezcan sin que nadie lo note.
    """
    salida = construir_pipeline().fit_transform(entrada, objetivo)

    for bandera in COLUMNAS_PROTEGIDAS_DE_VARIANZA:
        assert bandera in salida.columns, f"{bandera} se cayó de la matriz"


def test_el_suelo_a_cero_no_distingue_las_protegidas_de_las_demas(entrada, objetivo):
    """La limitación declarada, fijada: hoy no están protegidas, solo vigiladas.

    Se prueba subiendo el suelo, que es lo único que las pone en peligro. Si algún día se
    implementa el desvío del plan, este test tiene que cambiar y el de arriba seguir en verde.
    """
    from src.features.params import PARAMS, Parametro

    viejo = PARAMS["app_umbral_varianza"]
    PARAMS["app_umbral_varianza"] = Parametro(0.5, "dominio", viejo.descripcion, viejo.fuente)
    salida = construir_pipeline().fit_transform(entrada, objetivo)

    assert not set(COLUMNAS_PROTEGIDAS_DE_VARIANZA) & set(salida.columns)


def test_las_protegidas_son_binarias_de_la_matriz(entrada):
    """Guardián: una protegida que no esté en el bucket binario no la filtra la varianza."""
    for bandera in COLUMNAS_PROTEGIDAS_DE_VARIANZA:
        assert bandera in BINARIAS
        assert bandera in entrada.columns
