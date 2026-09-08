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
    CATEGORICAS_OHE,
    COL_DIA,
    COL_EDUCACION,
    COL_FRANJA,
    COL_HORA,
    COL_OCUPACION,
    COL_ORGANIZACION,
    DIA_FIN_DE_SEMANA,
    DIA_LABORABLE,
    FRANJA_FUERA,
    JERARQUIA_EDUCACION,
    NIVEL_SIN_OCUPACION,
    NUMERICAS,
    aplicar_dominio,
    columnas_declaradas,
    construir_pipeline,
    franja_horaria,
    informe_buckets,
    verificar_contrato_columnas,
)

N = 60
COLUMNAS_ENTRADA = 101


@pytest.fixture
def entrada():
    """Las 101 columnas que ve el pipeline, con valores plausibles y variedad en cada bucket.

    Se arma desde las listas declaradas y no a mano: si mañana entra una columna nueva al
    contrato, el fixture la trae sola y no hay dos sitios que sincronizar.
    """
    frame = pd.DataFrame({c: np.linspace(1.0, 100.0, N) for c in NUMERICAS})
    frame[COL_HORA] = list(range(24)) + list(range(24)) + list(range(12))
    for c in BINARIAS:
        frame[c] = ([0] * (N - 5)) + ([1] * 5)
    for c in CATEGORICAS_OHE:
        if c == COL_FRANJA:
            continue
        frame[c] = ["uno", "dos"] * (N // 2)
    frame[COL_DIA] = (["MONDAY"] * 40) + (["SATURDAY"] * 20)
    frame[COL_EDUCACION] = [JERARQUIA_EDUCACION[i % len(JERARQUIA_EDUCACION)] for i in range(N)]
    # con riesgo diferencial: alternándolas, las dos salían con la misma tasa, el WoE era
    # constante y el VarianceThreshold se llevaba la columna
    frame[COL_ORGANIZACION] = (["banca"] * 45) + (["obra"] * 15)
    frame[COL_OCUPACION] = (["oficio"] * 50) + ([None] * 10)
    return frame


@pytest.fixture
def objetivo():
    return pd.Series(([0] * 50) + ([1] * 10))


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
    [(0, FRANJA_FUERA), (5, FRANJA_FUERA), (6, "manana"), (11, "manana"), (12, "tarde"),
     (17, "tarde"), (18, FRANJA_FUERA), (23, FRANJA_FUERA)],
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
    assert (salida[COL_OCUPACION] == NIVEL_SIN_OCUPACION).sum() == 10


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

    assert (una[COL_DIA] == DIA_FIN_DE_SEMANA).sum() == 20
    assert (dos[COL_DIA] == DIA_FIN_DE_SEMANA).sum() == 20


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
    assert fila[COL_EDUCACION] == -1, "el ordinal sale fuera de la escala por abajo"
    columnas_suite = [c for c in p.get_feature_names_out() if c.startswith("NAME_TYPE_SUITE")]
    assert salida[columnas_suite].to_numpy().sum() == 0, (
        "el OHE con handle_unknown='ignore' tiene que dejar la fila a ceros"
    )


def test_el_fixture_usa_categorias_que_de_verdad_no_estaban(entrada):
    """Guardián: con una categoría ya vista, los dos tests de arriba no prueban nada."""
    for columna, valor_nuevo in NUEVAS.items():
        assert valor_nuevo not in set(entrada[columna]), f"{columna} ya traía ese valor"
