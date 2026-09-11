"""Tests del registro de parámetros del pipeline (src/features/params.py).

Lo que protegen: que ningún corte entre sin declarar de dónde sale, que el mismo umbral no se
escriba dos veces con valores distintos (el patrón del 52.500 frente al 52.497 de bureau), y que
usar un parámetro todavía sin fijar falle en vez de colarse como None.
"""

import pytest
import yaml

from src.config import RAIZ, cargar_config
from src.features.agg_bureau import COLUMNAS_SIN_RECETA
from src.features.params import (
    CORTES_POR_FEATURE,
    PARAMS,
    PROCEDENCIAS,
    REAJUSTABLES,
    Parametro,
    con_contraste_pendiente,
    cortes_de,
    features_con_corte,
    fijar_operativo,
    operativos_pendientes,
    parametro,
    por_procedencia,
    reajustables,
    sin_fijar,
    valor,
)


def test_toda_procedencia_esta_en_el_vocabulario_cerrado():
    assert set(REAJUSTABLES) < set(PROCEDENCIAS)
    for nombre, p in PARAMS.items():
        assert p.procedencia in PROCEDENCIAS, nombre


def test_todo_parametro_declara_descripcion_y_fuente():
    for nombre, p in PARAMS.items():
        assert p.descripcion.strip(), f"{nombre} sin descripción"
        assert p.fuente.strip(), f"{nombre} sin fuente"


def test_un_parametro_de_dominio_no_puede_quedar_sin_valor():
    with pytest.raises(ValueError, match="dominio"):
        Parametro(None, "dominio", "d", "f")


def test_procedencia_fuera_del_vocabulario_revienta():
    with pytest.raises(ValueError, match="procedencia"):
        Parametro(1, "heredado", "d", "f")


def test_solo_estimado_y_medido_pueden_estar_sin_fijar():
    for nombre, p in sin_fijar().items():
        assert p.procedencia in REAJUSTABLES, nombre


@pytest.mark.parametrize("nombre", sorted(sin_fijar()))
def test_usar_un_parametro_sin_fijar_falla_en_vez_de_devolver_none(nombre):
    """Uno por uno, no solo el primero: cada pendiente tiene que bloquear a quien lo use."""
    with pytest.raises(ValueError, match="sin fijar"):
        valor(nombre)


def test_los_pendientes_sin_referencia_son_los_declarados():
    """Deuda declarada, no olvidos. Si aparece uno nuevo, que se note aquí."""
    assert set(sin_fijar()) == {
        "bureau_count_cola",
        "umbral_continuas_rb",
        "min_denominador_proporcion",
        "prev_ratio_rechazo_min_solicitudes",
    }


def test_un_parametro_fijado_devuelve_su_valor():
    assert valor("dias_por_anio") == 365.25
    assert valor("umbral_flags_pp") == 2.0


def test_parametro_no_declarado_revienta():
    with pytest.raises(KeyError, match="no declarado"):
        parametro("umbral_inventado")


def test_las_procedencias_particionan_el_registro():
    total = sum(len(por_procedencia(p)) for p in PROCEDENCIAS)
    assert total == len(PARAMS)
    assert set(reajustables()) == set(por_procedencia("estimado")) | set(por_procedencia("medido"))


def test_el_centinela_del_dataset_esta_declarado_una_sola_vez():
    """365243 es el mismo código en application_train y en previous_application.

    Declararlo por tabla es el patrón que en bureau dio 52.500 en un sitio y 52.497 en otro
    para el mismo control.
    """
    con_ese_valor = [n for n, p in PARAMS.items() if p.valor_referencia == 365243]
    assert con_ese_valor == ["centinela_365243"], con_ese_valor


def test_no_hay_dos_parametros_con_la_misma_descripcion():
    """Misma descripción es copia y pega: el mismo corte con dos nombres."""
    vistas = {}
    for nombre, p in PARAMS.items():
        d = p.descripcion.strip()
        assert d not in vistas, f"{nombre} repite la descripción de {vistas.get(d)}"
        vistas[d] = nombre


# --- frontera entre las cifras del EDA y el código del pipeline ------------------------------
# `valor()` impide que un corte declarado se consuma sin refijarlo sobre el split. Lo que
# ninguna puerta cubría es la **cifra suelta**: un 8,0729 escrito a mano dentro de una función
# no pasa por `valor()` ni aparece en PARAMS, y es exactamente el mismo problema, una medición
# hecha sobre la población completa usada como si fuera una constante.
#
# Estas son las que el EDA midió **contra el objetivo**. Como constante de un test valen, que
# ahí solo comprueban que el cómputo no se ha movido y no transforman ni una fila. Dentro de
# `src/` serían un parámetro del EDA consumido sin refijar.
#
# Los recuentos de forma no están aquí a propósito y no son fuga: 122 columnas, 307.511 filas,
# las 19 que se eliminan o las 15 del bloque edificio son estructura del dataset, valen igual
# en train, en validación, en application_test y en la API, y no salen de estimar nada.
CIFRAS_MEDIDAS_CONTRA_EL_TARGET = {
    8.0729, 8.0734, 8.07,      # tasa de default global, cruda y limpia
    24_825,                    # positivos
    6.96, 7.03, 9.23, 6.99,    # tripartita del bloque edificio
    7.05, 9.22,                # la misma sin TOTALAREA_MODE en el denominador
    7.72, 10.34, 2.62,         # bandera del buró y su delta
    8.09, 3.53,                # bandera del círculo social
    8.52, 7.50, 9.31, 7.77,    # los dos scores externos
    1.02, 1.55,                # sus deltas en puntos porcentuales
    # las que publica el punto 1.4, todas remedidas sobre los 245.993 de entrenamiento
    10.944, 8.929, 8.569, 5.379, 1.504,   # tasa por nivel de NAME_EDUCATION_TYPE
    8.490, 7.720, 8.026,                  # tasa por franja horaria
    12.500, 4.110,                        # tasa de Industry: type 13 y type 12
    2.0415, 0.8920, 0.4840,               # WoE de Industry: type 8, crudo y los dos suavizados
    0.3755, 0.6548, 0.7550, 0.7924,       # WoE de los tipos 13 y 12 y los dos extremos
    0.0443, 0.0286, 0.0446, 0.0284,       # correlación de FLAG_DOCUMENT_3 y 6, tabla y train
    0.0207, 0.186,                        # Fisher de las raras y P(0 positivos en 20 obs)
    8.631, 11.634,                        # el hueco por el que NAME_HOUSING_TYPE no se agrupa
    # bureau, que el bloque 2 empieza a citar y agg_bureau.py va a tener delante
    10.12, 7.73, 10.1249, 7.7301, 2.3951, # default sin y con historial de buro, y su delta
    7.50,                                 # el nivel "sin cuota reportada" de la tripartita
    12.719, 8.5752, 8.273, 2.6011,        # los cuatro efectos mas altos de la receta de bureau
    0.1828, 0.1569,                       # rank-biserial del ratio de deuda y del ritmo anual
    6.12, 16.15,                          # los extremos del gradiente de BUREAU_CREDITS_PER_YEAR
    16.99, 13.16, 7.82,                   # default de deuda mayor que credito, grosero y base
    # las que publica el 2.3, sobre los 210.875 clientes de train con historial
    10.09, 8.37, 8.31,                    # el tramo pico de vencimiento y sus dos vecinos
    9.35, 5.77, 2.47, 1.91,               # la ventana: tasas extremas, delta a 180 y a 730 dias
    2.33, 1.94,                           # la cola del conteo con 18 y con 17 creditos
}  # fmt: skip
# Los recorridos en puntos porcentuales de esos mismos agrupamientos (los 2,2 de NAME_TYPE_SUITE,
# los 4,24 y 1,80 de NAME_FAMILY_STATUS, los 3,00 del hueco) se quedan fuera a propósito, y por
# lo mismo que los enteros pequeños de las `estimado`: son indistinguibles de una constante
# estructural, y el 3,00 chocaría de frente con el factor de winsorización y con el ratio de
# deuda sobre crédito, que valen 3 los dos. Lo que las cubre son las dos tasas de arriba, que
# son de donde salen.

# params.py es la excepción, y por diseño: ahí una cifra medida está obligada a declarar su
# procedencia, y `valor()` la bloquea hasta que alguien la refija sobre solo_train(). El
# problema que persigue este test es la cifra suelta fuera del registro.
MODULO_DEL_REGISTRO = "params.py"


def test_ninguna_cifra_medida_contra_el_target_vive_en_src():
    """Una tasa de default escrita a mano en `src/` es un corte del EDA sin refijar.

    No pasa por `valor()`, no aparece en PARAMS y nadie la va a remedir sobre el split, así que
    esquiva entera la disciplina que el resto del registro impone. El escaneo es del árbol
    sintáctico y no del texto: una cifra citada en un comentario o en un docstring no cuenta,
    que documentar el hallazgo del EDA es justo lo que hay que hacer.

    Es una lista cerrada, no una regla general: cubre las cifras que el EDA publicó, y una
    medición nueva hay que añadirla aquí. Lo que garantiza es que las publicadas no se cuelen.
    """
    import ast

    from src.config import RAIZ

    encontradas, constantes = [], 0
    for fichero in sorted((RAIZ / "src").rglob("*.py")):
        if fichero.name == MODULO_DEL_REGISTRO:
            continue
        for nodo in ast.walk(ast.parse(fichero.read_text())):
            if not isinstance(nodo, ast.Constant) or isinstance(nodo.value, bool):
                continue
            if not isinstance(nodo.value, (int, float)):
                continue
            constantes += 1
            if nodo.value in CIFRAS_MEDIDAS_CONTRA_EL_TARGET:
                ruta = fichero.relative_to(RAIZ)
                encontradas.append(f"{ruta}:{nodo.lineno} -> {nodo.value}")

    assert constantes > 50, f"el escaneo solo vio {constantes} constantes, revisar el recorrido"
    assert not encontradas, (
        "cifras medidas contra el TARGET dentro de src/, que nadie va a refijar sobre el "
        f"split: {encontradas}"
    )


# --- frontera entre config.yaml y params.py -------------------------------------------------
# config.yaml lleva infraestructura y params.py lleva cortes de modelado. La frontera se
# comprueba en las dos direcciones: que no haya la misma clave en los dos sitios, y que las
# claves concretas que se migraron o se eliminaron no vuelvan a colarse en config.yaml.

CORTES_QUE_NO_VUELVEN_A_CONFIG = (
    "min_iv",
    "n_bins_max",
    "correlation_threshold",
    "missing_threshold",
)


def _claves_de_config():
    cfg = cargar_config()
    cfg.pop("paths", None)
    return {k for bloque in cfg.values() for k in bloque}


def test_ninguna_clave_vive_en_los_dos_ficheros():
    solapadas = set(PARAMS) & _claves_de_config()
    assert not solapadas, f"mismo concepto declarable en dos sitios: {sorted(solapadas)}"


def test_los_cortes_de_modelado_ya_no_estan_en_config_yaml():
    """El 0,95 de correlation_threshold contra el 0,70 de aquí era el mismo control duplicado."""
    presentes = set(CORTES_QUE_NO_VUELVEN_A_CONFIG) & _claves_de_config()
    assert not presentes, f"volvieron a config.yaml: {sorted(presentes)}"


def test_los_dos_cortes_migrados_estan_declarados_aqui():
    for nombre in ("min_iv", "n_bins_max"):
        assert nombre in PARAMS, f"{nombre} se sacó de config.yaml y no llegó a params.py"
    assert valor("min_iv") == 0.02
    assert valor("n_bins_max") == 10


def test_el_umbral_de_redundancia_es_uno_solo_y_es_el_del_eda():
    assert valor("redundancia_pearson") == 0.70
    assert "correlation_threshold" not in PARAMS


def test_el_umbral_unico_de_nulos_no_se_migro():
    """El EDA decide los nulos columna a columna con IV, no con un umbral único."""
    assert "missing_threshold" not in PARAMS


def test_config_yaml_conserva_la_infraestructura():
    """Migrar los cortes no puede llevarse por delante lo que el split y la API necesitan."""
    cfg = cargar_config()
    assert set(cfg["dataset"]) == {"target_col", "id_col", "test_size", "random_state"}
    for bloque in ("paths", "modeling", "monitoring", "api", "mlflow"):
        assert cfg[bloque], f"bloque {bloque} vacío o ausente"


def test_los_umbrales_metodologicos_coinciden_con_las_recetas_del_eda():
    """El alfa y los umbrales que las recetas ya publicaron no se pueden mover por descuido."""
    assert valor("redundancia_pearson") == 0.70
    assert valor("redundancia_cramer") == 0.50
    assert valor("banda_revision_pearson") < valor("redundancia_pearson")
    assert valor("umbral_flags_pp") == 2.0


# --- cruce entre las recetas del EDA y el registro de cortes --------------------------------

TABLAS = ["bureau", "bureau_balance", "previous_application"]
# lo que cada agregación añade sin receta, con su motivo declarado en su módulo
SIN_RECETA = {"bureau": set(COLUMNAS_SIN_RECETA)}


def _receta(tabla):
    return yaml.safe_load((RAIZ / "config" / f"{tabla}_features.yaml").read_text())


@pytest.mark.parametrize("tabla", TABLAS)
def test_cada_feature_del_mapa_existe_en_su_receta(tabla):
    """El mapa no puede referirse a features que ni la receta ni la agregación declaran."""
    de_la_receta = {f["nombre"] for f in _receta(tabla)["features"]} | SIN_RECETA.get(tabla, set())
    del_mapa = set(CORTES_POR_FEATURE[tabla])
    assert del_mapa <= de_la_receta, f"features inventadas: {sorted(del_mapa - de_la_receta)}"


@pytest.mark.parametrize("tabla", TABLAS)
def test_cada_feature_con_corte_apunta_a_un_parametro_declarado(tabla):
    """Cada corte que lleva dentro una feature tiene que existir en el registro y ser accionable.

    Accionable significa reajustable sobre el split, o de dominio pero con su contraste
    declarado; un corte de dominio a secas dentro de una feature provisional sería un valor
    que nadie va a revisar.
    """
    for feature, cortes in CORTES_POR_FEATURE[tabla].items():
        assert cortes, f"{tabla}.{feature} está en el mapa sin ningún corte"
        for corte in cortes:
            p = parametro(corte)  # revienta si no está declarado
            accionable = p.procedencia in REAJUSTABLES or p.contraste_pendiente is not None
            assert accionable, f"{tabla}.{feature} depende de {corte}, que nadie revisa"


@pytest.mark.parametrize("tabla", TABLAS)
def test_una_feature_con_corte_solo_es_firme_si_su_descarte_es_estructural(tabla):
    """Si su construcción depende de un umbral que se puede mover, casi siempre es provisional.

    La excepción legítima es el descarte por redundancia estructural contra una hermana
    construida desde el mismo corte: ahí la redundancia se mantiene con cualquier valor del
    umbral, porque las dos salen del mismo booleano. Caso de PREV_EARLY_SETTLED_COUNT y
    _RATIO, que correlacionan 0,8586 y 0,8539 con la bandera que sí se conserva.
    """
    receta = {f["nombre"]: f for f in _receta(tabla)["features"]}
    for feature, cortes in CORTES_POR_FEATURE[tabla].items():
        # lo que no tiene receta no se decidió en firme contra nada
        if feature not in receta or receta[feature]["firmeza"] == "provisional":
            continue
        assert (
            receta[feature]["decision"] == "descartar"
        ), f"{tabla}.{feature} es firme, depende de {cortes} y no es un descarte"
        hermanas = [
            otra
            for otra, sus_cortes in CORTES_POR_FEATURE[tabla].items()
            if otra != feature
            and set(sus_cortes) & set(cortes)
            and receta[otra]["decision"] != "descartar"
        ]
        assert hermanas, (
            f"{tabla}.{feature} se descarta en firme por redundancia, pero ninguna feature "
            f"viva comparte sus cortes {cortes}"
        )


def test_todo_corte_reajustable_lo_usa_alguna_feature_o_es_transversal():
    """Un corte reajustable que no usa nadie es un valor huérfano.

    Los transversales no aparecen en el mapa porque no pertenecen a una feature concreta:
    son los percentiles de application_train y los tres pendientes de la capa 2b.
    """
    usados = set(features_con_corte())
    transversales = {n for n in PARAMS if n.startswith("app_")} | {
        "umbral_continuas_rb",
        "min_denominador_proporcion",
    }
    huerfanos = set(reajustables()) - usados - transversales
    assert not huerfanos, f"cortes reajustables que no usa ninguna feature: {sorted(huerfanos)}"


def test_el_suelo_del_denominador_es_dominio_con_su_contraste_declarado():
    """Reclasificarlo no puede llevarse por delante el contraste que la receta pide."""
    p = parametro("suelo_anios_denominador")
    assert p.procedencia == "dominio"
    assert p.contraste_pendiente is not None
    assert "suelo_anios_denominador" in con_contraste_pendiente()
    # las dos features que lo llevan dentro, una por tabla
    assert cortes_de("bureau", "BUREAU_CREDITS_PER_YEAR") == ("suelo_anios_denominador",)
    assert cortes_de("previous_application", "PREV_APPLICATIONS_PER_YEAR") == (
        "suelo_anios_denominador",
    )


def test_el_cap_de_la_antiguedad_del_coche_es_estimado():
    """64 era exactamente el p99, así que el criterio de negocio no sostenía el valor."""
    assert parametro("app_cap_p99_own_car_age").procedencia == "estimado"
    assert "app_own_car_age_max" not in PARAMS


def test_cortes_de_una_tabla_desconocida_revienta():
    with pytest.raises(KeyError, match="sin mapa de cortes"):
        cortes_de("installments", "LO_QUE_SEA")


def test_una_feature_sin_corte_devuelve_tupla_vacia():
    assert cortes_de("bureau", "BUREAU_ACTIVE_COUNT") == ()


# --- valor operativo: la referencia del EDA no basta por sí sola para usar un reajustable ---
# De la capa que fija estos valores sobre solo_train() está la 2a del winsorizador, en
# `ajustar_capa2a()`; el resto de la 2a y la 2b todavía no, así que para esos cortes estos tests
# describen el contrato que tendrán que cumplir y no un flujo ya integrado.


def test_todo_reajustable_empieza_sin_operativo_fijado():
    """Al importar el módulo nadie ha fijado nada, y así tiene que llegar a cada test.

    Lo sostiene la fixture `restaurar_params` de conftest.py: `ajustar_capa2a()` ya fija los diez
    cortes del winsorizador, así que sin ella este test dependería de qué haya corrido antes.
    """
    assert set(operativos_pendientes()) == set(reajustables())


@pytest.mark.parametrize("nombre", sorted(reajustables()))
def test_valor_revienta_para_cualquier_reajustable_sin_operativo(nombre):
    """No solo los tres sin referencia: los que ya tienen cifra del EDA tampoco valen tal cual."""
    with pytest.raises(ValueError, match="sin fijar"):
        valor(nombre)


def test_fijar_operativo_permite_usar_el_valor():
    fijar_operativo("app_winsor_amt_income_total", 1_500_000, n_train=245_993)
    assert valor("app_winsor_amt_income_total") == 1_500_000
    assert "app_winsor_amt_income_total" not in operativos_pendientes()
    # la referencia del EDA queda intacta al lado, como rastro de auditoría
    assert parametro("app_winsor_amt_income_total").valor_referencia == 1_417_500


def test_refijar_uno_ya_fijado_revienta_en_vez_de_pisarlo():
    """Sin esta guarda, dos ajustes sobre poblaciones distintas los resuelve un upsert.

    Gana el último y nadie se entera: es el mecanismo del orden del registro que en el EDA dejó
    una feature con el efecto de una población y la justificación de otra. Aquí la consecuencia
    sería un límite ajustado sobre una partición con el n de otra.
    """
    fijar_operativo("app_winsor_cnt_children", 9, n_train=245_993)
    with pytest.raises(ValueError, match="ya está fijado"):
        fijar_operativo("app_winsor_cnt_children", 99, n_train=12)
    # y el primero sigue en pie, que el intento fallido no puede dejarlo a medias
    assert valor("app_winsor_cnt_children") == 9
    assert parametro("app_winsor_cnt_children").n_train_operativo == 245_993


def test_refijar_con_sobrescribir_explicito_si_pisa():
    """La dirección contraria: pedirlo a las claras sí vale, como en `construir_split()`."""
    fijar_operativo("app_winsor_cnt_children", 9, n_train=245_993)
    fijar_operativo("app_winsor_cnt_children", 99, n_train=12, sobrescribir=True)
    assert valor("app_winsor_cnt_children") == 99
    assert parametro("app_winsor_cnt_children").n_train_operativo == 12


def test_fijar_operativo_exige_n_train_positivo():
    with pytest.raises(ValueError, match="positivo"):
        fijar_operativo("app_winsor_cnt_children", 8, n_train=0)


def test_fijar_operativo_sobre_un_dominio_revienta():
    with pytest.raises(ValueError, match="no es reajustable"):
        fijar_operativo("redundancia_pearson", 0.8, n_train=1000)


def test_un_dominio_no_admite_valor_operativo_directo():
    with pytest.raises(ValueError, match="no tiene valor operativo"):
        Parametro(1, "dominio", "d", "f", valor_operativo=2, n_train_operativo=10)


def test_valor_operativo_sin_n_train_asociado_revienta():
    with pytest.raises(ValueError, match="n_train_operativo"):
        Parametro(None, "medido", "d", "f", valor_operativo=2)


def test_la_referencia_del_eda_no_se_usa_hasta_que_alguien_la_declara_sobre_train():
    """El número que ya vivía en PARAMS no basta solo, aunque sea el mismo que se acabe fijando."""
    referencia = parametro("prev_plazo_largo_cuotas").valor_referencia
    with pytest.raises(ValueError, match="sin fijar"):
        valor("prev_plazo_largo_cuotas")
    fijar_operativo("prev_plazo_largo_cuotas", referencia, n_train=245_993)
    assert valor("prev_plazo_largo_cuotas") == referencia
