"""Fixtures compartidas por toda la suite.

Existe por una sola cosa: `PARAMS` es un diccionario a nivel de módulo y `fijar_operativo()`
escribe en él, así que cualquier test que fije un corte se lo deja fijado a los que corran
después. Sin restaurar, `test_todo_reajustable_empieza_sin_operativo_fijado` y la parametrizada
de `test_valor_revienta_para_cualquier_reajustable_sin_operativo` pasan a depender del orden de
la suite: comprobado con un test que fija `app_winsor_cnt_children` y no restaura, dos rojos en
test_params.py.

Va autouse y no a petición porque la protección tiene que valer para el test que todavía no
está escrito. Cuatro ficheros llevaban su propia copia de este mismo snapshot, tres de ellas
nacidas al fijar los primeros cortes en la capa 2a, y la que se olvidase de ponerla rompía a
otro fichero y no a sí misma.
"""

import pytest

from src.features.params import PARAMS


@pytest.fixture(autouse=True)
def restaurar_params():
    """Snapshot de PARAMS para deshacer cualquier fijar_operativo() que haga el test.

    Los Parametro son inmutables (frozen), así que una copia superficial del dict basta:
    fijar_operativo() reemplaza la entrada entera, nunca muta un Parametro existente.
    """
    original = dict(PARAMS)
    yield
    PARAMS.clear()
    PARAMS.update(original)
