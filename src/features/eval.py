"""Evaluadores de señal de features agregadas a nivel cliente (Fase B del EDA).

Encapsulan el z-test de proporciones de las flags y el Mann-Whitney/rank-biserial de las
continuas, con el control de composición integrado: una feature pasada como string se
auto-condiciona por notna() (excluye a los clientes sin filas en la tabla auxiliar); como
tupla con fill0=True se mide global (los mete en el grupo 0 y contamina la baseline). Ver
CLAUDE.md §8.5. Reutilizable por las tablas auxiliares restantes (bureau_balance,
previous_application, POS_CASH_balance, credit_card_balance, installments_payments).

Uso:
    ev = EvaluadorSenal(df_cliente, TARGET)
    ev.evaluar_flags([("HAS_X", valores), ("HAS_X", valores, poblacion)])
    ev.evaluar_continuas(["RATIO_X", ("COUNT_X", True, poblacion)])
    # acumuladores para el ranking consolidado:
    ev.fb_flags, ev.fb_cont, ev.fb_comparisons
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm


class EvaluadorSenal:
    """Mide la señal de flags/continuas contra el TARGET con control de composición."""

    def __init__(self, df: pd.DataFrame, target: Any) -> None:
        self.df = df
        self.target = np.asarray(target)
        self.fb_flags = []  # acumulador de flags (upsert por feature)
        self.fb_cont = []  # acumulador de continuas (upsert por feature)
        self.fb_comparisons = []  # toda comparación emitida (familia Bonferroni de referencia)

    @staticmethod
    def _registrar(acc: list, entry: dict) -> None:
        """Upsert por nombre: la medición condicionada reemplaza a la global de la misma feature."""
        for i, e in enumerate(acc):
            if e["feature"] == entry["feature"]:
                acc[i] = entry
                return
        acc.append(entry)

    def evaluar_flags(self, flags_list: Sequence[tuple]) -> None:
        """Métricas de flags binarias. Item: (nombre, valores[, poblacion, etiqueta])."""
        rows = []
        for item in flags_list:
            name, values = item[0], item[1]
            pop = item[2] if len(item) > 2 else None
            etiqueta = (
                item[3]
                if len(item) > 3
                else ("global (nulo=0)" if pop is None else "con historial")
            )

            v = np.asarray(values)
            p = np.ones(len(v), bool) if pop is None else np.asarray(pop)
            m1, m0 = p & (v == 1), p & (v == 0)
            n1, n0 = int(m1.sum()), int(m0.sum())

            if n1 == 0 or n0 == 0:
                continue

            s1, s0 = self.target[m1].sum(), self.target[m0].sum()
            p1, p0 = s1 / n1, s0 / n0
            pp = (s1 + s0) / (n1 + n0)

            # z-test de proporciones
            z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
            pv = 2 * norm.sf(abs(z))

            rows.append(
                {
                    "Variable": name,
                    "Población": etiqueta,
                    "Tasa (1)": f"{round(p1*100, 2)}% (n={n1})",
                    "Tasa (0)": f"{round(p0*100, 2)}% (n={n0})",
                    "Delta": f"{(p1-p0)*100:+.2f} pp",
                    "z-stat": f"{round(z, 1)}",
                    "p-valor": f"{pv}",
                }
            )
            self.fb_comparisons.append((name, etiqueta))
            # la última medición de cada feature manda: listar siempre global primero
            # y condicionada después
            self._registrar(
                self.fb_flags,
                {
                    "feature": name,
                    "tipo": "flag",
                    "poblacion": etiqueta,
                    "efecto": (p1 - p0) * 100,
                    "p": pv,
                    "n_pos": n1,
                },
            )

        from IPython.display import display  # import perezoso: no acopla src.features a IPython

        display(pd.DataFrame(rows).style.hide(axis="index"))

    def evaluar_continuas(self, cont_list: Sequence[Any]) -> None:
        """Mann-Whitney U + rank-biserial. Item: nombre | (nombre, fill0[, poblacion, etiqueta]).

        Sin fill0 el notna() ya excluye a los clientes sin historial (auto-condicionada);
        con fill0=True los mete en el grupo 0 y contamina la medición (control de composición).
        """
        rows = []
        for item in cont_list:
            if isinstance(item, tuple):
                name, fill0 = item[0], item[1]
                pop = item[2] if len(item) > 2 else None
                etiqueta = item[3] if len(item) > 3 else None
            else:
                name, fill0, pop, etiqueta = item, False, None, None

            s = self.df[name].fillna(0) if fill0 else self.df[name]
            mask = s.notna().values
            if pop is not None:
                mask = mask & np.asarray(pop)

            if etiqueta is None:
                if pop is not None:
                    etiqueta = "con historial"
                elif fill0:
                    etiqueta = "global (fill 0)"
                else:
                    etiqueta = "no nulos (auto-cond.)"

            vals = s.values[mask]
            t = self.target[mask]
            x, y = vals[t == 0], vals[t == 1]

            if len(x) == 0 or len(y) == 0:
                continue

            u, pv = mannwhitneyu(x, y, alternative="two-sided")
            r = abs((2 * u) / (len(x) * len(y)) - 1)

            rows.append(
                {
                    "Variable": name,
                    "Población": etiqueta,
                    "n": f"{len(vals)}",
                    "Rank-Biserial": f"{round(r, 4)}",
                    "p-valor": f"{pv}",
                    "Mediana (0)": f"{round(np.median(x), 1)}",
                    "Mediana (1)": f"{round(np.median(y), 1)}",
                }
            )
            self.fb_comparisons.append((name, etiqueta))
            self._registrar(
                self.fb_cont,
                {
                    "feature": name,
                    "tipo": "continua",
                    "poblacion": etiqueta,
                    "efecto": r,
                    "p": pv,
                    "n_pos": len(vals),
                },
            )

        from IPython.display import display  # import perezoso: no acopla src.features a IPython

        display(pd.DataFrame(rows).style.hide(axis="index"))
