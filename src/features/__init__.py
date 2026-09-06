"""src.features — feature engineering: WoE, aggregations, ratios, listado de categorías,
recomendaciones de codificación, evaluadores de señal."""

from src.features.eval import EvaluadorSenal
from src.features.selection import obtener_categorias, recomendar_codificacion

__all__: list[str] = ["obtener_categorias", "recomendar_codificacion", "EvaluadorSenal"]
