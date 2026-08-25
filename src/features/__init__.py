"""src.features — ingeniería de features: WoE, aggregations, ratios, listado de categorías, recomendaciones de codificación, evaluadores de señal."""
from src.features.selection import obtener_categorias, recomendar_codificacion
from src.features.eval import EvaluadorSenal

__all__: list[str] = ["obtener_categorias", "recomendar_codificacion", "EvaluadorSenal"]
