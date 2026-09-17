"""Единый энкодер: одна и та же модель используется при индексации и при запросе.

Модель грузится один раз (lru_cache) и переиспользуется.
Для моделей семейства E5 автоматически подставляются служебные префиксы
"query: " и "passage: " — без них качество поиска заметно падает.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

from . import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_model(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    model_name = name or config.MODEL_NAME
    logger.info("loading encoder model=%s", model_name)
    return SentenceTransformer(model_name)


def _encode(texts: list[str], batch_size: int) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # косинус = скалярное произведение
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_passages(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Векторизация фрагментов документов (используется при индексации)."""
    items = list(texts)
    if config.E5_PREFIXES:
        items = [f"passage: {t}" for t in items]
    return _encode(items, batch_size)


def embed_queries(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Векторизация вопросов пользователя."""
    items = list(texts)
    if config.E5_PREFIXES:
        items = [f"query: {t}" for t in items]
    return _encode(items, batch_size)


def embed_query(text: str) -> list[float]:
    return embed_queries([text])[0]


# Совместимость со старым именем.
embed_texts = embed_passages


def model_version() -> str:
    return config.MODEL_NAME
