"""Поиск релевантных чанков и построение ответа.

Ответ строится ТОЛЬКО из текста найденных чанков (экстрактивно):
генеративной модели нет, поэтому «сочинить» факт сервис не может.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from . import chunker, config
from .backends import EncoderBackend

logger = logging.getLogger(__name__)

# Короткие и служебные слова не участвуют в лексической проверке.
_STOPWORDS = {
    "как", "что", "где", "когда", "кто", "чем", "для", "при", "или", "если",
    "нужно", "надо", "можно", "какой", "какая", "какие", "сколько", "почему",
    "это", "того", "тому", "быть", "есть", "меня", "мне", "мой", "моя",
}
_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass
class Hit:
    file: str
    chunk_id: int
    score: float
    text: str

    def as_source(self) -> dict[str, Any]:
        return {"file": self.file, "chunk_id": self.chunk_id, "score": self.score}


# --- Лексическая проверка ------------------------------------------------


def _stems(text: str) -> set[str]:
    """Грубая нормализация: обрезаем окончание, оставляя первые 5 символов."""
    words = _WORD_RE.findall(text.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in _STOPWORDS}


def lexical_overlap(question: str, text: str) -> int:
    """Сколько значимых слов вопроса встречается во фрагменте."""
    return len(_stems(question) & _stems(text))


# --- Поиск ---------------------------------------------------------------


def search(question: str, store, encoder: EncoderBackend, top_k: int | None = None) -> list[Hit]:
    """Векторный поиск top-k чанков по вопросу."""
    top_k = top_k or config.TOP_K
    # Вопрос проходит через тот же модуль нормализации, что и документы.
    normalized = chunker.normalize(question)
    vector = encoder.embed_query(normalized)
    raw = store.search(vector, top_k)
    return [Hit(**item) for item in raw]


def filter_hits(question: str, hits: list[Hit]) -> list[Hit]:
    """Отсекает фрагменты ниже порога схожести и не связанные с вопросом лексически."""
    relevant = [h for h in hits if h.score >= config.SCORE_THRESHOLD]
    if config.LEXICAL_GUARD:
        relevant = [
            h
            for h in relevant
            if lexical_overlap(question, h.text) >= config.LEXICAL_MIN_OVERLAP
        ]
    return relevant


# --- Ответ ---------------------------------------------------------------


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_answer(question: str, hits: list[Hit], encoder: Optional[EncoderBackend] = None) -> str:
    """Склейка топ-2 чанков + экстрактивное резюме.

    Из текста топ-чанков выбираются предложения, наиболее близкие к вопросу.
    Ни одно слово не добавляется от себя, кроме служебного префикса.

    ``encoder`` опционален, чтобы тесты могли вызывать функцию без модели
    (пустой список ``hits`` возвращается раньше, чем энкодер понадобится).
    """
    if not hits:
        return config.NO_ANSWER

    top = hits[: config.ANSWER_CHUNKS]

    sentences: list[str] = []
    for hit in top:
        for sentence in chunker.split_sentences(hit.text):
            if len(sentence) >= 25 and sentence not in sentences:
                sentences.append(sentence)

    if not sentences:
        return config.ANSWER_PREFIX + top[0].text[:500].strip()

    if encoder is None:
        # Деградация без энкодера: просто берём первые N предложений.
        keep = list(range(min(config.MAX_ANSWER_SENTENCES, len(sentences))))
    else:
        q_vec = encoder.embed_query(chunker.normalize(question))
        s_vecs = encoder.embed_passages(sentences)
        scored = sorted(
            ((_dot(q_vec, v), i) for i, v in enumerate(s_vecs)),
            key=lambda x: x[0],
            reverse=True,
        )
        keep = sorted(i for _, i in scored[: config.MAX_ANSWER_SENTENCES])
    summary = " ".join(sentences[i] for i in keep).strip()

    if not summary.endswith((".", "!", "?", "…")):
        summary += "."
    return config.ANSWER_PREFIX + summary


def answer_question(question: str, store, encoder: EncoderBackend) -> tuple[dict[str, Any], int]:
    """Главный сценарий. Возвращает (payload, число релевантных чанков)."""
    hits = search(question, store, encoder)
    relevant = filter_hits(question, hits)

    meta = store.meta() or {}
    model_version = meta.get("model", encoder.model_version())

    if not relevant:
        best = hits[0].score if hits else 0.0
        logger.debug("below threshold: best=%.4f thr=%.2f", best, config.SCORE_THRESHOLD)
        return (
            {"answer": config.NO_ANSWER, "sources": [], "model_version": model_version},
            0,
        )

    payload = {
        "answer": build_answer(question, relevant, encoder),
        "sources": [h.as_source() for h in relevant[: config.TOP_K]],
        "model_version": model_version,
    }
    return payload, len(relevant)
