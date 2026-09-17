"""Единый модуль нарезки текста.

Используется И при индексации (scripts/build_index.py), И при обработке
запроса (app/rag.py) — другого кода нарезки в проекте нет.
Зависимостей от сторонних библиотек нет, поэтому модуль легко тестируется.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа."""

    chunk_id: int  # порядковый номер внутри файла, начиная с 0
    text: str
    n_words: int


def normalize(text: str) -> str:
    """Приводит текст к единому виду: убирает мусорные пробелы и переносы."""
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_words: int | None = None,
    overlap: float | None = None,
) -> list[Chunk]:
    """Режет текст на пересекающиеся окна фиксированной длины (в словах).

    :param chunk_words: размер окна в словах (по умолчанию config.CHUNK_WORDS)
    :param overlap: доля перекрытия 0.1-0.2 (по умолчанию config.CHUNK_OVERLAP)
    """
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap

    if chunk_words <= 0:
        raise ValueError("chunk_words должен быть > 0")
    if chunk_words > 600:
        raise ValueError("chunk_words > 600 слов — нарушение требований ЛР")
    if not 0.0 <= overlap < 0.5:
        raise ValueError("overlap должен быть в диапазоне [0.0, 0.5)")

    text = normalize(text)
    if not text:
        return []

    words = text.split(" ")
    step = max(1, int(round(chunk_words * (1.0 - overlap))))

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        piece = " ".join(window).strip()
        if piece:
            chunks.append(Chunk(chunk_id=len(chunks), text=piece, n_words=len(window)))
        if start + chunk_words >= len(words):
            break
        start += step

    return chunks


_HEADING_RE = re.compile(r"^#{1,6}\s")


def split_sentences(text: str) -> list[str]:
    """Разбиение на предложения (для экстрактивного ответа).

    Строки-заголовки markdown отбрасываются: в индексе они полезны, а в тексте
    ответа выглядят мусором.
    """
    sentences: list[str] = []
    for line in normalize(text).split("\n"):
        line = line.strip()
        if not line or _HEADING_RE.match(line):
            continue
        for part in re.split(r"(?<=[.!?…])\s+", line):
            part = part.strip(" -–—•*\t|")
            if part:
                sentences.append(part)
    return sentences
