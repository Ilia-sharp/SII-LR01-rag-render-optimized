"""Единая точка конфигурации сервиса.

Все пути — относительные (через pathlib), абсолютных хардкодов нет.
Любой параметр переопределяется переменной окружения.
"""

from __future__ import annotations

import os
from pathlib import Path

# Корень проекта = каталог, в котором лежит папка app/
BASE_DIR = Path(__file__).resolve().parents[1]

# --- Пути ---------------------------------------------------------------
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", BASE_DIR / ".chroma"))
META_PATH = CHROMA_DIR / "index_meta.json"

# --- Модель -------------------------------------------------------------
# Для Render Free Tier (512 МБ RAM) используется компактная модель
# cointegrated/rubert-tiny2 (~29 МБ int8 ONNX, 312-dim, контекст 512 токенов).
# Старая конфигурация intfloat/multilingual-e5-small (~470 МБ) слишком тяжёлая
# для Render Free: на этапе загрузки она использует ~550 МБ RAM и OOM'ит.
#
# Чтобы переключиться обратно на e5-small (например, локально):
#   1. RAG_MODEL=intfloat/multilingual-e5-small
#   2. RAG_SCORE_THRESHOLD=0.80
#   3. Пересоберите bundle: python scripts/build_onnx_bundle.py
#      (это первая версия скрипта — для e5-small)
MODEL_NAME = os.getenv("RAG_MODEL", "cointegrated/rubert-tiny2")

# Модели семейства E5 требуют префиксов "query: " / "passage: ".
# rubert-tiny2 — НЕ E5, префиксы не нужны.
E5_PREFIXES = "e5" in MODEL_NAME.lower()

# --- Хранилище ----------------------------------------------------------
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "documents")

# --- Нарезка на чанки ---------------------------------------------------
# ~220 слов ≈ 300-450 токенов для русского текста (требование: 256-512 токенов,
# не более 600 слов), перекрытие 15 % (требование: 10-20 %).
CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "220"))
CHUNK_OVERLAP = float(os.getenv("RAG_CHUNK_OVERLAP", "0.15"))

# --- Поиск и ответ ------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "3"))                  # сколько фрагментов в sources
ANSWER_CHUNKS = int(os.getenv("RAG_ANSWER_CHUNKS", "2"))  # склейка топ-2 чанков
MAX_ANSWER_SENTENCES = int(os.getenv("RAG_MAX_ANSWER_SENTENCES", "3"))


def _default_threshold(model_name: str) -> str:
    """Порог схожести зависит от модели.

    У E5 косинусные схожести «сжаты» (нерелевантная пара ≈ 0.75), поэтому
    порог высокий. У классических sentence-transformers шкала растянута.
    rubert-tiny2 калиброван на ~0.35 (см. scripts/build_onnx_bundle_tiny.py).
    Калибровка на своём корпусе: python scripts/calibrate.py
    """
    name = model_name.lower()
    if "e5" in name:
        return "0.80"
    if "rubert-tiny" in name:
        return "0.35"
    return "0.35"


SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", _default_threshold(MODEL_NAME)))

# Дополнительная защита от ответа не по теме: в найденном чанке должно
# встретиться хотя бы одно значимое слово из вопроса.
# Отключается через RAG_LEXICAL_GUARD=0.
LEXICAL_GUARD = os.getenv("RAG_LEXICAL_GUARD", "1") not in {"0", "false", "False"}
LEXICAL_MIN_OVERLAP = int(os.getenv("RAG_LEXICAL_MIN_OVERLAP", "1"))

NO_ANSWER = "Не нашёл информации по вашему вопросу"
ANSWER_PREFIX = "Согласно документам: "

# --- Прочее -------------------------------------------------------------
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO")


def as_dict() -> dict:
    """Снимок конфигурации для логов и метаданных индекса."""
    return {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "lexical_guard": LEXICAL_GUARD,
    }
