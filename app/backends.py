"""Плагинные бэкенды энкодера и хранилища.

Две реализации одного интерфейса:

1. **Heavy** (`RAG_BACKEND=heavy`, по умолчанию для локальной разработки).
   - sentence-transformers + torch (~1.5 ГБ RAM)
   - chromadb векторное хранилище
   - берётся модель `intfloat/multilingual-e5-small` с HuggingFace

2. **Light** (`RAG_BACKEND=light`, для Render Free 512 МБ).
   - onnxruntime + tokenizers (~250 МБ RAM)
   - numpy-массив вместо ChromaDB (18 чанков = 28 КБ)
   - использует готовый bundle из desktop/bundle/

Логика поиска и сборки ответа (`app/rag.py`) одна и та же для обоих
бэкендов — отличаются только encode_query / encode_passages / search_in_index.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import config

logger = logging.getLogger(__name__)

# Какой бэкенд использовать. По умолчанию для Render — light.
BACKEND = os.getenv("RAG_BACKEND", "light").lower()
if BACKEND not in {"light", "heavy"}:
    BACKEND = "light"


# Путь к предсобранному ONNX-bundle (используется только в light-режиме).
BUNDLE_DIR = Path(os.getenv("RAG_BUNDLE_DIR", config.BASE_DIR / "desktop" / "bundle"))
ONNX_MODEL_DIR = BUNDLE_DIR / "model"
ONNX_INDEX_PATH = BUNDLE_DIR / "index.npz"


class EncoderBackend(Protocol):
    """Единый интерфейс энкодера для app/rag.py."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def model_version(self) -> str: ...


class VectorStore(Protocol):
    """Единый интерфейс векторного хранилища."""

    def is_ready(self) -> bool: ...
    def count(self) -> int: ...
    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]: ...
    def meta(self) -> dict[str, Any]: ...
    def reason(self) -> str: ...


# ---------------------------------------------------------------------------
# Light backend: ONNX Runtime + numpy
# ---------------------------------------------------------------------------


@dataclass
class LightIndexMeta:
    n_files: int
    n_chunks: int
    files: list[str]
    model: str
    threshold: float


class _LightEncoder:
    """Адаптер OnnxEncoder под интерфейс, ожидаемый app/rag.py."""

    def __init__(self):
        from desktop.onnx_encoder import OnnxEncoder  # type: ignore

        if not ONNX_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"ONNX-bundle не найден: {ONNX_MODEL_DIR}. "
                f"Соберите его: python scripts/build_onnx_bundle_tiny.py"
            )
        info = self._load_info()
        model_name = info.get("model", config.MODEL_NAME)
        # Префиксы E5 берём из метаданных bundle (по умолчанию True для e5-small,
        # False для rubert-tiny2). Метаданные приоритетнее config.E5_PREFIXES,
        # т.к. конфиг может быть переопределён env-переменной.
        self._use_prefixes = info.get("e5_prefixes", "e5" in model_name.lower())
        # memory_efficient=True: критично для Render Free (512 МБ RAM).
        # Экономит ~80 МБ ценой ~5-10 % скорости.
        self._encoder = OnnxEncoder(
            ONNX_MODEL_DIR,
            use_prefixes=self._use_prefixes,
            memory_efficient=True,
        )
        self._info = info

    def _load_info(self) -> dict:
        import json

        info_path = BUNDLE_DIR / "index.json"
        if not info_path.exists():
            return {}
        return json.loads(info_path.read_text(encoding="utf-8")).get("info", {})

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.encode_queries([text])[0].tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode_queries(texts).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode_passages(texts).tolist()

    def model_version(self) -> str:
        return self._info.get("model", config.MODEL_NAME)


class LightVectorStore:
    """Numpy-хранилище: 18 чанков, поиск — простое матричное умножение."""

    def __init__(self):
        import numpy as np
        from desktop.onnx_encoder import load_index  # type: ignore

        if not ONNX_INDEX_PATH.exists():
            self._ready = False
            self._reason = (
                f"Индекс не найден: {ONNX_INDEX_PATH}. "
                f"Соберите bundle: python scripts/build_onnx_bundle.py"
            )
            return

        self._vectors, self._chunks, self._info = load_index(ONNX_INDEX_PATH)
        self._np = np
        self._ready = True
        self._reason = ""

    def is_ready(self) -> bool:
        return self._ready

    def count(self) -> int:
        return int(self._vectors.shape[0]) if self._ready else 0

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self._ready:
            return []
        from desktop.onnx_encoder import search as np_search

        hits = np_search(self._vectors, self._np.array(vector, dtype="float32"), top_k)
        result = []
        for idx, score in hits:
            ch = self._chunks[idx]
            result.append({
                "file": ch["file"],
                "chunk_id": ch["chunk_id"],
                "score": round(float(score), 4),
                "text": ch.get("text", ""),
            })
        return result

    def meta(self) -> dict[str, Any]:
        if not self._ready:
            return {}
        return {
            "model": self._info.get("model", config.MODEL_NAME),
            "n_files": self._info.get("n_files", 0),
            "n_chunks": self._info.get("n_chunks", 0),
            "files": self._info.get("files", []),
            "built_for": self._info.get("built_for", "light"),
        }

    def reason(self) -> str:
        return self._reason


# ---------------------------------------------------------------------------
# Heavy backend: sentence-transformers + ChromaDB (для локальной разработки)
# ---------------------------------------------------------------------------


class _HeavyEncoder:
    """Sentence-transformers обёртка. Делегирует в app/embedder."""

    def __init__(self):
        # Импортируем тут, чтобы в light-режиме torch вообще не загружался.
        from . import embedder
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_queries(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_passages(texts)

    def model_version(self) -> str:
        return self._embedder.model_version()


class HeavyVectorStore:
    """ChromaDB-backed store. Делегирует в app/store + app/embedder для сборки meta."""

    def __init__(self):
        from . import store

        self._store = store
        meta = store.load_meta()
        if meta is None:
            self._ready = False
            self._reason = "Индекс не построен. Запустите: python scripts/build_index.py"
            return
        if meta.get("model") != config.MODEL_NAME:
            self._ready = False
            self._reason = (
                f"Индекс построен моделью {meta.get('model')}, а сервис запущен с "
                f"{config.MODEL_NAME}. Переиндексируйте: python scripts/build_index.py"
            )
            return

        try:
            self._collection = store.get_collection()
        except Exception as exc:  # noqa: BLE001
            self._ready = False
            self._reason = f"Ошибка открытия ChromaDB: {exc}"
            return

        if self._collection.count() == 0:
            self._ready = False
            self._reason = "Индекс пуст. Запустите: python scripts/build_index.py"
            return

        self._ready = True
        self._reason = ""
        self._meta = meta

    def is_ready(self) -> bool:
        return self._ready

    def count(self) -> int:
        return int(self._collection.count()) if self._ready else 0

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self._ready:
            return []
        return self._store.query(self._collection, vector, top_k)

    def meta(self) -> dict[str, Any]:
        if not self._ready:
            return {}
        return self._meta

    def reason(self) -> str:
        return self._reason


# ---------------------------------------------------------------------------
# Фабрика
# ---------------------------------------------------------------------------


def get_encoder() -> EncoderBackend:
    """Возвращает энкодер в зависимости от RAG_BACKEND."""
    if BACKEND == "light":
        return _LightEncoder()
    return _HeavyEncoder()


def get_store() -> VectorStore:
    """Возвращает векторное хранилище в зависимости от RAG_BACKEND."""
    if BACKEND == "light":
        return LightVectorStore()
    return HeavyVectorStore()


def warmup_encoder(encoder: EncoderBackend) -> None:
    """Прогрев: первый прогон onnxruntime самый медленный."""
    if BACKEND == "light":
        encoder.embed_query("разогрев модели")
    else:
        # Heavy-бэкенд сам греется при загрузке модели.
        encoder.embed_query("разогрев")
