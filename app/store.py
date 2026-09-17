"""Векторное хранилище (ChromaDB, persistent) и метаданные индекса."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_ADD_BATCH = 500  # Chroma не любит слишком большие батчи


def get_client():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client=None):
    """Коллекция с косинусной метрикой (по умолчанию Chroma использует L2)."""
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client=None):
    """Полностью пересоздаёт коллекцию — индексация всегда «с нуля»."""
    client = client or get_client()
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:  # коллекции ещё нет — это нормально
        logger.debug("collection %s не существовала", config.COLLECTION_NAME)
    return get_collection(client)


def add_chunks(collection, ids, embeddings, documents, metadatas) -> None:
    for i in range(0, len(ids), _ADD_BATCH):
        sl = slice(i, i + _ADD_BATCH)
        collection.add(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )


def query(collection, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Возвращает список словарей: file, chunk_id, score (косинусная схожесть), text."""
    res = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]
    for text, meta, dist in zip(documents, metadatas, distances):
        # для space="cosine" Chroma отдаёт distance = 1 - cosine_similarity
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        hits.append(
            {
                "file": meta.get("file", "unknown"),
                "chunk_id": int(meta.get("chunk_id", -1)),
                "score": round(score, 4),
                "text": text or "",
            }
        )
    return hits


# --- Метаданные индекса -------------------------------------------------


def save_meta(extra: dict[str, Any]) -> None:
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **config.as_dict(),
        **extra,
    }
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_meta() -> dict[str, Any] | None:
    if not config.META_PATH.exists():
        return None
    try:
        return json.loads(config.META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("index_meta.json повреждён")
        return None
