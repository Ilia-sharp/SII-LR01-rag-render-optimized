#!/usr/bin/env python
"""Индексация корпуса: python scripts/build_index.py

Читает все документы из data/, режет их единым модулем app/chunker.py,
векторизует тем же энкодером, что используется при запросе, и складывает
в персистентное хранилище ChromaDB (.chroma/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Позволяет запускать скрипт как `python scripts/build_index.py` из корня проекта.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import chunker, config, embedder, loader, store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Построение векторного индекса")
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="папка с документами")
    p.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    p.add_argument("--overlap", type=float, default=config.CHUNK_OVERLAP)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    logger.info("data_dir=%s", args.data_dir)
    documents = loader.load_documents(args.data_dir)
    logger.info("найдено документов: %d", len(documents))

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    per_file: dict[str, int] = {}

    for doc in documents:
        chunks = chunker.split_into_chunks(doc.text, args.chunk_words, args.overlap)
        per_file[doc.name] = len(chunks)
        for ch in chunks:
            ids.append(f"{doc.name}::{ch.chunk_id}")
            texts.append(ch.text)
            metadatas.append(
                {"file": doc.name, "chunk_id": ch.chunk_id, "n_words": ch.n_words}
            )

    if not texts:
        logger.error("не получилось нарезать ни одного чанка — проверьте data/")
        return 1

    logger.info("чанков всего: %d | модель: %s", len(texts), config.MODEL_NAME)
    logger.info("векторизация (первый запуск скачивает модель, это может занять 1-2 минуты)...")
    embeddings = embedder.embed_passages(texts, batch_size=args.batch_size)

    collection = store.reset_collection()
    store.add_chunks(collection, ids, embeddings, texts, metadatas)

    store.save_meta(
        {
            "n_files": len(documents),
            "n_chunks": len(texts),
            "files": per_file,
            "data_dir": str(args.data_dir),
        }
    )

    elapsed = time.perf_counter() - started
    print("\n--- Индекс построен -------------------------------------")
    for name, n in per_file.items():
        print(f"  {name:<28} чанков: {n}")
    print(f"  Итого: {len(documents)} файлов, {len(texts)} чанков")
    print(f"  Модель: {config.MODEL_NAME}")
    print(f"  Хранилище: {config.CHROMA_DIR}")
    print(f"  Время: {elapsed:.1f} c")
    print("---------------------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
