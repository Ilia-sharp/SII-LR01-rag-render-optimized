#!/usr/bin/env python
"""Быстрая проверка ответа без запуска сервера.

Пример: python scripts/ask.py "За сколько дней подавать заявление на отпуск?"

Поддерживает оба бэкенда (light = ONNX+numpy по умолчанию,
heavy = torch+chromadb для локальной разработки).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import backends, rag  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print('Использование: python scripts/ask.py "ваш вопрос"')
        return 2

    question = " ".join(sys.argv[1:])

    try:
        encoder = backends.get_encoder()
        store = backends.get_store()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка инициализации бэкенда ({backends.BACKEND}): {exc}")
        print("Возможно, нужно собрать bundle: python scripts/build_onnx_bundle_tiny.py")
        return 1

    if not store.is_ready():
        print(f"Индекс не готов: {store.reason()}")
        return 1

    started = time.perf_counter()
    payload, n = rag.answer_question(question, store, encoder)
    latency_ms = (time.perf_counter() - started) * 1000

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[backend={backends.BACKEND} chunks_found={n} latency_ms={latency_ms:.1f}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
