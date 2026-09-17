#!/usr/bin/env python
"""Калибровка порога схожести: python scripts/calibrate.py

Прогоняет вопросы «по корпусу» и «вне корпуса», печатает лучшие score
и подсказывает подходящее значение RAG_SCORE_THRESHOLD.

Поддерживает оба бэкенда (light = ONNX+numpy по умолчанию,
heavy = torch+chromadb для локальной разработки).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import backends, config, rag  # noqa: E402

IN_CORPUS = [
    "За сколько дней нужно подать заявление на отпуск?",
    "Какой лимит на проживание в гостинице в Москве?",
    "Как сбросить пароль от корпоративной учётной записи?",
    "Сколько дней длится испытательный срок?",
    "Какая компенсация за спортзал?",
    "Когда выплачивается аванс?",
    "Сколько дней в неделю нужно быть в офисе при гибридном формате?",
    "Как оплачивается переработка?",
    "Куда сообщать о фишинговом письме?",
]

OUT_OF_CORPUS = [
    "Как приготовить борщ из марсианской свёклы?",
    "Какая максимальная скорость у гепарда?",
    "Кто выиграл чемпионат мира по футболу в 1998 году?",
    "Как настроить квантовый компьютер дома?",
    "Какая столица Австралии?",
]


def main() -> int:
    try:
        encoder = backends.get_encoder()
        store = backends.get_store()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка инициализации бэкенда: {exc}")
        return 1

    if not store.is_ready():
        print(f"Индекс не готов: {store.reason()}")
        return 1

    def best_scores(questions: list[str]) -> list[float]:
        out = []
        for q in questions:
            hits = rag.search(q, store, encoder)
            score = hits[0].score if hits else 0.0
            overlap = rag.lexical_overlap(q, hits[0].text) if hits else 0
            src = f"{hits[0].file}#{hits[0].chunk_id}" if hits else "-"
            print(f"  {score:.4f} | overlap={overlap} | {src:<28} | {q}")
            out.append(score)
        return out

    print(f"Бэкенд: {backends.BACKEND}")
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Текущий порог: {config.SCORE_THRESHOLD} | lexical_guard={config.LEXICAL_GUARD}\n")

    print("ВОПРОСЫ ПО КОРПУСУ (score должен быть выше порога):")
    good = best_scores(IN_CORPUS)
    print("\nВОПРОСЫ ВНЕ КОРПУСА (должны отсекаться):")
    bad = best_scores(OUT_OF_CORPUS)

    print("\n--- Итог -------------------------------------------------")
    print(f"  по корпусу : min={min(good):.4f} max={max(good):.4f}")
    print(f"  вне корпуса: min={min(bad):.4f} max={max(bad):.4f}")
    if min(good) > max(bad):
        suggested = (min(good) + max(bad)) / 2
        print(f"  Рекомендуемый порог: RAG_SCORE_THRESHOLD={suggested:.2f}")
    else:
        print("  Классы пересекаются — порог не разделяет их полностью.")
        print("  Оставьте включённым lexical_guard (RAG_LEXICAL_GUARD=1).")
    print("----------------------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

