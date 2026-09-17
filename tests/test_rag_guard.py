"""Тесты отсечения нерелевантных фрагментов (модель не требуется)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, rag  # noqa: E402

CHUNK = (
    "Заявление на отпуск подаётся не позднее чем за 14 рабочих дней до даты "
    "начала отпуска через портал HR-Self-Service."
)


def test_lexical_overlap_positive():
    assert rag.lexical_overlap("За сколько дней подать заявление на отпуск?", CHUNK) >= 2


def test_lexical_overlap_zero_for_foreign_topic():
    assert rag.lexical_overlap("Как приготовить борщ из марсианской свёклы?", CHUNK) == 0


def test_filter_drops_low_score():
    hits = [rag.Hit(file="a.md", chunk_id=0, score=config.SCORE_THRESHOLD - 0.2, text=CHUNK)]
    assert rag.filter_hits("Когда подавать заявление на отпуск?", hits) == []


def test_filter_keeps_relevant():
    hits = [rag.Hit(file="a.md", chunk_id=0, score=0.99, text=CHUNK)]
    assert len(rag.filter_hits("Когда подавать заявление на отпуск?", hits)) == 1


def test_filter_drops_off_topic_even_with_high_score():
    hits = [rag.Hit(file="a.md", chunk_id=0, score=0.99, text=CHUNK)]
    assert rag.filter_hits("Какая максимальная скорость у гепарда?", hits) == []


def test_no_hits_gives_no_answer_text():
    assert rag.build_answer("любой вопрос", []) == config.NO_ANSWER
