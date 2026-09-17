"""Тесты единого модуля нарезки (запускаются без тяжёлых зависимостей)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import chunker, config  # noqa: E402


def make_text(n_words: int) -> str:
    return " ".join(f"слово{i}" for i in range(n_words))


def test_empty_text_returns_no_chunks():
    assert chunker.split_into_chunks("") == []
    assert chunker.split_into_chunks("   \n\n ") == []


def test_short_text_is_single_chunk():
    chunks = chunker.split_into_chunks(make_text(50))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0
    assert chunks[0].n_words == 50


def test_chunk_size_limit_respected():
    chunks = chunker.split_into_chunks(make_text(2000), chunk_words=200, overlap=0.15)
    assert all(c.n_words <= 200 for c in chunks)
    assert all(c.n_words <= 600 for c in chunks)  # требование ЛР


def test_overlap_is_applied():
    chunks = chunker.split_into_chunks(make_text(500), chunk_words=100, overlap=0.2)
    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_chunk_ids_are_sequential():
    chunks = chunker.split_into_chunks(make_text(1000), chunk_words=150, overlap=0.1)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_all_words_are_covered():
    text = make_text(777)
    chunks = chunker.split_into_chunks(text, chunk_words=120, overlap=0.15)
    covered = " ".join(c.text for c in chunks)
    assert "слово776" in covered
    assert "слово0" in covered


def test_normalization_collapses_whitespace():
    assert chunker.normalize("а   б\t\tв\r\nг") == "а б в\nг"


def test_too_large_chunk_is_rejected():
    with pytest.raises(ValueError):
        chunker.split_into_chunks(make_text(100), chunk_words=900)


def test_default_config_within_lab_requirements():
    assert 0.10 <= config.CHUNK_OVERLAP <= 0.20
    assert config.CHUNK_WORDS <= 600


def test_sentence_split():
    sentences = chunker.split_sentences("Первое предложение. Второе! Третье?")
    assert len(sentences) == 3
