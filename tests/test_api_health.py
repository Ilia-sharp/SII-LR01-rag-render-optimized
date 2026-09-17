"""Проверка /health и корректного ответа /ask без индекса.

Тест не грузит модель и не требует построенного индекса.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_without_index_returns_503_or_valid_schema(client):
    response = client.post("/ask", json={"question": "Тестовый вопрос про отпуск"})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert set(body) == {"answer", "sources", "model_version"}
        assert isinstance(body["sources"], list)
