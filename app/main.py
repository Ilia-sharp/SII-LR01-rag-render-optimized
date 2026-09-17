"""FastAPI-сервис: POST /ask, GET /health.

Логи содержат только хеш и длину вопроса, latency_ms и количество чанков —
полного текста вопроса/ответа в логах нет.

Поддерживаются два бэкенда (см. app/backends.py):
  - light (по умолчанию, для Render): ONNX Runtime + numpy, ~250 МБ RAM
  - heavy (для локальной разработки): sentence-transformers + chromadb
Переключается переменной окружения RAG_BACKEND=light|heavy.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import backends, config, rag

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("rag.api")

state: dict = {"encoder": None, "store": None, "ready": False, "reason": "not loaded"}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Вопрос пользователя")


class Source(BaseModel):
    file: str
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_version: str


def _load_index() -> None:
    """Инициализация бэкенда и индекса при старте сервиса."""
    try:
        encoder = backends.get_encoder()
        store = backends.get_store()
    except Exception as exc:  # noqa: BLE001
        state["ready"] = False
        state["reason"] = f"Ошибка инициализации бэкенда: {exc}"
        logger.exception("backend init failed")
        return

    if not store.is_ready():
        state["ready"] = False
        state["reason"] = store.reason()
        logger.warning(state["reason"])
        return

    state["encoder"] = encoder
    state["store"] = store
    state["ready"] = True
    state["reason"] = ""
    meta = store.meta()
    logger.info(
        "index loaded backend=%s model=%s chunks=%d files=%s",
        backends.BACKEND,
        meta.get("model"),
        store.count(),
        meta.get("n_files"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_index()
    if state["ready"]:
        # Прогреваем энкодер, чтобы первый запрос не ждал загрузку/инициализацию.
        backends.warmup_encoder(state["encoder"])
        logger.info("encoder warmed up")
    yield


app = FastAPI(
    title="RAG-ассистент по документам",
    version="1.0.0",
    description="ЛР1 по дисциплине «Системы искусственного интеллекта»",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    meta = state["store"].meta() if state["store"] else {}
    return {
        "service": "rag-assistant",
        "backend": backends.BACKEND,
        "index_ready": state["ready"],
        "model_version": meta.get("model", config.MODEL_NAME),
        "chunks": meta.get("n_chunks"),
        "files": meta.get("n_files"),
        "endpoints": ["POST /ask", "GET /health", "GET /info"],
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """Страница демонстрации. Тот же интерфейс использует и desktop-сборка."""
    ui_path = config.BASE_DIR / "app" / "ui.html"
    if not ui_path.exists():
        return HTMLResponse(
            "<h1>RAG-ассистент</h1><p>Интерфейс не найден. "
            "Доступны POST /ask, GET /health, GET /docs.</p>"
        )
    meta = state["store"].meta() if state["store"] else {}
    files = meta.get("files") or []
    page = (
        ui_path.read_text(encoding="utf-8")
        .replace("{{MODEL}}", str(meta.get("model", config.MODEL_NAME)))
        .replace("{{FILES}}", str(meta.get("n_files", "—")))
        .replace("{{CHUNKS}}", str(meta.get("n_chunks", "—")))
        .replace("{{THRESHOLD}}", str(config.SCORE_THRESHOLD))
        .replace("{{FILELIST}}", ", ".join(files) if files else "индекс не построен")
    )
    return HTMLResponse(page)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not state["ready"]:
        raise HTTPException(status_code=503, detail=state["reason"])

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Вопрос не может быть пустым")

    q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    started = time.perf_counter()
    try:
        payload, n_chunks = rag.answer_question(question, state["store"], state["encoder"])
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.exception("ask failed q_hash=%s latency_ms=%.1f", q_hash, latency_ms)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервиса")

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ask q_hash=%s q_len=%d chunks_found=%d latency_ms=%.1f answered=%s",
        q_hash,
        len(question),
        n_chunks,
        latency_ms,
        n_chunks > 0,
    )
    return AskResponse(**payload)
