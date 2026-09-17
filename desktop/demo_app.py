#!/usr/bin/env python
"""Демонстрационное приложение (то, что становится RAG-Demo.exe).

Поднимает локальный сервер на 127.0.0.1, открывает браузер и отвечает
на вопросы по корпусу. Работает офлайн: модель и индекс лежат внутри exe.

Логика поиска и сборки ответа — та же, что в серверной версии:
используются app/chunker.py и app/rag.py, меняется только транспорт
(stdlib http.server вместо FastAPI) и хранилище (numpy вместо ChromaDB).
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# Внутри exe ресурсы распаковываются в sys._MEIPASS.
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import chunker, config, rag  # noqa: E402
from onnx_encoder import OnnxEncoder, load_index, search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("rag.demo")

HOST = "127.0.0.1"
PORT_RANGE = range(8765, 8800)

STATE: dict = {}


def ui_path() -> Path:
    """В exe страница лежит рядом, в исходниках — в app/ (общая с сервером)."""
    for candidate in (BASE_DIR / "ui.html", PROJECT_ROOT / "app" / "ui.html"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ui.html не найден")


class _EncoderAdapter:
    """Подставляется вместо app.embedder, чтобы переиспользовать rag.build_answer."""

    def __init__(self, encoder: OnnxEncoder):
        self.encoder = encoder

    def embed_query(self, text: str) -> list[float]:
        return self.encoder.encode_queries([text])[0].tolist()

    def embed_passages(self, texts) -> list[list[float]]:
        return [v.tolist() for v in self.encoder.encode_passages(list(texts))]

    embed_texts = embed_passages

    def model_version(self) -> str:
        return STATE.get("info", {}).get("model", config.MODEL_NAME)


def load_everything() -> None:
    model_dir = BASE_DIR / "bundle" / "model"
    index_path = BASE_DIR / "bundle" / "index.npz"

    logger.info("загрузка модели из %s", model_dir)
    started = time.perf_counter()
    vectors, meta, info = load_index(index_path)
    encoder = OnnxEncoder(model_dir, use_prefixes="e5" in info.get("model", "").lower())

    STATE.update(vectors=vectors, meta=meta, info=info, encoder=encoder)
    rag.embedder = _EncoderAdapter(encoder)  # единая логика ответа

    # Прогрев: первый прогон onnxruntime самый долгий.
    encoder.encode_queries(["разогрев"])
    logger.info(
        "готово за %.1f c | модель=%s | файлов=%d | чанков=%d",
        time.perf_counter() - started,
        info.get("model"),
        info.get("n_files"),
        info.get("n_chunks"),
    )


def answer(question: str) -> tuple[dict, int]:
    vectors: np.ndarray = STATE["vectors"]
    meta: list[dict] = STATE["meta"]
    encoder: OnnxEncoder = STATE["encoder"]

    q_vec = encoder.encode_queries([chunker.normalize(question)])[0]
    hits = [
        rag.Hit(
            file=meta[i]["file"],
            chunk_id=meta[i]["chunk_id"],
            score=round(score, 4),
            text=meta[i]["text"],
        )
        for i, score in search(vectors, q_vec, config.TOP_K)
    ]
    relevant = rag.filter_hits(question, hits)

    model_version = STATE["info"].get("model", config.MODEL_NAME)
    if not relevant:
        return (
            {"answer": config.NO_ANSWER, "sources": [], "model_version": model_version},
            0,
        )
    return (
        {
            "answer": rag.build_answer(question, relevant),
            "sources": [h.as_source() for h in relevant],
            "model_version": model_version,
        },
        len(relevant),
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGDemo/1.0"

    def log_message(self, fmt, *args):  # отключаем шумный лог http.server
        return

    def _send(self, code: int, payload: dict | None = None, html: str | None = None) -> None:
        if html is not None:
            body = html.encode("utf-8")
            ctype = "text/html; charset=utf-8"
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            page = ui_path().read_text(encoding="utf-8")
            info = STATE["info"]
            page = (
                page.replace("{{MODEL}}", str(info.get("model")))
                .replace("{{FILES}}", str(info.get("n_files")))
                .replace("{{CHUNKS}}", str(info.get("n_chunks")))
                .replace("{{THRESHOLD}}", str(config.SCORE_THRESHOLD))
                .replace("{{FILELIST}}", ", ".join(info.get("files", [])))
            )
            self._send(200, html=page)
        elif self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/info":
            self._send(200, STATE["info"])
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/ask":
            self._send(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            question = str(data.get("question", "")).strip()
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"detail": "некорректный JSON"})
            return

        if not question:
            self._send(422, {"detail": "Вопрос не может быть пустым"})
            return

        q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
        started = time.perf_counter()
        try:
            payload, n_chunks = answer(question)
        except Exception:  # noqa: BLE001
            logger.exception("ask failed q_hash=%s", q_hash)
            self._send(500, {"detail": "внутренняя ошибка"})
            return
        latency_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "ask q_hash=%s q_len=%d chunks_found=%d latency_ms=%.1f answered=%s",
            q_hash,
            len(question),
            n_chunks,
            latency_ms,
            n_chunks > 0,
        )
        payload["latency_ms"] = round(latency_ms, 1)
        self._send(200, payload)


def find_port() -> int:
    for port in PORT_RANGE:
        with socket.socket() as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    raise RuntimeError("не найден свободный порт")


def main() -> int:
    print("=" * 62)
    print("  RAG-ассистент по документам — демонстрация (ЛР1)")
    print("  Работает офлайн. Чтобы закрыть — закройте это окно.")
    print("=" * 62)

    try:
        load_everything()
    except Exception as exc:  # noqa: BLE001
        logger.exception("не удалось загрузить модель или индекс")
        print(f"\nОШИБКА: {exc}")
        input("\nНажмите Enter, чтобы закрыть...")
        return 1

    port = find_port()
    url = f"http://{HOST}:{port}"
    server = ThreadingHTTPServer((HOST, port), Handler)

    print(f"\n  Интерфейс: {url}")
    print(f"  API:       POST {url}/ask, GET {url}/health\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
