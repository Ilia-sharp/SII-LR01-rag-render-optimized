"""Энкодер на onnxruntime: без torch и sentence-transformers.

Один и тот же класс используется при сборке индекса (prepare_bundle.py)
и внутри exe (demo_app.py) — это гарантирует, что документы и вопросы
кодируются абсолютно одинаково.

Пулинг — mean pooling по last_hidden_state с учётом attention_mask,
далее L2-нормализация. Именно так работает multilingual-e5-small.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MAX_LENGTH = 512


class OnnxEncoder:
    def __init__(self, model_dir: Path, use_prefixes: bool = True, max_length: int = MAX_LENGTH,
                 memory_efficient: bool = False):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            raise FileNotFoundError(f"В {model_dir} нет model.onnx и/или tokenizer.json")

        self.use_prefixes = use_prefixes
        self.max_length = max_length

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_token = "<pad>"
        pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:  # запасной вариант для BERT-подобных словарей
            pad_token, pad_id = "[PAD]", self.tokenizer.token_to_id("[PAD]") or 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        opts = ort.SessionOptions()
        # В memory-efficient режиме убираем агрессивную оптимизацию графа
        # и отключаем арену памяти CPU — экономит ~80 МБ RAM ценой ~5-10 % скорости.
        if memory_efficient:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            opts.enable_cpu_mem_arena = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        # Лимит потоков (важно для 0.1 CPU на Render Free)
        opts.intra_op_num_threads = int(__import__("os").environ.get("ORT_NUM_THREADS", "1"))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    # --- внутреннее -----------------------------------------------------

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = ids
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]  # (batch, seq, dim)

        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        vectors = summed / counts
        norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return (vectors / norms).astype(np.float32)

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        items = [f"{prefix}{t}" for t in texts] if self.use_prefixes else list(texts)
        chunks = [
            self._encode_batch(items[i : i + batch_size])
            for i in range(0, len(items), batch_size)
        ]
        return np.vstack(chunks)

    # --- публичное ------------------------------------------------------

    def encode_passages(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        return self._encode(list(texts), PASSAGE_PREFIX, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        return self._encode(list(texts), QUERY_PREFIX, batch_size)


# --- Индекс: обычный numpy-массив вместо ChromaDB ------------------------
# В корпусе десятки чанков, поэтому полный перебор занимает микросекунды,
# а exe не тянет за собой sqlite/hnswlib и собирается без проблем.


def save_index(path: Path, vectors: np.ndarray, meta: list[dict], info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors.astype(np.float32))
    payload = {"info": info, "chunks": meta}
    path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_index(path: Path) -> tuple[np.ndarray, list[dict], dict]:
    path = Path(path)
    vectors = np.load(path)["vectors"]
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return vectors, payload["chunks"], payload["info"]


def search(vectors: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Косинусная схожесть: векторы нормированы, поэтому это скалярное произведение."""
    scores = vectors @ query_vector
    order = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in order]
