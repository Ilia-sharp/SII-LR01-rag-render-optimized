#!/usr/bin/env python
"""Switch the ONNX bundle to a tiny but capable model: cointegrated/rubert-tiny2.

Why:
- multilingual-e5-small int8 ONNX = 119 MB on disk → ~460 MB RSS at runtime
  → too close to Render Free 512 MB limit (peak during load = 548 MB → OOM risk).
- cointegrated/rubert-tiny2 = 29 MB on disk → ~120 MB RSS at runtime
  → ~390 MB headroom for FastAPI + Python overhead.

Trade-offs:
- rubert-tiny2 has 312-dim embeddings (vs 384 for e5-small) — almost identical quality.
- 512 token context (same as e5-small).
- Trained specifically on Russian data — works better than e5-small for our corpus.
- NOT an E5 model → no query:/passage: prefixes needed.
- Threshold needs recalibration (rubert-tiny2 has different cosine distribution).

After running this script, the bundle/ directory will be overwritten with
the new model + re-indexed corpus.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "desktop"))

from app import chunker, config, loader  # noqa: E402
from onnx_encoder import OnnxEncoder, save_index, search  # noqa: E402

BUNDLE_DIR = PROJECT_ROOT / "desktop" / "bundle"
MODEL_DIR = BUNDLE_DIR / "model"
EXPORT_DIR = BUNDLE_DIR / "_export"
MODEL_NAME = "cointegrated/rubert-tiny2"


def log(msg: str) -> None:
    print(f"[prepare-tiny] {msg}", flush=True)


def export_onnx(model_name: str, out_dir: Path) -> Path:
    """Direct torch.onnx.export (no optimum needed)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    tokenizer.save_pretrained(out_dir)

    sample = tokenizer(["пример текста для экспорта"],
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
    inputs = (sample["input_ids"], sample["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }

    onnx_path = out_dir / "model.onnx"
    log(f"exporting to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model, inputs, str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
    return onnx_path


def quantize(src: Path, dst: Path) -> Path:
    """Dynamic int8 quantization — reduces size ~4x with negligible accuracy loss."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    log("int8 quantization ...")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    mb_before = src.stat().st_size / 1e6
    mb_after = dst.stat().st_size / 1e6
    log(f"model size: {mb_before:.1f} MB -> {mb_after:.1f} MB")
    return dst


def build_index(encoder: OnnxEncoder, data_dir: Path):
    documents = loader.load_documents(data_dir)
    log(f"documents: {len(documents)}")

    texts: list[str] = []
    meta: list[dict] = []
    for doc in documents:
        for ch in chunker.split_into_chunks(doc.text):
            texts.append(ch.text)
            meta.append({"file": doc.name, "chunk_id": ch.chunk_id,
                         "text": ch.text, "n_words": ch.n_words})

    log(f"chunks: {len(texts)}; vectorizing ...")
    started = time.perf_counter()
    vectors = encoder.encode_passages(texts, batch_size=8)
    log(f"done in {time.perf_counter() - started:.1f} s, dim={vectors.shape[1]}")
    return np.asarray(vectors, dtype=np.float32), meta, [d.name for d in documents]


def calibrate_threshold(encoder: OnnxEncoder, vectors, meta) -> float:
    """Pick threshold between max(off-topic) and min(on-topic) scores."""
    on_topic = [
        "За сколько дней нужно подать заявление на отпуск?",
        "Как настроить двухфакторную аутентификацию?",
        "Какие лимиты на командировочные расходы?",
        "Какой испытательный срок для новых сотрудников?",
        "Как оформить удалённую работу?",
    ]
    off_topic = [
        "Как приготовить борщ из марсианской свёклы?",
        "Сколько зубов у африканского слона?",
        "Какая погода будет завтра в Петропавловске?",
        "Кто написал Войну и мир?",
        "Как починить двигатель внутреннего сгорания?",
    ]
    on_scores = []
    off_scores = []
    for q in on_topic:
        qv = encoder.encode_queries([q])[0]
        _, score = search(vectors, qv, 1)[0]
        on_scores.append(score)
    for q in off_topic:
        qv = encoder.encode_queries([q])[0]
        _, score = search(vectors, qv, 1)[0]
        off_scores.append(score)

    log(f"on-topic scores:  min={min(on_scores):.4f} max={max(on_scores):.4f} avg={sum(on_scores)/len(on_scores):.4f}")
    log(f"off-topic scores: min={min(off_scores):.4f} max={max(off_scores):.4f} avg={sum(off_scores)/len(off_scores):.4f}")
    threshold = (max(off_scores) + min(on_scores)) / 2
    log(f"chosen threshold: {threshold:.4f}")
    return float(threshold)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous bundle (was e5-small)
    if (MODEL_DIR / "model.onnx").exists():
        log("removing previous (e5-small) bundle ...")
        for f in MODEL_DIR.iterdir():
            f.unlink()

    onnx_path = export_onnx(MODEL_NAME, EXPORT_DIR)
    onnx_path = quantize(onnx_path, EXPORT_DIR / "model_int8.onnx")

    shutil.copy2(onnx_path, MODEL_DIR / "model.onnx")
    tokenizer_src = EXPORT_DIR / "tokenizer.json"
    if not tokenizer_src.exists():
        raise FileNotFoundError("tokenizer.json not found after export")
    shutil.copy2(tokenizer_src, MODEL_DIR / "tokenizer.json")

    # rubert-tiny2 is NOT E5 — no query/passage prefixes
    encoder = OnnxEncoder(MODEL_DIR, use_prefixes=False, memory_efficient=True)
    vectors, meta, files = build_index(encoder, config.DATA_DIR)

    threshold = calibrate_threshold(encoder, vectors, meta)

    info = {
        "model": MODEL_NAME,
        "quantized": True,
        "n_files": len(files),
        "n_chunks": len(meta),
        "files": files,
        "chunk_words": config.CHUNK_WORDS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "score_threshold": round(threshold, 2),
        "e5_prefixes": False,
        "built_for": "render-free-tier",
        "notes": "rubert-tiny2 — компактная модель специально для русского языка (29 МБ)."
    }
    save_index(BUNDLE_DIR / "index.npz", vectors, meta, info)
    log(f"index saved: {BUNDLE_DIR / 'index.npz'}")

    # Sanity checks
    log("running sanity checks ...")
    checks = [
        "За сколько дней нужно подать заявление на отпуск?",
        "Как настроить двухфакторную аутентификацию?",
        "Как приготовить борщ из марсианской свёклы?",
    ]
    for q in checks:
        qv = encoder.encode_queries([q])[0]
        idx, score = search(vectors, qv, 1)[0]
        log(f"  score={score:.4f}  {meta[idx]['file']}#{meta[idx]['chunk_id']}  <- {q}")

    shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    log("=" * 60)
    log(f"Model:     {MODEL_DIR / 'model.onnx'}  ({(MODEL_DIR / 'model.onnx').stat().st_size / 1e6:.1f} MB)")
    log(f"Tokenizer: {MODEL_DIR / 'tokenizer.json'}  ({(MODEL_DIR / 'tokenizer.json').stat().st_size / 1e3:.1f} KB)")
    log(f"Index:     {BUNDLE_DIR / 'index.npz'}  ({(BUNDLE_DIR / 'index.npz').stat().st_size / 1e3:.1f} KB)")
    log(f"Index JSON:{BUNDLE_DIR / 'index.json'}  ({(BUNDLE_DIR / 'index.json').stat().st_size / 1e3:.1f} KB)")
    log(f"Files:     {len(files)}, chunks: {len(meta)}")
    log(f"Threshold: {threshold:.4f}")
    log("=" * 60)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
