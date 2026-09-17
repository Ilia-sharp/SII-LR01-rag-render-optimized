#!/usr/bin/env python
"""Export multilingual-e5-small to ONNX (int8 quantized) + build numpy index.

This script produces a self-contained bundle for Render deployment:
    bundle/model/model.onnx       (~120 MB after int8 quantization)
    bundle/model/tokenizer.json
    bundle/index.npz               (vectors, ~kB)
    bundle/index.json              (chunk metadata + info)

Result is committed to the repo so Render does NOT need to download
the model or run heavy ML deps at deploy time.
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
MODEL_NAME = "intfloat/multilingual-e5-small"


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


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
    log(f"model size: {mb_before:.0f} MB -> {mb_after:.0f} MB")
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


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists():
        log("ONNX model already exists, skipping export")
        onnx_path = MODEL_DIR / "model.onnx"
    else:
        # Export model
        onnx_path = export_onnx(MODEL_NAME, EXPORT_DIR)
        # Quantize
        onnx_path = quantize(onnx_path, EXPORT_DIR / "model_int8.onnx")
        # Move to final location
        shutil.copy2(onnx_path, MODEL_DIR / "model.onnx")
        # Move tokenizer.json (it was saved by save_pretrained in EXPORT_DIR)
        tokenizer_src = EXPORT_DIR / "tokenizer.json"
        if not tokenizer_src.exists():
            raise FileNotFoundError("tokenizer.json not found after export")
        shutil.copy2(tokenizer_src, MODEL_DIR / "tokenizer.json")

    # Build encoder
    encoder = OnnxEncoder(MODEL_DIR, use_prefixes="e5" in MODEL_NAME.lower())

    # Build index
    vectors, meta, files = build_index(encoder, config.DATA_DIR)

    info = {
        "model": MODEL_NAME,
        "quantized": True,
        "n_files": len(files),
        "n_chunks": len(meta),
        "files": files,
        "chunk_words": config.CHUNK_WORDS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "score_threshold": 0.80,  # for E5
        "built_for": "render-free-tier",
    }
    save_index(BUNDLE_DIR / "index.npz", vectors, meta, info)
    log(f"index saved: {BUNDLE_DIR / 'index.npz'}")

    # Sanity check
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

    # Cleanup export dir
    shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    # Final report
    log("=" * 60)
    log(f"Model:     {MODEL_DIR / 'model.onnx'}  ({(MODEL_DIR / 'model.onnx').stat().st_size / 1e6:.1f} MB)")
    log(f"Tokenizer: {MODEL_DIR / 'tokenizer.json'}  ({(MODEL_DIR / 'tokenizer.json').stat().st_size / 1e3:.1f} KB)")
    log(f"Index:     {BUNDLE_DIR / 'index.npz'}  ({(BUNDLE_DIR / 'index.npz').stat().st_size / 1e3:.1f} KB)")
    log(f"Index JSON:{BUNDLE_DIR / 'index.json'}  ({(BUNDLE_DIR / 'index.json').stat().st_size / 1e3:.1f} KB)")
    log(f"Files:     {len(files)}, chunks: {len(meta)}")
    log("=" * 60)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
