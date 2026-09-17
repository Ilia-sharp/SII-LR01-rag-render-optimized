#!/usr/bin/env python
"""Подготовка содержимого exe: модель в ONNX + готовый индекс.

Запускается ОДИН РАЗ на машине с интернетом (обычно из build_exe.bat):

    python desktop/prepare_bundle.py

Результат в desktop/bundle/:
    model/model.onnx       — энкодер (по умолчанию квантованный int8, ~120 МБ)
    model/tokenizer.json   — токенизатор
    index.npz + index.json — векторы и текст чанков корпуса из data/

После этого exe работает полностью офлайн.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import chunker, config, loader  # noqa: E402
from onnx_encoder import OnnxEncoder, save_index  # noqa: E402

BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"
MODEL_DIR = BUNDLE_DIR / "model"
EXPORT_DIR = BUNDLE_DIR / "_export"


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


# --- 1. Экспорт модели в ONNX -------------------------------------------


def export_onnx(model_name: str, out_dir: Path) -> Path:
    """Сначала пробуем optimum, при неудаче — прямой torch.onnx.export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from optimum.exporters.onnx import main_export

        log(f"экспорт через optimum: {model_name}")
        main_export(
            model_name_or_path=model_name,
            output=out_dir,
            task="feature-extraction",
            opset=14,
        )
        return out_dir / "model.onnx"
    except Exception as exc:  # noqa: BLE001
        log(f"optimum не сработал ({type(exc).__name__}: {exc}); пробую torch.onnx.export")
        return export_onnx_fallback(model_name, out_dir)


def export_onnx_fallback(model_name: str, out_dir: Path) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    tokenizer.save_pretrained(out_dir)

    sample = tokenizer(["пример текста"], return_tensors="pt", padding=True, truncation=True)
    inputs = (sample["input_ids"], sample["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }

    onnx_path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
    return onnx_path


# --- 2. Квантизация ------------------------------------------------------


def quantize(src: Path, dst: Path) -> Path:
    """Динамическая int8-квантизация: размер падает примерно вчетверо."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        log("квантизация int8...")
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        mb_before = src.stat().st_size / 1e6
        mb_after = dst.stat().st_size / 1e6
        log(f"размер модели: {mb_before:.0f} МБ -> {mb_after:.0f} МБ")
        return dst
    except Exception as exc:  # noqa: BLE001
        log(f"квантизация не удалась ({type(exc).__name__}: {exc}); беру модель как есть")
        return src


# --- 3. Сборка индекса ---------------------------------------------------


def build_index(encoder: OnnxEncoder, data_dir: Path) -> tuple:
    import numpy as np

    documents = loader.load_documents(data_dir)
    log(f"документов: {len(documents)}")

    texts: list[str] = []
    meta: list[dict] = []
    for doc in documents:
        for ch in chunker.split_into_chunks(doc.text):
            texts.append(ch.text)
            meta.append({"file": doc.name, "chunk_id": ch.chunk_id, "text": ch.text})

    log(f"чанков: {len(texts)}; векторизация...")
    started = time.perf_counter()
    vectors = encoder.encode_passages(texts)
    log(f"готово за {time.perf_counter() - started:.1f} c, размерность {vectors.shape[1]}")
    return np.asarray(vectors), meta, [d.name for d in documents]


# --- main ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка модели и индекса для exe")
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--no-quantize", action="store_true", help="оставить fp32 (крупнее, точнее)")
    parser.add_argument("--keep-export", action="store_true", help="не удалять временные файлы экспорта")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = export_onnx(args.model, EXPORT_DIR)
    if not args.no_quantize:
        onnx_path = quantize(onnx_path, EXPORT_DIR / "model_int8.onnx")

    shutil.copy2(onnx_path, MODEL_DIR / "model.onnx")
    tokenizer_src = EXPORT_DIR / "tokenizer.json"
    if not tokenizer_src.exists():
        raise FileNotFoundError("tokenizer.json не найден после экспорта")
    shutil.copy2(tokenizer_src, MODEL_DIR / "tokenizer.json")

    encoder = OnnxEncoder(MODEL_DIR, use_prefixes="e5" in args.model.lower())
    vectors, meta, files = build_index(encoder, args.data_dir)

    info = {
        "model": args.model,
        "quantized": not args.no_quantize,
        "n_files": len(files),
        "n_chunks": len(meta),
        "files": files,
        "chunk_words": config.CHUNK_WORDS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "score_threshold": config.SCORE_THRESHOLD,
    }
    save_index(BUNDLE_DIR / "index.npz", vectors, meta, info)
    log(f"индекс сохранён: {BUNDLE_DIR / 'index.npz'}")

    # Проверка вменяемости до упаковки в exe.
    from onnx_encoder import search

    checks = [
        "За сколько дней нужно подать заявление на отпуск?",
        "Как приготовить борщ из марсианской свёклы?",
    ]
    print("\n[prepare] контрольные запросы:")
    for q in checks:
        qv = encoder.encode_queries([q])[0]
        idx, score = search(vectors, qv, 1)[0]
        print(f"  score={score:.4f}  {meta[idx]['file']}#{meta[idx]['chunk_id']}  <- {q}")
    print(f"[prepare] порог отсечения: {config.SCORE_THRESHOLD}\n")

    if not args.keep_export:
        shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    print(json.dumps(info, ensure_ascii=False, indent=2))
    log("bundle готов, можно собирать exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
