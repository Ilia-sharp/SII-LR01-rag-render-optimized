#!/usr/bin/env bash
# Сборка исполняемого файла на macOS/Linux (аналог build_exe.bat).
# PyInstaller собирает бинарник ТОЛЬКО под ту ОС, где он запущен.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] виртуальное окружение сборки"
python3 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install -q -U pip

echo "[2/5] PyTorch (CPU, только для экспорта модели)"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || pip install -q torch

echo "[3/5] зависимости сборки"
pip install -q -r desktop/requirements-build.txt

echo "[4/5] экспорт модели в ONNX и построение индекса"
python desktop/prepare_bundle.py

echo "[5/5] сборка бинарника"
pyinstaller --noconfirm --clean desktop/rag_demo.spec

echo "ГОТОВО: dist/RAG-Demo"
