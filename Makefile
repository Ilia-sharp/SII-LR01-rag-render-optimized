.PHONY: install install-dev index index-tiny run run-heavy test smoke clean

# Default: lightweight-режим для Render Free Tier (ONNX + rubert-tiny2).
install:
	python -m venv venv && ./venv/bin/pip install -U pip && ./venv/bin/pip install -r requirements.txt

# Heavy: для локальной разработки с полным стеком (torch + chromadb).
install-dev:
	python -m venv venv && ./venv/bin/pip install -U pip && ./venv/bin/pip install -r requirements-dev.txt

# Сборка bundle для Render (rubert-tiny2, по умолчанию).
index-tiny:
	python scripts/build_onnx_bundle_tiny.py

# Сборка bundle с e5-small (тяжёлый, для максимальной точности).
index-e5:
	python scripts/build_onnx_bundle.py

# Сборка индекса в ChromaDB (только для heavy-бэкенда).
index:
	python scripts/build_index.py

# Запуск с light-бэкендом (по умолчанию, для совместимости с Render).
run:
	RAG_BACKEND=light uvicorn app.main:app --reload --port 8000

# Запуск с heavy-бэкендом (нужен requirements-dev.txt и построенный .chroma/).
run-heavy:
	RAG_BACKEND=heavy RAG_MODEL=intfloat/multilingual-e5-small uvicorn app.main:app --reload --port 8000

test:
	pytest -q

smoke:
	bash scripts/smoke_test.sh

clean:
	rm -rf .chroma __pycache__ app/__pycache__ tests/__pycache__ .pytest_cache desktop/bundle/_export
