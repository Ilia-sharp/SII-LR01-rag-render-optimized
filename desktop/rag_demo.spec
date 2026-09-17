# -*- mode: python ; coding: utf-8 -*-
"""Сборка RAG-Demo.exe.

Запуск:  pyinstaller --noconfirm --clean desktop/rag_demo.spec
Результат: dist/RAG-Demo.exe (один файл, работает офлайн).

ONEFILE=False соберёт папку dist/RAG-Demo/ — стартует в разы быстрее,
удобно, если носите демо на флешке.
"""

from pathlib import Path

ONEFILE = True

SPEC_DIR = Path(SPECPATH).resolve()          # noqa: F821 (SPECPATH даёт PyInstaller)
PROJECT_ROOT = SPEC_DIR.parent
BUNDLE = SPEC_DIR / "bundle"

datas = [
    (str(BUNDLE / "model" / "model.onnx"), "bundle/model"),
    (str(BUNDLE / "model" / "tokenizer.json"), "bundle/model"),
    (str(BUNDLE / "index.npz"), "bundle"),
    (str(BUNDLE / "index.json"), "bundle"),
    (str(PROJECT_ROOT / "app" / "ui.html"), "."),
]

# Тяжёлые библиотеки нужны только на этапе подготовки модели,
# внутрь exe они не попадают.
excludes = [
    "torch", "torchvision", "torchaudio",
    "transformers", "sentence_transformers", "optimum",
    "chromadb", "fastapi", "uvicorn", "starlette",
    "scipy", "pandas", "matplotlib", "PIL", "sklearn",
    "IPython", "notebook", "pytest", "tkinter",
]

a = Analysis(
    [str(SPEC_DIR / "demo_app.py")],
    pathex=[str(PROJECT_ROOT), str(SPEC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "app", "app.chunker", "app.config", "app.rag",
        "onnx_encoder", "tokenizers", "onnxruntime", "numpy",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="RAG-Demo",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=True,          # окно с логами — видно latency_ms и число чанков
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(
        pyz, a.scripts, [], exclude_binaries=True,
        name="RAG-Demo", debug=False, strip=False, upx=False, console=True,
    )
    coll = COLLECT(
        exe, a.binaries, a.datas, strip=False, upx=False, name="RAG-Demo",
    )
