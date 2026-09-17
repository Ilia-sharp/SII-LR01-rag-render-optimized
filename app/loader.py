"""Чтение документов корпуса: .txt, .md, .pdf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Document:
    path: Path
    name: str  # имя файла, попадает в sources[].file
    text: str


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251", errors="ignore")


def _read_pdf(path: Path) -> str:
    """PDF читаем через pymupdf, при его отсутствии — через pypdf."""
    try:
        import fitz  # pymupdf

        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return _read_text_file(path)
    raise ValueError(f"Неподдерживаемый формат файла: {path.name}")


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """Собирает все поддерживаемые файлы из data/ (включая подпапки)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Папка с документами не найдена: {data_dir}")

    files = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"В {data_dir} нет файлов с расширениями {sorted(config.SUPPORTED_SUFFIXES)}"
        )

    documents: list[Document] = []
    for path in files:
        text = read_document(path)
        if text and text.strip():
            documents.append(Document(path=path, name=path.name, text=text))
    return documents
