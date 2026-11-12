from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from schemas import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}


def _read_docx(content: bytes) -> str:
    """Extract paragraph text from a DOCX document using the standard library."""
    with zipfile.ZipFile(io.BytesIO(content), "r") as docx:
        xml = docx.read("word/document.xml")

    root = ElementTree.fromstring(xml)
    word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []

    for paragraph in root.iter(f"{word_namespace}p"):
        parts = []
        for element in paragraph.iter():
            if element.tag == f"{word_namespace}t":
                parts.append(element.text or "")
            elif element.tag == f"{word_namespace}tab":
                parts.append("\t")
            elif element.tag in (f"{word_namespace}br", f"{word_namespace}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))

    return "\n".join(paragraphs)


def find_archive(source: str | Path) -> Path:
    path = Path(source).expanduser()

    if path.is_file():
        return path

    filename = path.name
    data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
    for candidate in (data_dir / filename, Path("data") / filename):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Не найден архив {source!s}. Проверьте путь или переменную DATA_DIR."
    )


def _read_pdf(content: bytes) -> str:
    """Извлекает текст из PDF: база знаний может прийти не только в TXT/DOCX."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        return _read_docx(content)
    if suffix == ".pdf":
        return _read_pdf(content)
    return content.decode("utf-8-sig")


def read_archive(source: str | Path) -> list[Document]:
    archive_path = find_archive(source)
    documents: list[Document] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        filenames = sorted(name for name in archive.namelist() if not name.endswith("/"))

        for filename in filenames:
            if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                text = _read_document(filename, archive.read(filename)).strip()
            except Exception as error:
                print(
                    f"[archive] пропущен {filename}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if text:
                documents.append(
                    Document(
                        id=len(documents) + 1,
                        filename=filename,
                        text=text,
                    )
                )

        if not documents:
            # Последний шанс: вдруг документы лежат с неожиданным расширением.
            for filename in filenames:
                try:
                    text = archive.read(filename).decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if len(text) > 100:
                    documents.append(
                        Document(id=len(documents) + 1, filename=filename, text=text)
                    )

    if not documents:
        raise ValueError(
            f"В архиве {archive_path} не найдено читаемых документов"
        )

    return documents
