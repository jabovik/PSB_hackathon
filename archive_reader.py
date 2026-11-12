from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from schemas import Document


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


def read_archive(source: str | Path) -> list[Document]:
    archive_path = find_archive(source)
    documents: list[Document] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        filenames = sorted(
            name for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".txt", ".md", ".docx"}
        )

        for filename in filenames:
            content = archive.read(filename)
            suffix = Path(filename).suffix.lower()
            text = (
                _read_docx(content)
                if suffix == ".docx"
                else content.decode("utf-8-sig")
            ).strip()
            if text:
                documents.append(
                    Document(
                        id=len(documents) + 1,
                        filename=filename,
                        text=text,
                    )
                )

    if not documents:
        raise ValueError(
            f"В архиве {archive_path} не найдено документов TXT, MD или DOCX"
        )

    return documents
