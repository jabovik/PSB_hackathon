from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from schemas import Document


MetadataValue = str | int


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

# Ядро
def _extract_metadata(filename: str) -> dict[str, MetadataValue]:
    path = Path(filename)
    archive_name = path.name
    raw_stem = Path(archive_name).stem
    category_prefix, separator, raw_title = raw_stem.partition("__")

    if not separator:
        raw_title = raw_stem
        category_prefix = ""

    title = raw_title.replace("_", " ").strip()
    metadata: dict[str, MetadataValue] = {
        "title": title,
        "file_format": path.suffix.lower().lstrip("."),
    }

    category = " ".join(category_prefix.casefold().split())

    if category == "документы компании":
        metadata["collection"] = "company_documents"
        _add_company_metadata(raw_title, metadata)
    elif category == "законы":
        metadata["collection"] = "laws"
        _add_law_metadata(raw_stem.partition("__")[2], metadata)
    else:
        metadata["collection"] = "unclassified"

    return metadata

# Метаданные документов компании
def _add_company_metadata(
    title: str,
    metadata: dict[str, MetadataValue],
) -> None:
    normalized_title = title.casefold()

    if normalized_title.startswith("договор "):
        metadata["document_type"] = "contract"
        match = re.search(
            r"(?P<number>[А-ЯЁA-Z]{1,4}-\d{1,2}-\d{4})"
            r"(?:\s+(?P<counterparty>.+))?$",
            title,
            flags=re.IGNORECASE,
        )
        if match:
            metadata["document_number"] = match.group("number")
            counterparty = (match.group("counterparty") or "").strip()
            if counterparty:
                metadata["counterparty"] = counterparty

    elif normalized_title.startswith("доп соглашение"):
        metadata["document_type"] = "amendment"
        amendment_match = re.search(
            r"доп\s+соглашение\s+(\d+)", title, flags=re.IGNORECASE
        )
        if amendment_match:
            metadata["amendment_number"] = int(amendment_match.group(1))

        related_match = re.search(
            r"к\s+договору\s+([А-ЯЁA-Z]{1,4}-\d{1,2}-\d{4})",
            title,
            flags=re.IGNORECASE,
        )
        if related_match:
            metadata["related_document_number"] = related_match.group(1)

    elif normalized_title.startswith("письмо"):
        metadata["document_type"] = "compliance_letter"
        number_match = re.search(
            r"\b([А-ЯЁA-Z]{1,3}-\d{1,3})\b", title, flags=re.IGNORECASE
        )
        date_match = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b", title)
        if number_match:
            metadata["document_number"] = number_match.group(1)
        if date_match:
            metadata["document_date"] = date_match.group(1)

    elif normalized_title.startswith("регламент"):
        metadata["document_type"] = "internal_regulation"
        revision_match = re.search(
            r"\bред\.?\s*(\d{4})\b", title, flags=re.IGNORECASE
        )
        if revision_match:
            metadata["revision_year"] = int(revision_match.group(1))

        if re.search(r"\bутрат\w*\s+силу\b", title, flags=re.IGNORECASE):
            metadata["status"] = "not_in_force"

    elif normalized_title.startswith("служебная записка"):
        metadata["document_type"] = "memo"

    else:
        metadata["document_type"] = "other_company_document"

#Метаданные законов
def _add_law_metadata(
    title: str,
    metadata: dict[str, MetadataValue],
) -> None:
    metadata["document_type"] = "law_article"
    match = re.match(
        r"(?P<act>.+?)_ст(?P<article>\d+(?:\.\d+)*)_(?P<edition>.+)$",
        title,
        flags=re.IGNORECASE,
    )
    if not match:
        return

    act_label = match.group("act").replace("152ФЗ", "152-ФЗ")
    metadata["act_label"] = act_label
    metadata["article_number"] = match.group("article")
    metadata["edition_label"] = match.group("edition").replace("_", " ")


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
                        metadata=_extract_metadata(filename),
                    )
                )

    if not documents:
        raise ValueError(
            f"В архиве {archive_path} не найдено документов TXT, MD или DOCX"
        )

    return documents
