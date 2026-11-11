from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import Chunk, Document

DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 150

LEGAL_SEPARATORS = [
    r"\n(?=Статья\s+\d+)",
    r"\n(?=\d+(?:\.\d+)*\.\s+[А-ЯЁ])",
    r"\n(?=\d+(?:\.\d+)*\))",
    r"\n\s*\n",
    r"\n",
    r"(?<=[.!?])\s+",
    " ",
    "",
]


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(
    document: Document,
    start_chunk_id: int = 1,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:

    if not isinstance(document, Document):
        raise TypeError("document должен быть объектом app.schemas.Document")
    if start_chunk_id < 1:
        raise ValueError("start_chunk_id должен быть положительным")
    if max_chars <= 0:
        raise ValueError("max_chars должен быть положительным")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars должен быть от 0 до max_chars - 1")

    text = _clean_text(document.text)
    if not text:
        return []

    prefix = f"Документ: {document.filename}\n\n"
    body_size = max_chars - len(prefix)
    if body_size <= 0:
        raise ValueError("Имя файла слишком длинное для выбранного max_chars")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body_size,
        chunk_overlap=min(overlap_chars, body_size - 1),
        separators=LEGAL_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        length_function=len,
    )

    pieces = splitter.split_text(text)
    chunks: list[Chunk] = []

    for piece in pieces:
        chunk_text = prefix + piece.strip()
        chunks.append(
            Chunk(
                id=start_chunk_id + len(chunks),
                document_id=document.id,
                text=chunk_text,
                embedding=None,
            )
        )

    return chunks
