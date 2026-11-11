from collections.abc import Sequence

from app.schemas import Chunk, Document

from .document_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
)


def chunk_documents(
    documents: Sequence[Document],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:

    all_chunks: list[Chunk] = []
    next_chunk_id = 1
    seen_document_ids: set[int] = set()

    for document in documents:
        if document.id in seen_document_ids:
            raise ValueError(f"Повторяется Document.id={document.id}")
        seen_document_ids.add(document.id)

        chunks = chunk_document(
            document=document,
            start_chunk_id=next_chunk_id,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(chunks)
        next_chunk_id += len(chunks)

    return all_chunks
