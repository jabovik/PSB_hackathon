from __future__ import annotations

from pathlib import Path

from schemas import Chunk
from embedding_client import EmbeddingClient

from archive_reader import read_archive
from document_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
)

DEFAULT_EMBEDDING_BATCH_SIZE = 16


class Ingester:
    """Read an archive, create chunks, and attach embeddings to each chunk."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
        overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size должен быть положительным")

        self.embedding_client = embedding_client
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.embedding_batch_size = embedding_batch_size

    def ingest(self, source: str | Path) -> list[Chunk]:
        chunks = chunk_documents(
            read_archive(source),
            max_chars=self.max_chunk_chars,
            overlap_chars=self.overlap_chars,
        )

        if self.embedding_client is not None:
            # Bound request size while avoiding a separate API call for every chunk.
            for start in range(0, len(chunks), self.embedding_batch_size):
                batch = chunks[start : start + self.embedding_batch_size]
                self.embedding_client.embed_chunks(batch)

        return chunks
