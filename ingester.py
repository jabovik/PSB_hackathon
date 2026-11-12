from __future__ import annotations

import os
import sys
import time
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
#: Сколько секунд максимум тратим на эмбеддинги, чтобы не съесть лимит платформы.
DEFAULT_EMBEDDING_BUDGET_SECONDS = 300.0


class Ingester:
    """Read an archive, create chunks, and attach embeddings to each chunk."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
        overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        embedding_budget_seconds: float | None = None,
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size должен быть положительным")

        self.embedding_client = embedding_client
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.embedding_batch_size = embedding_batch_size
        self.embedding_budget_seconds = (
            embedding_budget_seconds
            if embedding_budget_seconds is not None
            else float(os.getenv("INGEST_BUDGET_SECONDS", str(DEFAULT_EMBEDDING_BUDGET_SECONDS)))
        )

    def ingest(self, source: str | Path) -> list[Chunk]:
        chunks = chunk_documents(
            read_archive(source),
            max_chars=self.max_chunk_chars,
            overlap_chars=self.overlap_chars,
        )

        if self.embedding_client is not None:
            # Bound request size while avoiding a separate API call for every chunk.
            started = time.monotonic()
            for start in range(0, len(chunks), self.embedding_batch_size):
                if time.monotonic() - started > self.embedding_budget_seconds:
                    print(
                        f"[ingester] бюджет эмбеддингов {self.embedding_budget_seconds:.0f} c "
                        f"исчерпан, оставшиеся чанки пойдут только в BM25",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                batch = chunks[start : start + self.embedding_batch_size]
                try:
                    self.embedding_client.embed_chunks(batch)
                except Exception as error:
                    # Без эмбеддингов остаётся BM25: лучше ответить, чем упасть целиком.
                    print(
                        f"[ingester] не удалось получить эмбеддинги для чанков "
                        f"{start}..{start + len(batch)}: {type(error).__name__}: {error}; "
                        f"оставшиеся чанки пойдут только в BM25",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

        return chunks
