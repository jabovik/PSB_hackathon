from __future__ import annotations

from pathlib import Path

import httpx

from app.schemas import Chunk, Document
from embeddings import EmbeddingClient

from .archive_reader import read_archive
from .documents_chunker import chunk_documents


class Ingester:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        max_chunk_chars: int = 1800,
        overlap_chars: int = 140,
        embedding_batch_size: int = 32,
        timeout: float = 60.0,
    ):
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size должен быть положительным")
        if max_chunk_chars < 300:
            raise ValueError("max_chunk_chars должен быть не меньше 300")

        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.embedding_batch_size = embedding_batch_size
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []
        self._http_client: httpx.Client | None = None

        if embedding_client is None:
            self._http_client = httpx.Client(timeout=timeout)
            try:
                self.embedding_client = EmbeddingClient.from_env(self._http_client)
            except Exception:
                self._http_client.close()
                self._http_client = None
                raise
        else:
            self.embedding_client = embedding_client

    def create_embeddings(self, chunks: list[Chunk]) -> list[list[float]]:
        vectors: list[list[float]] = []

        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            batch_vectors = self.embedding_client.embed_texts(
                [chunk.text for chunk in batch]
            )

            for chunk, vector in zip(batch, batch_vectors, strict=True):
                chunk.embedding = vector
                vectors.append(vector)

        return vectors

    def ingest(self, source: str | Path) -> list[Chunk]:
        documents = read_archive(source)
        chunks = chunk_documents(
            documents,
            max_chars=self.max_chunk_chars,
            overlap_chars=self.overlap_chars,
        )

        vectors = self.create_embeddings(chunks)

        self.documents = documents
        self.chunks = chunks
        self.embeddings = vectors
        return self.chunks

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "Ingester":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
