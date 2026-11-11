"""Application facade that connects ingestion, retrieval, and generation."""

from __future__ import annotations

from pathlib import Path

import httpx

from app.schemas import Chunk
from embeddings import EmbeddingClient
from generation import generate_answer
from generation.client import LLMClient
from generation.config import GenConfig
from ingestion.Ingester import Ingester
from retrieval.Retriever import Retriever
from retrieval.reranker import Reranker


class App:
    """Load a knowledge base, answer questions, and own the API clients."""

    def __init__(
        self,
        ingester: Ingester,
        embedding_client: EmbeddingClient,
        reranker: Reranker,
        llm_client: LLMClient,
        http_client: httpx.Client,
    ) -> None:
        self.ingester = ingester
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.llm_client = llm_client
        self._http_client = http_client
        self._retriever: Retriever | None = None

    @classmethod
    def from_env(cls) -> App:
        """Build clients from environment settings and share the HTTP connection."""
        http_client = httpx.Client(timeout=60.0)
        try:
            embedding_client = EmbeddingClient.from_env(http_client)
            reranker = Reranker.from_env(http_client)
            llm_client = LLMClient(GenConfig.from_env())
            ingester = Ingester(embedding_client)
        except Exception:
            http_client.close()
            raise

        return cls(
            ingester=ingester,
            embedding_client=embedding_client,
            reranker=reranker,
            llm_client=llm_client,
            http_client=http_client,
        )

    def ingest(self, source: Path) -> list[Chunk]:
        """Ingest a source and replace the active searchable knowledge base."""
        chunks = self.ingester.ingest(source)
        self._retriever = Retriever(
            chunks,
            embedding_client=self.embedding_client,
            reranker=self.reranker,
        )
        return chunks

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> str:
        """Retrieve relevant chunks and generate an answer grounded in them."""
        if self._retriever is None:
            raise RuntimeError("Сначала загрузите базу знаний вызовом ingest(source)")

        chunks = self._retriever.retrieve(
            question,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return generate_answer(
            question,
            [chunk.text for chunk in chunks],
            client=self.llm_client,
        )

    def close(self) -> None:
        self.llm_client.close()
        self._http_client.close()

    def __enter__(self) -> App:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
