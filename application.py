"""Application facade that connects ingestion, retrieval, and generation."""

from __future__ import annotations

from pathlib import Path

import httpx

from schemas import Chunk
from embedding_client import EmbeddingClient
from generator import generate_answer
from llm_client import LLMClient
from gen_config import GenConfig
from ingester import Ingester
from retriever import Retriever
from reranker import Reranker

#: Если модель отказалась, пробуем ещё раз с более широким контекстом.
EXPANDED_TOP_K = 12
EXPANDED_CANDIDATE_K = 24
#: Сколько чанков максимум уходит в генератор и у скольких верхних
#: результатов добавляем соседей по документу.
MAX_CONTEXTS = 12
NEIGHBOR_SEED_COUNT = 2
_REFUSAL_MARKERS = ("В предоставленной базе знаний", "вне компетенции ассистента")


def _is_refusal(answer: str) -> bool:
    return any(marker in answer for marker in _REFUSAL_MARKERS)


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

    def _expand_with_neighbors(self, chunks: list[Chunk]) -> list[Chunk]:
        """Добавляет соседние чанки того же документа для верхних результатов.

        Режет документы по пунктам, поэтому факт может оказаться на границе
        двух чанков. Соседи закрывают такие обрывы, не меняя индекс.
        """
        if self._retriever is None or len(chunks) >= MAX_CONTEXTS:
            return chunks

        by_id = {chunk.id: chunk for chunk in self._retriever.chunks}
        expanded: list[Chunk] = []
        seen: set[int] = set()

        for position, chunk in enumerate(chunks):
            candidates = [chunk]
            if position < NEIGHBOR_SEED_COUNT:
                previous = by_id.get(chunk.id - 1)
                following = by_id.get(chunk.id + 1)
                if previous is not None and previous.document_id == chunk.document_id:
                    candidates.insert(0, previous)
                if following is not None and following.document_id == chunk.document_id:
                    candidates.append(following)

            for candidate in candidates:
                if candidate.id in seen:
                    continue
                seen.add(candidate.id)
                expanded.append(candidate)
                if len(expanded) >= MAX_CONTEXTS:
                    return expanded

        return expanded

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
        answer = generate_answer(
            question,
            [chunk.text for chunk in self._expand_with_neighbors(chunks)],
            client=self.llm_client,
        )

        if _is_refusal(answer) and top_k < EXPANDED_TOP_K:
            expanded = self._retriever.retrieve(
                question,
                top_k=EXPANDED_TOP_K,
                candidate_k=max(candidate_k, EXPANDED_CANDIDATE_K),
            )
            if len(expanded) > len(chunks):
                answer = generate_answer(
                    question,
                    [chunk.text for chunk in self._expand_with_neighbors(expanded)],
                    client=self.llm_client,
                )

        return answer

    def close(self) -> None:
        self.llm_client.close()
        self._http_client.close()

    def __enter__(self) -> App:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
