import os
from collections.abc import Sequence

import httpx

from app.schemas import Chunk


class EmbeddingClient:
    """Client for the OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        api_key: str,
        model: str = "baai/bge-m3",
    ):
        if not base_url.strip():
            raise ValueError("Для эмбеддингов нужен base_url")
        if not api_key.strip():
            raise ValueError("Для эмбеддингов нужен api_key")
        self.client = client
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls, client: httpx.Client):
        base_url = (
            os.getenv("LLM_BASE_URL", "").strip()
            or os.getenv("BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
        )
        api_key = (
            os.getenv("EMBEDDER_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        model = os.getenv("EMB_MODEL", "baai/bge-m3").strip() or "baai/bge-m3"
        return cls(client, base_url, api_key, model)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": list(texts),
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("Ответ сервиса эмбеддингов не содержит список data")

        vectors: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError("Некорректный элемент в ответе сервиса эмбеддингов")
            try:
                index = int(item.get("index", position))
                raw_vector = item["embedding"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Некорректный элемент в ответе сервиса эмбеддингов") from error

            if index < 0 or index >= len(texts) or vectors[index] is not None:
                raise ValueError("Некорректный индекс embedding")
            if not isinstance(raw_vector, list) or not raw_vector:
                raise ValueError("Эмбеддинг должен быть непустым списком чисел")
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as error:
                raise ValueError("Эмбеддинг содержит нечисловые значения") from error

            # The configured embedding model returns normalized vectors, so
            # keep its values as-is instead of calculating the norm again.
            if not any(value != 0.0 for value in vector):
                raise ValueError("Получен нулевой эмбеддинг")
            vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise ValueError("Получены не все embeddings")
        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return list(chunks)
