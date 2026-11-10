import os

import httpx

from app.schemas import SearchResult


class Reranker:
    """Rerank an existing candidate list through an OpenAI-compatible /rerank API."""

    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        api_key: str,
        model: str = "cohere/rerank-5-fast",
    ):
        if not base_url.strip():
            raise ValueError("Для реранкера нужен base_url")
        if not api_key.strip():
            raise ValueError("Для реранкера нужен api_key")
        self.client = client
        self.url = base_url.rstrip("/") + "/rerank"
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
            os.getenv("RERANKER_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        model = os.getenv("RERANKER_MODEL", "cohere/rerank-5-fast").strip()
        return cls(client, base_url, api_key, model or "cohere/rerank-5-fast")

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0 or not candidates:
            return []

        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": question,
                "documents": [result.chunk.text for result in candidates],
                "top_n": min(top_k, len(candidates)),
            },
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("results", payload.get("data"))
        if not isinstance(items, list):
            raise ValueError("Ответ реранкера не содержит список results")

        reranked: list[SearchResult] = []
        seen_indices: set[int] = set()
        for rank, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= len(candidates) or index in seen_indices:
                continue

            seen_indices.add(index)
            raw_score = item.get("relevance_score", item.get("score", 1.0 / (rank + 1)))
            reranked.append(SearchResult(chunk=candidates[index].chunk, score=float(raw_score)))
            if len(reranked) == top_k:
                break

        return reranked
