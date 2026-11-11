from math import sqrt

from schemas import Chunk, SearchResult


class VectorSearch:

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0 or not query_embedding:
            return []

        query_norm = sqrt(sum(value * value for value in query_embedding)) # norm calculation
        if query_norm == 0:
            return []

        ranked: list[SearchResult] = []
        for chunk in self.chunks:
            embedding = chunk.embedding
            if embedding is None:
                continue
            if len(embedding) != len(query_embedding):
                raise ValueError(
                    f"Размер эмбеддинга фрагмента {chunk.id} ({len(embedding)}) "
                    f"не совпадает с размером запроса ({len(query_embedding)})"
                )

            chunk_norm = sqrt(sum(value * value for value in embedding)) # norm calculation
            if chunk_norm == 0:
                continue

            score = sum(q * d for q, d in zip(query_embedding, embedding)) / (query_norm * chunk_norm)
            ranked.append(SearchResult(chunk=chunk, score=float(score)))

        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]