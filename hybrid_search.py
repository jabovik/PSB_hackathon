from schemas import Chunk, SearchResult
from bm25 import BM25Search
from vector_search import VectorSearch


class HybridSearch:
    """Combines keyword and vector results with reciprocal rank fusion."""

    def __init__(self, chunks: list[Chunk], rrf_constant: int = 60): # 60 is a common choice for RRF constant
        if rrf_constant < 0:
            raise ValueError("rrf_constant должен быть неотрицательным")
        self.chunks = chunks
        self.rrf_constant = rrf_constant
        self.bm25 = BM25Search(chunks)
        self.vector = VectorSearch(chunks)

    def search(
        self,
        question: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        # Get every candidate so fusion can compare ranks from both methods.
        keyword_results = self.bm25.search(question, top_k=len(self.chunks))
        vector_results = (
            self.vector.search(query_embedding, top_k=len(self.chunks))
            if query_embedding is not None
            else []
        )
        if not vector_results:
            return keyword_results[:top_k]
        if not keyword_results:
            return vector_results[:top_k]

        chunks_by_id = { # link chunk IDs to chunks for easy access during fusion
            result.chunk.id: result.chunk
            for result in (*keyword_results, *vector_results)
        }
        fused_scores: dict[int, float] = {}
        for results in (keyword_results, vector_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + ( # sum scores from both methods
                    1.0 / (self.rrf_constant + rank) # reciprocal rank fusion score calculation
                )

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [
            SearchResult(chunk=chunks_by_id[chunk_id], score=fused_scores[chunk_id])
            for chunk_id in ranked_ids[:top_k]
        ]
