from app.schemas import Chunk
from retrieval.hybrid_search import HybridSearch


class Retriever:
    """Convenience facade: hybrid search when vectors are supplied, BM25 otherwise."""

    def __init__(
        self,
        chunks: list[Chunk],
        question: str,
        query_embedding: list[float] | None = None,
    ):
        self.chunks = chunks
        self.question = question
        self.query_embedding = query_embedding
        self.hybrid_search = HybridSearch(chunks)

    def retrieve(self, top_k: int = 5) -> list[Chunk]:
        results = self.hybrid_search.search(
            self.question,
            query_embedding=self.query_embedding,
            top_k=top_k,
        )
        return [result.chunk for result in results]
