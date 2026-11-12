import httpx

from schemas import Chunk
from embedding_client import EmbeddingClient
from hybrid_search import HybridSearch
from reranker import Reranker

class Retriever:
    """Create query embeddings, retrieve candidates, then optionally rerank them."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_client: EmbeddingClient | None = None,
        reranker: Reranker | None = None,
    ):
        self.chunks = chunks
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.hybrid_search = HybridSearch(chunks)

    def retrieve(self, question: str, top_k: int = 5, candidate_k: int = 20) -> list[Chunk]:
        if top_k <= 0:
            return []
        candidate_k = min(len(self.chunks), max(top_k, candidate_k))

        query_embedding = None
        if self.embedding_client is not None:
            try:
                query_embedding = self.embedding_client.embed_text(question)
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                pass
        results = self.hybrid_search.search(
            question,
            query_embedding=query_embedding,
            top_k=candidate_k,
        )
        if self.reranker is not None and results:
            try:
                reranked = self.reranker.rerank(question, results, top_k=top_k)
                if reranked:
                    results = reranked
                else:
                    results = results[:top_k]
            except (httpx.HTTPError, TypeError, ValueError) as error:
                results = results[:top_k] # use the original order if reranking fails
        else:
            results = results[:top_k]
        return [result.chunk for result in results]
