from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Keep imports working both in unittest discovery and when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Chunk, SearchResult
from embeddings import EmbeddingClient
from retrieval.Retriever import Retriever
from retrieval.bm25 import BM25Search
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from retrieval.vector_search import VectorSearch


class BM25SearchTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(0, "doc-1", "Погода в Лондоне"),
            Chunk(1, "doc-1", "Нормативный акт 6767"),
            Chunk(2, "doc-1", "Расторжение договора с ПСБ"),
        ]
        self.search = BM25Search(self.chunks)

    def test_returns_matching_chunk_first(self):
        results = self.search.search("Расторжение договора", top_k=3)

        self.assertTrue(results)
        self.assertIs(results[0].chunk, self.chunks[2])
        self.assertGreater(results[0].score, 0)

    def test_returns_empty_when_there_are_no_matching_terms(self):
        self.assertEqual([], self.search.search("космический корабль"))

    def test_handles_empty_corpus_and_non_positive_top_k(self):
        self.assertEqual([], BM25Search([]).search("договор"))
        self.assertEqual([], self.search.search("договор", top_k=0))


class VectorSearchTests(unittest.TestCase):
    def setUp(self):
        self.best_match = Chunk(0, "doc-1", "первый", [1.0, 0.0])
        self.other_match = Chunk(1, "doc-1", "второй", [0.0, 1.0])
        self.no_embedding = Chunk(2, "doc-1", "без вектора")
        self.search = VectorSearch([self.other_match, self.no_embedding, self.best_match])

    def test_ranks_by_cosine_similarity(self):
        results = self.search.search([1.0, 0.0])

        self.assertEqual([self.best_match, self.other_match], [result.chunk for result in results])
        self.assertAlmostEqual(1.0, results[0].score)
        self.assertAlmostEqual(0.0, results[1].score)

    def test_skips_zero_vectors_and_respects_top_k(self):
        search = VectorSearch([
            Chunk(0, "doc-1", "zero", [0.0, 0.0]),
            self.best_match,
        ])

        self.assertEqual([self.best_match], [r.chunk for r in search.search([1.0, 0.0], top_k=1)])
        self.assertEqual([], search.search([0.0, 0.0]))
        self.assertEqual([], self.search.search([1.0, 0.0], top_k=0))

    def test_rejects_mismatched_embedding_dimensions(self):
        search = VectorSearch([Chunk(3, "doc-2", "wrong size", [1.0])])

        with self.assertRaises(ValueError):
            search.search([1.0, 0.0])


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.lexical_and_vector = Chunk(0, "doc-1", "contract termination", [1.0, 0.0])
        self.lexical_only = Chunk(1, "doc-1", "contract clause", [0.0, 1.0])
        self.vector_only = Chunk(2, "doc-1", "weather report", [0.9, 0.1])
        self.search = HybridSearch([self.lexical_and_vector, self.lexical_only, self.vector_only])

    def test_fuses_keyword_and_vector_ranks(self):
        results = self.search.search("contract", query_embedding=[1.0, 0.0], top_k=3)

        self.assertEqual(self.lexical_and_vector, results[0].chunk)
        self.assertEqual(3, len(results))
        self.assertGreater(results[0].score, results[1].score)

    def test_uses_bm25_when_no_query_embedding_is_provided(self):
        results = self.search.search("contract", top_k=5)

        self.assertTrue(results)
        self.assertNotIn(self.vector_only, [result.chunk for result in results])


class RetrieverTests(unittest.TestCase):
    def test_returns_chunks_in_search_order(self):
        relevant = Chunk(0, "doc-1", "Расторжение договора")
        unrelated = Chunk(1, "doc-1", "Погода в Лондоне")
        retriever = Retriever([unrelated, relevant])

        self.assertEqual([relevant], retriever.retrieve("расторжение договора", top_k=1))

    def test_creates_query_embedding_from_question(self):
        best = Chunk(0, "doc-1", "alpha", [1.0, 0.0])
        other = Chunk(1, "doc-1", "beta", [0.0, 1.0])
        embedding_client = Mock(spec=EmbeddingClient)
        embedding_client.embed_text.return_value = [1.0, 0.0]
        retriever = Retriever([other, best], embedding_client=embedding_client)

        self.assertEqual([best], retriever.retrieve("meaning of life", top_k=1))
        embedding_client.embed_text.assert_called_once_with("meaning of life")

    def test_sends_candidates_to_reranker_and_returns_its_order(self):
        first = Chunk(0, "doc-1", "contract first")
        second = Chunk(1, "doc-1", "contract second")
        reranker = Mock()
        reranker.rerank.side_effect = lambda question, candidates, top_k: [
            candidates[1], candidates[0]
        ][:top_k]
        retriever = Retriever([first, second], reranker=reranker)

        results = retriever.retrieve("contract", top_k=2, candidate_k=2)

        self.assertEqual([second, first], results)
        reranker.rerank.assert_called_once()


class RerankerTests(unittest.TestCase):
    def test_posts_candidates_and_returns_reranked_results(self):
        first = Chunk(0, "doc-1", "first text")
        second = Chunk(1, "doc-1", "second text")
        candidates = [
            SearchResult(chunk=first, score=0.4),
            SearchResult(chunk=second, score=0.3),
        ]
        client = Mock()
        response = client.post.return_value
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        reranker = Reranker(client, "https://api.example/v1/", "test-key", "test-model")

        results = reranker.rerank("question", candidates, top_k=1)

        self.assertEqual([second], [result.chunk for result in results])
        self.assertEqual(0.95, results[0].score)
        client.post.assert_called_once_with(
            "https://api.example/v1/rerank",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test-model",
                "query": "question",
                "documents": ["first text", "second text"],
                "top_n": 1,
            },
        )

    def test_returns_empty_for_non_positive_top_k(self):
        client = Mock()
        reranker = Reranker(client, "https://api.example/v1", "test-key")

        self.assertEqual([], reranker.rerank("question", [], top_k=0))
        client.post.assert_not_called()

    def test_from_env_reads_platform_settings(self):
        client = Mock()
        env = {
            "LLM_BASE_URL": "https://api.example/v1/",
            "RERANKER_API_KEY": "rerank-key",
            "RERANKER_MODEL": "custom-reranker",
        }

        with patch.dict("os.environ", env, clear=True):
            reranker = Reranker.from_env(client)

        self.assertIs(reranker.client, client)
        self.assertEqual("https://api.example/v1/rerank", reranker.url)
        self.assertEqual("rerank-key", reranker.api_key)
        self.assertEqual("custom-reranker", reranker.model)


if __name__ == "__main__":
    unittest.main()
