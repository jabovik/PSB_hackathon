from pathlib import Path
import sys
import unittest

# Keep imports working both in unittest discovery and when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Chunk
from retrieval.Retriever import Retriever
from retrieval.bm25 import BM25Search
from retrieval.hybrid_search import HybridSearch
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
        retriever = Retriever([unrelated, relevant], "расторжение договора")

        self.assertEqual([relevant], retriever.retrieve(top_k=1))


if __name__ == "__main__":
    unittest.main()
