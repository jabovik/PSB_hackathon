import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from unittest.mock import Mock

from application import MAX_CONTEXTS, App
from ingester import Ingester
from schemas import Chunk


class NeighborExpansionTests(unittest.TestCase):
    def make_app(self, chunks: list[Chunk]) -> App:
        app = App(Ingester(None), None, None, Mock(), Mock())
        app._retriever = Mock()
        app._retriever.chunks = chunks
        return app

    def test_adds_previous_and_next_for_top_hits(self):
        chunks = [Chunk(i, 1, f"текст {i}") for i in range(1, 6)]
        app = self.make_app(chunks)

        result = app._expand_with_neighbors([chunks[1], chunks[3]])

        self.assertEqual([1, 2, 3, 4, 5], [chunk.id for chunk in result])

    def test_does_not_cross_document_boundary(self):
        chunks = [
            Chunk(1, 1, "первый документ"),
            Chunk(2, 1, "первый документ, продолжение"),
            Chunk(3, 2, "второй документ"),
            Chunk(4, 2, "второй документ, продолжение"),
        ]
        app = self.make_app(chunks)

        result = app._expand_with_neighbors([chunks[1]])

        self.assertEqual([1, 2], [chunk.id for chunk in result])

    def test_respects_max_contexts(self):
        chunks = [Chunk(i, 1, f"текст {i}") for i in range(1, 31)]
        app = self.make_app(chunks)
        selected = [chunks[index] for index in (2, 5, 8, 11, 14, 17, 20, 23)]

        result = app._expand_with_neighbors(selected)

        self.assertEqual(MAX_CONTEXTS, len(result))

    def test_returns_chunks_as_is_without_retriever(self):
        app = App(Ingester(None), None, None, Mock(), Mock())
        chunks = [Chunk(1, 1, "текст")]

        self.assertEqual(chunks, app._expand_with_neighbors(chunks))


if __name__ == "__main__":
    unittest.main()
