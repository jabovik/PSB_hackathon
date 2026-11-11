from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

# Keep imports working both in unittest discovery and when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import Chunk
from embedding_client import EmbeddingClient


class EmbeddingClientTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.response = self.client.post.return_value
        self.embedder = EmbeddingClient(
            self.client,
            "https://api.example/v1/",
            "test-key",
            "test-model",
        )

    def test_returns_normalized_vectors_in_input_order(self):
        self.response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [0.6, 0.8]},
            ]
        }

        vectors = self.embedder.embed_texts(["first", "second"])

        self.assertEqual([[0.6, 0.8], [0.0, 1.0]], vectors)
        self.client.post.assert_called_once_with(
            "https://api.example/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test-model",
                "input": ["first", "second"],
                "encoding_format": "float",
            },
        )

    def test_embed_chunks_stores_returned_embeddings(self):
        self.response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.6, 0.8]}]
        }
        chunk = Chunk(0, "doc-1", "chunk text")

        result = self.embedder.embed_chunks([chunk])

        self.assertIs(result[0], chunk)
        self.assertEqual([0.6, 0.8], chunk.embedding)

    def test_empty_input_does_not_call_api(self):
        self.assertEqual([], self.embedder.embed_texts([]))
        self.client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
