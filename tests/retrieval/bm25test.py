from pathlib import Path
import sys

# Running this file directly puts tests/retrieval on sys.path, not the project
# root where the retrieval and app packages live.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import retrieval.bm25
from app.schemas import Chunk
import unittest
class BM25Test(unittest.TestCase):
    def test_obv(self):
        # BM25 tokenizes words but does not normalize Russian word forms.
        question = "Расторжение договора"
        chunks = [Chunk(0,0,"Погода в Лондоне"), Chunk(0,0,"Нормативный акт 6767"), Chunk(0,0,"Расторжение договора с ПСБ...")]
        bm25 = retrieval.bm25.BM25Search(chunks=chunks)
        results = bm25.search(question,3)
        self.assertEqual(chunks[2], results[0].chunk)
if __name__ == '__main__':
    unittest.main()
