import re
from rank_bm25 import BM25Okapi

from app.schemas import Chunk, SearchResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Search:

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None

    def search(self, question: str, top_k: int = 20) -> list[SearchResult]:
        if top_k <= 0 or self.bm25 is None:
            return []

        tokenized_question = tokenize(question)
        if not tokenized_question:
            return []

        scores = self.bm25.get_scores(tokenized_question)
        query_terms = set(tokenized_question)
        matching = [
            (index, score)
            for index, (chunk, score) in enumerate(zip(self.chunks, scores))
            if query_terms.intersection(tokenize(chunk.text))
        ]
        matching.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=self.chunks[index], score=float(score))
            for index, score in matching[:top_k]
        ]
