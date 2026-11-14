"""Минимальный BM25Okapi для локального Docker-стенда, без numpy."""

from __future__ import annotations

import math


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = [list(document) for document in corpus]
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in self.corpus]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.size = len(self.corpus)

        frequencies: dict[str, int] = {}
        for document in self.corpus:
            for term in set(document):
                frequencies[term] = frequencies.get(term, 0) + 1
        self.idf = {
            term: math.log(1.0 + (self.size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.size
        average = self.average_length or 1.0
        for term in query:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for index, document in enumerate(self.corpus):
                frequency = document.count(term)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / average
                )
                scores[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return scores
