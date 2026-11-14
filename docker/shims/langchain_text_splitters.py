"""Минимальная замена langchain-text-splitters для локального Docker-стенда.

Режет текст по окну chunk_size с перекрытием, стараясь резать по границе
строки или предложения. API совместим с тем, что использует document_chunker.py.
"""

from __future__ import annotations


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators=None,
        is_separator_regex: bool = False,
        keep_separator: bool = False,
        length_function=len,
        **kwargs,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.length_function = length_function

    def _cut(self, text: str) -> int:
        window = text[: self.chunk_size]
        newline = window.rfind("\n")
        sentence = window.rfind(". ")
        boundary = max(newline, sentence)
        if boundary > self.chunk_size * 0.6:
            return boundary + 1
        return len(window)

    def split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._cut(text[start:])
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks
