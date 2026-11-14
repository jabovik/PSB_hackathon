"""Локальный OpenAI-совместимый прокси без внешних моделей и зависимостей.

Векторизация — feature hashing по словам и трёхбуквенным n-граммам: работает
для русского текста, ничего не скачивает и не требует numpy/fastembed.
Реранк — косинус по этим же векторам. Чат проксируется в DeepSeek.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 384
WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")


def _features(text: str) -> list[str]:
    lowered = text.lower().replace("ё", "е")
    words = WORD_RE.findall(lowered)
    features = list(words)
    for word in words:
        padded = f" {word} "
        features.extend(padded[index : index + 3] for index in range(len(padded) - 2))
    return features


def embed_text(text: str) -> list[float]:
    vector = [0.0] * DIM
    for feature in _features(text):
        index = zlib.crc32(feature.encode("utf-8")) % DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True, "model": "feature-hash"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "bad json"})
            return

        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/embeddings"):
            inputs = payload.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            self._send(
                200,
                {
                    "data": [
                        {"index": index, "embedding": embed_text(text)}
                        for index, text in enumerate(inputs)
                    ]
                },
            )
        elif path.endswith("/rerank"):
            query = str(payload.get("query") or "")
            documents = [str(document) for document in payload.get("documents") or []]
            top_n = min(int(payload.get("top_n") or len(documents)), len(documents))
            query_vector = embed_text(query)
            scores = [cosine(query_vector, embed_text(document)) for document in documents]
            order = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)
            self._send(
                200,
                {
                    "results": [
                        {"index": index, "relevance_score": scores[index]}
                        for index in order[:top_n]
                    ]
                },
            )
        elif path.endswith("/chat/completions"):
            upstream = os.environ["UPSTREAM_BASE_URL"].rstrip("/")
            key = os.environ["UPSTREAM_API_KEY"]
            request = urllib.request.Request(
                f"{upstream}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read().decode("utf-8")
            except Exception as error:
                self._send(502, {"error": f"{type(error).__name__}: {error}"})
                return
            self._send(200, json.loads(body))
        else:
            self.send_error(404)

    def log_message(self, *args) -> None:
        pass


def serve(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    return server
