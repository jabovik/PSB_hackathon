from __future__ import annotations

import random
import sys
import time

import httpx

from .config import GenConfig


class LLMClient:
    def __init__(self, cfg: GenConfig) -> None:
        self._cfg = cfg
        self._http = httpx.Client(timeout=cfg.timeout)

    def close(self) -> None:
        self._http.close()

    def chat(self, messages: list[dict]) -> str | None:
        if not self._cfg.base_url or not self._cfg.api_key:
            print("[gen] не заданы LLM_BASE_URL или LLM_API_KEY", file=sys.stderr)
            return None

        url = f"{self._cfg.base_url}/chat/completions"
        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}

        delay = 1.0
        for attempt in range(3):
            try:
                response = self._http.post(url, json=payload, headers=headers)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        "retryable", request=response.request, response=response
                    )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt == 2:
                    print(f"[gen] chat failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return None
                time.sleep(delay + random.random() * 0.5)
                delay *= 2
        return None