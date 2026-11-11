from __future__ import annotations

import os
from dataclasses import dataclass


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


@dataclass(frozen=True)
class GenConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float

    @classmethod
    def from_env(cls) -> "GenConfig":
        return cls(
            base_url=_env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/"),
            api_key=_env("LLM_API_KEY", "OPENAI_API_KEY"),
            model=_env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL"),
            temperature=float(_env("TEMPERATURE", default="0.2")),
            max_tokens=int(_env("MAX_TOKENS", default="512")),
            timeout=float(_env("LLM_TIMEOUT", default="60")),
        )