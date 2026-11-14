from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


# Load .env when running locally. Docker Compose also passes the same file.
load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    gateway_api_key: str

    deepseek_api_key: str
    deepseek_base_url: str
    llm_model: str

    siliconflow_api_key: str
    siliconflow_base_url: str
    emb_model: str

    reranker_enabled: bool
    reranker_model: str
    reranker_top_n: int

    http_timeout: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        gateway_api_key=os.getenv("GATEWAY_API_KEY", "").strip(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-flash").strip(),
        siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", "").strip(),
        siliconflow_base_url=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ).rstrip("/"),
        emb_model=os.getenv("EMB_MODEL", "BAAI/bge-m3").strip(),
        reranker_enabled=_as_bool(os.getenv("RERANKER_ENABLED"), False),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        ).strip(),
        reranker_top_n=max(1, int(os.getenv("RERANKER_TOP_N", "5"))),
        http_timeout=max(5.0, float(os.getenv("HTTP_TIMEOUT", "120"))),
    )
