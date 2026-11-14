from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class GatewayRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(GatewayRequest):
    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)


class EmbeddingsRequest(GatewayRequest):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = "float"


class RerankRequest(GatewayRequest):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    model: str | None = None
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    missing = []
    if not settings.gateway_api_key:
        missing.append("GATEWAY_API_KEY")
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.siliconflow_api_key:
        missing.append("SILICONFLOW_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.http_timeout, connect=15.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(
    title="Model API Gateway",
    description=(
        "A small authenticated proxy exposing the three model endpoints "
        "used by the hackathon README."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def require_gateway_key(request: Request, call_next):
    # Keep health and interactive API documentation accessible for local checks.
    public_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if request.url.path not in public_paths:
        expected = get_settings().gateway_api_key
        auth_header = request.headers.get("Authorization", "")
        supplied = auth_header.removeprefix("Bearer ").strip()
        if not expected or not hmac.compare_digest(supplied, expected):
            return Response(
                content='{"detail":"Invalid or missing gateway API key"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


async def _proxy_post(
    request: Request,
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> Response:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        upstream_response = await request.app.state.http.post(
            url,
            headers=headers,
            json=payload,
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Upstream model API timed out: {url}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach upstream model API: {url}",
        ) from exc

    # Preserve the upstream JSON body and status code. This keeps the response
    # compatible with the API formats expected by the starter code.
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get(
            "content-type", "application/json"
        ),
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/completions", tags=["models"])
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Response:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=503, detail="DeepSeek API key is not configured")

    payload = body.model_dump(exclude_none=True)
    payload["model"] = settings.llm_model

    return await _proxy_post(
        request,
        url=f"{settings.deepseek_base_url}/chat/completions",
        api_key=settings.deepseek_api_key,
        payload=payload,
    )


@app.post("/embeddings", tags=["models"])
async def embeddings(body: EmbeddingsRequest, request: Request) -> Response:
    settings = get_settings()
    if not settings.siliconflow_api_key:
        raise HTTPException(
            status_code=503, detail="Embedding API key is not configured"
        )

    payload = body.model_dump(exclude_none=True)
    payload["model"] = settings.emb_model

    return await _proxy_post(
        request,
        url=f"{settings.siliconflow_base_url}/embeddings",
        api_key=settings.siliconflow_api_key,
        payload=payload,
    )


@app.post("/rerank", tags=["models"])
async def rerank(body: RerankRequest, request: Request) -> Response:
    settings = get_settings()
    if not settings.reranker_enabled:
        raise HTTPException(
            status_code=503,
            detail="Reranker is disabled. Set RERANKER_ENABLED=true to enable it.",
        )
    if not settings.siliconflow_api_key:
        raise HTTPException(
            status_code=503, detail="Reranker API key is not configured"
        )

    payload = body.model_dump(exclude_none=True)
    payload["model"] = settings.reranker_model
    payload["top_n"] = body.top_n or settings.reranker_top_n

    return await _proxy_post(
        request,
        url=f"{settings.siliconflow_base_url}/rerank",
        api_key=settings.siliconflow_api_key,
        payload=payload,
    )
