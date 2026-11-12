"""Генерация ответа: сборка сообщений для LLM, вызов и чистка ответа."""

from __future__ import annotations

from llm_client import LLMClient
from gen_config import GenConfig
from gen_postprocess import FALLBACK, clean_answer, verify_citations
from gen_prompt import build_messages

_client: LLMClient | None = None


def _default_client() -> LLMClient:
    """Клиент создаётся лениво и переиспользуется между вызовами."""
    global _client
    if _client is None:
        _client = LLMClient(GenConfig.from_env())
    return _client


def generate_answer(
    question: str,
    contexts: list[str],
    *,
    client: LLMClient | None = None,
) -> str:
    """Отвечает на вопрос по найденным фрагментам.

    ``client`` нужен для подмены в тестах; по умолчанию используется общий
    DeepSeek-клиент из настроек окружения. Если фрагментов нет или модель не
    ответила, возвращается ``FALLBACK``.
    """
    if not contexts:
        return FALLBACK

    messages = build_messages(question, contexts)
    raw = (client or _default_client()).chat(messages)
    return verify_citations(clean_answer(raw), contexts)
