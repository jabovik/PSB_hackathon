"""Приведение ответа LLM к виду, который годится для submission.csv.

Ответ модели — сырой текст: он может быть пустым, содержать переводы строк,
обрамляющие кавычки, markdown-обёртку или префикс «Ответ:». В CSV ответ должен
быть одной строкой, поэтому всё это вычищается здесь, а не в вызывающем коде.
"""

from __future__ import annotations

import re

#: Что отдаём, если модель не ответила или ответ оказался пустым.
FALLBACK = "Не удалось получить ответ."

#: Лимит платформы на один ответ в submission.csv.
MAX_ANSWER_CHARS = 10_000

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*")
#: Markdown-разметка внутри ответа: жирный текст и инлайн-код.
_MARKUP_RE = re.compile(r"\*\*|`")
_ANSWER_PREFIX_RE = re.compile(r"^(?:ответ|answer)\s*:\s*", re.IGNORECASE)
_QUOTE_PAIRS = (
    ('"""', '"""'),
    ("'''", "'''"),
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
    ("„", "“"),
    ("`", "`"),
    ("*", "*"),
)


def _strip_wrapping(text: str) -> str:
    """Снимает парные обёртки (кавычки, markdown), повторяя до устойчивого вида."""
    changed = True
    while changed:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                changed = True
    return text


def clean_answer(raw: str | None, fallback: str = FALLBACK) -> str:
    """Возвращает ответ одной строкой или ``fallback``, если ответа нет.

    Пустой ввод, ``None`` и ответ из одних пробелов дают ``fallback``. Иначе
    ответ очищается от markdown-обёртки, префикса «Ответ:» и обрамляющих
    кавычек, а все пробельные последовательности (включая переводы строк)
    сжимаются в одиночные пробелы.
    """
    if raw is None:
        return fallback

    text = raw.strip()
    if not text:
        return fallback

    text = _FENCE_RE.sub(" ", text)
    text = _MARKUP_RE.sub("", text)
    text = _strip_wrapping(text)
    text = _ANSWER_PREFIX_RE.sub("", text)
    text = " ".join(text.split())
    text = _strip_wrapping(text)
    text = text.strip()
    text = _shorten_no_information_answer(text)

    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"

    return text or fallback


#: Ссылка на пункт/статью/часть: «п. 4.2», «ст. 70», «ч. 2».
_REF_RE = re.compile(r"(?:п\.|пункт|ст\.|статья|ч\.|часть)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_SOURCE_RE = re.compile(r"Источник:.*$", re.IGNORECASE)
_NO_INFORMATION_RE = re.compile(
    r"^(?:в предоставленных документах|в предоставленной базе знаний|в базе знаний)\s+"
    r"(?:нет сведений|нет информации|нет документов)\s+о\s+"
    r"(?P<topic>.+?)(?:\s+[—–]\s+|[,;.]|$)",
    re.IGNORECASE,
)


def _shorten_no_information_answer(text: str) -> str:
    """Keep only the missing subject when the answer is a full refusal."""
    match = _NO_INFORMATION_RE.match(text)
    if not match:
        return text
    topic = match.group("topic").strip().rstrip(" .,:;—–")
    if not topic:
        return text
    return f"В предоставленных документах нет сведений о {topic}."


def _has_number(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", text) is not None


_CITATION_STOP_WORDS = {
    "п", "пункт", "ст", "статья", "ч", "часть", "договор", "документ",
    "источник", "ред", "редакция", "от", "по", "в", "и", "№",
}


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def _citation_matches_context(citation: str, number: str, contexts: list[str]) -> bool:
    """Require the cited number and at least one document-specific label together."""
    label = _REF_RE.sub(" ", citation)
    label_words = [
        word for word in _normalize_words(label)
        if word not in _CITATION_STOP_WORDS and len(word) > 1
    ]
    for context in contexts:
        context_words = _normalize_words(context)
        if not _has_number(number, context):
            continue
        if not label_words or any(word in context_words for word in label_words):
            return True
    return False


def verify_citations(answer: str, contexts: list[str]) -> str:
    """Убирает ссылки на пункты, которых нет в переданном контексте.

    Модель иногда переносит номер пункта из соседнего документа. Для судьи
    ссылка на несуществующее положение — выдуманный факт, поэтому такие
    ссылки вырезаем, а не оставляем на удачу.
    """
    if not contexts:
        return answer

    def keep_citation(citation: str, original: str) -> str:
        numbers = _REF_RE.findall(citation)
        if numbers and not all(
            _citation_matches_context(citation, number, contexts)
            for number in numbers
        ):
            return ""
        return original

    result = _PAREN_RE.sub(
        lambda match: keep_citation(match.group(1), match.group(0)), answer
    )
    result = _SOURCE_RE.sub(
        lambda match: keep_citation(match.group(0), match.group(0)), result
    )
    result = re.sub(r"\s+([,.;:])", r"\1", result)
    result = " ".join(result.split()).strip()

    return result or FALLBACK
