"""Приведение ответа LLM к виду, который годится для submission.csv.

Ответ модели — сырой текст: он может быть пустым, содержать переводы строк,
обрамляющие кавычки, markdown-обёртку или префикс «Ответ:». В CSV ответ должен
быть одной строкой, поэтому всё это вычищается здесь, а не в вызывающем коде.
"""

from __future__ import annotations

import re

#: Что отдаём, если модель не ответила или ответ оказался пустым.
FALLBACK = "Не удалось получить ответ."

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

    return text or fallback
