import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from postprocess import FALLBACK, clean_answer


class CleanAnswerTests(unittest.TestCase):
    def test_returns_fallback_for_none(self):
        self.assertEqual(FALLBACK, clean_answer(None))

    def test_returns_fallback_for_blank_text(self):
        self.assertEqual(FALLBACK, clean_answer("  \n\t  "))

    def test_uses_custom_fallback(self):
        self.assertEqual("пусто", clean_answer(None, fallback="пусто"))

    def test_keeps_plain_answer(self):
        self.assertEqual("Лимит — 100 000 ₽.", clean_answer("Лимит — 100 000 ₽."))

    def test_collapses_newlines_and_spaces_into_one_line(self):
        raw = "  Лимит — 100 000 ₽.\n\n  Оформление бесплатно.\t"
        self.assertEqual("Лимит — 100 000 ₽. Оформление бесплатно.", clean_answer(raw))

    def test_strips_answer_prefix(self):
        self.assertEqual("Лимит — 100 000 ₽.", clean_answer("Ответ: Лимит — 100 000 ₽."))
        self.assertEqual("Лимит — 100 000 ₽.", clean_answer("ответ:  Лимит — 100 000 ₽."))

    def test_strips_wrapping_quotes(self):
        self.assertEqual("Копилка", clean_answer('"«Копилка»"'))
        self.assertEqual("Копилка", clean_answer("«Копилка»"))

    def test_strips_markdown_fences_and_bold(self):
        raw = "```text\n**Копилка** — бесплатно.\n```"
        self.assertEqual("Копилка — бесплатно.", clean_answer(raw))

    def test_keeps_inner_quotes(self):
        self.assertEqual('Счёт «Копилка» бесплатен.', clean_answer('Счёт «Копилка» бесплатен.'))

    def test_returns_fallback_when_only_prefix_left(self):
        self.assertEqual(FALLBACK, clean_answer("Ответ:"))


if __name__ == "__main__":
    unittest.main()
