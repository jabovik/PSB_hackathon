import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from gen_postprocess import FALLBACK, clean_answer, verify_citations


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

    def test_caps_answer_at_platform_limit(self):
        self.assertLessEqual(len(clean_answer("слово " * 5000)), 10_000)


class VerifyCitationsTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            "Документ: договор аренды.txt\n\n3.1. Арендная плата — 450 000 рублей.",
            "Документ: регламент.txt\n\n2.4. Договоры свыше 10 000 000 рублей согласуются с директором.",
        ]

    def test_keeps_citation_present_in_context(self):
        answer = "Арендная плата — 450 000 рублей (п. 3.1, договор аренды)."

        self.assertEqual(answer, verify_citations(answer, self.contexts))

    def test_drops_citation_absent_from_context(self):
        answer = "Арендная плата — 450 000 рублей (п. 9.9, договор аренды)."

        self.assertEqual(
            "Арендная плата — 450 000 рублей.",
            verify_citations(answer, self.contexts),
        )

    def test_keeps_parentheses_without_references(self):
        answer = "Арендная плата — 450 000 рублей (без НДС)."

        self.assertEqual(answer, verify_citations(answer, self.contexts))

    def test_drops_source_tail_with_absent_reference(self):
        answer = "Арендная плата — 450 000 рублей. Источник: договор, п. 9.9."

        self.assertNotIn("9.9", verify_citations(answer, self.contexts))

    def test_returns_fallback_when_only_citation_left(self):
        self.assertEqual(FALLBACK, verify_citations("(п. 9.9)", self.contexts))


if __name__ == "__main__":
    unittest.main()
