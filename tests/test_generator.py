import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from generator import generate_answer
from gen_postprocess import FALLBACK


class FakeClient:
    """Подменяет LLMClient.chat: запоминает сообщения, отдаёт заданный ответ."""

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str | None:
        self.calls.append(messages)
        return self.reply


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_cleaned_model_answer(self):
        client = FakeClient("  Лимит — 100 000 ₽.\n\nОформление бесплатно.  ")

        answer = generate_answer("Какой лимит?", ["лимит 100 000"], client=client)

        self.assertEqual("Лимит — 100 000 ₽. Оформление бесплатно.", answer)

    def test_sends_question_and_contexts_to_model(self):
        client = FakeClient("ок")

        generate_answer("Вопрос про лимит", ["первый фрагмент", "второй фрагмент"], client=client)

        self.assertEqual(1, len(client.calls))
        system, user = client.calls[0]
        self.assertEqual("system", system["role"])
        self.assertEqual("user", user["role"])
        self.assertIn("Вопрос про лимит", user["content"])
        self.assertIn("первый фрагмент", user["content"])
        self.assertIn("второй фрагмент", user["content"])

    def test_limits_contexts_to_twelve(self):
        client = FakeClient("ок")
        contexts = [f"фрагмент {i}" for i in range(1, 15)]

        generate_answer("вопрос", contexts, client=client)

        user = client.calls[0][1]["content"]
        self.assertIn("[12] фрагмент 12", user)
        self.assertNotIn("фрагмент 13", user)

    def test_does_not_call_model_without_contexts(self):
        client = FakeClient("ок")

        answer = generate_answer("вопрос", [], client=client)

        self.assertEqual(FALLBACK, answer)
        self.assertEqual([], client.calls)

    def test_falls_back_when_model_returns_none(self):
        self.assertEqual(FALLBACK, generate_answer("вопрос", ["контекст"], client=FakeClient(None)))

    def test_falls_back_when_model_returns_blank(self):
        self.assertEqual(FALLBACK, generate_answer("вопрос", ["контекст"], client=FakeClient("   ")))

    def test_drops_citation_that_is_not_in_context(self):
        client = FakeClient("Арендная плата — 450 000 рублей (п. 9.9, договор аренды).")

        answer = generate_answer("Какая арендная плата?", ["3.1. Арендная плата — 450 000 рублей."], client=client)

        self.assertNotIn("9.9", answer)
        self.assertIn("450 000", answer)


if __name__ == "__main__":
    unittest.main()
