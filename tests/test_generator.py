import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from generation import generate_answer


class GenerateAnswerStubTests(unittest.TestCase):
    def test_returns_first_context(self):
        answer = generate_answer("Какой лимит?", ["лимит 100 000", "карта бесплатна, честно при честно"])

        self.assertEqual("лимит 100 000", answer)

    def test_truncates_long_context(self):
        answer = generate_answer("вопрос", ["x" * 500])

        self.assertEqual(200, len(answer))

    def test_returns_stub_when_no_contexts(self):
        self.assertEqual("нет данных", generate_answer("вопрос", []))


if __name__ == "__main__":
    unittest.main()