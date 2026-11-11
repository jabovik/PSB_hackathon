"""Платформенная точка входа: загрузка данных и создание submission.csv."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from application import App
from postprocess import FALLBACK


def _load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [(row["id"], row["question"]) for row in reader]


def _answer_one(
    app: App,
    position: int,
    question: str,
    candidate_k: int,
    top_k: int,
) -> tuple[int, str]:
    try:
        return position, app.ask(
            question,
            candidate_k=candidate_k,
            top_k=top_k,
        )
    except Exception:
        return position, FALLBACK


def _write_submission(
    questions: list[tuple[str, str]],
    answers: list[str],
) -> None:
    with Path("submission.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for (question_id, _), answer in zip(questions, answers, strict=True):
            writer.writerow([question_id, answer or FALLBACK])


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    questions = _load_questions(data_dir / "questions.csv")
    knowledge_base_path = data_dir / "knowledge-base.zip"

    # TOP_K задаёт размер выборки перед реранжированием,
    # RERANKER_TOP_N — число чанков, передаваемых генератору.
    candidate_k = int(os.getenv("TOP_K", "4"))
    top_k = int(os.getenv("RERANKER_TOP_N", "2"))
    worker_count = int(os.getenv("MAX_WORKERS", "2"))
    candidate_k = max(candidate_k, top_k)

    answers = [FALLBACK] * len(questions)
    _write_submission(questions, answers)

    try:
        with App.from_env() as app:
            app.ingest(knowledge_base_path)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _answer_one,
                        app,
                        position,
                        question,
                        candidate_k,
                        top_k,
                    )
                    for position, (_, question) in enumerate(questions)
                ]

                for future in as_completed(futures):
                    position, answer = future.result()
                    answers[position] = answer
    except Exception:
        pass

    _write_submission(questions, answers)


if __name__ == "__main__":
    main()
