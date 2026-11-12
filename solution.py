"""Платформенная точка входа: загрузка данных и создание submission.csv."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from application import App
from gen_postprocess import FALLBACK

KNOWLEDGE_BASE_NAMES = ("kb.zip", "knowledge_base.zip", "knowledge-base.zip")

#: Оставляем запас до лимита платформы в 20 минут: очередь и запись CSV.
DEFAULT_DEADLINE_SECONDS = 1000.0


def _find_knowledge_base(data_dir: Path) -> Path:
    """Платформа кладёт базу под разными именами — берём первый найденный архив."""
    for name in KNOWLEDGE_BASE_NAMES:
        path = data_dir / name
        if path.is_file():
            return path
    archives = sorted(data_dir.glob("*.zip"))
    if archives:
        return archives[0]
    raise FileNotFoundError(
        f"В {data_dir} нет архива базы знаний (искали {', '.join(KNOWLEDGE_BASE_NAMES)})"
    )


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
    deadline: float,
) -> tuple[int, str]:
    if time.monotonic() >= deadline:
        return position, FALLBACK
    try:
        return position, app.ask(
            question,
            candidate_k=candidate_k,
            top_k=top_k,
        )
    except Exception:
        return position, FALLBACK


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    questions = _load_questions(data_dir / "questions.csv")

    # Глубину поиска берём не меньше разумного минимума: на 2 фрагментах
    # модель слишком часто отказывается, хотя ответ в базе есть.
    candidate_k = max(int(os.getenv("TOP_K", "4")), 24)
    top_k = max(int(os.getenv("RERANKER_TOP_N", "2")), 8)
    worker_count = int(os.getenv("MAX_WORKERS", "2"))
    candidate_k = max(candidate_k, top_k)
    deadline = time.monotonic() + float(
        os.getenv("DEADLINE_SECONDS", str(DEFAULT_DEADLINE_SECONDS))
    )

    answers = [FALLBACK] * len(questions)
    try:
        knowledge_base_path = _find_knowledge_base(data_dir)
        print(
            f"[solution] questions={len(questions)} kb={knowledge_base_path.name}",
            file=sys.stderr,
            flush=True,
        )
        with App.from_env() as app:
            chunks = app.ingest(knowledge_base_path)
            print(f"[solution] ingested chunks={len(chunks)}", file=sys.stderr, flush=True)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _answer_one,
                        app,
                        position,
                        question,
                        candidate_k,
                        top_k,
                        deadline,
                    )
                    for position, (_, question) in enumerate(questions)
                ]

                for future in as_completed(futures):
                    position, answer = future.result()
                    answers[position] = answer
    except Exception as error:
        # Пустой сабмит платформа не принимает, поэтому пишем CSV даже при сбое.
        print(
            f"[solution] база знаний не обработана: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )

    answered = sum(1 for answer in answers if answer != FALLBACK)
    print(f"[solution] answered={answered}/{len(answers)}", file=sys.stderr, flush=True)

    output_path = Path(os.getenv("OUTPUT_FILE", "submission.csv"))
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for (question_id, _), answer in zip(questions, answers, strict=True):
            writer.writerow([question_id, answer])


if __name__ == "__main__":
    main()
