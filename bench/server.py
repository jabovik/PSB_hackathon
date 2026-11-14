"""Локальный тестовый стенд: вопросы -> App.ask -> ответы, время, отказы, судья."""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from application import App
from gen_config import GenConfig
from ingester import Ingester
from llm_client import LLMClient

DATA = ROOT / "data"
BENCH_DIR = Path(__file__).resolve().parent
KB_ARCHIVE = Path(tempfile.mkdtemp(prefix="psb_bench_")) / "kb.zip"

JUDGE_PROMPT = """Ты — строгий LLM-судья юридического ассистента.
Сравни ответ команды с эталоном и оцени по шкале:
1.00 — верно, полно, не хуже эталона;
0.75 — в основном верно, мелкие недочёты;
0.50 — сопоставимо, но часть фактов потеряна;
0.25 — заметно хуже: неполно или неточно;
0.00 — грубая ошибка, отказ при наличии ответа или выдуманный факт.
Верни строго JSON: {"score": <число>, "reason": "<кратко>"}."""

REFUSAL_MARKERS = ("В предоставленной базе знаний", "вне компетенции")


def build_knowledge_base() -> None:
    with zipfile.ZipFile(KB_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DATA.glob("*.txt")):
            archive.write(path, arcname=path.name)


build_knowledge_base()
app = FastAPI(title="PSB test bench")


def parse_questions(text: str) -> list[tuple[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows: list[tuple[str, str]] = []
    for row in reader:
        question_id = (row.get("id") or "").strip()
        question = (row.get("question") or "").strip()
        if question_id and question:
            rows.append((question_id, question))
    return rows


def parse_golden(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    reader = csv.DictReader(io.StringIO(text))
    return {
        (row.get("id") or "").strip(): (row.get("answer") or "").strip()
        for row in reader
        if (row.get("id") or "").strip()
    }


def judge(client: LLMClient, golden: str, answer: str) -> tuple[float, str]:
    raw = client.chat(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Эталон:\n{golden}\n\nОтвет команды:\n{answer}"},
        ]
    )
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
        score = max(0.0, min(1.0, float(verdict.get("score", 0.0))))
        return score, str(verdict.get("reason", ""))[:200]
    except (ValueError, TypeError):
        return 0.0, f"не разобрал ответ судьи: {raw!r}"[:200]


def demo_text(name: str) -> str:
    path = DATA / name
    return path.read_text(encoding="utf-8-sig") if path.is_file() else ""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BENCH_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/demo")
def demo() -> dict:
    return {
        "questions": demo_text("demo_questions.csv"),
        "golden": demo_text("demo_output.csv"),
    }


@app.post("/api/run")
def run(payload: dict) -> dict:
    questions_text = str(payload.get("questions_text") or "").strip()
    golden_text = str(payload.get("golden_text") or "").strip()
    top_k = max(1, min(20, int(payload.get("top_k") or 8)))
    candidate_k = max(top_k, min(40, int(payload.get("candidate_k") or 12)))
    use_judge = bool(payload.get("judge"))

    questions = parse_questions(questions_text)
    if not questions:
        return {"error": "Не разобрал вопросы: нужен CSV с колонками id,question"}

    cfg = GenConfig.from_env()
    if not cfg.base_url or not cfg.api_key or not cfg.model:
        return {"error": "Нет LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (проверьте .env)"}

    golden = parse_golden(golden_text)
    client = LLMClient(cfg)
    instance = App(Ingester(None), None, None, client, httpx.Client())

    results = []
    scores = []
    refusals = 0
    total_started = time.monotonic()
    try:
        instance.ingest(KB_ARCHIVE)
        for question_id, question in questions:
            started = time.monotonic()
            answer = instance.ask(question, top_k=top_k, candidate_k=candidate_k)
            elapsed = time.monotonic() - started

            refused = any(marker in answer for marker in REFUSAL_MARKERS)
            refusals += int(refused)
            docs = []
            if instance._retriever is not None:
                found = instance._retriever.retrieve(
                    question, top_k=top_k, candidate_k=candidate_k
                )
                docs = [
                    {
                        "preview": " ".join(chunk.text.split())[:160],
                        "chars": len(chunk.text),
                    }
                    for chunk in found[:3]
                ]

            score = None
            reason = ""
            if use_judge and question_id in golden:
                score, reason = judge(client, golden[question_id], answer)
                scores.append(score)

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "answer": answer,
                    "seconds": round(elapsed, 2),
                    "refused": refused,
                    "score": score,
                    "reason": reason,
                    "docs": docs,
                }
            )
    finally:
        instance.close()

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["id", "answer"])
    writer.writerows((row["id"], row["answer"]) for row in results)

    return {
        "results": results,
        "submission": output.getvalue(),
        "summary": {
            "questions": len(results),
            "refusals": refusals,
            "total_seconds": round(time.monotonic() - total_started, 1),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "judged": len(scores),
        },
    }
