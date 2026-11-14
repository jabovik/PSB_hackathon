"""Точка входа стенда: готовит данные, поднимает прокси и запускает solution.py."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, "/app")

DATA_SRC = Path(os.getenv("DATA_SRC", "/data"))
DATA_DIR = Path("/tmp/data")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/out"))


def prepare_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    questions = DATA_SRC / "questions.csv"
    if not questions.is_file():
        questions = DATA_SRC / "demo_questions.csv"
    if not questions.is_file():
        raise SystemExit(f"В {DATA_SRC} нет questions.csv или demo_questions.csv")
    shutil.copy2(questions, DATA_DIR / "questions.csv")

    for name in ("knowledge-base.zip", "knowledge_base.zip", "kb.zip"):
        source = DATA_SRC / name
        if source.is_file():
            shutil.copy2(source, DATA_DIR / "knowledge-base.zip")
            break
    else:
        with zipfile.ZipFile(
            DATA_DIR / "knowledge-base.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(DATA_SRC.glob("*.txt")):
                archive.write(path, arcname=path.name)


def main() -> None:
    prepare_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from proxy import serve

    server = serve("127.0.0.1", 8000)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1):
                ready = True
                break
        except Exception:
            time.sleep(0.5)
    if not ready:
        raise SystemExit("Локальный прокси не поднялся")

    key = os.environ["UPSTREAM_API_KEY"]
    os.environ.update(
        {
            "LLM_BASE_URL": "http://127.0.0.1:8000",
            "LLM_API_KEY": key,
            "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-chat"),
            "DATA_DIR": str(DATA_DIR),
            "OUTPUT_FILE": str(OUTPUT_DIR / "submission.csv"),
            "EMB_MODEL": os.getenv("EMB_MODEL", "local-feature-hash"),
            "RERANKER_MODEL": os.getenv("RERANKER_MODEL", "local-cosine"),
            "TOP_K": os.getenv("TOP_K", "4"),
            "RERANKER_TOP_N": os.getenv("RERANKER_TOP_N", "2"),
            "MAX_WORKERS": os.getenv("MAX_WORKERS", "2"),
        }
    )

    import solution

    solution.main()
    server.shutdown()

    output = OUTPUT_DIR / "submission.csv"
    if output.is_file():
        print(f"[bench] готово: {output} ({output.stat().st_size} байт)", flush=True)
    else:
        print("[bench] submission.csv не создан", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
