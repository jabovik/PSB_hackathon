from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = ROOT / "PSB_hackathon.zip"

SUBMISSION_FILES = (
    "solution.py",
    "application.py",
    "archive_reader.py",
    "bm25.py",
    "document_chunker.py",
    "embedding_client.py",
    "generation_config.py",
    "generator.py",
    "hybrid_search.py",
    "ingester.py",
    "llm_client.py",
    "postprocess.py",
    "prompt.py",
    "reranker.py",
    "retriever.py",
    "schemas.py",
    "vector_search.py",
    "requirements.txt",
    "README.md",
)


def main() -> None:
    with ZipFile(ARCHIVE_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for filename in SUBMISSION_FILES:
            archive.write(ROOT / filename, arcname=filename)


if __name__ == "__main__":
    main()
