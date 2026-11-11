from __future__ import annotations

import os
import zipfile
from pathlib import Path

from app.schemas import Document


def find_archive(source: str | Path) -> Path:
    path = Path(source).expanduser()

    if path.is_file():
        return path

    filename = path.name
    data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
    for candidate in (data_dir / filename, Path("data") / filename):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Не найден архив {source!s}. Проверьте путь или переменную DATA_DIR."
    )


def read_archive(source: str | Path) -> list[Document]:
    archive_path = find_archive(source)
    documents: list[Document] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        filenames = sorted(
            name for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith(".txt")
        )

        for filename in filenames:
            text = archive.read(filename).decode("utf-8-sig").strip()
            if text:
                documents.append(
                    Document(
                        id=len(documents) + 1,
                        filename=filename,
                        text=text,
                    )
                )

    if not documents:
        raise ValueError(f"В архиве {archive_path} не найдено TXT-документов")

    return documents
