# Локальный Docker-стенд

Эмуляция платформенного запуска: `solution.py` читает `DATA_DIR/questions.csv` и `knowledge-base.zip`, ходит в локальный OpenAI-совместимый прокси (векторный поиск + реранк) и DeepSeek (генерация), пишет `submission.csv`.

## Запуск

```powershell
docker compose up --build
```

Результат: `out/submission.csv`.

Ключ и модель берутся из `.env` в корне проекта (`LLM_API_KEY`, `LLM_MODEL`), compose подхватывает его автоматически.

## Данные

- `data/questions.csv` — если нет, берётся `data/demo_questions.csv`;
- `data/knowledge-base.zip` / `knowledge_base.zip` / `kb.zip` — если нет, архив собирается из `data/*.txt`.

## Что внутри

- `docker/proxy.py` — stdlib-прокси:
  - `POST /embeddings` — feature hashing по словам и 3-граммам (384 измерения, без скачивания моделей);
  - `POST /rerank` — косинус по этим же векторам;
  - `POST /chat/completions` — проксирует в `{UPSTREAM_BASE_URL}/chat/completions`.
- `docker/shims/` — лёгкие замены `langchain-text-splitters` и `rank-bm25`, чтобы образ собирался без тяжёлых зависимостей.
- `docker/entrypoint.py` — готовит данные, поднимает прокси и запускает `solution.main()`.

## Ограничения

- Векторизация — **feature hashing**, не `BAAI/bge-m3`: семантика слабее, поэтому часть вопросов может уходить в отказ. Настоящие эмбеддинги и Qwen3-реранкер доступны только на платформе.
- `data/`, `out/`, `.venv`, `.git` в образ не попадают (`.dockerignore`).

## Проверка

Прогон на демо-вопросах: `answered=18/18`, `refused=3`, 21.8 с, `out/submission.csv` 7.8 КБ. Локальный судья по эталону — 0.736 (ниже, чем на «настоящих» чанкере/BM25, из-за hashing-векторов).
