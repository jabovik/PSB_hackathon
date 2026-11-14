# Model API Gateway

Локальный FastAPI-шлюз для трёх API-вызовов из README стартера:

- `POST /chat/completions` → DeepSeek API;
- `POST /embeddings` → BGE-M3 через SiliconFlow;
- `POST /rerank` → BGE Reranker через SiliconFlow (опционально).

Снаружи все три маршрута доступны на одном `LLM_BASE_URL`. Внутренние URL и ключи провайдеров разделены: ключ DeepSeek не используется для SiliconFlow и наоборот.

## 1. Модели и провайдеры

| Назначение | Провайдер | Model ID |
|---|---|---|
| LLM | DeepSeek | `deepseek-flash` (DeepSeek-V4.1-Flash) |
| Embeddings | SiliconFlow | `BAAI/bge-m3` |
| Reranker | SiliconFlow | `BAAI/bge-reranker-v2-m3` (выключен по умолчанию) |

BGE-M3 в этом стенде вызывается как обычная dense-embedding модель через `/embeddings`. У модели есть и sparse/ColBERT возможности, но этот совместимый маршрут их не возвращает.

## 2. Запуск

Нужны Docker Engine и Docker Compose.

```bash
cp .env.example .env
```

Заполнить в `.env`:

- `DEEPSEEK_API_KEY` — ключ DeepSeek;
- `SILICONFLOW_API_KEY` — ключ SiliconFlow;
- `GATEWAY_API_KEY` — придумать отдельный длинный локальный ключ.

Значения `LLM_API_KEY`, `EMBEDDER_API_KEY` и `RERANKER_API_KEY` в `.env` должны совпадать с `GATEWAY_API_KEY`: это ключ доступа к локальному шлюзу, а не ключи upstream-провайдеров.

Запуск:

```bash
docker compose up --build -d
```

Проверка состояния:

```bash
docker compose ps
docker compose logs -f model-api
curl http://127.0.0.1:8000/health
```

Остановить:

```bash
docker compose down
```

Сервис по умолчанию привязан только к `127.0.0.1`, чтобы не выставлять локальный шлюз в сеть.

## 3. Проверка трёх POST-запросов

Выполнить в каталоге проекта после заполнения `.env`:

```bash
set -a
. ./.env
set +a

curl -sS http://127.0.0.1:8000/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat.json

curl -sS http://127.0.0.1:8000/embeddings \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/embeddings.json
```

Reranker по умолчанию выключен. Для его включения указать в `.env`:

```dotenv
RERANKER_ENABLED=true
```

Перезапустить сервис и отправить:

```bash
curl -sS http://127.0.0.1:8000/rerank \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/rerank.json
```

## 4. Подключение стартера из README

Если `solution.py` запускается на хостовой машине, в его окружении:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8000
LLM_API_KEY=<то же значение, что GATEWAY_API_KEY>
EMBEDDER_API_KEY=<то же значение, что GATEWAY_API_KEY>
RERANKER_API_KEY=<то же значение, что GATEWAY_API_KEY>
LLM_MODEL=deepseek-flash
EMB_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

Если `solution.py` запускается в другом контейнере той же Docker Compose-сети, вместо `127.0.0.1` используйте `model-api` и подключите контейнер к этой сети.

## 5. API-контракт

Тела запросов соответствуют README стартера:

### Chat

```http
POST /chat/completions
Authorization: Bearer <GATEWAY_API_KEY>
Content-Type: application/json
```

```json
{
  "messages": [
    {"role": "user", "content": "Привет!"}
  ],
  "temperature": 0.2,
  "max_tokens": 256
}
```

Поле `model` необязательно: шлюз подставляет `LLM_MODEL` из `.env`.

### Embeddings

```http
POST /embeddings
Authorization: Bearer <GATEWAY_API_KEY>
Content-Type: application/json
```

```json
{
  "input": ["текст 1", "текст 2"],
  "encoding_format": "float"
}
```

Шлюз подставляет модель из `EMB_MODEL`.

### Rerank

```http
POST /rerank
Authorization: Bearer <GATEWAY_API_KEY>
Content-Type: application/json
```

```json
{
  "query": "вопрос",
  "documents": ["фрагмент 1", "фрагмент 2"],
  "top_n": 2
}
```

Шлюз подставляет `RERANKER_MODEL` и значение `RERANKER_TOP_N`, если `top_n` не передан. При `RERANKER_ENABLED=false` маршрут возвращает HTTP 503.

## 6. Коды ошибок

- `401` — нет или неверный локальный `GATEWAY_API_KEY`;
- `422` — некорректное тело запроса;
- `502` — ошибка соединения с upstream API;
- `504` — timeout upstream API;
- `503` — нужный маршрут отключён или не настроен.

Ответы и HTTP-коды upstream-провайдера проксируются без изменения тела.
