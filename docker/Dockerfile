FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY docker/requirements.txt /app/docker-requirements.txt
RUN pip install --no-cache-dir -r /app/docker-requirements.txt

COPY *.py /app/
# Лёгкие замены langchain-text-splitters и rank-bm25: без тяжёлых зависимостей.
COPY docker/shims/ /app/
COPY docker/proxy.py docker/entrypoint.py /app/

ENTRYPOINT ["python", "/app/entrypoint.py"]
