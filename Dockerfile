# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Модели (YuNet, SFace) и шрифт подписей скачиваются при сборке с проверкой
# SHA-256 — контейнеру не нужен доступ к GitHub при старте.
RUN python -c "from app.detector import ensure_models; ensure_models()"

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown app:app /app/data
USER app

# Токены, база лиц и снимки — в томе, переживают пересоздание контейнера
ENV ROSDOMOFON_DATA_DIR=/app/data
VOLUME /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/auth/status', timeout=4)"

# Внутри контейнера слушаем все интерфейсы; наружу публиковать только на
# localhost или в доверенную сеть — своей авторизации у сервиса нет.
CMD ["python", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
