FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade "pip==26.0.1" \
    && python -m pip install --no-cache-dir "poetry==2.2.1"

ARG APP_UID=1000
ARG APP_GID=1000
RUN set -eux; \
    groupadd -g "${APP_GID}" app || groupadd app; \
    useradd -m -u "${APP_UID}" -g "${APP_GID}" -s /bin/bash app

WORKDIR /app

COPY pyproject.toml poetry.lock ./
# Retry install to mitigate transient PyPI timeouts in CI/builders
RUN set -euo pipefail; \
    poetry --version; \
    for i in 1 2 3; do \
      if poetry install --no-root --no-interaction --no-ansi; then \
        break; \
      fi; \
      echo "Poetry install failed (attempt $i), retrying in 10s..."; \
      sleep 10; \
      if [ "$i" = "3" ]; then \
        echo "Poetry install failed after 3 attempts"; \
        exit 1; \
      fi; \
    done

COPY . /app

RUN mkdir -p /app/.cache /app/reports /tmp/poetry-cache \
    && chown -R app:app /app /tmp/poetry-cache

USER app

CMD ["bash", "-lc", "poetry run python -m src.main --help"]
