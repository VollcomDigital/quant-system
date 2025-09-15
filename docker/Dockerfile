FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION}

# Ensure poetry is on a standard PATH location
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry || true

WORKDIR /app

COPY pyproject.toml ./
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

CMD ["bash", "-lc", "poetry run python -m src.main --help"]
