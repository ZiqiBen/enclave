FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system enclave \
    && useradd --system --gid enclave --create-home enclave

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY sql ./sql
RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p /app/data/uploads \
    && chown -R enclave:enclave /app/data

USER enclave
EXPOSE 8000
CMD ["/app/.venv/bin/enclave-local"]
