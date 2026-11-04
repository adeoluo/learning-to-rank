# syntax=docker/dockerfile:1

# ---- builder ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=rust:1-slim /usr/local/cargo /usr/local/cargo
COPY --from=rust:1-slim /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /application
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable || uv sync --no-dev --no-editable

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 application
WORKDIR /application
COPY --from=builder --chown=application:application /application /application
ENV PATH="/application/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER application
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import learning_to_rank" || exit 1
CMD ["python", "-m", "learning_to_rank"]
