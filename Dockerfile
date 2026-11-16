# One image for the producer, both consumers, and the dashboard.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1

# Dependencies first, so code changes do not reinstall PyTorch.
# On Linux, uv takes PyTorch from its CPU-only index (see pyproject.toml).
# The cache mount keeps uv's download cache out of the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY dashboard ./dashboard
COPY checkpoints ./checkpoints
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
