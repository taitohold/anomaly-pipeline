.PHONY: install test lint data

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# Needs a .env file with your Alpaca key (see .env.example)
data:
	uv run --env-file .env python scripts/download.py
	uv run python scripts/build_features.py
	uv run python scripts/split_counts.py
