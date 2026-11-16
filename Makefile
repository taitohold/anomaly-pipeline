.PHONY: install test lint data earnings train compare analysis

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

# Needs SEC_USER_AGENT in .env (your name and email, as the SEC requires)
earnings:
	uv run --env-file .env python scripts/download_earnings.py

train:
	uv run python scripts/train.py

compare:
	uv run python scripts/compare.py

# Needs results/comparison_events.csv from `make compare` (it cross-checks against it)
analysis:
	uv run python scripts/error_analysis.py
