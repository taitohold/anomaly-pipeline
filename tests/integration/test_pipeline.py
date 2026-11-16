"""End-to-end test: the whole pipeline in Docker Compose, on synthetic bars.

Starts Redpanda, Postgres and both consumers, replays a file of synthetic
bars through the producer, then checks the scores and flags written to
Postgres against the offline detectors run on the same bars.

Needs Docker running, and ports 55432, 19092 and 8501 free (stop `make up`
first). Run with `make integration`.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

from anomaly.detectors import AutoencoderDetector, zscore_scores
from anomaly.features import build_features
from anomaly.synthetic import make_bars
from anomaly.train import load_checkpoint

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "anomaly-integration"  # separate from the `make up` stack
DSN = f"postgresql://anomaly:anomaly@localhost:{os.environ.get('PG_PORT', '55432')}/anomaly"
SYMBOLS = ["AAPL", "JPM", "MSFT"]  # real tickers, so the model's scaler knows them
TIMEOUT = 180


def compose(*args, env=None, check=True):
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        check=check,
        capture_output=True,
        text=True,
    )


def fetch_scores() -> pd.DataFrame:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT symbol, date, detector, score, flagged FROM scores")
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["symbol", "date", "detector", "score", "flagged"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["symbol", "date", "detector"]).sort_index()


def expected_scores(bars: pd.DataFrame) -> pd.DataFrame:
    """What the offline code says the consumers should write."""
    thresholds = json.loads((ROOT / "checkpoints/thresholds.json").read_text())
    features = build_features(bars)
    detectors = {
        "z-score": zscore_scores,
        "autoencoder": AutoencoderDetector(
            *load_checkpoint(ROOT / "checkpoints/autoencoder.pt")
        ).score,
    }
    parts = []
    for name, detector in detectors.items():
        scored = features.assign(score=detector(features), detector=name).dropna(subset=["score"])
        scored["flagged"] = scored["score"] > thresholds[name]["threshold"]
        parts.append(scored[["symbol", "date", "detector", "score", "flagged"]])
    return pd.concat(parts).set_index(["symbol", "date", "detector"]).sort_index()


def wait_for_rows(n: int) -> pd.DataFrame:
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            scores = fetch_scores()
            if len(scores) >= n:
                return scores
        except psycopg.OperationalError:
            pass
        time.sleep(2)
    raise TimeoutError(f"expected {n} score rows within {TIMEOUT}s")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    replay_dir = tmp_path_factory.mktemp("replay")
    bars = make_bars(SYMBOLS, "2024-06-03", "2024-12-31", seed=11)
    bars.to_csv(replay_dir / "bars.csv", index=False)
    env = {"REPLAY_DIR": str(replay_dir), "REPLAY_START": "2024-06-03", "REPLAY_DELAY": "0"}

    compose("down", "-v", check=False)
    services = ["redpanda", "postgres", "zscore-consumer", "model-consumer"]
    try:  # everything after "up" is inside try, so a failed start still cleans up
        up = compose("up", "-d", "--build", "--wait", *services, env=env, check=False)
        assert up.returncode == 0, up.stderr[-2000:]
        sent = compose("run", "--rm", "producer", env=env)
        assert "sent" in sent.stdout, sent.stdout + sent.stderr
        expected = expected_scores(bars)
        yield {"env": env, "expected": expected, "written": wait_for_rows(len(expected))}
    finally:
        logs = compose("logs", "--no-color", "--tail", "20", check=False).stdout
        print(logs)
        compose("down", "-v", check=False)


def test_every_scorable_day_gets_a_score_from_both_detectors(pipeline):
    expected, written = pipeline["expected"], pipeline["written"]
    assert set(written.index) == set(expected.index)
    assert set(written.index.get_level_values("detector")) == {"z-score", "autoencoder"}


def test_scores_and_flags_match_the_offline_detectors(pipeline):
    expected, written = pipeline["expected"], pipeline["written"].loc[pipeline["expected"].index]
    # The consumer runs PyTorch on Linux in Docker; the expected scores come from
    # PyTorch on the host. Their float32 rounding differs by up to about 1.5e-7,
    # which is large relative to scores near zero, hence the absolute tolerance.
    # Flags use a threshold near 1.8, so this cannot flip one.
    np.testing.assert_allclose(written["score"], expected["score"], rtol=1e-5, atol=1e-6)
    assert (written["flagged"] == expected["flagged"]).all()
    assert written["flagged"].any(), "the test data should produce at least one flag"


def test_replaying_the_same_bars_adds_no_rows(pipeline):
    before = len(fetch_scores())
    compose("run", "--rm", "producer", env=pipeline["env"])
    time.sleep(10)  # let the consumers read the duplicate bars
    assert len(fetch_scores()) == before
