# anomaly-pipeline

A PyTorch autoencoder that learns what normal daily price and volume movement
looks like for 30 large US stocks and flags what doesn't fit. It is compared
against a z-score rule on the same test data and will run inside a streaming
pipeline (Redpanda → PostgreSQL → Streamlit) next to that rule.

**Status: work in progress.** Milestone 1 of 7 (project skeleton). There are no
results yet. See [SPEC.md](SPEC.md) for the goal, [PLAN.md](PLAN.md) for the
design, and [DECISIONS.md](DECISIONS.md) for the choices made so far.

This detects unusual movement only. It does not predict prices or give trading
signals.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
make install   # create the environment
make test      # run the tests
make lint      # check style
```

## Origin

The pipeline design (REST API → Redpanda → z-score detector → PostgreSQL →
Streamlit) comes from an earlier team project. That code is not in this repo.
Everything here is new and written solo. Full credit details will be added
with the final README.
