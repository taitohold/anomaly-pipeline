"""Error analysis (milestone 5). Writes results/error_analysis.md and CSVs.

Sections:
1. Recall when every detector raises the same number of alarms on test.
2. Chance floor: "catches" made by alarms on untouched data.
3. Who catches what, per anomaly kind.
4. Smallest size each detector reliably catches.
5. False alarms on the untouched test period.
6. The real days the model flags most strongly.

Section 1 sets thresholds on the untouched test period. That is analysis only:
the headline thresholds in results/comparison.md come from validation.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly.analysis import agreement, alarm_peaks, overlap_counts, reliable_size
from anomaly.earnings import earnings_mask
from anomaly.evaluate import (
    alarm_rate,
    calibrate_threshold,
    find_alarms,
    match,
    to_positions,
    touches,
)
from anomaly.inject import KINDS, inject_anomalies
from anomaly.runs import BARS, EARNINGS, PASSES, build_detectors, prepare

VAL_TARGETS = {"monthly": 1.0, "quarterly": 1 / 3}
VAL_LABELS = {"monthly": "1 per stock per month", "quarterly": "1 per stock per quarter"}
TEST_RATES = [0.25, 0.5, 1.0, 2.0]
TOP_N = 20
RESULTS = Path("results")

started = time.time()
bars = pd.read_csv(BARS, parse_dates=["date"])
clean = prepare(bars)
detectors = build_detectors(clean)
names = list(detectors)
mask = earnings_mask(pd.read_csv(EARNINGS), clean) if EARNINGS.exists() else None

scored_clean = {n: clean.assign(score=fn(clean)) for n, fn in detectors.items()}
test_clean = {n: f[f["split"] == "test"] for n, f in scored_clean.items()}

val_threshold = {
    (n, label): calibrate_threshold(f[f["split"] == "val"], target)
    for n, f in scored_clean.items()
    for label, target in VAL_TARGETS.items()
}
test_threshold = {(n, r): calibrate_threshold(test_clean[n], r) for n in names for r in TEST_RATES}
achieved = {
    (n, r): alarm_rate(test_clean[n].assign(flag=test_clean[n]["score"] > test_threshold[(n, r)]))
    for n in names
    for r in TEST_RATES
}
clean_alarms = {
    (n, label): find_alarms(
        test_clean[n].assign(flag=test_clean[n]["score"] > val_threshold[(n, label)])
    )
    for n in names
    for label in VAL_TARGETS
}

caught_rows, chance_rows = [], []
for seed in range(PASSES):
    injected, events = inject_anomalies(bars, seed=seed)
    frame = prepare(injected)
    test = frame[frame["split"] == "test"]
    events = to_positions(events, frame)
    events["event_id"] = [f"{seed}-{i}" for i in range(len(events))]
    for n, fn in detectors.items():
        scored = test.assign(score=fn(frame))
        settings = [(f"val {label}", val_threshold[(n, label)]) for label in VAL_TARGETS]
        settings += [(f"test {r}", test_threshold[(n, r)]) for r in TEST_RATES]
        for setting, threshold in settings:
            ev, _ = match(events, find_alarms(scored.assign(flag=scored["score"] > threshold)))
            caught_rows.append(
                ev[["event_id", "kind", "size", "caught"]].assign(detector=n, setting=setting)
            )
        for label in VAL_TARGETS:
            ev, _ = match(events, clean_alarms[(n, label)])
            chance_rows.append(
                ev[["event_id", "kind", "caught"]].assign(detector=n, setting=f"val {label}")
            )
    print(f"pass {seed + 1}/{PASSES} done ({time.time() - started:.0f}s)")

caught = pd.concat(caught_rows, ignore_index=True)
chance = pd.concat(chance_rows, ignore_index=True)

# Cross-check: recall at the validation thresholds must match results/comparison.md.
comparison = pd.read_csv(RESULTS / "comparison_events.csv")
for label in VAL_TARGETS:
    mine = caught[caught["setting"] == f"val {label}"].groupby(["detector", "kind"])["caught"].sum()
    theirs = (
        comparison[comparison["setting"] == VAL_LABELS[label]]
        .groupby(["detector", "kind"])["caught"]
        .sum()
    )
    if not mine.sort_index().equals(theirs.sort_index()):
        raise SystemExit(f"Cross-check failed at the {label} setting: does not match comparison.md")
print("Cross-check passed: recall at validation thresholds matches results/comparison.md")


def cell(c: int, n: int) -> str:
    return f"{c}/{n} ({c / n:.0%})"


def recall_cells(rows: pd.DataFrame) -> list[str]:
    return [
        cell(
            int(rows.loc[rows["detector"] == n, "caught"].sum()), int((rows["detector"] == n).sum())
        )
        for n in names
    ]


def header(first: list[str]) -> list[str]:
    cols = [*first, *names]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


n_events = caught["event_id"].nunique()
md = [
    "# Error analysis",
    "",
    (
        f"Planted-anomaly numbers use the same {n_events:,} events as `results/comparison.md` "
        f"({PASSES} passes). The cross-check that recall at the validation thresholds matches "
        "that file passed."
    ),
    "",
    "## 1. Recall at equal false-alarm rates on test",
    "",
    (
        "Each detector's threshold is set so it raises the stated number of alarms per stock "
        "per month on the **untouched** test period. Every detector then gets the same alarm "
        "budget on the data it is scored on. This uses test data, so it is analysis only; the "
        "headline table sets thresholds on validation."
    ),
    "",
    *header(["Alarms per stock per month"]),
]
for r in TEST_RATES:
    md.append(
        f"| {r:g} | " + " | ".join(recall_cells(caught[caught["setting"] == f"test {r}"])) + " |"
    )
spread = max(abs(achieved[(n, r)] - r) for n in names for r in TEST_RATES)
md += [
    "",
    f"Achieved rates on untouched test are within {spread:.2f} of each target.",
    "",
    "By kind, at 0.5 alarms per stock per month:",
    "",
    *header(["Kind"]),
]
half = caught[caught["setting"] == "test 0.5"]
for kind in KINDS:
    md.append(f"| {kind} | " + " | ".join(recall_cells(half[half["kind"] == kind])) + " |")

md += [
    "",
    "## 2. Chance floor",
    "",
    (
        "Alarms raised on the **untouched** test period, scored against the planted events' "
        "windows. Nothing was planted in that data, so any catch here is coincidence. Real "
        "recall should be judged against this floor."
    ),
    "",
]
for label in VAL_TARGETS:
    md += [
        f"Threshold {VAL_LABELS[label]} (set on validation): chance / real recall",
        "",
        *header(["Kind"]),
    ]
    ch = chance[chance["setting"] == f"val {label}"]
    real = caught[caught["setting"] == f"val {label}"]
    for kind in KINDS:
        pairs = [
            f"{ch[(ch['kind'] == kind) & (ch['detector'] == n)]['caught'].mean():.0%} / "
            f"{real[(real['kind'] == kind) & (real['detector'] == n)]['caught'].mean():.0%}"
            for n in names
        ]
        md.append(f"| {kind} | " + " | ".join(pairs) + " |")
    md.append("")

quarterly = caught[caught["setting"] == "val quarterly"]
overlap = overlap_counts(quarterly, names)
md += [
    "## 3. Who catches what",
    "",
    (
        "At the quarterly validation threshold. Each event is counted once: caught by all "
        "four, by exactly one, by some (two or three), or by none."
    ),
    "",
    "| Kind | " + " | ".join(overlap.columns) + " |",
    "|" + "---|" * (len(overlap.columns) + 1),
]
for kind, row in overlap.iterrows():
    md.append(f"| {kind} | " + " | ".join(str(int(v)) for v in row) + " |")

md += [
    "",
    "## 4. Smallest size caught reliably",
    "",
    (
        "Smallest size from which a detector catches at least 90% of events at that size and "
        "every larger one. Spike and level-shift sizes are in units of the stock's recent "
        "daily volatility (σ); burst and surge sizes are multipliers. A dash means no size "
        "reaches 90%."
    ),
    "",
]
for label in VAL_TARGETS:
    part = caught[caught["setting"] == f"val {label}"]
    md += [f"Threshold {VAL_LABELS[label]}:", "", *header(["Kind"])]
    sizes = {}
    for n in names:
        recall = (
            part[part["detector"] == n]
            .groupby(["kind", "size"])["caught"]
            .agg(caught="sum", events="count")
            .reset_index()
        )
        sizes[n] = reliable_size(recall)
    for kind in KINDS:
        unit = "σ" if kind in ("spike", "level_shift") else "×"
        values = []
        for n in names:
            s = sizes[n].get(kind)
            values.append("—" if s is None else (f"{s}σ" if unit == "σ" else f"×{s}"))
        md.append(f"| {kind} | " + " | ".join(values) + " |")
    md.append("")

# 5. False alarms on untouched test data, quarterly validation threshold.
alarms_q = {n: clean_alarms[(n, "quarterly")] for n in names}
dates = clean.set_index(["symbol", "pos"])["date"]
stock_days = len(test_clean[names[0]])
summary_rows = []
for n in names:
    al = alarms_q[n]
    others = pd.concat([alarms_q[o] for o in names if o != n])
    summary_rows.append(
        {
            "detector": n,
            "alarms": len(al),
            "per stock per month": len(al) * 21 / stock_days,
            "on earnings days": touches(al, mask).mean() if mask is not None else np.nan,
            "also flagged by z-score": agreement(al, alarms_q["z-score"]).mean()
            if n != "z-score"
            else np.nan,
            "also flagged by another detector": agreement(al, others).mean(),
        }
    )
fa_summary = pd.DataFrame(summary_rows)
md += [
    "## 5. Alarms on the untouched test period",
    "",
    (
        "Quarterly validation threshold, real 2025-2026 data with nothing planted. Some of "
        "these are real events, some are false alarms; there is no way to tell them apart "
        'without labels. "Also flagged" means another detector raised an alarm in the same '
        "stock within one trading day."
    ),
    "",
    (
        "| Detector | Alarms | Per stock per month | On earnings days "
        "| Also flagged by z-score | Also flagged by another detector |"
    ),
    "|---|---|---|---|---|---|",
]


def pct(v) -> str:
    return "—" if pd.isna(v) else f"{v:.0%}"


for r in fa_summary.itertuples(index=False):
    md.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {pct(r[3])} | {pct(r[4])} | {pct(r[5])} |")

by_month = (
    pd.DataFrame(
        {
            n: alarms_q[n]
            .assign(
                month=[
                    dates[(s, p)].strftime("%Y-%m")
                    for s, p in zip(alarms_q[n]["symbol"], alarms_q[n]["a_start"])
                ]
            )
            .groupby("month")
            .size()
            for n in names
        }
    )
    .fillna(0)
    .astype(int)
)
busiest = by_month.sum(axis=1).nlargest(3)
md += [
    "",
    "Alarms by month (month the alarm started):",
    "",
    *header(["Month"]),
    *[f"| {m} | " + " | ".join(str(v) for v in row) + " |" for m, row in by_month.iterrows()],
    "",
    "Busiest months, all detectors together: "
    + ", ".join(f"{m} ({v})" for m, v in busiest.items())
    + ". Look these up to see what moved the market.",
    "",
]
top_stocks = {n: alarms_q[n]["symbol"].value_counts().head(5) for n in names}
md += ["Stocks with the most alarms:", "", *header(["Rank"])]
for i in range(5):
    md.append(
        f"| {i + 1} | "
        + " | ".join(
            f"{s.index[i]} ({s.iloc[i]})" if i < len(s) else "" for s in top_stocks.values()
        )
        + " |"
    )

# 6. The model's strongest alarms on real, untouched test data.
model = "autoencoder"
peaks = alarm_peaks(test_clean[model], alarms_q[model]).nlargest(TOP_N, "peak_score")
features_at = clean.set_index(["symbol", "pos"])
top = pd.DataFrame(
    {
        "date": [dates[(s, p)].date() for s, p in zip(peaks["symbol"], peaks["peak_pos"])],
        "symbol": peaks["symbol"].to_numpy(),
        "move": [
            np.expm1(features_at.loc[(s, p), "log_return"])
            for s, p in zip(peaks["symbol"], peaks["peak_pos"])
        ],
        "volume vs 20-day median": [
            np.exp(features_at.loc[(s, p), "rel_volume"])
            for s, p in zip(peaks["symbol"], peaks["peak_pos"])
        ],
        "earnings": touches(peaks, mask).to_numpy() if mask is not None else False,
        "z-score too": agreement(peaks, alarms_q["z-score"]).to_numpy(),
        "forest too": agreement(peaks, alarms_q["isolation forest"]).to_numpy(),
        "score": peaks["peak_score"].to_numpy(),
    }
)


def yes(flag) -> str:
    return "yes" if flag else ""


md += [
    "",
    "## 6. Real days the model flags most strongly",
    "",
    (
        f"The autoencoder's {TOP_N} highest-scoring alarms on the untouched test period "
        "(quarterly validation threshold), one row per alarm at its highest-scoring day. "
        '"Move" is that day\'s close-to-close change. The last column is for you: look each '
        "one up."
    ),
    "",
    "| Date | Stock | Move | Volume vs 20-day median | Earnings window | Z-score too | Forest too | What happened |",
    "|---|---|---|---|---|---|---|---|",
    *[
        f"| {r.date} | {r.symbol} | {r.move:+.1%} | ×{r[3]:.1f} | {yes(r.earnings)} | "
        f"{yes(r[5])} | {yes(r[6])} | |"
        for r in top.itertuples(index=False)
    ],
    "",
    "Generated by `scripts/error_analysis.py`.",
]

RESULTS.mkdir(exist_ok=True)
(RESULTS / "error_analysis.md").write_text("\n".join(md) + "\n")
caught.to_csv(RESULTS / "error_analysis_events.csv", index=False)
fa_summary.to_csv(RESULTS / "error_analysis_alarms.csv", index=False)
by_month.to_csv(RESULTS / "error_analysis_by_month.csv")
top.to_csv(RESULTS / "error_analysis_top_days.csv", index=False)
chance.to_csv(RESULTS / "error_analysis_chance.csv", index=False)
pd.concat(
    {n: alarms_q[n]["symbol"].value_counts() for n in names}, names=["detector", "symbol"]
).rename("alarms").to_csv(RESULTS / "error_analysis_by_stock.csv")
print("\n".join(md))
print(f"\nSaved to {RESULTS}/ in {time.time() - started:.0f}s")
