"""Score all four detectors on anomalies planted into the test period.

Needs checkpoints/autoencoder.pt (run scripts/train.py first). Writes
results/comparison.md (the table), results/comparison_summary.csv, and
results/comparison_events.csv (one row per planted event per detector setting).
"""

import time
from pathlib import Path

import pandas as pd

from anomaly.detectors import (
    AutoencoderDetector,
    IsolationForestDetector,
    robust_scores,
    zscore_scores,
)
from anomaly.earnings import earnings_mask
from anomaly.evaluate import (
    DAYS_PER_MONTH,
    alarm_rate,
    calibrate_threshold,
    find_alarms,
    match,
    to_positions,
    touches,
    with_positions,
)
from anomaly.features import build_features
from anomaly.inject import KINDS, inject_anomalies
from anomaly.split import assign_split
from anomaly.train import load_checkpoint

PASSES = 10
TARGETS = {"1 per stock per month": 1.0, "1 per stock per quarter": 1 / 3}
FIXED = ("z-score", "|z| > 3 (the original rule)", 3.0)
RESULTS = Path("results")
EARNINGS = Path("data/raw/earnings.csv")
CHECKPOINT = Path("checkpoints/autoencoder.pt")

started = time.time()
bars = pd.read_csv("data/raw/bars.csv", parse_dates=["date"])


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    features = build_features(raw)
    features["split"] = assign_split(features["date"])
    return with_positions(features)


clean = prepare(bars)
forest = IsolationForestDetector(seed=0).fit(clean)
autoencoder = AutoencoderDetector(*load_checkpoint(CHECKPOINT))
DETECTORS = {
    "z-score": zscore_scores,
    "robust": robust_scores,
    "isolation forest": forest.score,
    "autoencoder": autoencoder.score,
}

# Thresholds come from the untouched validation period, never from test.
settings = []
for name, detector in DETECTORS.items():
    val = clean.assign(score=detector(clean)).query("split == 'val'")
    for label, target in TARGETS.items():
        settings.append((name, label, calibrate_threshold(val, target)))
val_z = clean.assign(score=zscore_scores(clean)).query("split == 'val'")
fixed_val_rate = alarm_rate(val_z.assign(flag=val_z["score"] > FIXED[2]))
settings.append(FIXED)

mask = earnings_mask(pd.read_csv(EARNINGS), clean) if EARNINGS.exists() else None

event_rows, alarm_rows = [], []
for seed in range(PASSES):
    injected, events = inject_anomalies(bars, seed=seed)
    frame = prepare(injected)
    test = frame[frame["split"] == "test"]
    events = to_positions(events, frame)
    for name, detector in DETECTORS.items():
        scored = test.assign(score=detector(frame))
        for det, label, threshold in settings:
            if det != name:
                continue
            alarms = find_alarms(scored.assign(flag=scored["score"] > threshold))
            ev, al = match(events, alarms)
            on_earnings = touches(al, mask) if mask is not None else pd.Series(False, al.index)
            event_rows.append(ev.assign(detector=name, setting=label, seed=seed))
            alarm_rows.append(
                {
                    "detector": name,
                    "setting": label,
                    "seed": seed,
                    "alarms": len(al),
                    "true": int(al["true"].sum()),
                    "false_on_earnings": int((~al["true"] & on_earnings).sum()),
                    "stock_days": len(scored),
                }
            )
    print(f"pass {seed + 1}/{PASSES} done ({time.time() - started:.0f}s)")

events_all = pd.concat(event_rows, ignore_index=True)
summary = pd.DataFrame(alarm_rows).groupby(["detector", "setting"], sort=False).sum()
summary = summary.drop(columns="seed")
summary["false"] = summary["alarms"] - summary["true"]
summary["precision"] = summary["true"] / summary["alarms"]
summary["precision_ex_earnings"] = summary["true"] / (
    summary["alarms"] - summary["false_on_earnings"]
)
summary["false_per_stock_month"] = summary["false"] * DAYS_PER_MONTH / summary["stock_days"]
recall = (
    events_all.groupby(["detector", "setting", "kind", "size"], sort=False)["caught"]
    .agg(caught="sum", events="count")
    .reset_index()
)


def cell(caught: int, n: int) -> str:
    return f"{caught}/{n} ({caught / n:.0%})"


def size_label(kind: str, size) -> str:
    return f"{size}σ" if kind in ("spike", "level_shift") else f"×{size}"


def table(label: str) -> list[str]:
    names = [d for d, s, _ in settings if s == label]
    lines = [
        "| Kind | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    part = recall[recall["setting"] == label]
    for kind in [*KINDS, "all"]:
        rows = part if kind == "all" else part[part["kind"] == kind]
        cells = []
        for name in names:
            r = rows[rows["detector"] == name]
            cells.append(cell(int(r["caught"].sum()), int(r["events"].sum())))
        title = "**All kinds**" if kind == "all" else kind
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    for metric, title in [
        ("precision", "Precision"),
        ("precision_ex_earnings", "Precision, earnings days set aside"),
        ("false_per_stock_month", "False alarms per stock per month (test)"),
    ]:
        values = []
        for name in names:
            s = summary.loc[(name, label)]
            if metric == "precision":
                values.append(cell(int(s["true"]), int(s["alarms"])))
            elif metric == "precision_ex_earnings":
                kept = int(s["alarms"] - s["false_on_earnings"])
                values.append(cell(int(s["true"]), kept) if mask is not None else "n/a")
            else:
                values.append(f"{s[metric]:.2f}")
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    return lines


n_events = len(events_all) // len(settings)
test_days = clean.loc[clean["split"] == "test", "date"]
earnings_note = (
    " (SEC EDGAR 8-K item 2.02)." if mask is not None else " (not run: no earnings data)."
)
md = [
    "# Detectors compared on planted anomalies",
    "",
    (
        "**Every number here comes from anomalies planted into real test-period bars "
        f"({test_days.min().date()} to {test_days.max().date()}), not from labelled real "
        f"events.** {PASSES} passes, {n_events:,} planted events in all. "
        "Cells show caught/planted (recall)."
    ),
    "",
    (
        "Thresholds are set on the untouched validation period (2023-2024) to give the "
        "alarm rate named in each heading. Real unusual days in the test period count as "
        "false alarms, which pulls precision down for every detector; the earnings row sets "
        "aside false alarms that fall within a day of an earnings release" + earnings_note
    ),
    "",
]
for label in TARGETS:
    md += [f"## Threshold: {label} on validation", "", *table(label), ""]
md += [
    f"## The original rule as-is: {FIXED[1]}",
    "",
    f"On validation this fixed threshold gives {fixed_val_rate:.2f} alarms per stock per month.",
    "",
    *table(FIXED[1]),
    "",
    f"## Recall by size, threshold {next(iter(TARGETS))}",
    "",
    "| Kind | Size | " + " | ".join(DETECTORS) + " |",
    "|---|---|" + "---|" * len(DETECTORS),
]
part = recall[recall["setting"] == next(iter(TARGETS))]
for (kind, size), rows in part.groupby(["kind", "size"], sort=True):
    cells = [
        cell(int(r["caught"].iloc[0]), int(r["events"].iloc[0]))
        for r in (rows[rows["detector"] == d] for d in DETECTORS)
    ]
    md.append(f"| {kind} | {size_label(kind, size)} | " + " | ".join(cells) + " |")
md += [
    "",
    "## Thresholds used",
    "",
    "| Detector | Setting | Threshold |",
    "|---|---|---|",
    *[f"| {d} | {s} | {t:.4f} |" for d, s, t in settings],
    "",
    "Generated by `scripts/compare.py`.",
]

RESULTS.mkdir(exist_ok=True)
(RESULTS / "comparison.md").write_text("\n".join(md) + "\n")
summary.to_csv(RESULTS / "comparison_summary.csv")
events_all.to_csv(RESULTS / "comparison_events.csv", index=False)
print("\n".join(md))
print(f"\nSaved to {RESULTS}/ in {time.time() - started:.0f}s")
