"""Build the README's number blocks from the files in results/.

Every number in the README comes from here, so none is ever copied by hand.
The README marks each block with

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

and fill() replaces what sits between the markers. tests/test_readme.py fails
if the README is out of date with the results files.
"""

import json
import re
from pathlib import Path

import pandas as pd

from anomaly.inject import KINDS

DETECTORS = ["z-score", "robust", "isolation forest", "autoencoder"]
MONTHLY, QUARTERLY = "1 per stock per month", "1 per stock per quarter"
ORIGINAL_RULE = "|z| > 3 (the original rule)"
SIZE_UNITS = {"spike": "σ", "level_shift": "σ", "vol_burst": "×", "volume_surge": "×"}


def cell(caught: int, n: int) -> str:
    return f"{caught}/{n} ({caught / n:.0%})"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def recall_rows(events: pd.DataFrame, detectors: list[str]) -> list[list[str]]:
    rows = []
    for kind in [*KINDS, None]:
        part = events if kind is None else events[events["kind"] == kind]
        label = "**All kinds**" if kind is None else kind
        cells = []
        for d in detectors:
            hits = part.loc[part["detector"] == d, "caught"]
            cells.append(cell(int(hits.sum()), len(hits)))
        rows.append([label, *cells])
    return rows


def data_block(r: Path) -> str:
    splits = pd.read_csv(r / "split_counts.csv")
    rows = [
        [s.split, f"{s.first_day} to {s.last_day}", f"{s.rows:,}", f"{s.windows:,}"]
        for s in splits.itertuples()
    ]
    stocks = int(splits["stocks"].iloc[0])
    return (
        f"{stocks} stocks, daily bars. Rows are stock-days with full features; windows are "
        "30-day sequences that lie entirely inside one split.\n\n"
        + table(["Split", "Days", "Rows", "Windows"], rows)
    )


def training_block(r: Path) -> str:
    history = pd.read_csv(r / "training_history.csv")
    best = history.loc[history["val_loss"].idxmin()]
    return (
        f"Trained for {len(history)} epochs on CPU; early stopping kept epoch "
        f"{int(best['epoch'])} (validation loss {best['val_loss']:.4f}, "
        f"training loss {best['train_loss']:.4f})."
    )


def headline_block(r: Path) -> str:
    events = pd.read_csv(r / "comparison_events.csv")
    summary = pd.read_csv(r / "comparison_summary.csv").set_index(["detector", "setting"])
    n = events[(events["setting"] == MONTHLY) & (events["detector"] == "z-score")]
    out = [
        (
            f"**{len(n):,} planted events** ({n['seed'].nunique()} passes over the test "
            "period). Cells are caught/planted."
        ),
    ]
    for label in (QUARTERLY, MONTHLY):
        part = events[events["setting"] == label]
        rows = recall_rows(part, DETECTORS)
        for metric, title in [
            ("precision", "Precision"),
            ("precision_ex_earnings", "Precision, earnings days set aside"),
            ("false_per_stock_month", "False alarms per stock per month on test"),
        ]:
            values = []
            for d in DETECTORS:
                s = summary.loc[(d, label)]
                if metric == "precision":
                    values.append(cell(int(s["true"]), int(s["alarms"])))
                elif metric == "precision_ex_earnings":
                    values.append(cell(int(s["true"]), int(s["alarms"] - s["false_on_earnings"])))
                else:
                    values.append(f"{s[metric]:.2f}")
            rows.append([title, *values])
        out += [
            "",
            f"**Threshold set on validation for {label}:**",
            "",
            table(["Kind", *DETECTORS], rows),
        ]
    original = events[events["setting"] == ORIGINAL_RULE]
    s = summary.loc[("z-score", ORIGINAL_RULE)]
    out += [
        "",
        (
            "The original rule as it stands, |z| > 3, catches "
            f"{cell(int(original['caught'].sum()), len(original))} with precision "
            f"{cell(int(s['true']), int(s['alarms']))}."
        ),
    ]
    return "\n".join(out)


def equal_budget_block(r: Path) -> str:
    events = pd.read_csv(r / "error_analysis_events.csv")
    # Settings are stored as "test 0.25", "test 1.0", ...; keep the stored names.
    settings = sorted(
        (s for s in events["setting"].unique() if s.startswith("test ")),
        key=lambda s: float(s.split()[1]),
    )
    rows = []
    for setting in settings:
        rate = float(setting.split()[1])
        part = events[events["setting"] == setting]
        cells = []
        for d in DETECTORS:
            hits = part.loc[part["detector"] == d, "caught"]
            cells.append(cell(int(hits.sum()), len(hits)))
        rows.append([f"{rate:g}", *cells])
    half = events[events["setting"] == "test 0.5"]
    return "\n".join(
        [
            table(["Alarms per stock per month", *DETECTORS], rows),
            "",
            "By kind, at 0.5 alarms per stock per month:",
            "",
            table(["Kind", *DETECTORS], recall_rows(half, DETECTORS)),
        ]
    )


def chance_block(r: Path) -> str:
    """At the monthly validation threshold: recall from coincidence (alarms on
    untouched data) next to real recall."""

    def rate(path: str) -> pd.Series:
        df = pd.read_csv(r / path)
        return df[df["setting"] == "val monthly"].groupby(["kind", "detector"])["caught"].mean()

    chance, real = rate("error_analysis_chance.csv"), rate("error_analysis_events.csv")
    rows = [
        [kind, *[f"{chance[(kind, d)]:.0%} / {real[(kind, d)]:.0%}" for d in DETECTORS]]
        for kind in KINDS
    ]
    return table(["Kind (chance / real)", *DETECTORS], rows)


def real_data_block(r: Path) -> str:
    alarms = pd.read_csv(r / "error_analysis_alarms.csv").set_index("detector")
    by_stock = pd.read_csv(r / "error_analysis_by_stock.csv")
    rows = []
    for d in DETECTORS:
        a = alarms.loc[d]
        top = by_stock[by_stock["detector"] == d].nlargest(1, "alarms").iloc[0]
        rows.append(
            [
                d,
                int(a["alarms"]),
                f"{a['per stock per month']:.2f}",
                f"{a['on earnings days']:.0%}",
                f"{top['symbol']} ({int(top['alarms'])})",
            ]
        )
    return table(
        ["Detector", "Alarms", "Per stock per month", "On earnings days", "Most-flagged stock"],
        rows,
    )


def service_block(r: Path, thresholds: Path) -> str:
    flags = pd.read_csv(r / "service_flags.csv").set_index("metric")["value"]
    t = json.loads(thresholds.read_text())

    def v(name: str) -> str:
        x = flags[name]
        return f"{x:g}" if x != int(x) else f"{int(x):,}"

    return table(
        [
            "Detector",
            "Threshold",
            "Validation alarms per stock per month",
            "Flagged days, 2025-2026",
            "Per stock per month",
        ],
        [
            [
                "z-score",
                f"{t['z-score']['threshold']:g}",
                f"{t['z-score']['val_alarms_per_stock_month']:.2f}",
                v("z-score flagged days"),
                v("z-score flagged days per stock per month"),
            ],
            [
                "autoencoder",
                f"{t['autoencoder']['threshold']:.3f}",
                f"{t['autoencoder']['val_alarms_per_stock_month']:.2f}",
                v("autoencoder flagged days"),
                v("autoencoder flagged days per stock per month"),
            ],
        ],
    ) + (
        f"\n\nFlagged by both: {v('flagged by both')} days; z-score only: "
        f"{v('flagged by z-score only')}; autoencoder only: {v('flagged by autoencoder only')}."
    )


def render(results: Path, thresholds: Path) -> dict[str, str]:
    return {
        "data": data_block(results),
        "training": training_block(results),
        "headline": headline_block(results),
        "equal-budget": equal_budget_block(results),
        "chance": chance_block(results),
        "real-data": real_data_block(results),
        "service": service_block(results, thresholds),
    }


def fill(readme: str, blocks: dict[str, str]) -> str:
    """Replace each generated block in the README text. Every block must have
    its markers in the README, and every marked block must be known."""
    pattern = re.compile(
        r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n).*?(\n<!-- END GENERATED: (?P=name) -->)",
        re.DOTALL,
    )
    found = {m["name"] for m in pattern.finditer(readme)}
    if found != set(blocks):
        raise ValueError(f"README markers {sorted(found)} do not match blocks {sorted(blocks)}")
    return pattern.sub(lambda m: m[1] + blocks[m["name"]] + m[3], readme)
