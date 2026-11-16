# DECISIONS

One entry per major decision: what we chose, what the alternatives were, and why.
Full reasoning for the milestone 0 entries is in PLAN.md.

## 2026-11-15 — Milestone 0

### 1. Rebuild the pipeline from scratch
- **Chose:** write the whole pipeline (producer, z-score consumer, schema, dashboard) new in this repo.
- **Alternatives:** extend the original team repo.
- **Why:** the original repo and code are lost. The design is credited to the team project; all code here is new.

### 2. Data source: Alpaca Market Data API, free Basic plan
- **Chose:** Alpaca, called with plain `requests`.
- **Alternatives:** Tiingo (fallback), Massive/Polygon (2 years only), Finnhub (no free history), Yahoo via `yfinance` (unofficial scraper).
- **Why:** free bars since 2016, split adjustment built in, 200 requests/min, and terms that allow personal non-commercial use.
- **Consequences:** raw bars are never committed, the dashboard runs locally only, the README shows derived results only, CI uses synthetic bars.
- **Resolved 2026-11-15:** the free plan served full-market (SIP) history. Downloaded 81,990 bars, 2016-01-04 to 2026-11-13, with no missing trading days.

### 3. Bar size: daily
- **Chose:** daily bars, 2016 to present.
- **Alternatives:** 15-minute bars.
- **Why:** flagged dates can be looked up directly, no time-of-day pattern to remove, ten years of market regimes, fast training. Cost: a small dataset (~82k rows, estimate).

### 4. Universe: 30 large US stocks listed before 2016
- **Chose:** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, JPM, V, MA, UNH, JNJ, PG, XOM, CVX, HD, KO, PEP, WMT, COST, DIS, NFLX, ADBE, CRM, INTC, CSCO, BAC, PFE, MRK.
- **Alternatives:** current Dow 30 (some members lack history back to 2016).
- **Why:** every stock has the full period. Survivorship bias is noted as a limitation.

### 5. Time split
- **Chose:** train 2016–2022, validation 2023–2024, test 2025 to latest. A window belongs to a split only if every bar is inside it.
- **Alternatives:** random split (forbidden by the spec: it leaks the future).
- **Why:** mirrors real use, where a model only ever sees the past.

### 6. Features and scaling
- **Chose:** log return, 20-day rolling volatility, log volume relative to its 20-day median. One mean and standard deviation per stock per feature, fitted on the training period. One model for all stocks.
- **Alternatives:** adding high/low range; one global scaler; one model per stock.
- **Why:** these are the three the spec names; per-stock scaling makes stocks comparable; ~1,700 windows per stock is too few for separate models.

### 7. Window and model size
- **Chose:** 30-day window; LSTM autoencoder with hidden size 32, one layer each side, bottleneck of 8 (~10,500 parameters).
- **Alternatives:** longer windows; larger models.
- **Why:** ~51k training windows (estimate) cannot support a big model, and a tight bottleneck is what forces unusual windows to reconstruct badly.

### 8. Model anomaly score: error on the newest bar
- **Chose:** reconstruction error of the last bar in the window.
- **Alternatives:** error over the whole window (the spec's literal wording).
- **Why:** a whole-window score keeps one bad day flagged for 30 bars. Newest-bar error points at the day it happened and suits a live consumer.

### 9. Z-score baseline definition
- **Chose:** rolling 20-bar z-score of log returns, flag when |z| > 3.
- **Alternatives:** the original rule's exact settings (not recoverable).
- **Why:** a standard form of the rule. It sees returns only, so the README compares the model mainly against detectors 2 and 3, which see all features.

### 10. Thresholds come from validation, never test
- **Chose:** set each detector's threshold on the untouched validation period at one alarm per stock per month; also report one per quarter.
- **Alternatives:** tuning thresholds on the test set.
- **Why:** tuning on test data would inflate every number.

### 11. Tooling
- **Chose:** `uv`, CSV files, `confluent-kafka`, `psycopg`, `ruff`, `make`, Python 3.13.
- **Alternatives and reasons:** see PLAN.md section 9.

## 2026-11-15 — Milestone 2

### 12. Tests and CI use synthetic bars
- **Chose:** a seeded random-walk generator (`anomaly.synthetic.make_bars`).
- **Alternatives:** a small committed sample of real bars.
- **Why:** Alpaca's terms forbid redistributing its data, and CI then needs no API key.

### 13. Raw data is written once
- **Chose:** the download script refuses to run if `data/raw/bars.csv` exists, and records the feed and dates in `bars_source.txt`.
- **Alternatives:** overwrite on every run.
- **Why:** CLAUDE.md says raw files are never modified, and every result should trace back to one fixed download.

### 14. CPU-only PyTorch on Linux
- **Chose:** uv installs PyTorch from its CPU-only index on Linux, and the normal build on macOS (139 MB).
- **Alternatives:** the default Linux build (539 MB plus several GB of CUDA libraries).
- **Why:** CI has no GPU, so the CUDA libraries would only slow every run.

### 15. Dataset returns tensors only, with a separate index
- **Chose:** `__getitem__` returns just the window tensor; `self.index[i]` holds (symbol, end date).
- **Alternatives:** return (tensor, symbol, date) tuples.
- **Why:** the training loop only needs tensors, and DataLoader cannot batch strings or dates without extra code.

### 16. Claude writes all four core pieces
- **Chose:** the Dataset class, autoencoder model, training loop, and anomaly injection function are written by Claude (AI), not by hand. Decided 2026-11-15, while writing the Dataset.
- **Alternatives:** write them by hand from given signatures and tests (the original plan in SPEC.md); a guided walkthrough.
- **Why:** the owner's choice. To keep the project honest, the README must say these pieces are AI-written, and the owner should still be able to explain every line.

## 2026-11-15 — Milestone 3

### 17. How planted events are scored
- **Chose:** consecutive flagged days in one stock form one alarm. An event is caught if an alarm overlaps its days or the 2 trading days after. A month is 21 trading days. Precision is true alarms / all alarms.
- **Alternatives:** count flagged days instead of alarms; precision as caught events / (caught + false alarms).
- **Why:** a real alert system notifies once per run of flagged days. Known side effect: one long event (a volatility burst) can set off several alarms that all count as true, which flatters precision slightly.

### 18. Chance-level recall
- **Observed:** the z-score sees only returns, yet "catches" 18% of volume surges. That is coincidence: at about 1 alarm per stock per month, a random alarm lands in a 3-to-5-day catch window roughly 15-20% of the time.
- **Consequence:** recall near 18% at the monthly threshold means no skill. Milestone 5 should measure this floor directly (alarms on the untouched test set, scored against the same event windows) rather than estimate it.

### 19. Earnings days as an outside check on precision
- **Chose:** SEC EDGAR 8-K item 2.02 filings mark earnings releases; precision is also reported with false alarms within 1 trading day of the reaction day set aside.
- **Alternatives:** treat the biggest real moves as true events (biased toward the z-score, which defines "big" the same way); inject into fully synthetic data (loses real market behaviour); a README note only.
- **Why:** an outside source that does not depend on any detector. It misses non-earnings news, so precision still reads low.
- **Assumption to verify:** EDGAR's acceptance times are US Eastern despite the "Z" suffix. The download script prints filings by hour so this can be checked.

## Still open
- Team project name, what it was for, and whether teammates are named (needed for the README in milestone 7).
