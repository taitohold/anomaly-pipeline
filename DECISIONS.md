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
- **Time zone (checked 2026-11-15):** I first assumed EDGAR's acceptance times were Eastern despite the "Z". The first download showed that was wrong: the times cluster at 10-12h and 20-21h, which only makes sense as UTC (6-8am and 4-5pm Eastern). The code now converts from UTC.
- **Known gap:** Exxon (XOM) has only 2 item 2.02 filings since 2023, against a median of 16 per stock, so most of its earnings days are not masked.

## 2026-11-15 — Milestone 4

### 20. Overfit check: 4 windows for 3,000 steps, not 64
- **Chose:** the sanity check memorises 4 non-overlapping windows (pass: loss below 5% of start). Real data reached 2.3%; synthetic data reached 0.8%.
- **Alternatives:** 64 windows (the plan), or a wider model for the check.
- **Why:** measured, the 10,475-parameter model memorises slowly. On 64 windows of synthetic noise it was still at 26% of the starting loss after 3,000 steps. The check is meant to prove the loop can learn, and 4 windows does that in under 20 seconds.

### 21. Seed before building the model
- **Found:** two runs with the same TrainConfig seed differed, because the model's starting weights are drawn when it is built, before train() seeds anything.
- **Chose:** call set_seed() before building the model (scripts/train.py does).

### 22. Train on CPU, not the Apple GPU
- **Measured:** 3.4s per epoch on CPU, 1.5s on MPS, after warm-up.
- **Chose:** CPU by default (`--device` can override).
- **Why:** a full run is about 3 minutes either way, CPU results repeat exactly (MPS does not fully guarantee it), and every README number must come out the same on a rerun.

### 23. The first full run hit the 50-epoch cap
- **Observed:** validation loss was still falling at epoch 50 (0.3765), so early stopping never triggered. A longer run would probably improve the model a little.
- **Also observed:** validation loss is below training loss. Expected: training (2016-2022) includes the 2020 crash and 2022; validation (2023-2024) was calmer and easier to rebuild.

### 24. The autoencoder overshoots its alarm budget on test
- **Observed:** thresholds set on validation for 1 alarm per stock per month gave the autoencoder 1.58 false alarms per stock per month on test, against 1.27 for the isolation forest and 0.85 for the z-score. Part of its recall lead at that setting comes from raising more alarms.
- **Likely cause:** 2025-2026 looks less like the calm validation period, and reconstruction error rises across the board.
- **Consequence:** compare at the quarterly setting too, where the overshoot is small. Milestone 5 should add a diagnostic comparing recall at equal false-alarm rates on the test period. That uses test data, so it is analysis only and never sets a reported threshold.

### 25. Retrained with a 150-epoch cap
- **Chose:** raise the default cap from 50 to 150 epochs. Early stopping ended the run at epoch 134, keeping epoch 129 (536s on CPU).
- **Result:** validation loss fell 11% (0.3765 to 0.3334), but detection barely moved. Recall at the monthly setting went from 1634/1800 (91%) to 1618/1800 (90%); at the quarterly setting from 1354/1800 (75%) to 1388/1800 (77%), with false alarms on test rising from 0.62 to 0.69 per stock per month.
- **Lesson:** a lower reconstruction loss does not by itself mean better anomaly detection. The model is kept because it is what `make train` reproduces and early stopping chose it, not because it detects better.

## 2026-11-16 — Milestone 5 (findings from results/error_analysis.md)

### 26. At an equal alarm budget, the autoencoder leads only when alarms are scarce
- **Method:** thresholds set so every detector raises the same number of alarms on the untouched test period (analysis only; headline thresholds stay on validation).
- **Found:** overall recall at 0.25 alarms per stock per month: autoencoder 1086/1800 (60%), z-score 881/1800 (49%). At 1: autoencoder and isolation forest tie at 83%. At 2: the forest and robust z-score (95%) beat the autoencoder (92%).
- **But by kind, at 0.5:** the z-score beats the autoencoder on spikes (82% against 76%), level shifts (72% against 64%) and volatility bursts (82% against 80%). The autoencoder's overall lead comes from volume surges, which the z-score cannot see (70% against 9%).
- **Answer to #24:** the autoencoder's earlier spike win over the z-score at the quarterly validation threshold came from raising more alarms on test. At an equal budget, the simple rule still wins on price-only anomalies, as the spec expected.

### 27. Chance floors are high for long events
- At 1 alarm per stock per month, coincidence alone "catches" 39-48% of volatility bursts (their catch window is 7-12 days long) and 12-26% of one-day events. The z-score's 18% on volume surges equals its chance floor exactly, so it shows no skill there.

### 28. Intel dominates the model's alarms on real data
- At the quarterly threshold on untouched test data, 83 of the autoencoder's 443 alarms (19%) are Intel, and 60 of the forest's 379.
- **Likely cause:** per-stock scaling is fitted on 2016-2022. Intel's 2024-2026 moves are far larger than its training-period ones, so ordinary days for the new Intel look unusual to both learned detectors.
- **Possible fix, not done:** scale with a rolling window instead of fixed training statistics. It belongs in the README's limitations.

## 2026-11-16 — Milestone 6

### 29. The service reuses the offline feature and detector code
- **Chose:** each consumer keeps the last 50 bars per stock and runs the same build_features and detector functions as the offline evaluation.
- **Alternatives:** a separate incremental implementation (faster, but a second copy of the logic to keep in sync).
- **Why:** it rules out train/serve skew by construction. A unit test checks streamed scores equal offline scores for every day: exactly for the z-score, and to float32 rounding (at most 2.4e-7 relative) for the autoencoder. Cost: about 167 bars per second for the model consumer, plenty for daily bars.

### 30. Service thresholds: the original rule, and the model matched to it
- **Chose:** the z-score consumer flags |z| > 3. The model's threshold is set on validation to give the same alarm rate (0.4727 per stock per month). Stored in checkpoints/thresholds.json by scripts/thresholds.py.
- **Observed on the real 2025-2026 replay:** the z-score flagged 306 days (0.46 per stock per month, on target); the model flagged 705 (1.05, 2.3 times its target). They agreed on 168 days. The model's threshold does not hold in the newer regime: the same drift as #24 and #28.

### 31. Delivery and restarts
- **Chose:** commit the database write before the Kafka offsets (at-least-once), with upserts keyed on (symbol, date, detector) so reprocessing changes nothing. On startup a consumer reads the topic from the beginning to rebuild its in-memory windows, and ignores bars not newer than the latest it has.
- **Alternatives:** store the windows in Postgres and resume from committed offsets.
- **Why:** simple and correct for a replay of daily bars. It would not scale to an unbounded live topic; noted as a limitation.

### 32. Model files are committed
- **Chose:** commit checkpoints/autoencoder.pt (49 KB of weights, no market data) and thresholds.json, so `make up` runs without retraining and CI's integration test can load the model.

### 33. Ports and image size
- **Postgres is published on host port 55432** (override with PG_PORT): 5433 was already taken by another project's container on the development machine.
- **Image size:** uv's download cache was being saved in the image (3.9 GB). A build-time cache mount brought it to 2.2 GB; PyTorch (CPU build) is about 690 MB of that.

## 2026-11-16 — Milestone 7

### 34. README numbers are generated, and a test enforces it
- **Chose:** scripts/readme_results.py writes every number block in the README from results/, between marker comments. tests/test_readme.py fails if the README differs, so a hand-edited or stale number breaks CI. Checked by editing one number: the test failed.
- **Alternatives:** copy numbers by hand.
- **Why:** CLAUDE.md requires every README number to come from a rerunnable script. Narrative text avoids numbers that are not in a generated block.

### 35. Download end dates are pinned
- **Chose:** bars end 2026-11-14 (last bar 2026-11-13), and earnings filings after that date are dropped.
- **Alternatives:** download "up to yesterday".
- **Why:** with a moving end date, every rerun would add days and change every number. Pinned, `make reproduce` should give identical results.

### 36. The README says plainly who wrote the code
- **Chose:** the README states that all code was written by Claude (AI) under the owner's direction, extending #16, which covered only the four core pieces. In practice Claude wrote all of the code.
- **Why:** an accurate account; the project's purpose is to show work honestly.

## Still open
- Team project: named in the README as Market Anomaly Pipeline (2026-11-16). Its purpose and teammates are not described; add them if you want them credited.
