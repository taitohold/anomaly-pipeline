# PLAN: Learned anomaly detector for a market data pipeline

Milestone 0 output. Nothing here is built yet. Every number marked *estimate* is
arithmetic or a guess, not a measurement; measured numbers start in milestone 2.

## 1. What changed from the spec

The spec assumed I would review the existing team pipeline. You told me the repo
and code are lost and you don't remember the data API, so:

- **The pipeline is rebuilt from scratch in this repo**, following the same shape
  as the original: REST API → Redpanda → z-score detector → PostgreSQL → Streamlit.
- **"The existing z-score rule" becomes a re-implementation**, not the original code.
- **Milestone 6 grows.** It now includes the producer, the z-score consumer, the
  database schema, and the dashboard, not only the model consumer.
- **The README has to say this plainly**: the pipeline *design* comes from a team
  project (credited by name), all code in this repo is new and written solo, and
  the original code was not available. On a resume this is "rebuilt and extended",
  not "added to the team's production pipeline".

## 2. Data source

### Recommendation: Alpaca Market Data API, free Basic plan

| Option | Free history | Limits | Verdict |
|---|---|---|---|
| **Alpaca (Basic)** | Daily and minute bars since 2016, split/dividend adjustment flag | 200 requests/min | **Recommended** |
| Tiingo (free) | Daily bars going back decades, full-market volume | 50 requests/hour, 1,000/day | Fallback. Good daily data, but no path to intraday or live bars |
| Massive (was Polygon) | 2 years | 5 requests/min | Rejected: 2 years is too little to split three ways |
| Finnhub (free) | Stock candles reported as paid-only since ~2025 | 60 requests/min | Rejected: no free history |
| Yahoo via `yfinance` | Long | Unofficial | Rejected: it is an unofficial scraper, so there are no terms that clearly allow this |

Why Alpaca: one documented REST endpoint returns bars for many symbols at once,
it handles stock splits for us (NVDA, TSLA, AAPL, GOOGL and AMZN all split in this
period, and an unadjusted split looks like a −75% crash), and the same account can
later serve minute bars or a live feed if we ever want them.

### What the terms say

I read Alpaca's Terms and Conditions PDF and its docs. The relevant parts:

- Use is for "your own personal and non-commercial purposes". A resume project fits.
- Content may not be "copied, reproduced, republished, uploaded, posted, publicly
  displayed ... or distributed" to another server or website. Alpaca's support
  page says the same in one line: API data cannot be redistributed.
- Making the data available to other people through your own application needs
  30 days' written notice to Alpaca.

What that means for this repo (my reading, not legal advice):

1. **No raw bars are ever committed.** `data/` is in `.gitignore`. Anyone cloning
   the repo runs the download script with their own free key.
2. **The dashboard runs locally only.** It is not deployed to a public URL.
3. **The README publishes derived results only**: metrics, sample sizes, flagged
   dates, anomaly scores. No tables of raw prices.
4. **CI and tests use synthetic bars** made by a seeded random-walk generator, so
   they need no key and no real data.

### Things I could not fully confirm

- The Terms PDF I retrieved has a 2021 file date. Read the current version when
  you sign up; if it differs on the points above, tell me.
- **Which feed the free plan may use for history.** Alpaca's market data page
  lists Basic as: history since 2016, only the latest 15 minutes blocked. Its
  paper trading page says a paper-only account is "only entitled to receive and
  make use of IEX market data" (IEX is one exchange, about 2.5% of market volume).
  Plan: check the plan page inside your account at the start of milestone 2. If
  full-market (SIP) history is allowed, use it. If not, use `feed=iex`, where
  prices are still accurate for large stocks but volume is a small sample of the
  real total. Either way the README states which feed the numbers came from.

### What you need to do

Create a free Alpaca **paper-only** account (email sign-up), generate an API key
pair, and put it in a local `.env` file. I will give you `.env.example` in
milestone 1. I never need to see the key.

## 3. Bar size

### Recommendation: daily bars

| | Daily | 15-minute |
|---|---|---|
| Rows (30 stocks, 2016 → now) | ~82,000 *estimate* | ~2.1 million *estimate* |
| Training time | A few minutes *estimate* | Tens of minutes or more *estimate* |
| Extra work | None | Must remove the time-of-day pattern (volume and volatility are always high at the open and close) and handle overnight gaps |
| "What happened on this date?" check | Direct: look up the date | Indirect: timestamp → day → news |
| Fits a streaming pipeline | One record per stock per day; we replay history fast | More natural |
| Works on IEX-only volume | Yes, daily totals are large | Thin: few IEX trades per bar |

Why daily:

- Ten years covers several different market regimes (2018, the 2020 crash, 2022).
- The spec's sanity check is "list the top real dates so I can look them up".
  Daily bars make that a one-step lookup.
- Intraday bars have a strong time-of-day pattern. Without removing it, every
  detector flags the market open every day. Removing it is real extra work that
  adds nothing to what the project has to prove.
- Fast training means you can iterate on a training loop you wrote yourself.

Cost of choosing daily: the dataset is small for deep learning, so the model
must be small, and a stream of one bar per stock per day is less exciting than
an intraday stream. The replay producer hides the second point (it plays back
ten years in minutes).

## 4. Universe, splits, features

**Tickers (30, all listed before 2016 so every stock has full history):**
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, JPM, V, MA, UNH, JNJ, PG, XOM,
CVX, HD, KO, PEP, WMT, COST, DIS, NFLX, ADBE, CRM, INTC, CSCO, BAC, PFE, MRK.

Picking today's large companies is survivorship bias. That matters for predicting
returns; it matters little for "is this bar unusual". It goes in Limitations.

**Time split (never shuffled):**

| Split | Period | Trading days | Bars (×30) | Windows of 30 |
|---|---|---|---|---|
| Train | 2016-01 → 2022-12 | ~1,762 | ~52,900 | ~51,400 |
| Validation | 2023-01 → 2024-12 | ~502 | ~15,100 | ~14,200 |
| Test | 2025-01 → latest | ~469 | ~14,100 | ~13,200 |

All *estimates*; the milestone 2 script prints the real counts.

Rules that the milestone 2 tests will enforce:

- A window belongs to a split only if **every** bar in it is inside that split.
- Scaling statistics are computed from training-period rows only.
- Rolling features look backwards only, so a validation row may use earlier
  training bars as history. That is past data, not leakage.

**Features (three per bar, as in the spec):**

1. Log return, close to close.
2. Rolling volatility: standard deviation of the last 20 log returns.
3. Relative volume: log of today's volume over the median of the last 20 days.

**Scaling:** one mean and standard deviation per stock per feature, from the
training period. **One model for all 30 stocks**, because 1,700 windows per stock
is too few to train 30 separate models.

## 5. Model and training

**Window length: 30 trading days** (about six weeks). Long enough to hold the
20-day rolling features plus context, short enough to leave ~51k training windows.

**LSTM autoencoder, about 10,500 parameters:**

```
input  (30 bars × 3 features)
  → LSTM encoder, hidden size 32, 1 layer
  → linear layer to an 8-number summary (the bottleneck)
  → that summary repeated 30 times
  → LSTM decoder, hidden size 32, 1 layer
  → linear layer back to 3 features per bar
```

The bottleneck is the point: the model cannot memorise 90 numbers in 8, so it
learns the common patterns and reconstructs unusual windows badly.

**Training:** mean squared error, Adam, batch size 256, up to 50 epochs, early
stopping on validation loss, fixed seed, checkpoint of the best epoch.

**Time on your laptop (M3, 16 GB):** *estimate* under 5 minutes for a full run.
Before the full run I will time one epoch and tell you the real figure. The code
detects cuda, mps, or cpu as the spec asks, but for a model this small the Apple
GPU (mps) is often slower than the CPU, so I will time both and use the faster.

**One alternative if there is time:** a plain dense autoencoder on the flattened
window. It answers "does the LSTM's sense of order actually help?".

## 6. Scoring design

**Injection happens on raw bars, before features are built.** The injected price
and volume then flow through the same feature code as real data, so rolling
volatility reacts the way it would to a real event.

| Kind | What is planted | Sizes |
|---|---|---|
| Spike | Close jumps for one bar, then returns | 2 to 8 × the stock's recent daily volatility |
| Level shift | Price jumps and stays there | 2 to 8 × recent volatility |
| Volatility burst | Returns scaled up for 5 to 10 bars | ×2 to ×5 |
| Volume surge | Volume multiplied for 1 to 3 bars, price untouched | ×3 to ×10 |

Note for when you write it: in return terms a level shift is **one** outlier
return, and a spike is **two** opposite ones. They differ only by the reversal.

**Counting:**

- An event is *caught* if a detector flags any bar from its start to a few bars
  after its end. Consecutive flagged bars count as one alarm.
- A *false alarm* is an alarm outside every planted event.
- **Thresholds are set on the untouched validation period** so that each detector
  raises one alarm per stock per month there. The test set is never used to pick
  a threshold. I will also report one alarm per stock per quarter, because one
  per month on daily bars is a loose threshold (about 1 bar in 21).
- Target sample: ~180 planted events per pass, 10 seeded passes, so roughly 450
  per kind *estimate*. Every percentage is reported with its count.

**Known weakness to state in the README:** the test period contains real unusual
days. A detector that correctly flags one is charged a false alarm.

## 7. Detectors and fairness

| # | Detector | Sees |
|---|---|---|
| 1 | Rolling z-score, 20 bars, flag when \|z\| > 3 | Returns only, like the original price rule |
| 2 | Median / MAD version of the same | All three features, flag on the worst one |
| 3 | Isolation forest (scikit-learn) | All three features |
| 4 | LSTM autoencoder | Windows of all three features |

Detector 1 looks at price only, so it cannot catch a volume surge by design.
Beating it there proves little. The honest comparison for the model is against
detectors 2 and 3, which see the same features. The results table and README
will say so.

**Anomaly score for the model: decision needed (see §10).** The spec says "how
badly it reconstructs the window". If the score is the error over the whole
window, one bad bar stays inside 30 consecutive windows and the flag stays lit
for six weeks. Scoring the error on the **newest bar only** points at the day it
happened and is what a live consumer needs.

## 8. Pipeline (milestone 6)

```
replay producer ──► Redpanda topic "bars" (keyed by symbol)
                        ├─► z-score consumer   (consumer group 1) ─┐
                        └─► model consumer     (consumer group 2) ─┼─► PostgreSQL ─► Streamlit
```

- The producer replays saved bars from `data/raw/` at a chosen speed.
- Keying by symbol keeps each stock's bars in order.
- Two consumer groups each receive every bar independently.
- Both write to one `flags` table with a `detector` column. The primary key is
  (symbol, time, detector), so replaying the same data twice writes nothing new.
- The model consumer loads one saved file holding the weights, the scaling
  statistics, and the threshold. It needs 49 bars per stock before its first score.
- A live mode that polls Alpaca is **not** planned. Replay is enough to
  demonstrate and test the service.

## 9. Repo and tools

```
src/anomaly/   download, features, split, dataset*, inject*, detectors, model*, train*, evaluate
src/service/   producer, consumers, db
dashboard/     Streamlit app
scripts/       one script per verify step
tests/
docker-compose.yml   Makefile   .github/workflows/ci.yml
SPEC.md  PLAN.md  DECISIONS.md  CLAUDE.md  README.md
```
`*` = "my turn" pieces: I give the signature and tests, you write the body.

| Tool | For | Alternative I passed on |
|---|---|---|
| `uv` (already installed) | Python environment with a lockfile | venv + requirements.txt: no exact pins |
| `requests` | Calling Alpaca's REST endpoint | Alpaca's SDK: hides the HTTP you should be able to explain |
| CSV files | Storing bars | Parquet: needs another dependency for 82k rows |
| `confluent-kafka` | Talking to Redpanda | kafka-python: less maintained |
| `psycopg` | Talking to PostgreSQL | SQLAlchemy: an ORM for two tables is overkill |
| `ruff` | Lint and format | flake8 + black: two tools instead of one |
| `make` | The one-command rerun (`make all`) | A shell script: no per-step targets |

Python 3.13 (what your laptop has). Docker 29 and the GitHub CLI are installed
and you are logged in to GitHub, so milestone 1 can create the repo and CI.

## 10. Decisions I need from you

1. **Data source:** Alpaca free plan (recommended) or Tiingo?
2. **Bar size:** daily (recommended) or 15-minute?
3. **The original z-score rule:** do you remember its window, threshold, or
   whether it ran on price or on returns? If not, I use 20 bars on log returns,
   flag at |z| > 3.
4. **Model score:** error on the newest bar (recommended) or on the whole window
   as the spec literally says?
5. **Credit:** what was the team project called, and for what course or event?
   Should teammates be named in the README?
6. **Repo:** public or private on GitHub, and what name? (The folder is spelled
   "Anomoly"; I'd name the repo `anomaly-pipeline`.)
7. **Ticker list** in §4: OK, or swap any?

## 11. Risks

- **The model may lose.** On spikes the z-score rule will probably win. The
  model's best chance is volatility bursts and volume surges. Whatever happens
  gets reported per kind.
- **Regime change.** If 2025–26 is more volatile than 2016–22, reconstruction
  error rises across the whole test period and false alarms go up.
- **Planted anomalies are not real ones.** Good numbers here show the detectors
  find the shapes we planted, nothing more.
- **Small data.** ~51k training windows is small. That is why the model is ~10k
  parameters.

## 12. Next step

After you answer §10, I record the choices in DECISIONS.md and send the short
plan for milestone 1 (skeleton, one passing test, CI) for your OK.

## Sources

- Alpaca market data plans: https://docs.alpaca.markets/docs/about-market-data-api
- Alpaca bars endpoint: https://docs.alpaca.markets/reference/stockbars
- Alpaca paper trading: https://docs.alpaca.markets/docs/paper-trading
- Alpaca Terms and Conditions: https://files.alpaca.markets/disclosures/library/TermsAndConditions.pdf
- Alpaca on redistribution: https://alpaca.markets/support/redistribute-alpaca-api
- Tiingo plans: https://api.tiingo.com/products/iex-api
