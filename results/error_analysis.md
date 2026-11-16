# Error analysis

Planted-anomaly numbers use the same 1,800 events as `results/comparison.md` (10 passes). The cross-check that recall at the validation thresholds matches that file passed.

## 1. Recall at equal false-alarm rates on test

Each detector's threshold is set so it raises the stated number of alarms per stock per month on the **untouched** test period. Every detector then gets the same alarm budget on the data it is scored on. This uses test data, so it is analysis only; the headline table sets thresholds on validation.

| Alarms per stock per month | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| 0.25 | 881/1800 (49%) | 638/1800 (35%) | 517/1800 (29%) | 1086/1800 (60%) |
| 0.5 | 1108/1800 (62%) | 1078/1800 (60%) | 1137/1800 (63%) | 1306/1800 (73%) |
| 1 | 1268/1800 (70%) | 1408/1800 (78%) | 1486/1800 (83%) | 1496/1800 (83%) |
| 2 | 1445/1800 (80%) | 1703/1800 (95%) | 1706/1800 (95%) | 1663/1800 (92%) |

Achieved rates on untouched test are within 0.00 of each target.

By kind, at 0.5 alarms per stock per month:

| Kind | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| spike | 371/450 (82%) | 277/450 (62%) | 277/450 (62%) | 343/450 (76%) |
| level_shift | 324/450 (72%) | 225/450 (50%) | 232/450 (52%) | 288/450 (64%) |
| vol_burst | 371/450 (82%) | 305/450 (68%) | 313/450 (70%) | 362/450 (80%) |
| volume_surge | 42/450 (9%) | 271/450 (60%) | 315/450 (70%) | 313/450 (70%) |

## 2. Chance floor

Alarms raised on the **untouched** test period, scored against the planted events' windows. Nothing was planted in that data, so any catch here is coincidence. Real recall should be judged against this floor.

Threshold 1 per stock per month (set on validation): chance / real recall

| Kind | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| spike | 16% / 91% | 24% / 80% | 26% / 87% | 24% / 89% |
| level_shift | 12% / 82% | 22% / 70% | 21% / 80% | 18% / 83% |
| vol_burst | 39% / 92% | 44% / 83% | 48% / 94% | 48% / 95% |
| volume_surge | 18% / 18% | 30% / 84% | 28% / 96% | 30% / 92% |

Threshold 1 per stock per quarter (set on validation): chance / real recall

| Kind | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| spike | 6% / 73% | 6% / 49% | 10% / 66% | 11% / 79% |
| level_shift | 5% / 65% | 5% / 39% | 6% / 56% | 6% / 68% |
| vol_burst | 16% / 73% | 16% / 57% | 17% / 74% | 21% / 85% |
| volume_surge | 7% / 7% | 8% / 39% | 12% / 79% | 14% / 77% |

## 3. Who catches what

At the quarterly validation threshold. Each event is counted once: caught by all four, by exactly one, by some (two or three), or by none.

| Kind | events | all | only z-score | only robust | only isolation forest | only autoencoder | some | none |
|---|---|---|---|---|---|---|---|---|
| level_shift | 450 | 147 | 26 | 2 | 0 | 16 | 150 | 109 |
| spike | 450 | 184 | 22 | 5 | 2 | 16 | 162 | 59 |
| vol_burst | 450 | 216 | 10 | 1 | 0 | 21 | 154 | 48 |
| volume_surge | 450 | 21 | 0 | 8 | 32 | 26 | 307 | 56 |

## 4. Smallest size caught reliably

Smallest size from which a detector catches at least 90% of events at that size and every larger one. Spike and level-shift sizes are in units of the stock's recent daily volatility (σ); burst and surge sizes are multipliers. A dash means no size reaches 90%.

Threshold 1 per stock per month:

| Kind | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| spike | 3σ | 6σ | 6σ | 3σ |
| level_shift | 4σ | 6σ | 6σ | 4σ |
| vol_burst | ×3 | ×5 | ×3 | ×3 |
| volume_surge | — | ×10 | ×5 | ×5 |

Threshold 1 per stock per quarter:

| Kind | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| spike | 6σ | 8σ | 6σ | 6σ |
| level_shift | 6σ | — | 8σ | 6σ |
| vol_burst | ×5 | — | ×5 | ×3 |
| volume_surge | — | — | ×10 | ×10 |

## 5. Alarms on the untouched test period

Quarterly validation threshold, real 2025-2026 data with nothing planted. Some of these are real events, some are false alarms; there is no way to tell them apart without labels. "Also flagged" means another detector raised an alarm in the same stock within one trading day.

| Detector | Alarms | Per stock per month | On earnings days | Also flagged by z-score | Also flagged by another detector |
|---|---|---|---|---|---|
| z-score | 226 | 0.34 | 34% | — | 83% |
| robust | 209 | 0.31 | 24% | 38% | 53% |
| isolation forest | 379 | 0.57 | 26% | 41% | 81% |
| autoencoder | 443 | 0.66 | 12% | 27% | 72% |

Alarms by month (month the alarm started):

| Month | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| 2025-01 | 14 | 8 | 10 | 14 |
| 2025-02 | 12 | 9 | 13 | 14 |
| 2025-03 | 9 | 1 | 16 | 16 |
| 2025-04 | 31 | 20 | 67 | 61 |
| 2025-05 | 6 | 22 | 15 | 20 |
| 2025-06 | 7 | 9 | 12 | 15 |
| 2025-07 | 10 | 11 | 13 | 13 |
| 2025-08 | 8 | 5 | 15 | 9 |
| 2025-09 | 8 | 7 | 10 | 10 |
| 2025-10 | 11 | 13 | 11 | 9 |
| 2025-11 | 10 | 10 | 14 | 12 |
| 2025-12 | 5 | 7 | 11 | 27 |
| 2026-01 | 19 | 11 | 13 | 22 |
| 2026-02 | 11 | 7 | 11 | 21 |
| 2026-03 | 5 | 12 | 11 | 16 |
| 2026-04 | 14 | 7 | 17 | 24 |
| 2026-05 | 6 | 10 | 10 | 19 |
| 2026-06 | 5 | 5 | 27 | 32 |
| 2026-07 | 8 | 8 | 21 | 25 |
| 2026-08 | 5 | 10 | 12 | 12 |
| 2026-09 | 6 | 10 | 20 | 14 |
| 2026-10 | 7 | 4 | 14 | 19 |
| 2026-11 | 9 | 3 | 16 | 19 |

Busiest months, all detectors together: 2025-04 (179), 2026-06 (69), 2026-01 (65). Look these up to see what moved the market.

Stocks with the most alarms:

| Rank | z-score | robust | isolation forest | autoencoder |
|---|---|---|---|---|
| 1 | JPM (11) | CSCO (12) | INTC (60) | INTC (83) |
| 2 | AAPL (10) | UNH (12) | MRK (27) | MRK (31) |
| 3 | CSCO (10) | META (11) | CSCO (24) | CSCO (24) |
| 4 | INTC (10) | MSFT (10) | UNH (21) | PEP (24) |
| 5 | MA (10) | AMZN (9) | META (17) | UNH (24) |

## 6. Real days the model flags most strongly

The autoencoder's 20 highest-scoring alarms on the untouched test period (quarterly validation threshold), one row per alarm at its highest-scoring day. "Move" is that day's close-to-close change. The last column is for you: look each one up.

| Date | Stock | Move | Volume vs 20-day median | Earnings window | Z-score too | Forest too | What happened |
|---|---|---|---|---|---|---|---|
| 2026-08-19 | MRK | +12.6% | ×4.2 |  | yes | yes | |
| 2025-09-18 | INTC | +22.8% | ×8.6 |  | yes | yes | |
| 2026-04-24 | INTC | +23.6% | ×2.7 | yes | yes | yes | |
| 2026-10-30 | INTC | -28.6% | ×4.9 | yes | yes | yes | |
| 2025-04-09 | AAPL | +15.3% | ×3.6 |  | yes | yes | |
| 2025-05-13 | UNH | -17.8% | ×7.0 |  |  | yes | |
| 2026-07-30 | MSFT | +15.5% | ×3.7 | yes | yes | yes | |
| 2025-04-09 | INTC | +18.8% | ×2.1 |  |  | yes | |
| 2025-08-15 | UNH | +12.0% | ×4.2 |  | yes | yes | |
| 2025-04-17 | UNH | -22.4% | ×5.9 | yes | yes | yes | |
| 2025-04-09 | DIS | +11.9% | ×2.9 |  | yes | yes | |
| 2025-05-07 | DIS | +10.8% | ×3.5 | yes |  | yes | |
| 2026-08-27 | CRM | +22.6% | ×4.9 | yes | yes | yes | |
| 2025-04-09 | WMT | +9.6% | ×1.7 |  | yes | yes | |
| 2026-11-05 | INTC | -10.3% | ×2.6 |  |  | yes | |
| 2026-05-08 | INTC | +14.0% | ×1.9 |  |  | yes | |
| 2026-01-23 | INTC | -17.0% | ×3.0 | yes | yes | yes | |
| 2026-04-29 | INTC | +12.1% | ×2.0 |  |  | yes | |
| 2026-06-08 | INTC | +11.2% | ×1.1 |  |  | yes | |
| 2025-04-09 | AMZN | +12.0% | ×2.7 |  | yes | yes | |

Generated by `scripts/error_analysis.py`.
