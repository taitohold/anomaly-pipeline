# SPEC: Learned anomaly detector for my market data pipeline

Save this file as SPEC.md in the project folder, next to CLAUDE.md.

## What this is

My existing pipeline flags unusual stock price movements with a z-score rule. This project trains a PyTorch model that learns what normal price and volume movement looks like and flags what doesn't fit. It then compares the model against the z-score rule on the same test data, and runs the model inside the pipeline next to the old rule.

## Why I'm building it

I'm applying for software engineering internships. My resume has no deep learning code I wrote myself. This project has to show that I can write and train a PyTorch model by hand, measure it honestly against a simple rule, and ship it as a service inside a real pipeline, with tests, CI, and Docker.

## The existing pipeline

REST API, then Redpanda (Kafka), then a z-score detector, then PostgreSQL, then a Streamlit dashboard. I built it with a team, so:

* Do this work in my own new repo. Credit the original team project in the README and say clearly which parts are new.
* Ask me for the pipeline code and which market data API it used before planning.

## Data

* Historical price and volume bars for about 30 large US stocks. In milestone 0, choose a source whose terms allow this use, and tell me the bar size (daily or minute) you recommend and why.
* Split by time: train on the earlier period, validate on the next, test on the most recent. Never shuffle across time.
* Fit all scaling on the training period only.

## The hard part: scoring

Real anomalies have no labels, so the test needs planted ones.

* Write a function that injects anomalies into held-out data. Include at least four kinds: a single-bar spike, a level shift, a burst of volatility, and a volume surge with no price move. Vary the sizes.
* Score every detector on the same injected test set: precision, recall, and recall at a fixed false-alarm rate (for example one false alarm per stock per month).
* As a sanity check, list the top real dates the model flags on untouched data, so I can look up what happened on them.
* The README must say the main numbers come from planted anomalies.
* The model may not beat the z-score rule on simple spikes. Report what happens for each anomaly kind, including the ones where the rule wins.

## Detectors to compare

1. The existing z-score rule.
2. A robust version using the median and median absolute deviation.
3. Isolation forest from scikit-learn.
4. My PyTorch model: an autoencoder over sliding windows of returns, rolling volatility, and volume. The anomaly score is how badly it reconstructs the window. Start with an LSTM autoencoder. Try one alternative only if there is time.

## Pieces I write myself

Give me the function signature and tests for each of these, and let me write the body. Review it after.

* The sliding-window Dataset class
* The autoencoder model class
* The training loop
* The anomaly injection function

## Stack

* Python, PyTorch, pandas, NumPy, scikit-learn.
* Redpanda, PostgreSQL, and Streamlit, matching the existing pipeline.
* Docker Compose to run everything locally. No paid cloud services.
* pytest, GitHub Actions.
* Training must run on my laptop. Detect the device (cuda, mps, or cpu). Tell me the expected training time before any long run.

## Milestones

Each one ends with something I can run and see.

0. Plan only, no code. Review the existing pipeline, confirm the data source and its terms, and propose bar size, window length, and model size. Write PLAN.md.
1. Skeleton. Repo layout, one passing test, CI running on push.
2. Data. A download script into data/raw/, feature building into data/processed/, the time split, and the window Dataset. Verify: tests that no test-period data reaches training or scaling, and a script that prints row counts for each split.
3. Injection and baselines. The injection function, then detectors 1 to 3 scored on the injected test set. Verify: one script prints the comparison table per anomaly kind, with counts, and saves it.
4. PyTorch model. The autoencoder, a training loop with a fixed random seed, validation after each epoch, early stopping, and saved checkpoints. Verify: it can overfit a tiny subset first. Then train fully and add the model to the same comparison table.
5. Error analysis. Which anomaly kinds and sizes each detector misses, the false alarms, and the top real dates the model flags.
6. Service. A consumer that reads records from Redpanda, keeps a rolling window per stock, scores it with the saved model, and writes flags to PostgreSQL next to the z-score flags. The dashboard shows both. Verify: an integration test with Docker Compose that replays saved data through the pipeline and checks the flags written.
7. Document. README with setup steps, an architecture diagram, the results table, the error analysis, and limitations.

## Reproducibility

One command reruns download, training, and evaluation and produces the numbers in the README.

## Out of scope

* Predicting prices or giving trading signals. This detects unusual movement and nothing more.
* Paid data or paid cloud hosting.
* Transformers or models that need a rented GPU.
* Changing the team's original repo.

## Kickoff prompt

Read CLAUDE.md and SPEC.md. Do milestone 0 only: ask me for the existing pipeline code and the data API it used, check the data source's terms, propose bar size, window length, and model size for my laptop, list the decisions you need from me, and write PLAN.md. Do not write any code yet.
