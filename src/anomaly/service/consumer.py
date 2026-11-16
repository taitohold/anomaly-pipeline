"""Read bars from Redpanda, score each day, and write scores to PostgreSQL.

Run with: python -m anomaly.service.consumer --detector z-score
          python -m anomaly.service.consumer --detector autoencoder

Each detector is its own consumer group, so each one receives every bar.

Delivery: a batch's database write is committed before its Kafka offsets, so
a crash can only cause bars to be processed again, never skipped. Writes are
upserts, so processing a bar twice changes nothing.

Restarts: the rolling windows live in memory, so on startup a consumer reads
the topic from the beginning to rebuild them. That is simple and correct for a
replay demo. A long-running system would store its windows instead.
"""

import argparse
import json
import time
from pathlib import Path

from confluent_kafka import OFFSET_BEGINNING, Consumer

from anomaly.detectors import AutoencoderDetector, zscore_scores
from anomaly.service.db import connect, write_bars, write_scores
from anomaly.service.kafka import TOPIC, bootstrap, ensure_topic
from anomaly.service.online import OnlineScorer, decode_bar
from anomaly.train import load_checkpoint

THRESHOLDS = Path("checkpoints/thresholds.json")
CHECKPOINT = Path("checkpoints/autoencoder.pt")
BATCH = 500


def build_scorer(detector: str) -> OnlineScorer:
    if detector == "z-score":
        return OnlineScorer(zscore_scores)
    return OnlineScorer(AutoencoderDetector(*load_checkpoint(CHECKPOINT)).score)


def from_the_beginning(consumer, partitions):
    for p in partitions:
        p.offset = OFFSET_BEGINNING
    consumer.assign(partitions)


def run(detector: str) -> None:
    threshold = json.loads(THRESHOLDS.read_text())[detector]["threshold"]
    scorer = build_scorer(detector)
    store_bars = detector == "z-score"
    ensure_topic()
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap(),
            "group.id": f"detector-{detector}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([TOPIC], on_assign=from_the_beginning)
    conn = connect()
    print(f"{detector} consumer started, threshold {threshold:.4f}", flush=True)

    processed, started = 0, time.time()
    while True:
        messages = consumer.consume(num_messages=BATCH, timeout=1.0)
        if not messages:
            continue
        bars, scores = [], []
        for message in messages:
            if message.error():
                raise RuntimeError(message.error())
            bar = decode_bar(message.value())
            score = scorer.add(bar)
            bars.append(bar)
            if score is not None:
                scores.append(
                    {
                        "symbol": bar["symbol"],
                        "date": bar["date"].date(),
                        "detector": detector,
                        "score": score,
                        "threshold": threshold,
                        "flagged": score > threshold,
                    }
                )
        if store_bars:
            write_bars(conn, [{**b, "date": b["date"].date()} for b in bars])
        write_scores(conn, scores)
        conn.commit()  # database first ...
        consumer.commit(asynchronous=False)  # ... then offsets: at-least-once
        processed += len(messages)
        print(
            f"{detector}: {processed:,} bars, {len(scores)} scores in last batch "
            f"({processed / (time.time() - started):.0f} bars/s)",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, choices=["z-score", "autoencoder"])
    run(parser.parse_args().detector)


if __name__ == "__main__":
    main()
