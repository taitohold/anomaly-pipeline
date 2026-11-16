"""Replay saved daily bars into Redpanda, one trading day at a time.

Run with: python -m anomaly.service.producer --file data/raw/bars.csv
"""

import argparse
import time

import pandas as pd
from confluent_kafka import Producer

from anomaly.service.kafka import TOPIC, bootstrap, ensure_topic
from anomaly.service.online import encode_bar


def replay(bars: pd.DataFrame, delay: float) -> int:
    """Send every bar, day by day, keyed by stock. Returns the number sent."""
    ensure_topic()
    producer = Producer({"bootstrap.servers": bootstrap(), "enable.idempotence": True})
    sent = 0
    for _, day in bars.sort_values(["date", "symbol"]).groupby("date", sort=True):
        for bar in day.to_dict("records"):
            producer.produce(TOPIC, key=bar["symbol"], value=encode_bar(bar))
            sent += 1
        producer.poll(0)  # serve delivery callbacks
        if delay:
            time.sleep(delay)
    producer.flush()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/raw/bars.csv")
    parser.add_argument("--start", default=None, help="first date to send (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between trading days")
    args = parser.parse_args()

    bars = pd.read_csv(args.file, parse_dates=["date"])
    if args.start:
        bars = bars[bars["date"] >= pd.Timestamp(args.start)]
    started = time.time()
    sent = replay(bars, args.delay)
    days = bars["date"].nunique()
    print(
        f"sent {sent:,} bars over {days} trading days in {time.time() - started:.0f}s", flush=True
    )


if __name__ == "__main__":
    main()
