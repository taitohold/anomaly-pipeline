"""Kafka (Redpanda) settings shared by the producer and the consumers."""

import os
import time

from confluent_kafka.admin import AdminClient, NewTopic

TOPIC = "bars"
PARTITIONS = 3  # bars are keyed by stock, so each stock stays in order within its partition


def bootstrap() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")


def ensure_topic(timeout: float = 60.0) -> None:
    """Create the topic if it does not exist, waiting for the broker to come up."""
    admin = AdminClient({"bootstrap.servers": bootstrap()})
    deadline = time.time() + timeout
    while True:
        try:
            if TOPIC in admin.list_topics(timeout=5).topics:
                return
            futures = admin.create_topics([NewTopic(TOPIC, PARTITIONS, 1)])
            futures[TOPIC].result()
            return
        except Exception as error:  # broker not ready yet, or topic created by someone else
            if "TOPIC_ALREADY_EXISTS" in str(error):
                return
            if time.time() > deadline:
                raise
            time.sleep(1)
