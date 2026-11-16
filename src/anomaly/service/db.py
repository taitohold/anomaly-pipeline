"""PostgreSQL writes for the consumers. Every write is an upsert keyed on the
table's primary key, so processing the same bar twice changes nothing."""

import os

import psycopg


def connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL", "postgresql://anomaly:anomaly@localhost:55432/anomaly")
    return psycopg.connect(url)


def write_bars(conn: psycopg.Connection, bars: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO bars (symbol, date, close, volume)
            VALUES (%(symbol)s, %(date)s, %(close)s, %(volume)s)
            ON CONFLICT (symbol, date) DO UPDATE
                SET close = EXCLUDED.close, volume = EXCLUDED.volume
            """,
            bars,
        )


def write_scores(conn: psycopg.Connection, scores: list[dict]) -> None:
    """Each dict has symbol, date, detector, score, threshold, flagged."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO scores (symbol, date, detector, score, threshold, flagged)
            VALUES (%(symbol)s, %(date)s, %(detector)s, %(score)s, %(threshold)s, %(flagged)s)
            ON CONFLICT (symbol, date, detector) DO UPDATE
                SET score = EXCLUDED.score,
                    threshold = EXCLUDED.threshold,
                    flagged = EXCLUDED.flagged
            """,
            scores,
        )
