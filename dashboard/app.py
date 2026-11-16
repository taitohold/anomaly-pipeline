"""Streamlit dashboard: the z-score rule and the autoencoder, side by side.

Reads the bars and scores tables the consumers write. Local only: the data
comes from Alpaca, whose terms do not allow publishing it.
"""

import os

import altair as alt
import pandas as pd
import psycopg
import streamlit as st

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://anomaly:anomaly@localhost:55432/anomaly"
)
COLORS = alt.Scale(domain=["z-score", "autoencoder"], range=["#d62728", "#1f77b4"])

st.set_page_config(page_title="Anomaly flags", layout="wide")
st.title("Unusual price and volume moves")
st.caption(
    "The original z-score rule (|z| > 3) next to the LSTM autoencoder, set to flag "
    "about as often on validation data. Flags mark unusual movement only; they are "
    "not trading signals."
)


@st.cache_data(ttl=5)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


if st.button("Refresh"):
    st.cache_data.clear()

counts = query(
    """
    SELECT detector, count(*) AS days_scored, count(*) FILTER (WHERE flagged) AS flags,
           max(date) AS latest_day
    FROM scores GROUP BY detector ORDER BY detector DESC
    """
)
if counts.empty:
    st.info("No scores yet. Start the replay with `make replay` and refresh in a few seconds.")
    st.stop()

agree = query(
    """
    SELECT count(*) FILTER (WHERE z.flagged AND m.flagged) AS both,
           count(*) FILTER (WHERE z.flagged AND NOT m.flagged) AS only_z,
           count(*) FILTER (WHERE m.flagged AND NOT z.flagged) AS only_model
    FROM scores z JOIN scores m USING (symbol, date)
    WHERE z.detector = 'z-score' AND m.detector = 'autoencoder'
    """
).iloc[0]

cols = st.columns(len(counts) + 3)
for col, row in zip(cols, counts.itertuples()):
    col.metric(
        f"{row.detector} flags", f"{row.flags:,}", help=f"out of {row.days_scored:,} days scored"
    )
cols[-3].metric("Flagged by both", int(agree["both"]))
cols[-2].metric("Only z-score", int(agree["only_z"]))
cols[-1].metric("Only autoencoder", int(agree["only_model"]))

symbols = query("SELECT DISTINCT symbol FROM bars ORDER BY symbol")["symbol"].tolist()
symbol = st.selectbox("Stock", symbols)

bars = query("SELECT date, close, volume FROM bars WHERE symbol = %s ORDER BY date", (symbol,))
flags = query(
    """
    SELECT s.date, s.detector, s.score, s.threshold, b.close
    FROM scores s JOIN bars b USING (symbol, date)
    WHERE s.symbol = %s AND s.flagged ORDER BY s.date
    """,
    (symbol,),
)

price = (
    alt.Chart(bars)
    .mark_line(color="#999")
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False)),
    )
)
marks = (
    alt.Chart(flags)
    .mark_point(size=90, filled=True, opacity=0.85)
    .encode(
        x="date:T",
        y="close:Q",
        color=alt.Color("detector:N", scale=COLORS, title="Flagged by"),
        shape=alt.Shape("detector:N", title="Flagged by"),
        tooltip=[
            "date:T",
            "detector:N",
            alt.Tooltip("score:Q", format=".2f"),
            alt.Tooltip("threshold:Q", format=".2f"),
        ],
    )
)
st.altair_chart((price + marks).properties(height=380), use_container_width=True)

st.subheader("Latest flags, all stocks")
recent = query(
    """
    WITH moves AS (
        SELECT symbol, date,
               100 * (close / lag(close) OVER (PARTITION BY symbol ORDER BY date) - 1) AS move
        FROM bars
    )
    SELECT s.date, s.symbol, s.detector, round(s.score::numeric, 2) AS score,
           round(m.move::numeric, 1) AS move_pct
    FROM scores s JOIN moves m USING (symbol, date)
    WHERE s.flagged
    ORDER BY s.date DESC, s.symbol LIMIT 50
    """
)
# Renamed here, not in SQL: psycopg reads "%" in a query as a parameter placeholder.
st.dataframe(
    recent.rename(columns={"move_pct": "move %"}), use_container_width=True, hide_index=True
)
