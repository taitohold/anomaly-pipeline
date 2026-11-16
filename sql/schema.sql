-- Runs once, when the Postgres container first starts with an empty volume.

-- Daily bars, written by the z-score consumer (the original pipeline's storage role).
CREATE TABLE IF NOT EXISTS bars (
    symbol  text             NOT NULL,
    date    date             NOT NULL,
    close   double precision NOT NULL,
    volume  double precision NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- One row per stock, day and detector. The primary key makes every write an
-- upsert, so replaying the same bars never adds rows.
CREATE TABLE IF NOT EXISTS scores (
    symbol     text             NOT NULL,
    date       date             NOT NULL,
    detector   text             NOT NULL,
    score      double precision NOT NULL,
    threshold  double precision NOT NULL,
    flagged    boolean          NOT NULL,
    scored_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, date, detector)
);

CREATE INDEX IF NOT EXISTS scores_flagged ON scores (detector, date) WHERE flagged;
