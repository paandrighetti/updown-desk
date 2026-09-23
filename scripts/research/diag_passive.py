"""Diagnostics for the passive quoting replay: are the v0.2 losses an artifact of the replay?

Read-only, derived tables and one raw hour per end of the recording. Runs inside the
reporter image, where DuckDB is installed and the data sits under /data:

  docker compose run --rm -v /opt/updown-desk/scripts:/scripts reporter \
      python /scripts/diag_passive.py

0. Event mix of one raw CLOB hour on the first and last recorded day, and one recorded
   best_bid_ask message: can the top of book be followed between two trades?
1. Side convention of last_trade_price. The documentation does not say whether `side` is
   the taker's or the maker's. Taker BUY prints should sit above the prior mid.
2. Book snapshot cadence: share of `book` snapshots arriving within 250 ms of a print on
   the same market. Near 1 means snapshots are trade-driven, so the v0.2 quotes were only
   recomputed after trades.
3. Quote age at print time: how old the latest snapshot (at least 250 ms older, the base
   latency) is when an Up print arrives, and where the print sits against its best price.
4. RTDS cadence and relay delay over the last derived day, per topic and symbol.
5. Markout of passive fills by time to expiry, copied from the latest report (H2).
"""

from __future__ import annotations

import glob
import os
import sys

import duckdb

ROOT = os.environ.get("UPDOWN_DATA_DIR", "/data")
REPORT = os.path.join(os.environ.get("UPDOWN_REPORTS_DIR", "/reports"), "latest.md")
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
LATENCY_MS = 250

RAW = "read_json({path!r}, columns={{'rx_ts':'BIGINT','msg':'JSON'}}, format='newline_delimited')"

con = duckdb.connect()
con.execute("SET memory_limit='1500MB'")
con.execute("SET temp_directory='/tmp/duckdb_spill'")


def show(title: str, sql: str, note: str = "") -> None:
    print(f"\n== {title}")
    if note:
        print(note)
    try:
        print(con.sql(sql).df().to_string(index=False))
    except duckdb.Error as exc:
        print(f"failed: {exc}")


def view(name: str, table: str, days: int, cols: str = "*") -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "derived", table, "*.parquet")))[-days:]
    if not files:
        raise SystemExit(f"no derived {table} files under {ROOT}")
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT {cols} "
        f"FROM read_parquet({files!r}, union_by_name=true)"
    )


def raw_event_mix() -> None:
    clob = os.path.join(ROOT, "raw", "clob")
    files = sorted(glob.glob(os.path.join(clob, "*.jsonl.gz"))) or sorted(
        glob.glob(os.path.join(clob, "*.jsonl"))
    )
    noon = [f for f in files if os.path.basename(f)[9:11] == "12"] or files
    picks = sorted({noon[0], noon[-1]}) if noon else []
    if not picks:
        print("\n== 0. no raw CLOB file found")
        return
    for path in picks:
        show(
            f"0. raw CLOB event mix, {os.path.basename(path)}",
            f"SELECT json_extract_string(msg, '$.event_type') AS event_type, count(*) AS n "
            f"FROM {RAW.format(path=path)} GROUP BY 1 ORDER BY 2 DESC",
        )
    show(
        f"0b. one best_bid_ask message as recorded, {os.path.basename(picks[-1])}",
        f"SELECT CAST(msg AS VARCHAR) AS msg FROM {RAW.format(path=picks[-1])} "
        f"WHERE json_extract_string(msg, '$.event_type') = 'best_bid_ask' LIMIT 1",
    )


def build_tables() -> None:
    view("books", "books", DAYS, "rx_ts, token, bid, ask")
    view("trades", "trades", DAYS, "rx_ts, token, price, side")
    view("windows_raw", "windows", DAYS + 1)
    con.execute(
        """
        CREATE OR REPLACE TABLE tok AS
        WITH w AS (
            SELECT * FROM windows_raw
            QUALIFY row_number() OVER (PARTITION BY slug ORDER BY rx_ts DESC) = 1
        )
        SELECT up_token AS token, 'up' AS leg, condition_id,
               start * 1000 AS t0, "end" * 1000 AS t1 FROM w
        UNION ALL
        SELECT down_token, 'down', condition_id, start * 1000, "end" * 1000 FROM w
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE p AS
        SELECT t.rx_ts, t.rx_ts - {LATENCY_MS} AS rx_prior, t.token, t.price, t.side,
               k.leg, k.condition_id
        FROM trades t JOIN tok k USING (token)
        WHERE t.rx_ts BETWEEN k.t0 AND k.t1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE s AS
        SELECT b.rx_ts, b.token, b.bid, b.ask, k.leg, k.condition_id
        FROM books b JOIN tok k USING (token)
        WHERE b.rx_ts BETWEEN k.t0 AND k.t1
        """
    )


def side_convention() -> None:
    show(
        f"1. side of last_trade_price against the prior mid (prior = at least {LATENCY_MS} ms "
        "older)",
        """
        WITH j AS (
            SELECT p.side, round(p.price - (s.bid + s.ask) / 2, 4) AS d
            FROM p ASOF JOIN s ON p.token = s.token AND p.rx_prior >= s.rx_ts
            WHERE s.bid IS NOT NULL AND s.ask IS NOT NULL
        )
        SELECT side, count(*) AS prints,
               round(avg((d > 0)::INT), 3) AS above_mid,
               round(avg((d = 0)::INT), 3) AS at_mid,
               round(avg((d < 0)::INT), 3) AS below_mid
        FROM j GROUP BY side ORDER BY side
        """,
        "Taker convention: BUY mostly above_mid and SELL mostly below_mid. The reverse "
        "means `side` is the maker's side and the v0.2 fill rule is inverted.",
    )


def snapshot_cadence() -> None:
    show(
        "2. prints and book snapshots per window, inside the window",
        """
        SELECT 'prints' AS kind, leg, count(*) AS n,
               round(count(*) * 1.0 / count(DISTINCT condition_id), 1) AS per_window
        FROM p GROUP BY leg
        UNION ALL
        SELECT 'snapshots', leg, count(*),
               round(count(*) * 1.0 / count(DISTINCT condition_id), 1)
        FROM s GROUP BY leg
        ORDER BY kind, leg
        """,
    )
    show(
        "2b. lag from the latest print on the same market to each book snapshot",
        """
        WITH pr AS (SELECT rx_ts, condition_id FROM p),
        j AS (
            SELECT s.leg, s.rx_ts - pr.rx_ts AS lag_ms
            FROM s ASOF LEFT JOIN pr
              ON s.condition_id = pr.condition_id AND s.rx_ts >= pr.rx_ts
        )
        SELECT leg, count(*) AS snapshots,
               round(avg(coalesce((lag_ms <= 250)::INT, 0)), 3) AS within_250ms,
               round(avg(coalesce((lag_ms <= 1000)::INT, 0)), 3) AS within_1s,
               quantile_cont(lag_ms, 0.5) AS median_lag_ms
        FROM j GROUP BY leg ORDER BY leg
        """,
        "within_250ms near 1: snapshots arrive only after trades. Chance level with one "
        "print every few seconds is around 0.1.",
    )


def quote_age() -> None:
    show(
        "3. Up prints: age of the snapshot the replay quoted from, and where the print sat",
        """
        WITH j AS (
            SELECT p.side, p.price, s.bid, s.ask, p.rx_ts - s.rx_ts AS age_ms
            FROM p ASOF JOIN s ON p.token = s.token AND p.rx_prior >= s.rx_ts
            WHERE p.leg = 'up' AND s.bid IS NOT NULL AND s.ask IS NOT NULL
        ), c AS (
            SELECT age_ms,
                   CASE WHEN side = 'SELL' THEN round(bid - price, 4)
                        ELSE round(price - ask, 4) END AS through
            FROM j
        )
        SELECT count(*) AS up_prints,
               round(quantile_cont(age_ms, 0.5) / 1000, 2) AS median_age_s,
               round(quantile_cont(age_ms, 0.75) / 1000, 2) AS p75_age_s,
               round(quantile_cont(age_ms, 0.9) / 1000, 2) AS p90_age_s,
               round(avg((through = 0)::INT), 3) AS at_prior_best,
               round(avg((through > 0)::INT), 3) AS through_prior_best,
               round(avg((through < 0)::INT), 3) AS inside_prior_best
        FROM c
        """,
        "through_prior_best: the print is beyond the best price of the snapshot the replay "
        "quoted from, so the market moved while the replay quote did not.",
    )


def feed_cadence() -> None:
    view("feeds1", "feeds", 1, "rx_ts, obs_ts, topic, symbol, px")
    show(
        "4. RTDS cadence and relay delay, last derived day",
        """
        WITH f AS (
            SELECT topic, symbol, rx_ts, obs_ts, px,
                   lag(rx_ts) OVER w AS prev_rx, lag(px) OVER w AS prev_px
            FROM feeds1
            WINDOW w AS (PARTITION BY topic, symbol ORDER BY rx_ts)
        )
        SELECT topic, symbol, count(*) AS ticks,
               quantile_cont(rx_ts - prev_rx, 0.5) AS median_gap_ms,
               quantile_cont(rx_ts - prev_rx, 0.9) AS p90_gap_ms,
               round(avg((px <> prev_px)::INT), 3) AS share_new_price,
               quantile_cont(rx_ts - obs_ts, 0.5) AS median_delay_ms
        FROM f WHERE prev_rx IS NOT NULL
        GROUP BY topic, symbol ORDER BY topic, symbol
        """,
        "median_delay_ms: receive time minus the source timestamp (NTP clocks, a few ms of "
        "error). A taker reading the exchange directly sees each move about this much earlier.",
    )


def h2_table() -> None:
    print("\n== 5. H2: markout of passive fills by time to expiry, latest report")
    try:
        with open(REPORT, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        print(f"no report at {REPORT}")
        return
    for i, line in enumerate(lines):
        if line.startswith("### Markout of passive fills"):
            print("\n".join(lines[i : i + 10]))
            return
    print("section not found")


def main() -> None:
    print(f"data {ROOT}, last {DAYS} derived days, prior-snapshot latency {LATENCY_MS} ms")
    raw_event_mix()
    build_tables()
    side_convention()
    snapshot_cadence()
    quote_age()
    feed_cadence()
    h2_table()


if __name__ == "__main__":
    main()
