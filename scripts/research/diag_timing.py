"""Second diagnostic: disk, event timing, book mirroring and the recorded wire formats.

Read-only. Runs inside the reporter image:

  docker compose run --rm -v /opt/updown-desk/scripts:/scripts reporter \
      python /scripts/diag_timing.py

1. Disk: usage of the filesystem holding /data, size per data directory, raw CLOB size per
   day, and the drop list a newly created container reads from .env.
2. Receive delay per CLOB event type over one raw hour: receive time minus the exchange
   timestamp carried by the message.
3. Each print paired with the book message of the same token whose exchange timestamp is
   nearest: how close the two timestamps are and which message is received first. A print
   received well after the book of its own trade is matched by the v0.2 replay against a
   quote computed from the post-trade book.
4. Mirror check on the last derived day: in Up and Down snapshots received together, is the
   Up best bid one minus the Down best ask, and the Up ask one minus the Down bid? If so the
   two books are one book seen from both sides, and a taker buying Down trades against the
   Up bids.
5. One recorded message of each CLOB type, truncated.
"""

from __future__ import annotations

import glob
import os
import shutil
from collections import defaultdict

import duckdb

ROOT = os.environ.get("UPDOWN_DATA_DIR", "/data")
RAW = "read_json({path!r}, columns={{'rx_ts':'BIGINT','msg':'JSON'}}, format='newline_delimited')"
GB = 1024**3

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


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass  # rotated or compressed while walking
    return total


def disk() -> None:
    print("\n== 1. disk")
    u = shutil.disk_usage(ROOT)
    print(
        f"filesystem holding {ROOT}: total {u.total / GB:.1f} GB, used {u.used / GB:.1f} GB, "
        f"free {u.free / GB:.1f} GB ({u.used / u.total:.0%} used)"
    )
    for sub in ("raw/clob", "raw/rtds", "raw/windows", "derived"):
        print(f"  {sub:<12} {dir_size(os.path.join(ROOT, sub)) / GB:7.2f} GB")
    per_day: dict[str, int] = defaultdict(int)
    for path in glob.glob(os.path.join(ROOT, "raw", "clob", "*.jsonl*")):
        try:
            per_day[os.path.basename(path)[:8]] += os.path.getsize(path)
        except OSError:
            pass
    print("  raw CLOB per day (the last day is in progress, its recent hours uncompressed):")
    for day in sorted(per_day):
        print(f"    {day}  {per_day[day] / GB:6.2f} GB")
    print(f"  UPDOWN_DROP_EVENTS in .env: {os.environ.get('UPDOWN_DROP_EVENTS', '')!r}")


def latest_noon_hour() -> str | None:
    files = sorted(glob.glob(os.path.join(ROOT, "raw", "clob", "*_12.jsonl.gz")))
    return files[-1] if files else None


def timing(path: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ev AS
        SELECT rx_ts,
               json_extract_string(msg, '$.event_type') AS et,
               json_extract_string(msg, '$.asset_id') AS token,
               try_cast(json_extract_string(msg, '$.timestamp') AS BIGINT) AS ts
        FROM {RAW.format(path=path)}
        """
    )
    show(
        f"2. receive time minus exchange timestamp, {os.path.basename(path)}",
        """
        SELECT et AS event_type, count(*) AS n, count(ts) AS with_ts,
               quantile_cont(rx_ts - ts, 0.1) AS p10_ms,
               quantile_cont(rx_ts - ts, 0.5) AS median_ms,
               quantile_cont(rx_ts - ts, 0.9) AS p90_ms
        FROM ev GROUP BY et ORDER BY n DESC
        """,
    )
    show(
        "3. each print against the book message of the same token with the nearest exchange "
        "timestamp",
        """
        WITH pr AS (
            SELECT row_number() OVER () AS id, rx_ts, token, ts
            FROM ev WHERE et = 'last_trade_price' AND ts IS NOT NULL
        ), bk AS (
            SELECT token, ts, min(rx_ts) AS rx_ts
            FROM ev WHERE et = 'book' AND ts IS NOT NULL GROUP BY token, ts
        ), cand AS (
            SELECT pr.id, pr.rx_ts AS p_rx, pr.ts AS p_ts, bk.rx_ts AS b_rx, bk.ts AS b_ts
            FROM pr ASOF JOIN bk ON pr.token = bk.token AND pr.ts >= bk.ts
            UNION ALL
            SELECT pr.id, pr.rx_ts, pr.ts, bk.rx_ts, bk.ts
            FROM pr ASOF JOIN bk ON pr.token = bk.token AND pr.ts <= bk.ts
        ), near AS (
            SELECT * FROM cand
            QUALIFY row_number() OVER (PARTITION BY id ORDER BY abs(p_ts - b_ts), b_rx) = 1
        )
        SELECT count(*) AS prints,
               round(avg((p_ts = b_ts)::INT), 3) AS same_ts,
               round(avg((abs(p_ts - b_ts) <= 50)::INT), 3) AS within_50ms,
               quantile_cont(b_ts - p_ts, 0.5) AS median_book_minus_print_ts,
               round(avg((p_rx > b_rx)::INT) FILTER (WHERE abs(p_ts - b_ts) <= 50), 3)
                   AS print_received_after_book,
               quantile_cont(p_rx - b_rx, 0.5) FILTER (WHERE abs(p_ts - b_ts) <= 50)
                   AS median_rx_gap_ms,
               quantile_cont(p_rx - b_rx, 0.9) FILTER (WHERE abs(p_ts - b_ts) <= 50)
                   AS p90_rx_gap_ms
        FROM near
        """,
        "If within_50ms is high and median_rx_gap_ms is several hundred ms, the print of a "
        "trade is received well after the book of the same trade.",
    )


def mirror() -> None:
    books = sorted(glob.glob(os.path.join(ROOT, "derived", "books", "*.parquet")))[-1:]
    wins = sorted(glob.glob(os.path.join(ROOT, "derived", "windows", "*.parquet")))[-2:]
    if not books or not wins:
        print("\n== 4. no derived books or windows")
        return
    show(
        f"4. mirror check, {os.path.basename(books[0])}: Up and Down snapshots received within "
        "50 ms of each other",
        f"""
        WITH w AS (
            SELECT up_token, down_token, condition_id
            FROM read_parquet({wins!r}, union_by_name=true)
            QUALIFY row_number() OVER (PARTITION BY condition_id ORDER BY rx_ts DESC) = 1
        ), b AS (SELECT rx_ts, token, bid, ask FROM read_parquet({books!r})),
        u AS (
            SELECT row_number() OVER () AS id, b.*, w.condition_id
            FROM b JOIN w ON b.token = w.up_token
        ),
        d AS (SELECT b.*, w.condition_id FROM b JOIN w ON b.token = w.down_token),
        cand AS (
            SELECT u.id, u.bid AS ub, u.ask AS ua, d.bid AS db, d.ask AS da,
                   abs(u.rx_ts - d.rx_ts) AS gap
            FROM u ASOF JOIN d ON u.condition_id = d.condition_id AND u.rx_ts >= d.rx_ts
            UNION ALL
            SELECT u.id, u.bid, u.ask, d.bid, d.ask, abs(u.rx_ts - d.rx_ts)
            FROM u ASOF JOIN d ON u.condition_id = d.condition_id AND u.rx_ts <= d.rx_ts
        ), j AS (
            SELECT * FROM cand WHERE gap <= 50
            QUALIFY row_number() OVER (PARTITION BY id ORDER BY gap) = 1
        )
        SELECT count(*) AS pairs,
               round(avg((round(ub + da, 4) = 1)::INT), 3) AS up_bid_is_1_minus_down_ask,
               round(avg((round(ua + db, 4) = 1)::INT), 3) AS up_ask_is_1_minus_down_bid
        FROM j
        WHERE ub IS NOT NULL AND ua IS NOT NULL AND db IS NOT NULL AND da IS NOT NULL
        """,
        "Both shares near 1: one book seen from both sides.",
    )


def samples(path: str) -> None:
    for event_type, n in (("price_change", 1500), ("book", 700), ("last_trade_price", 700)):
        show(
            f"5. one recorded {event_type} message, first {n} characters",
            f"SELECT substr(CAST(msg AS VARCHAR), 1, {n}) AS msg FROM {RAW.format(path=path)} "
            f"WHERE json_extract_string(msg, '$.event_type') = '{event_type}' LIMIT 1",
        )


def main() -> None:
    disk()
    path = latest_noon_hour()
    if path:
        timing(path)
    else:
        print("\n== 2-3. no compressed raw CLOB hour found")
    mirror()
    if path:
        samples(path)


if __name__ == "__main__":
    main()
