"""Read raw JSONL (plain or gzip) into typed pandas frames with DuckDB.

The wire payload is kept as a JSON column and extracted with JSON path functions, so
schema drift between message types never breaks loading.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import time

import duckdb
import pandas as pd

from .windows import base_symbol

_READ = "read_json({paths}, columns={{'rx_ts':'BIGINT','msg':'JSON'}}, format='newline_delimited')"


def _col(path: str, alias: str, cast: str | None = None) -> str:
    """SQL expression extracting a JSON path from the wire payload, optionally cast."""
    expr = f"json_extract_string(msg,'$.{path}')"
    if cast:
        expr = f"try_cast({expr} AS {cast})"
    return f'{expr} AS "{alias}"'


def _where(path: str, value: str) -> str:
    return f"WHERE json_extract_string(msg,'$.{path}') = '{value}'"


def _paths(root: str, source: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, "raw", source, "*.jsonl*")))
    return [f for f in files if f.endswith((".jsonl", ".jsonl.gz"))]


def _sql_list(paths: list[str]) -> str:
    return "[" + ",".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def _query(root: str, source: str, select: str, where: str = "") -> pd.DataFrame:
    paths = _paths(root, source)
    if not paths:
        return pd.DataFrame()
    sql = f"SELECT {select} FROM {_READ.format(paths=_sql_list(paths))} {where}"
    return duckdb.sql(sql).df()


def load_windows(root: str) -> pd.DataFrame:
    cols = ", ".join(
        [
            "rx_ts",
            _col("symbol", "symbol"),
            _col("start", "start", "BIGINT"),
            _col("end", "end", "BIGINT"),
            _col("slug", "slug"),
            _col("condition_id", "condition_id"),
            _col("up_token", "up_token"),
            _col("down_token", "down_token"),
            _col("tick_size", "tick_size", "DOUBLE"),
            _col("fees_enabled", "fees_enabled", "BOOLEAN"),
            _col("fee_rate", "fee_rate", "DOUBLE"),
            _col("fee_exponent", "fee_exponent", "DOUBLE"),
        ]
    )
    df = _query(root, "windows", cols)
    if df.empty:
        return df
    return df.sort_values("rx_ts").drop_duplicates("slug", keep="last").reset_index(drop=True)


def load_feeds(root: str) -> pd.DataFrame:
    """All RTDS price updates: topic, base symbol, price, observation time, receive time."""
    cols = ", ".join(
        [
            "rx_ts",
            _col("topic", "topic"),
            _col("payload.symbol", "rtds_symbol"),
            _col("payload.value", "px", "DOUBLE"),
            _col("payload.timestamp", "obs_ts", "BIGINT"),
        ]
    )
    df = _query(
        root,
        "rtds",
        cols,
        _where("type", "update") + " AND json_extract_string(msg,'$.payload.value') IS NOT NULL",
    )
    if df.empty:
        return df
    df["symbol"] = df["rtds_symbol"].map(base_symbol)
    return df.sort_values("rx_ts").reset_index(drop=True)


def load_books(root: str) -> pd.DataFrame:
    """Top of book from full 'book' snapshots: best bid/ask price and size per token."""
    cols = ", ".join(
        ["rx_ts", _col("asset_id", "token"), _col("bids", "bids"), _col("asks", "asks")]
    )
    df = _query(root, "clob", cols, _where("event_type", "book"))
    if df.empty:
        return df

    def top(levels: str | None, best):
        if not levels:
            return (None, None)
        rows = json.loads(levels)
        if not rows:
            return (None, None)
        lvl = best(rows, key=lambda r: float(r["price"]))
        return (float(lvl["price"]), float(lvl["size"]))

    bids = df["bids"].map(lambda s: top(s, max))
    asks = df["asks"].map(lambda s: top(s, min))
    out = pd.DataFrame(
        {
            "rx_ts": df["rx_ts"],
            "token": df["token"],
            "bid": [b[0] for b in bids],
            "bid_size": [b[1] for b in bids],
            "ask": [a[0] for a in asks],
            "ask_size": [a[1] for a in asks],
        }
    )
    return out.sort_values("rx_ts").reset_index(drop=True)


def load_resolutions(root: str) -> pd.DataFrame:
    cols = ", ".join(
        ["rx_ts", _col("market", "condition_id"), _col("winning_asset_id", "winning_token")]
    )
    df = _query(root, "clob", cols, _where("event_type", "market_resolved"))
    if df.empty:
        return df
    return (
        df.sort_values("rx_ts").drop_duplicates("condition_id", keep="last").reset_index(drop=True)
    )


def message_counts(root: str, source: str) -> pd.DataFrame:
    """Messages per UTC hour and the largest silent gap inside each hour, in seconds."""
    paths = _paths(root, source)
    if not paths:
        return pd.DataFrame()
    sql = f"""
        WITH t AS (
            SELECT rx_ts, rx_ts - lag(rx_ts) OVER (ORDER BY rx_ts) AS gap
            FROM {_READ.format(paths=_sql_list(paths))}
        )
        SELECT strftime(to_timestamp(rx_ts / 1000), '%Y-%m-%d %H') AS hour,
               count(*) AS n, max(gap) / 1000.0 AS max_gap_s
        FROM t GROUP BY 1 ORDER BY 1
    """
    return duckdb.sql(sql).df()


def compress_old_files(root: str, older_than_s: int = 7200) -> int:
    """Gzip raw files not written to for older_than_s seconds. Returns the number compressed."""
    n = 0
    cutoff = time.time() - older_than_s
    for path in glob.glob(os.path.join(root, "raw", "*", "*.jsonl")):
        if os.path.getmtime(path) < cutoff:
            with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.remove(path)
            n += 1
    return n
