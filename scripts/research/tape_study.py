"""What do the makers on these markets earn? Realized spread of every fill on the tape.

Read-only, derived tables only. Runs inside the reporter image:

  docker compose run --rm -v /opt/updown-desk/scripts:/scripts reporter \
      python /scripts/tape_study.py

Every print is a fill for some resting order. The realized spread of those fills measures
what the liquidity resting on these books earns, with no queue or latency model to get wrong.
A replay of our own quotes would only add where we would sit in the queue and how fast we
would move, so it is worth building only where the resting liquidity earns something.

Definitions. The two books are one book seen from both sides (Up bid = 1 - Down ask in 99.9 %
of paired snapshots), so every print is expressed on the Up token: a taker BUY of Down at q is
a taker SELL of Up at 1 - q. The maker holds the other side. For a fill at price p with maker
direction d (+1 bought Up, -1 sold Up), the realized spread at horizon h is
    rs_h = d * (mid(t + h) - p)
where mid is the Up mid of the last book snapshot received by t + h inside the window (a
snapshot follows every trade, the one of the trade itself included; near expiry the horizon
stops at the window end), and at resolution rs_res = d * (outcome - p). The rebate is an
estimate that favours the maker: 20 % of the taker fee of the market's fee schedule. Liquidity
reward pools are not counted. Averages are share-weighted; t-statistics are cluster-robust with
one cluster per window start, so the four assets of a quarter hour count as one observation.

Pre-registered on 23 September 2026, before any realized spread was computed on these data.
Samples: exploration = windows starting 10 to 15 September, confirmation = windows starting
16 to 22 September (UTC). Statistic: rs_30s + rebate, in cents per share. Buckets: time to
expiry, distance of the price from 0.5, and the absolute move of the Binance price relayed by
Polymarket over the 3 s before the print; plus one niche cell (at least 120 s to expiry, price
within 0.3 of 0.5, relayed move under 1 bp). Decision: a bucket qualifies if its statistic is
positive with t > 2 on both samples. If none qualifies, passive quoting at the touch on these
markets is abandoned. If some do, the full follower simulation is built, restricted to them.
"""

from __future__ import annotations

import glob
import math
import os
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.environ.get("UPDOWN_DATA_DIR", "/data")
REBATE_SHARE = 0.20
HORIZONS_S = (5, 30, 60)
SPOT_LOOKBACK_MS = 3000


def utc(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


SAMPLES = (
    ("exploration", utc("2026-09-10"), utc("2026-09-16")),
    ("confirmation", utc("2026-09-16"), utc("2026-09-23")),
)
TAU_EDGES = (0, 60, 120, 300, 600, 901)
DIST_EDGES = (0.0, 0.1, 0.3, 0.51)
SPOT_EDGES = (0.0, 1.0, 5.0, math.inf)

con = duckdb.connect()
con.execute("SET memory_limit='1500MB'")
con.execute("SET temp_directory='/tmp/duckdb_spill'")


def view(name: str, table: str, cols: str = "*", where: str = "") -> bool:
    files = sorted(glob.glob(os.path.join(ROOT, "derived", table, "*.parquet")))
    # a day without source messages is written as a zero-column file DuckDB cannot read
    files = [f for f in files if len(pq.read_schema(f)) > 0]
    if not files:
        return False
    sql = f"SELECT {cols} FROM read_parquet({files!r}, union_by_name=true)"
    con.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}" + (f" WHERE {where}" if where else ""))
    return True


def load() -> pd.DataFrame:
    for name, table in (("windows_raw", "windows"), ("books", "books"), ("trades", "trades")):
        if not view(name, table):
            raise SystemExit(f"no derived {table} files under {ROOT}")
    view("feeds", "feeds", "rx_ts, symbol, px", "topic = 'crypto_prices'")
    sources = []
    if view("res_stream", "resolutions", "rx_ts, condition_id, winning_token"):
        sources.append("SELECT condition_id, winning_token, 0 AS rank, rx_ts FROM res_stream")
    if view("res_gamma", "outcomes", "rx_ts, condition_id, winning_token"):
        sources.append("SELECT condition_id, winning_token, 1 AS rank, rx_ts FROM res_gamma")
    if not sources:
        sources.append("SELECT NULL::VARCHAR, NULL::VARCHAR, 0, NULL::BIGINT WHERE false")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE res AS
        SELECT condition_id, winning_token FROM ({" UNION ALL ".join(sources)})
        QUALIFY row_number() OVER (PARTITION BY condition_id ORDER BY rank, rx_ts) = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE w AS
        SELECT symbol, start, "end" AS wend, condition_id, up_token, down_token,
               coalesce(fees_enabled, false) AS fees_on, coalesce(fee_rate, 0) AS fee_rate,
               coalesce(fee_exponent, 1) AS fee_exp
        FROM windows_raw
        QUALIFY row_number() OVER (PARTITION BY slug ORDER BY rx_ts DESC) = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE tok AS
        SELECT up_token AS token, condition_id FROM w
        UNION ALL SELECT down_token, condition_id FROM w
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE s AS
        SELECT b.rx_ts, w.condition_id, (b.bid + b.ask) / 2 AS mid
        FROM books b JOIN w ON b.token = w.up_token
        WHERE b.rx_ts BETWEEN w.start * 1000 AND w.wend * 1000
          AND b.bid IS NOT NULL AND b.ask IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE p AS
        SELECT tr.rx_ts AS t, w.symbol, w.condition_id, w.start, w.wend, w.up_token,
               tr.token = w.up_token AS on_up,
               CASE WHEN tr.token = w.up_token THEN tr.price ELSE 1 - tr.price END AS p,
               CASE WHEN (tr.token = w.up_token) = (tr.side = 'BUY') THEN -1 ELSE 1 END AS d,
               tr.size AS shares,
               CASE WHEN w.fees_on
                    THEN {REBATE_SHARE} * w.fee_rate * pow(tr.price * (1 - tr.price), w.fee_exp)
                    ELSE 0 END AS rebate,
               tr.rx_ts + 5000 AS t5, tr.rx_ts + 30000 AS t30, tr.rx_ts + 60000 AS t60,
               tr.rx_ts - {SPOT_LOOKBACK_MS} AS t_back
        FROM trades tr JOIN tok k ON tr.token = k.token
        JOIN w ON k.condition_id = w.condition_id
        WHERE tr.rx_ts BETWEEN w.start * 1000 AND w.wend * 1000
          AND tr.size > 0 AND tr.price > 0 AND tr.price < 1
        """
    )
    return con.sql(
        """
        WITH a AS (
            SELECT p.*, s.mid AS mid5 FROM p
            ASOF LEFT JOIN s ON p.condition_id = s.condition_id AND p.t5 >= s.rx_ts
        ), b AS (
            SELECT a.*, s.mid AS mid30 FROM a
            ASOF LEFT JOIN s ON a.condition_id = s.condition_id AND a.t30 >= s.rx_ts
        ), c AS (
            SELECT b.*, s.mid AS mid60 FROM b
            ASOF LEFT JOIN s ON b.condition_id = s.condition_id AND b.t60 >= s.rx_ts
        ), e AS (
            SELECT c.*, f.px AS px_now FROM c
            ASOF LEFT JOIN feeds f ON c.symbol = f.symbol AND c.t >= f.rx_ts
        ), g AS (
            SELECT e.*, f.px AS px_back FROM e
            ASOF LEFT JOIN feeds f ON e.symbol = f.symbol AND e.t_back >= f.rx_ts
        )
        SELECT g.t, g.symbol, g.start, g.wend, g.on_up, g.p, g.d, g.shares, g.rebate,
               g.mid5, g.mid30, g.mid60, g.px_now, g.px_back,
               CASE WHEN r.winning_token IS NULL THEN NULL
                    WHEN r.winning_token = g.up_token THEN 1 ELSE 0 END AS outcome
        FROM g LEFT JOIN res r ON g.condition_id = r.condition_id
        """
    ).df()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["tau"] = df["wend"] - df["t"] / 1000.0
    for h in HORIZONS_S:
        df[f"rs_{h}s"] = df["d"] * (df[f"mid{h}"] - df["p"])
    df["rs_res"] = df["d"] * (df["outcome"] - df["p"])
    df["stat"] = df["rs_30s"] + df["rebate"]
    df["spot_bp"] = np.abs(np.log(df["px_now"] / df["px_back"])) * 1e4
    df["sample"] = None
    for name, lo, hi in SAMPLES:
        df.loc[(df["start"] >= lo) & (df["start"] < hi), "sample"] = name
    df["tau_b"] = pd.cut(df["tau"], TAU_EDGES, right=False)
    df["dist_b"] = pd.cut((df["p"] - 0.5).abs(), DIST_EDGES, right=False)
    df["spot_b"] = pd.cut(df["spot_bp"], SPOT_EDGES, right=False)
    df["niche"] = (df["tau"] >= 120) & ((df["p"] - 0.5).abs() < 0.3) & (df["spot_bp"] < 1.0)
    return df


def weighted(g: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Share-weighted mean and its cluster-robust t (one cluster per window start)."""
    g = g[g[col].notna()]
    if g.empty:
        return float("nan"), float("nan"), 0
    by = g.assign(sx=g["shares"] * g[col])[["start", "sx", "shares"]].groupby("start").sum()
    total = by["shares"].sum()
    m = by["sx"].sum() / total
    n = len(by)
    if n < 2:
        return float(m), float("nan"), n
    resid = by["sx"] - m * by["shares"]
    se = math.sqrt(n / (n - 1) * float((resid**2).sum())) / total
    return float(m), float(m / se) if se > 0 else float("nan"), n


def cents(x: float) -> float:
    return round(100 * x, 3)


def table(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_string(index=False) if rows else "no data"


def main() -> None:
    df = enrich(load())
    print(f"data {ROOT}: {len(df)} prints inside windows, {df['start'].nunique()} windows")
    df = df[df["sample"].notna()]

    print("\n== 1. coverage per sample")
    rows = []
    for name, g in df.groupby("sample", sort=False):
        own_buy = (g["on_up"] & (g["d"] == -1)) | (~g["on_up"] & (g["d"] == 1))
        rows.append(
            {
                "sample": name,
                "days": pd.to_datetime(g["start"], unit="s").dt.date.nunique(),
                "windows": g.groupby(["start", "symbol"]).ngroups,
                "prints": len(g),
                "shares": round(float(g["shares"].sum())),
                "on_down_token": round(float((~g["on_up"]).mean()), 3),
                "taker_buys_own_token": round(float(own_buy.mean()), 3),
                "with_mid30": round(float(g["mid30"].notna().mean()), 3),
                "with_spot": round(float(g["spot_bp"].notna().mean()), 3),
                "with_outcome": round(float(g["outcome"].notna().mean()), 3),
            }
        )
    print(table(rows))

    print("\n== 2. realized spread of maker fills, cents per share (t cluster-robust)")
    rows = []
    for name, g in df.groupby("sample", sort=False):
        row = {"sample": name}
        for col in ("rs_5s", "rs_30s", "rs_60s", "rs_res", "rebate", "stat"):
            m, t, _ = weighted(g, col)
            row[col] = cents(m)
            if col in ("rs_30s", "rs_res", "stat"):
                row[f"t_{col}"] = round(t, 2)
        rows.append(row)
    print(table(rows))
    print("stat = rs_30s + rebate. Positive means the resting liquidity earned money.")

    print("\n== 3. stat by bucket and sample")
    rows = []
    families = (("tau_s", "tau_b"), ("dist_from_0.5", "dist_b"), ("spot_move_bp", "spot_b"))
    for family, col in families:
        for bucket, gb in df.groupby(col, observed=True, sort=True):
            row = {"family": family, "bucket": str(bucket)}
            for name, g in gb.groupby("sample", sort=False):
                m, t, n = weighted(g, "stat")
                row[f"{name}_cents"] = cents(m)
                row[f"{name}_t"] = round(t, 2)
                row[f"{name}_windows"] = n
                row[f"{name}_shares"] = round(float(g["shares"].sum()))
            rows.append(row)
    niche = {"family": "niche", "bucket": "tau>=120, |p-0.5|<0.3, move<1bp"}
    for name, g in df[df["niche"]].groupby("sample", sort=False):
        m, t, n = weighted(g, "stat")
        niche.update(
            {
                f"{name}_cents": cents(m),
                f"{name}_t": round(t, 2),
                f"{name}_windows": n,
                f"{name}_shares": round(float(g["shares"].sum())),
            }
        )
    rows.append(niche)
    buckets = pd.DataFrame(rows)
    print(buckets.to_string(index=False))

    print("\n== 4. sanity: by token printed and by maker direction, stat in cents")
    rows = []
    for (name, on_up, d), g in df.groupby(["sample", "on_up", "d"], sort=False):
        m, t, _ = weighted(g, "stat")
        mr, tr, _ = weighted(g, "rs_res")
        rows.append(
            {
                "sample": name,
                "token": "up" if on_up else "down",
                "maker": "bought_up" if d == 1 else "sold_up",
                "shares": round(float(g["shares"].sum())),
                "stat": cents(m),
                "t": round(t, 2),
                "rs_res": cents(mr),
                "t_res": round(tr, 2),
            }
        )
    print(table(rows))

    print("\n== 5. pre-registered decision")
    need = [c for c in ("exploration_t", "confirmation_t") if c in buckets]
    if len(need) < 2:
        print("both samples are needed; decision not taken")
        return
    ok = (
        (buckets["exploration_cents"] > 0)
        & (buckets["exploration_t"] > 2)
        & (buckets["confirmation_cents"] > 0)
        & (buckets["confirmation_t"] > 2)
    )
    if ok.any():
        print("qualifying buckets, to be simulated with a follower maker:")
        print(buckets[ok][["family", "bucket"] + need].to_string(index=False))
    else:
        print("no bucket qualifies: passive quoting at the touch on these markets is abandoned")


if __name__ == "__main__":
    main()
