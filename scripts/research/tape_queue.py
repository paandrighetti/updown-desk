"""Back of the queue: do the fills an order joining the queue receives still earn?

Read-only follow-up to tape_study.py, whose pre-registered rule qualified one bucket: prices at
a distance of 0.1 to 0.3 from 0.5. Both scripts sit in /opt/updown-desk/scripts:

  docker compose run --rm -v /opt/updown-desk/scripts:/scripts reporter \
      python /scripts/tape_queue.py

The tape study averages every fill. A follower joins the back of the queue at the best price, so
it is filled only by trades large enough to consume the orders ahead of it, which are the trades
that exhaust the level and move the price (Glosten 1994: a unit deeper in the book is filled only
by larger, better informed orders). A print exhausted its level when the snapshot of its own
trade, received about 36 ms before the print, shows the best price on that side beyond the
print: best ask above the price of a taker buy, best bid below the price of a taker sell. Those
prints are the fills of the back of the queue; the others filled orders at the front or in the
middle. Prints without a snapshot in the 100 ms before them are left out of the split.

Pre-registered on 23 September 2026, before this split was computed. Same samples, statistic and
threshold as tape_study.py: in the qualifying bucket, rs_30s + rebate of the level-exhausting
prints must be positive with t > 2 on both samples. If so, the follower simulation is built on
that bucket. If not, it is not built: a follower joining the queue receives these fills. The
script also restates the tape-study verdict for the bucket on the confirmation sample as it
stands when run, and splits the statistic into spread and estimated rebate.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tape_study as ts  # noqa: E402

POCKET = (0.1, 0.3)
OWN_MAX_MS = 100
COLS = ("rs_5s", "rs_30s", "rs_60s", "rebate", "stat", "rs_res")


def load() -> pd.DataFrame:
    """tape_study's fills, plus the best bid and ask of the last Up snapshot before each print.

    tape_study.py stays exactly as it ran; its load() leaves the tables p, w and the books view
    in its connection, which this reads back. Prints of the same market in the same
    millisecond share one snapshot, so the join key (t, symbol, start) is many to one. Only
    the pocket is kept before the join: the full frame plus the join outgrew the reporter's
    memory limit once the confirmation sample reached seven days.
    """
    df = ts.load()
    dist = (df["p"] - 0.5).abs()
    df = ts.enrich(df[(dist >= POCKET[0]) & (dist < POCKET[1])].copy())
    own = ts.con.sql(
        f"""
        WITH sq AS (
            SELECT b.rx_ts, w.condition_id, b.bid, b.ask
            FROM books b JOIN w ON b.token = w.up_token
            WHERE b.rx_ts BETWEEN w.start * 1000 AND w.wend * 1000
              AND b.bid IS NOT NULL AND b.ask IS NOT NULL
        ), k AS (
            SELECT DISTINCT t, symbol, start, condition_id FROM p
            WHERE abs(p - 0.5) >= {POCKET[0]} AND abs(p - 0.5) < {POCKET[1]}
        )
        SELECT k.t, k.symbol, k.start, sq.bid AS own_bid, sq.ask AS own_ask, sq.rx_ts AS own_rx
        FROM k ASOF LEFT JOIN sq ON k.condition_id = sq.condition_id AND k.t >= sq.rx_ts
        """
    ).df()
    return df.merge(own, on=["t", "symbol", "start"], how="left", validate="many_to_one")


def classify(df: pd.DataFrame) -> pd.DataFrame:
    dist = (df["p"] - 0.5).abs()
    df = df[(dist >= POCKET[0]) & (dist < POCKET[1])].copy()
    own = (df["t"] - df["own_rx"]).between(0, OWN_MAX_MS)
    back = np.where(df["d"] == -1, df["own_ask"] > df["p"] + 1e-9, df["own_bid"] < df["p"] - 1e-9)
    df["queue"] = np.where(~own, "no_own_snapshot", np.where(back, "back", "front_or_middle"))
    return df


def row(name: str, cls: str, g: pd.DataFrame) -> dict:
    out = {"sample": name, "prints": cls, "shares": round(float(g["shares"].sum()))}
    for col in COLS:
        m, t, _ = ts.weighted(g, col)
        out[col] = ts.cents(m)
        if col in ("rs_30s", "stat"):
            out[f"t_{col}"] = round(t, 2)
    return out


def main() -> None:
    full = load()
    full = full[full["sample"].notna()]
    pocket = classify(full)
    samples = [name for name, _, _ in ts.SAMPLES]

    print("== 1. fee schedule of the windows, which drives the rebate estimate")
    print(
        ts.con.sql(
            "SELECT fees_on, fee_rate, fee_exp, count(*) AS windows FROM w "
            "GROUP BY ALL ORDER BY windows DESC"
        )
        .df()
        .to_string(index=False)
    )

    print(f"\n== 2. pocket {POCKET[0]} <= |p - 0.5| < {POCKET[1]}: share of shares by queue class")
    rows = []
    for name in samples:
        g = pocket[pocket["sample"] == name]
        total = g["shares"].sum()
        days = pd.to_datetime(g["start"], unit="s").dt.date
        r = {"sample": name, "days": days.nunique(), "last_day": str(days.max())}
        for cls in ("front_or_middle", "back", "no_own_snapshot"):
            r[cls] = round(float(g.loc[g["queue"] == cls, "shares"].sum() / total), 3)
        rows.append(r)
    print(ts.table(rows))

    print("\n== 3. pocket, cents per share, all prints and by queue class (t cluster-robust)")
    rows = []
    for name in samples:
        g = pocket[pocket["sample"] == name]
        rows.append(row(name, "all", g))
        for cls in ("front_or_middle", "back"):
            rows.append(row(name, cls, g[g["queue"] == cls]))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("stat = rs_30s + rebate; rebate = 20 % of the fee schedule's taker fee, an estimate.")

    print("\n== 4. pre-registered decisions")

    def passes(cls: str) -> bool:
        sel = table[table["prints"] == cls].set_index("sample")
        return all(sel.loc[s, "stat"] > 0 and sel.loc[s, "t_stat"] > 2 for s in samples)

    print(f"tape study, pocket on the samples as they stand: qualifies = {passes('all')}")
    if passes("back"):
        print("back of the queue: positive with t > 2 on both samples; build the follower")
    else:
        print("back of the queue: fails the rule; the follower simulation is not built")


if __name__ == "__main__":
    main()
