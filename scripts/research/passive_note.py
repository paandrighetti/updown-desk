# ruff: noqa: E501
# Line length: the Markdown templates below hold table rows that cannot be wrapped.
"""Render docs/passive-quoting.md and the README summary from the raw outputs in docs/results.

Every figure in the note is read from those files, which are the unedited outputs of the
research scripts next to this one, or from the scripts' own parameters, so none is typed by hand:

  python scripts/research/passive_note.py           write the note and the README block
  python scripts/research/passive_note.py --check   exit 1 if either no longer matches

The outputs are pandas and markdown tables. Text columns come first in every table and a text
value may contain spaces (a bucket label such as "[0.0, 0.1)"), so rows are read from the right:
the trailing fields are the numeric columns, whatever precedes them is text. Numbers are rounded
half up on their decimal value, as a reader checking the outputs would round them. The prose
states signs and significance in words; the `assumed` checks in tape() list what that wording
takes for granted, and the script fails if a regenerated output contradicts it.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "docs" / "results"
NOTE = REPO / "docs" / "passive-quoting.md"
README = REPO / "README.md"
START, END = "<!-- passive-study:start -->", "<!-- passive-study:end -->"
GREP_PREFIX = re.compile(r"^\d{1,4}[-:](?=\D|$)")  # grep -n line numbers in v02_passive_grid.txt


# --- reading the outputs -----------------------------------------------------------------------


def sections(name: str) -> dict[str, list[str]]:
    """Lines of a result file, keyed by the text of their '== ' header."""
    out: dict[str, list[str]] = {"": []}
    key = ""
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        line = GREP_PREFIX.sub("", line)
        if line.startswith("== "):
            key = line[3:].strip()
            out[key] = []
        else:
            out[key].append(line)
    return out


def section_keys(secs: dict[str, list[str]], prefix: str) -> list[str]:
    keys = [k for k in secs if k.startswith(prefix)]
    if not keys:
        raise KeyError(f"no section starting with {prefix!r}")
    return keys


def section(secs: dict[str, list[str]], prefix: str) -> list[str]:
    return secs[section_keys(secs, prefix)[0]]


def table(lines: list[str], first_col: str, n_text: int = 1) -> pd.DataFrame:
    """A pandas to_string table whose header starts with first_col; n_text leading columns."""
    start = next(i for i, ln in enumerate(lines) if ln.split()[:1] == [first_col])
    header = lines[start].split()
    n_num = len(header) - n_text
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip() or ": " in line or " = " in line:
            break
        tok = line.split()
        text = tok[: len(tok) - n_num]
        rows.append(text[: n_text - 1] + [" ".join(text[n_text - 1 :])] + tok[len(tok) - n_num :])
    df = pd.DataFrame(rows, columns=header)
    for col in header[n_text:]:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # a text column after the numbers, such as a date
    return df


def md_table(lines: list[str], first_col: str) -> pd.DataFrame:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"| {first_col} |"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    df = pd.DataFrame(rows, columns=header)
    for col in header:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    return df


def row(df: pd.DataFrame, **match) -> pd.Series:
    sel = df
    for col, val in match.items():
        sel = sel[sel[col] == val]
    if len(sel) != 1:
        raise ValueError(f"expected one row for {match}, found {len(sel)}")
    return sel.iloc[0]


def constant(script: str, name: str) -> float:
    """A module-level numeric constant of one of the research scripts."""
    src = (HERE / script).read_text(encoding="utf-8")
    return float(re.search(rf"^{name} = ([\d.]+)$", src, re.MULTILINE).group(1))


# --- formatting --------------------------------------------------------------------------------


def num(x: float, places: int, sign: bool = False) -> str:
    d = Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def c(x: float) -> str:
    """Cents with a sign, for tables and for values quoted with their t-statistic."""
    return num(x, 2, sign=True)


def t(x: float) -> str:
    return num(x, 1)


def pct(x: float) -> str:
    return num(100 * x, 1)


def mag(x: float) -> str:
    """Magnitude, for prose that states the sign in words."""
    return num(abs(x), 2)


def day(d: date) -> str:
    return f"{d.day} {d:%B}"


def span(first: date, last: date) -> str:
    if first.month == last.month:
        return f"{first.day} to {day(last)}"
    return f"{day(first)} to {day(last)}"


def ymd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def hours(keys: list[str]) -> str:
    """'hour from 12:00 UTC on 22 September' or 'hours from 12:00 UTC on 9 and 22 September'."""
    found = [re.search(r"(\d{8})_(\d{2})", k).groups() for k in keys]
    if len({h for _, h in found}) != 1:
        raise ValueError("the wording assumes the same hour of day in every file")
    days = [ymd(d) for d, _ in found]
    start = f"{found[0][1]}:00 UTC on"
    if len(days) == 1:
        return f"hour from {start} {day(days[0])}"
    if len({d.month for d in days}) != 1:
        raise ValueError("the wording assumes hours sampled within one month")
    return f"hours from {start} {' and '.join(str(d.day) for d in days)} {days[0]:%B}"


def cents_grid(lo: float, hi: float) -> list[int]:
    """One-cent prices, in cents, whose distance from 0.5 falls in [lo, hi) as the scripts
    compute it: in floating point, so a price at a decimal distance of exactly lo can miss."""
    return [i for i in range(1, 100) if lo <= abs(i / 100 - 0.5) < hi]


def price_ranges(cents: list[int]) -> str:
    runs: list[list[int]] = []
    for i in cents:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return " and from ".join(f"{r[0] / 100:.2f} to {r[-1] / 100:.2f}" for r in runs)


# --- values ----------------------------------------------------------------------------------


def v02() -> dict:
    lines = [ln for block in sections("v02_passive_grid.txt").values() for ln in block]
    text = "\n".join(lines)
    grid = md_table(lines, "anchor")
    # The report that ran the grid covers the last UPDOWN_REPORT_DAYS derived days.
    config = (REPO / "src" / "updown" / "config.py").read_text(encoding="utf-8")
    n_days = int(re.search(r'"UPDOWN_REPORT_DAYS", "(\d+)"', config).group(1))
    days = sorted(re.findall(r"derived (\d{8}) in", text))[-n_days:]
    prints = int(re.search(r"Prints in the tape: (\d+)", text).group(1))
    keys = ["half_spread", "latency_ms", "tau_min_s"]
    mid = grid[grid["anchor"] == "mid"].set_index(keys)["pnl_per_win"]
    model = grid[grid["anchor"] == "model"].set_index(keys)["pnl_per_win"]
    mo30 = grid.set_index(["anchor", "latency_ms", "tau_min_s", "half_spread"])["mo_30s"]
    wider_helps = mo30.xs(0.02, level="half_spread").gt(mo30.xs(0.01, level="half_spread"))
    h1 = row(grid, anchor="mid", half_spread=0.01, latency_ms=250, tau_min_s=120.0)
    # H2 is read on the report's base cell, which quotes to expiry (mid, 0.01, 250 ms, tau_min 0).
    # Each bucket's mo_30s averages only the fills that have a mid 30 s later, so the buckets
    # are quoted one by one rather than pooled with the shares column.
    tau = md_table(section(sections("diag_passive.txt"), "5. H2"), "tau_bucket")
    tau = tau.set_index("tau_bucket")["mo_30s"]
    late, early = tau[["[0, 60)", "[60, 120)"]], tau[["[300, 600)", "[600, 900)"]]
    n_cells, n_neg = len(grid), int((grid["pnl_per_win"] < 0).sum())
    abandon = int(((grid["mo_30s"] < -grid["half_spread"]) & (grid["pnl_ex_rebate"] < 0)).sum())
    if n_neg != n_cells:
        raise ValueError("the wording assumes every v0.2 cell lost money")
    return {
        "n_cells": n_cells,
        "n_win": f"{int(grid['n_win'].iloc[0]):,}",
        "loss_min": mag(grid["pnl_per_win"].max()),
        "loss_max": mag(grid["pnl_per_win"].min()),
        "t_lo": t(grid["t_stat"].min()),
        "t_hi": t(grid["t_stat"].max()),
        "h1_ex": c(h1["pnl_ex_rebate"] / h1["n_win"]),
        "h1_pnl": c(h1["pnl_per_win"]),
        "h1_t": t(h1["t_stat"]),
        "h1_verdict": "rejected" if h1["pnl_ex_rebate"] < 0 else "not rejected",
        "h2_late": " and ".join(c(100 * x) for x in late),
        "h2_early": " and ".join(c(100 * x) for x in early),
        "h2_verdict": "holds" if late.max() < min(0.0, early.min()) else "rejected",
        "h3_mid": int(mid.gt(model).sum()),
        "h3_verdict": "holds" if int(model.gt(mid).sum()) == 0 else "rejected",
        "n_pairs": len(mid),
        "abandon": abandon,
        "abandon_verdict": (
            f"met, over {n_days} days instead of the 14 required"
            if abandon == n_cells
            else "not met"
        ),
        "v02_days": n_days,
        "spread_helps": f"{int(wider_helps.sum())} of the" if wider_helps.any() else "none of the",
        "v02_prints": f"{prints:,}",
        "v02_span": span(ymd(days[0]), ymd(days[-1])),
    }


def diagnostics() -> dict:
    d = sections("diag_passive.txt")
    mix_keys = section_keys(d, "0. raw CLOB event mix")
    if len(mix_keys) != 2:
        raise ValueError("the wording assumes two sampled hours of the event mix")
    mixes = [table(d[k], "event_type").set_index("event_type")["n"] for k in mix_keys]
    side = table(section(d, "1. side"), "side").set_index("side")
    cad = table(section(d, "2. prints and book snapshots"), "kind", n_text=2)
    lag = table(section(d, "2b. lag"), "leg").set_index("leg")
    age = table(section(d, "3. Up prints"), "up_prints").iloc[0]
    feeds = table(section(d, "4. RTDS"), "topic", n_text=2)
    binance = row(feeds, topic="crypto_prices", symbol="btc")
    snap = cad[cad["kind"] == "snapshots"]["per_window"]
    if snap.nunique() != 1:
        raise ValueError("snapshot counts differ between the two tokens")
    buys, sells = side.loc["BUY", "prints"], side.loc["SELL", "prints"]
    tm = sections("diag_timing.txt")
    timing_key = section_keys(tm, "2. receive time")[0]
    delay = table(tm[timing_key], "event_type")
    pair = table(section(tm, "3. each print"), "prints").iloc[0]
    mirror_key = section_keys(tm, "4. mirror check")[0]
    mirror = table(tm[mirror_key], "pairs").iloc[0]
    main_types = delay[delay["event_type"].isin(["price_change", "book", "last_trade_price"])]
    return {
        "buy_above": pct(side.loc["BUY", "above_mid"]),
        "sell_below": pct(side.loc["SELL", "below_mid"]),
        "buy_share": pct(buys / (buys + sells)),
        "snap_pw": num(snap.iloc[0], 1),
        "prints_pw": num(cad[cad["kind"] == "prints"]["per_window"].sum(), 1),
        "pc_per_book": " and ".join(num(m["price_change"] / m["book"], 0) for m in mixes),
        "mix_hours": hours(mix_keys),
        "lag_share": pct(lag.loc["up", "within_250ms"]),
        "age_med": num(age["median_age_s"], 1),
        "age_p90": num(age["p90_age_s"], 1),
        "moved": pct(age["through_prior_best"] + age["inside_prior_best"]),
        "mirror_bid": pct(mirror["up_bid_is_1_minus_down_ask"]),
        "mirror_ask": pct(mirror["up_ask_is_1_minus_down_bid"]),
        "pairs": f"{int(mirror['pairs']):,}",
        "mirror_day": day(ymd(re.search(r"(\d{8})\.parquet", mirror_key).group(1))),
        "timing_hour": hours([timing_key]),
        "rx_min": num(main_types["median_ms"].min(), 0),
        "rx_max": num(main_types["median_ms"].max(), 0),
        "n_timed": f"{int(pair['prints']):,}",
        "rx_gap": num(pair["median_rx_gap_ms"], 0),
        "rx_gap_p90": num(pair["p90_rx_gap_ms"], 0),
        "within50": pct(pair["within_50ms"]),
        "print_first": pct(1 - pair["print_received_after_book"]),
        "gap": num(binance["median_gap_ms"], 0),
        "repeat": pct(1 - binance["share_new_price"]),
        "delay": num(binance["median_delay_ms"], 0),
        "lag_total": num((binance["median_delay_ms"] + binance["median_gap_ms"] / 2) / 1000, 1),
    }


SAMPLES = re.compile(r'\("(\w+)", utc\("([\d-]{10})"\), utc\("([\d-]{10})"\)\)')


def tape() -> dict:
    # Sample bounds as pre-registered in tape_study.py, end day exclusive.
    bounds = {
        name: (date.fromisoformat(a), date.fromisoformat(b) - timedelta(days=1))
        for name, a, b in SAMPLES.findall((HERE / "tape_study.py").read_text(encoding="utf-8"))
    }
    rebate_share = Decimal(str(constant("tape_study.py", "REBATE_SHARE")))
    lookback_s = constant("tape_study.py", "SPOT_LOOKBACK_MS") / 1000
    own_ms = int(constant("tape_queue.py", "OWN_MAX_MS"))
    families = {
        "tau_s": "time to expiry, s",
        "dist_from_0.5": "distance of the price from 0.5",
        "spot_move_bp": f"absolute move of the relayed Binance price over the prior {lookback_s:g} s, bp",
        "niche": "niche: at least 120 s to expiry, within 0.3 of 0.5, move under 1 bp",
    }
    s = sections("tape_study.txt")
    cov = table(section(s, "1. coverage"), "sample").set_index("sample")
    head = table(section(s, "2. realized spread"), "sample").set_index("sample")
    buckets = table(section(s, "3. stat by bucket"), "family", n_text=2)
    sanity = table(section(s, "4. sanity"), "sample", n_text=3)
    decision = section(s, "5. pre-registered decision")
    if not any("qualifying" in ln for ln in decision):
        raise ValueError("the note is written for a qualifying bucket")
    qualifying = table(decision, "family", n_text=2)
    q = sections("tape_queue.txt")
    fees = table(section(q, "1. fee schedule"), "fees_on").iloc[0]
    pocket_key = section_keys(q, "2. pocket")[0]
    shares = table(q[pocket_key], "sample").set_index("sample")
    split = table(section(q, "3. pocket, cents"), "sample", n_text=2)
    verdict = " ".join(section(q, "4. pre-registered"))
    if "build the follower" in verdict:
        raise ValueError("the note is written for the back of the queue failing the rule")
    lo, hi = map(
        float, re.search(r"pocket ([\d.]+) <= \|p - 0.5\| < ([\d.]+)", pocket_key).groups()
    )
    if len(qualifying) != 1 or row(qualifying, family="dist_from_0.5")["bucket"] != f"[{lo}, {hi})":
        raise ValueError("the queue split must be on the one qualifying bucket")
    pocket = row(buckets, family="dist_from_0.5", bucket=f"[{lo}, {hi})")
    near = row(buckets, family="dist_from_0.5", bucket="[0.0, 0.1)")
    pocket_cents = cents_grid(lo, hi)
    # Prices at a decimal distance of exactly hi are excluded by the open end of the bucket;
    # prices at exactly lo belong to it in decimals but miss it in floating point.
    upper = [i for i in range(1, 100) if abs(i - 50) == round(100 * hi)]
    missed = [
        i
        for i in range(1, 100)
        if round(100 * lo) <= abs(i - 50) < round(100 * hi) and i not in pocket_cents
    ]

    def cell(sample: str, prints: str) -> pd.Series:
        return row(split, sample=sample, prints=prints)

    ae, ac = cell("exploration", "all"), cell("confirmation", "all")
    fe, fc = cell("exploration", "front_or_middle"), cell("confirmation", "front_or_middle")
    be, bc = cell("exploration", "back"), cell("confirmation", "back")
    t_cols = buckets[["exploration_t", "confirmation_t"]]
    assumed = {
        "no sample earns the spread with t > 2": not (
            (head["rs_30s"] > 0) & (head["t_rs_30s"] > 2)
        ).any(),
        "the qualifying bucket's spread alone has |t| < 2": all(
            abs(r["t_rs_30s"]) < 2 for r in (ae, ac)
        ),
        "level-emptying fills lose before the rebate with t < -2": all(
            r["t_rs_30s"] < -2 for r in (be, bc)
        ),
        "level-emptying fills have |t| < 2 with the rebate": all(
            abs(r["t_stat"]) < 2 for r in (be, bc)
        ),
        "the other fills earn before the rebate with t > 2": all(
            r["rs_30s"] > 0 and r["t_rs_30s"] > 2 for r in (fe, fc)
        ),
        "the only bucket losing with t < -2 is near 0.5, in the exploration sample": (
            int((t_cols < -2).to_numpy().sum()) == 1 and near["exploration_t"] < -2
        ),
        "the edge prices outside the bucket are its upper edge and, by floating point, its lower edge": (
            not set(upper) & set(pocket_cents)
            and missed == [i for i in range(1, 100) if abs(i - 50) == round(100 * lo)]
        ),
    }
    broken = [k for k, ok in assumed.items() if not ok]
    if broken:
        raise ValueError("the wording no longer holds: " + "; ".join(broken))
    bids = sanity[sanity["maker"] == "bought_up"]
    asks = sanity[sanity["maker"] == "sold_up"]
    v = {
        "tape_span": span(bounds["exploration"][0], bounds["confirmation"][1]),
        "n_days": (bounds["confirmation"][1] - bounds["exploration"][0]).days + 1,
        "span_e": span(*bounds["exploration"]),
        "span_c": span(*bounds["confirmation"]),
        "down_bids": pct(bids.loc[bids["token"] == "down", "shares"].sum() / bids["shares"].sum()),
        "down_asks": pct(asks.loc[asks["token"] == "down", "shares"].sum() / asks["shares"].sum()),
        "fee_rate": f"{fees['fee_rate']:g}",
        "fee_shape": "p(1 - p)" if fees["fee_exp"] == 1 else f"(p(1 - p))^{fees['fee_exp']:g}",
        "rebate_share": f"{float(100 * rebate_share):g}",
        "rebate_rate": f"{(Decimal(str(fees['fee_rate'])) * rebate_share).normalize()}",
        "own_ms": own_ms,
        "lo": f"{lo:g}",
        "hi": f"{hi:g}",
        "pocket_prices": price_ranges(pocket_cents),
        "upper_prices": " and ".join(f"{i / 100:.2f}" for i in upper),
        "missed_prices": " and ".join(f"{i / 100:.2f}" for i in missed),
        "near_prices": price_ranges(cents_grid(0.0, 0.1)),
        "near_e": c(near["exploration_cents"]),
        "near_e_t": t(near["exploration_t"]),
        "near_c": c(near["confirmation_cents"]),
        "near_c_t": t(near["confirmation_t"]),
        "n_buckets": len(buckets),
        "n_better": int((buckets["confirmation_cents"] > buckets["exploration_cents"]).sum()),
        "pk_e": mag(pocket["exploration_cents"]),
        "pk_c": mag(pocket["confirmation_cents"]),
        "pk_e_t": t(pocket["exploration_t"]),
        "pk_c_t": t(pocket["confirmation_t"]),
        "pk_reb": num((ae["rebate"] + ac["rebate"]) / 2, 2),
        "pk_rs_e": c(ae["rs_30s"]),
        "pk_rs_c": c(ac["rs_30s"]),
        "pk_rs_e_t": t(ae["t_rs_30s"]),
        "pk_rs_c_t": t(ac["t_rs_30s"]),
        "back_e": mag(be["rs_30s"]),
        "back_c": mag(bc["rs_30s"]),
        "back_e_s": c(be["rs_30s"]),
        "back_c_s": c(bc["rs_30s"]),
        "back_e_t": t(be["t_rs_30s"]),
        "back_c_t": t(bc["t_rs_30s"]),
        "back_stat_e": c(be["stat"]),
        "back_stat_c": c(bc["stat"]),
        "back_stat_e_t": t(be["t_stat"]),
        "back_stat_c_t": t(bc["t_stat"]),
        "front_e": mag(fe["rs_30s"]),
        "front_c": mag(fc["rs_30s"]),
        "front_e_s": c(fe["rs_30s"]),
        "front_c_s": c(fc["rs_30s"]),
        "front_e_t": t(fe["t_rs_30s"]),
        "front_c_t": t(fc["t_rs_30s"]),
        # from the displayed values, so that a reader subtracting the table gets the same gap
        "gap_e": str(Decimal(c(fe["rs_30s"])) - Decimal(c(be["rs_30s"]))),
        "gap_c": str(Decimal(c(fc["rs_30s"])) - Decimal(c(bc["rs_30s"]))),
        "no_own_e": pct(shares.loc["exploration", "no_own_snapshot"]),
        "no_own_c": pct(shares.loc["confirmation", "no_own_snapshot"]),
    }
    for sample, tag in (("exploration", "e"), ("confirmation", "c")):
        v[f"prints_{tag}"] = f"{int(cov.loc[sample, 'prints']):,}"
        v[f"windows_{tag}"] = f"{int(cov.loc[sample, 'windows']):,}"
        v[f"days_{tag}"] = int(cov.loc[sample, "days"])
        v[f"rs30_{tag}"] = c(head.loc[sample, "rs_30s"])
        v[f"rs30_{tag}_t"] = t(head.loc[sample, "t_rs_30s"])
    v["headline_rows"] = "\n".join(
        f"| {sample} | {c(r['rs_5s'])} | {c(r['rs_30s'])} | {t(r['t_rs_30s'])} | "
        f"{c(r['rs_60s'])} | {c(r['rs_res'])} | {t(r['t_rs_res'])} | {num(r['rebate'], 2)} | "
        f"{c(r['stat'])} | {t(r['t_stat'])} |"
        for sample, r in head.iterrows()
    )

    def label(r: pd.Series) -> str:
        name = families[r["family"]]
        return name if r["family"] == "niche" else f"{name}: {r['bucket']}"

    v["bucket_rows"] = "\n".join(
        f"| {label(r)} | {c(r['exploration_cents'])} | {t(r['exploration_t'])} | "
        f"{c(r['confirmation_cents'])} | {t(r['confirmation_t'])} |"
        for _, r in buckets.iterrows()
    )
    kinds = {
        "all": "all",
        "front_or_middle": "that leave orders at the level",
        "back": "that empty the level",
    }
    v["split_rows"] = "\n".join(
        f"| {r['sample']} | {kinds[r['prints']]} | "
        f"{'100.0' if r['prints'] == 'all' else pct(shares.loc[r['sample'], r['prints']])} | "
        f"{c(r['rs_30s'])} | {t(r['t_rs_30s'])} | {num(r['rebate'], 2)} | {c(r['stat'])} | "
        f"{t(r['t_stat'])} |"
        for _, r in split.iterrows()
    )
    return v


# --- rendering -------------------------------------------------------------------------------

PREREGISTRATION = """\
> The taker result closes one question: the market is not mispriced against a diffusion fair
> value, so there is nothing to take. The passive replay asks the question a liquidity
> provider would ask instead: is the flow that hits resting quotes in these windows benign
> enough to earn the spread plus the maker rebate, and how does that depend on time to expiry?
> No orders are placed; the replay runs on the same recording, using the trade tape
> (`last_trade_price` events) that the taker replay did not need.
>
> Strategy: two-sided quotes on the Up token at `anchor +/- half_spread`, anchor either the
> market mid or the model fair value, shifted against inventory by a linear skew (the discrete
> analogue of the reservation price in Avellaneda and Stoikov, 2008), 20 shares a side,
> inventory capped at 100 shares and held to resolution. Quotes are rounded to the tick and kept
> one tick inside the opposite best, so a quote never takes.
>
> Fill model, every choice of which can only under-count fills: quotes computed at a book
> snapshot go live `latency_ms` after its receive time and stay live until the next snapshot's
> quotes go live, so a stale quote can be picked off during the latency. A bid is filled by
> taker SELL prints at or below it, an ask by taker BUY prints at or above it, on the same token.
> A quote that improves the displayed best is first in line; one that joins or sits behind it
> waits for the displayed best size to print at or through its price first. Matches through
> the Down token (mint and merge against Down orders) are not observed and are ignored.
> Maker rebate: 20 % of the taker fee on the fill (Polymarket's published crypto rate at the
> time of writing; a parameter, and the report shows PnL with and without it).
>
> Hypotheses written before the first run, with the statistic that decides each one:
>
> - H1. Quoting around the market mid, outside the last 120 s, earns a positive PnL per
>   window before rebates. Decided by the sign and t-statistic of window PnL in the cell
>   `mid, 0.01, 250 ms, tau_min 120` over at least 14 complete days.
> - H2. The flow hitting passive quotes in the last 120 s is informed: the 30 s markout of
>   fills in the `[0, 120)` time-to-expiry buckets is negative and below the markout of the
>   `[300, 900)` buckets. Decided by the markout table of the base cell.
> - H3. Anchoring quotes on the model fair value does not beat anchoring on the mid, since
>   the market is better calibrated than the model. Decided by comparing the two anchors cell
>   by cell at equal half spread, latency and `tau_min`.
> - Abandon criterion: if after 14 complete days every cell has a 30 s markout below minus
>   one half spread and a negative PnL before rebates, passive quoting on these windows is
>   declared unprofitable at 250 ms latency and the project moves on."""

NOTE_TEMPLATE = """\
# Passive quoting on Polymarket's 15-minute crypto markets

What a naive maker replay gets wrong, and what the liquidity resting on these books earns.
Up/Down markets on BTC, ETH, SOL and XRP, {tape_span} 2026.

> Generated by `scripts/research/passive_note.py` from the unedited outputs in `docs/results/`.
> Every measured value below is read from those outputs, and the generator fails if a regenerated
> output contradicts a sign or a significance level stated in words: to change the note, change
> the generator.

## Summary

- The first maker replay (v0.2) lost money in all {n_cells} pre-registered cells, by {loss_min}
  to {loss_max} USD per window with 20-share quotes ({n_win} windows, t from {t_lo} to {t_hi} with
  windows treated as independent). It measured its own design more than the market.
- Two measurements show why, and a third matters for faster replays. Full book snapshots arrive
  almost only with trades, so its quotes stood still between trades while the book kept changing:
  {pc_per_book} book-change messages per book message in the {mix_hours}. The Up and Down books
  are one book seen from both sides, and v0.2 ignored the Down token, whose purchases were
  {down_bids} % of the volume on the bid side of that book. The third measurement: in the
  {timing_hour}, the book update of a trade arrived {rx_gap} ms before its print in the median,
  which only matters to a replay that reacts within that time.
- On the tape, the resting liquidity does not earn the spread: its realized spread 30 s after the
  fill is {rs30_e} and {rs30_c} cents per share in the two samples (t = {rs30_e_t} and {rs30_c_t}),
  before the maker rebate.
- One pre-registered bucket passes on both samples, prices from {pocket_prices}, where the resting
  liquidity earns {pk_e} and {pk_c} cents per share with the estimated rebate (t = {pk_e_t} and
  {pk_c_t}). The rebate is about {pk_reb} cents of that; the spread alone is {pk_rs_e} and
  {pk_rs_c} cents (t = {pk_rs_e_t} and {pk_rs_c_t}), not distinguishable from zero.
- Queue position decides whether the rebate is kept. The fills that empty their price level, the
  only fills that complete the last order in the queue, lose {back_e} and {back_c} cents per share
  before the rebate (t = {back_e_t} and {back_c_t}) and are not distinguishable from zero with it
  ({back_stat_e} and {back_stat_c}, t = {back_stat_e_t} and {back_stat_c_t}). The other fills earn
  {front_e} and {front_c} before it (t = {front_e_t} and {front_c_t}). An order that joins the queue
  starts last, so a second pre-registered rule required the level-emptying fills to pass the test
  before any quote was simulated. They do not, and the study stopped there.

## Data and samples

The collector described in the README recorded every message of Polymarket's market websocket
(book snapshots, book changes and trades) and of its real-time price feed; since 22 September
2026 it no longer records book changes (`price_change`). A window is one market: one asset over
one quarter hour. The tape study uses every trade inside a window: {prints_e} prints over
{windows_e} windows ({days_e} days) in the exploration sample, windows starting {span_e}, and
{prints_c} prints over {windows_c} windows ({days_c} days) in the confirmation sample, windows
starting {span_c}, UTC.

Each of the two tape scripts states its decision rule in its docstring, written before the
statistic the rule governs was computed. The rules came in sequence. The tape study's rule (section 3) sent any
qualifying bucket to a simulation of a maker joining the queue. Before that simulation was built,
a second rule (section 4), written after the tape study's results and before the split by queue
position was computed, required the fills of the last order in the queue to pass the same test.
The outputs in `docs/results/` are the final runs on the complete samples: earlier runs of both
scripts used the confirmation days recorded at the time, and `tape_queue.py` was rewritten once
to fit in memory, without changing its rule. The scripts are committed together with their
results, so nothing in the repository history dates their rules. The v0.2 pre-registration is in
the 0.2.0 commit, whose author date is set by the author and proves nothing either.

## 1. The first replay (v0.2)

Pre-registered on 21 September 2026; the appendix reproduces the text as written. Quotes were
recomputed at each book snapshot, anchored on the mid or on a model fair value, and filled by
prints of the same token that reached their price. A quote that improved the displayed best was
first in line; one that joined or sat behind it waited for the displayed best size to print at
or through its price first. The grid crossed two anchors, two half spreads, two latencies and two stop times, and ran
in the daily report over the {n_win} windows of {v02_span}, with {v02_prints} prints on the tape.
PnL is the profit and loss of 20-share quotes held to resolution, in USD. The report's t is the
t-statistic of the window PnL after rebates, with windows treated as independent; it printed no
t before rebates.

| pre-registered test | measured | verdict on the replay as run |
|---|---|---|
| H1: quoting around the mid outside the last 120 s earns a positive PnL per window before rebates | {h1_ex} USD per window before rebates, {h1_pnl} after (t = {h1_t} after rebates) | {h1_verdict} |
| H2: fills in the last 120 s have a negative 30 s markout, below that of fills 300 s or more before expiry | {h2_late} cents per share in [0, 60) and [60, 120), against {h2_early} in [300, 600) and [600, 900); mid, 1 cent, 250 ms, quoting to expiry | {h2_verdict} |
| H3: the model anchor does not beat the mid anchor | mid better in {h3_mid} of {n_pairs} pairs of cells | {h3_verdict} |
| abandon if every cell has a 30 s markout below minus one half spread and a negative PnL before rebates | {abandon} of {n_cells} cells | {abandon_verdict} |

The report covered the last {v02_days} days, not the 14 the pre-registration asked for; once
section 2 had invalidated the replay, it was not run longer. A figure of the grid itself pointed
to the problem: doubling the half spread improved the 30 s markout in {spread_helps} {n_pairs}
pairs of cells. The markout is measured from the fill price, so if part of the flow reaching the
quotes carried no information, a wider quote would have kept more of the spread per share. It kept
none. That is consistent with quotes filled only once the price had moved through them, although
a wider quote is also reached only by larger moves.

## 2. Why v0.2 is not a verdict

| measurement | value | consequence |
|---|---|---|
| `side` field of trade messages | {buy_above} % of buys print above the prior mid, {sell_below} % of sells below | it is the taker's side, so the fill rule was the right way round |
| book snapshots per token and window | {snap_pw}, for {prints_pw} trades on the two tokens | a full snapshot of each token is sent with every trade, and almost only then |
| book-change messages (`price_change`) per book message | {pc_per_book}, {mix_hours} | v0.2 quotes stood still while the book moved |
| age of the snapshot quoted from, at the print | median {age_med} s, 90th percentile {age_p90} s | an effective latency of seconds, not the 250 ms of the grid |
| prints away from that snapshot's best price | {moved} % | the market had moved and the quote had not |
| Up bid = 1 - Down ask, Up ask = 1 - Down bid | {mirror_bid} % and {mirror_ask} % of {pairs} paired snapshots, {mirror_day} | one book seen from both sides |
| prints where the taker buys | {buy_share} % | bearish takers buy Down rather than sell Up |
| share of Down prints in the volume of each side of the combined book, {tape_span} | bid side (Up bids and Down asks): {down_bids} %; ask side: {down_asks} % | v0.2 counted Up prints only, so it missed most of the flow reaching its bids |
| receive time minus exchange timestamp, clock offset included, {timing_hour} | medians of {rx_min} to {rx_max} ms for snapshots, book changes and trades | no message type lags the others |
| book update of a trade against its print, same hour, {n_timed} prints | {within50} % paired within 50 ms; among them, book first by {rx_gap} ms (median) and {rx_gap_p90} ms (90th percentile), print no later than its book for {print_first} % | harmless at 250 ms; a replay that reacts faster must attach each print to its trade |
| Binance price relayed by Polymarket, one day | one tick every {gap} ms, {repeat} % repeated prices, {delay} ms behind Binance for BTC | a maker fed by the relay sees each move about {lag_total} s after a direct feed |

A purchase of Down fills either a Down ask or, through the exchange's MINT match, an Up bid. MINT
pairs a buy of Up with a buy of Down and mints the two tokens from collateral; MERGE does the
reverse for two sales (Polymarket's Conditional Token Framework exchange overview). The tape does
not say which match filled a print, hence the combined book. The lag test in section 2b of
`diag_passive.txt` looked for a print in the 250 ms before each snapshot and found one for
{lag_share} % of Up snapshots. The timing rows explain it: a trade's snapshot arrives before its
own print, so the latest earlier print belongs to the previous trade.

## 3. What the resting liquidity earns

Every print is a fill for some resting order, so the realized spread of all prints (Huang and
Stoll 1996) measures what the resting liquidity earns, with no queue or latency model to get
wrong. Prints of the Down token are expressed on the Up token: a taker buying Down at q sells Up at
1 - q. For a fill at price p with maker direction d (+1 bought Up, -1 sold Up), the realized spread
at horizon h is d x (mid(t + h) - p) per share, without the factor 2 of the usual definition,
where mid is the Up mid of the last book snapshot received by t + h inside the window. Near expiry
the horizon stops at the window end; at resolution the spread is d x (outcome - p). Taker fees
were introduced on these markets in January 2026 against latency arbitrage (Finance Magnates
2026). Under the schedule recorded here, takers pay {fee_rate} x {fee_shape} per share and makers
none, and {rebate_share} % of each market's fees fund a rebate, shared among its makers in
proportion to the fees their filled volume would have paid (Polymarket documentation). The rebate is therefore estimated at {rebate_rate} x
{fee_shape} per share. Liquidity reward pools are not counted. Averages are share-weighted;
t-statistics are cluster-robust with one cluster per quarter hour, so the four assets of a quarter
hour count once. Moves are in basis points (bp).

Realized spread of maker fills, cents per share:

| sample | 5 s | 30 s | t | 60 s | resolution | t | rebate | 30 s + rebate | t |
|---|---|---|---|---|---|---|---|---|---|
{headline_rows}

The statistic of the study is the 30 s realized spread plus the rebate. By bucket, cents per share:

| bucket | exploration | t | confirmation | t |
|---|---|---|---|---|
{bucket_rows}

Decision rule: a bucket qualifies if its statistic is positive with t > 2 on both samples. Only
one qualifies, a distance of {lo} to {hi} from 0.5. On the one-cent grid it holds prices from
{pocket_prices}: {upper_prices} sit at its excluded upper edge, and {missed_prices}, whose
distance computes to just under {lo} in floating point, fall in the first bucket. Close to 50/50, prices from {near_prices}, the statistic is {near_e} cents per share in
the exploration sample (t = {near_e_t}), the only significant loss in the table, and {near_c} in
the confirmation sample (t = {near_c_t}).

## 4. The back of the queue

A trade that empties its price level completes the last order in the queue at that moment; any
other trade stops short of it or fills it only in part. Deeper queue positions are reached only by larger trades, which carry
more information (Glosten 1994). An order that joins the queue starts last and moves forward as the
orders ahead of it are filled or cancelled, so it receives both kinds of fills but cannot avoid the
level-emptying ones. The split reads the last Up snapshot received in the {own_ms} ms before each
print, normally the snapshot of the print's own trade (section 2): a print emptied its level if
that snapshot shows no order left at the print's price on the maker's side. Within the qualifying
bucket, cents per share:

| sample | fills | share of volume, % | 30 s | t | rebate | 30 s + rebate | t |
|---|---|---|---|---|---|---|---|
{split_rows}

The second rule required the level-emptying fills to have a positive statistic with t > 2 on both
samples before a maker joining the queue was simulated. Their statistic is {back_stat_e} and
{back_stat_c} cents (t = {back_stat_e_t} and {back_stat_c_t}), so the simulation was not built.

The two kinds of fills differ by {gap_e} and {gap_c} cents per share at 30 s, which gives the order
of magnitude of what queue priority is worth per share filled here, on a one-cent tick. Moallemi
and Yuan (2017) find that queue value can be of the same order of magnitude as the bid-ask spread
on some large-tick stocks.

## 5. Conclusions

- Before the rebate, the resting liquidity on these markets earns close to nothing; in the one
  bucket that passes, it earns about the rebate. With continuous prices, competition between
  makers would hand the rebate to takers through tighter quotes, by the zero-profit logic of
  Glosten and Milgrom (1985). A one-cent tick stops quotes from narrowing once the spread is one
  tick, and the rent is then rationed by queue priority, as Yao and Ye (2018) show for large-tick
  stocks. The spread in the bucket was not measured here, so this reading is consistent with the
  data, not tested by it.
- Queue position decides whether the rebate is kept. On average the rebate is the only term
  distinguishable from zero, but by queue position the spread term is {front_e_s} and {front_c_s} cents for fills that
  leave orders at the level and {back_e_s} and {back_c_s} for fills that empty it. An order at the
  front takes part in every trade at its level until it is filled, so it receives both kinds of
  fills; the tape does not say what it earns.
- In the only bucket that passed, the fills that empty their price level earn nothing
  distinguishable from zero, even with the rebate. A maker that can neither rest early at a new
  price nor react faster than the others, which is the position of this project, has no
  demonstrated edge here, and the study stopped before measuring the mix of fills such a maker
  would get. Whether a faster quoter, one that cancels on a direct exchange feed, would earn more
  was not measured either.
- For anyone backtesting a maker on Polymarket: rebuild the book from `price_change` messages
  instead of `book` snapshots, treat the two tokens as one book and convert Down prints, attach
  each print to the book update of its trade, and measure the realized spread of the tape by queue
  position before simulating any quote.

## Limitations

- The rebate is an estimate: the recorded prints carry `fee_rate_bps` = "0", so neither fee nor
  rebate is on the tape. Without the rebate, the qualifying bucket earns only the spread shown in
  section 4, which is not distinguishable from zero.
- Future mids come from snapshots sent with trades and can be several seconds old. The mid of a
  large-tick book is not a martingale (Stoikov 2018), so the staleness may bias the 30 s realized
  spread, in a direction not measured.
- {no_own_e} % and {no_own_c} % of the bucket's volume had no Up snapshot in the {own_ms} ms before
  the print and is left out of the queue split. A print that arrives before the book update of its
  own trade is classified from the previous snapshot, if one arrived in the {own_ms} ms before it,
  and can be misclassified. Snapshots received in the same millisecond are ordered arbitrarily by
  the join, so a rerun can move a small number of shares between the two classes.
- The timing figures and the book-change counts come from single hours, the mirror check from one
  day: they establish the mechanics, not how these figures vary over the samples.
- The rule tests {n_buckets} buckets at t > 2 without a correction for multiple testing; requiring
  the threshold on two separate samples is what limits false positives.
- {n_days} days of one market family, recorded from one server. The confirmation sample does better
  than the exploration sample in {n_better} of {n_buckets} buckets; this span cannot tell a change of
  regime from noise.

## Reproduce

On the server, from the project directory:

```
docker compose run --rm -v "$PWD/scripts/research:/scripts" reporter python /scripts/tape_study.py
docker compose run --rm -v "$PWD/scripts/research:/scripts" reporter python /scripts/tape_queue.py
```

The scripts' docstrings give the path they had before they moved to `scripts/research/`.
`tape_queue.py` needs a large share of the server's 4 GB of memory: run it outside the daily
report, which starts at 06:00 UTC. The two diagnostics, `diag_passive.py` and `diag_timing.py`, read the latest raw hours,
which contain no `price_change` messages after 22 September, as well as the latest derived days
and `reports/latest.md`, which no longer has the v0.2 section, so a rerun differs from the outputs
kept in `docs/results/`. The v0.2 grid is the one the daily report of version 0.2.0 runs. Rebuild
this note with `python scripts/research/passive_note.py`; the test suite runs it with `--check`,
which fails when the note or the README summary no longer matches the results.

## References

- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book.
  Quantitative Finance 8(3), 217-224.
- Glosten, L. and Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with
  heterogeneously informed traders. Journal of Financial Economics 14(1), 71-100.
- Glosten, L. (1994). Is the electronic open limit order book inevitable? Journal of Finance 49(4),
  1127-1161.
- Huang, R. and Stoll, H. (1996). Dealer versus auction markets: a paired comparison of execution
  costs on NASDAQ and the NYSE. Journal of Financial Economics 41(3), 313-357.
- Moallemi, C. and Yuan, K. (2017). A model for queue position valuation in a limit order book.
  Columbia Business School Research Paper 17-70, SSRN 2996221.
- Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. Quantitative
  Finance 18(12), 1959-1966.
- Yao, C. and Ye, M. (2018). Why trading speed matters: a tale of queue rationing under price
  controls. Review of Financial Studies 31(6), 2157-2183.
- Polymarket. Conditional Token Framework (CTF) Exchange overview, matching scenarios NORMAL, MINT
  and MERGE. https://github.com/Polymarket/ctf-exchange/blob/main/docs/Overview.md
- Polymarket documentation. Fees. https://docs.polymarket.com/trading/fees
- Polymarket documentation. Maker Rebates Program.
  https://docs.polymarket.com/market-makers/maker-rebates
- Finance Magnates (7 January 2026). Polymarket introduces dynamic fees to curb latency arbitrage
  in short-term crypto markets.
  https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/

## Appendix: the v0.2 pre-registration, as written on 21 September 2026

{prereg}

Two statements in it proved wrong:

- "every choice of which can only under-count fills": quotes that stood still between trades were
  filled at prices that a quoter following the book would have left, so the model over-counted
  adverse fills, while ignoring the Down token under-counted fills, mostly on the bid side
  (section 2).
- "Matches through the Down token ... are not observed": they are observed, as prints of the Down
  token.
"""

README_TEMPLATE = """\
**Passive quoting, closed on 23 September 2026.** A first maker replay lost money in all
{n_cells} pre-registered cells, but it updated its quotes only after trades and ignored the flow of the Down
token, so it measured its own design more than the market. Measured on the tape instead, the
liquidity resting on these books has a realized spread of {rs30_e} and {rs30_c} cents per share at
30 s in the two samples, before the maker rebate. One pre-registered bucket passes, prices from
{pocket_prices}, and earns about the rebate. There, the fills that empty their price level lose
{back_e} and {back_c} cents per share before the rebate and are not distinguishable
from zero with it. A second rule, written before that split was computed, required them to pass,
so no quote was simulated. Full study, generated from the raw outputs:
[docs/passive-quoting.md](docs/passive-quoting.md)."""


def values() -> dict:
    return {**v02(), **diagnostics(), **tape(), "prereg": PREREGISTRATION}


def render() -> tuple[str, str]:
    v = values()
    readme = README.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        raise SystemExit(f"README.md lacks the {START} ... {END} markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    block = textwrap.fill(
        README_TEMPLATE.format(**v), width=96, break_long_words=False, break_on_hyphens=False
    )
    return NOTE_TEMPLATE.format(**v), f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the files are stale")
    args = parser.parse_args(argv)
    note, readme = render()
    if args.check:
        stale = [
            str(p.relative_to(REPO))
            for p, text in ((NOTE, note), (README, readme))
            if not p.exists() or p.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        return 0
    NOTE.write_text(note, encoding="utf-8")
    README.write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
