"""Build the daily markdown report from recorded data and the replay grid."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import passive, resolve, store, telegram
from .config import Settings
from .replay import build_contexts, checkpoints, feed_agreement, grid

log = logging.getLogger("updown.report")

THRESHOLDS = (0.01, 0.02, 0.05)
LATENCIES_MS = (0, 250, 1000, 3000)
# Passive quoting grid, pre-registered 2026-09-21 (see README). The base cell is the one
# whose fills feed the markout-by-time-to-expiry table.
PASSIVE_BASE = passive.Cell(anchor="mid", half_spread=0.01, latency_ms=250, tau_min_s=0.0)
PASSIVE_CELLS = tuple(
    passive.Cell(anchor=a, half_spread=h, latency_ms=lat, tau_min_s=tm)
    for a in ("mid", "model")
    for h in (0.01, 0.02)
    for lat in (250, 1000)
    for tm in (0.0, 120.0)
)


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _max_drawdown(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    return float((cum - peak).min()) if len(cum) else 0.0


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (th, lat), g in trades.sort_values("t_fill").groupby(["threshold", "latency_ms"]):
        pnl = g["pnl"].to_numpy()
        deployed = float((g["shares"] * g["price"]).sum())
        std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan")
        rows.append(
            {
                "threshold": th,
                "latency_ms": lat,
                "n": int(len(g)),
                "hit_rate": float(g["won"].mean()),
                "pnl_total": float(pnl.sum()),
                "pnl_per_trade": float(pnl.mean()),
                "t_stat": float(pnl.mean() / (std / np.sqrt(len(pnl))))
                if std and std > 0
                else float("nan"),
                "return_on_deployed": float(pnl.sum() / deployed) if deployed else float("nan"),
                "max_drawdown": _max_drawdown(pnl),
                "fees": float(g["fee"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_passive(windows: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """One row per passive cell: window-level PnL statistics plus fill-level markouts."""
    if windows.empty:
        return pd.DataFrame()
    rows = []
    for key, g in windows.sort_values("start").groupby(list(passive.CELL_KEYS), sort=False):
        pnl = g["pnl"].to_numpy()
        std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan")
        f = passive.select(fills, passive.Cell(*key)) if not fills.empty else fills
        sh = f["shares"].to_numpy() if not f.empty else np.empty(0)
        row = dict(zip(passive.CELL_KEYS, key, strict=True))
        row.update(
            {
                "n_win": int(len(g)),
                "shares_per_win": float((g["bought"] + g["sold"]).sum() / len(g)),
                "pnl_ex_rebate": float(g["pnl_ex_rebate"].sum()),
                "rebates": float(g["rebates"].sum()),
                "pnl": float(pnl.sum()),
                "pnl_per_win": float(pnl.mean()),
                "t_stat": float(pnl.mean() / (std / np.sqrt(len(pnl))))
                if std and std > 0
                else float("nan"),
                "max_drawdown": _max_drawdown(pnl),
            }
        )
        for col in ("mo_30s", "mo_res"):
            v = f[col].to_numpy() if not f.empty else np.empty(0)
            ok = ~np.isnan(v) if len(v) else np.zeros(0, dtype=bool)
            row[col] = float((v[ok] * sh[ok]).sum() / sh[ok].sum()) if ok.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def calibration(cp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier scores per checkpoint and a decile calibration table for model and market."""
    if cp.empty:
        return pd.DataFrame(), pd.DataFrame()
    cp = cp.dropna(subset=["model_p", "market_mid"]).copy()
    if cp.empty:
        return pd.DataFrame(), pd.DataFrame()
    cp["brier_model"] = (cp["model_p"] - cp["outcome"]) ** 2
    cp["brier_market"] = (cp["market_mid"] - cp["outcome"]) ** 2
    brier = (
        cp.groupby("offset_s")
        .agg(
            n=("outcome", "count"),
            brier_model=("brier_model", "mean"),
            brier_market=("brier_market", "mean"),
            mean_spread=("spread", "mean"),
        )
        .reset_index()
    )
    bins = np.linspace(0, 1, 11)
    cp["bucket"] = pd.cut(cp["model_p"], bins, include_lowest=True)
    cal = (
        cp.groupby("bucket", observed=True)
        .agg(
            n=("outcome", "count"),
            model_p=("model_p", "mean"),
            market_mid=("market_mid", "mean"),
            realized=("outcome", "mean"),
        )
        .reset_index()
    )
    cal["bucket"] = cal["bucket"].astype(str)
    return brier, cal


def build(settings: Settings) -> tuple[str, str]:
    root = settings.data_dir
    days = settings.report_days
    windows = store.load_derived(root, "windows", days=days)
    feeds = store.load_derived(
        root,
        "feeds",
        where=f"topic IN ('{settings.ref_feed}', '{settings.spot_feed}')",
        days=days,
    )
    books = store.load_derived(root, "books", days=days)
    tape = store.load_derived(root, "trades", days=days)
    resolutions = pd.concat(
        [
            store.load_derived(root, "resolutions", days=days + 1),
            store.load_derived(root, "outcomes", days=days + 1),
        ],
        ignore_index=True,
    )
    coverage = store.load_derived(root, "coverage", days=2)

    contexts = build_contexts(
        windows,
        feeds,
        books,
        resolutions,
        settings.ref_feed,
        spot_feed=settings.spot_feed,
        settlement=settings.settlement,
    )
    trades = grid(contexts, THRESHOLDS, LATENCIES_MS)
    summary = summarize_trades(trades)
    agreement = feed_agreement(windows, store.derived_files(root, "feeds", days=days), resolutions)
    brier, cal = calibration(checkpoints(contexts))
    t0 = time.monotonic()
    p_fills, p_windows = passive.grid(contexts, tape, PASSIVE_CELLS)
    p_summary = summarize_passive(p_windows, p_fills)
    p_markout = passive.markout_by_tau(passive.select(p_fills, PASSIVE_BASE))
    log.info("passive grid: %d cells in %.0f s", len(PASSIVE_CELLS), time.monotonic() - t0)
    n_prints = int(len(tape))
    prints_sized = float((tape["size"].notna() & (tape["size"] > 0)).mean()) if n_prints else 0.0

    n_ctx = len(contexts)
    n_res = sum(c.outcome is not None for c in contexts)
    sources = pd.Series([c.outcome_source for c in contexts]).value_counts().to_dict()
    n_sig = sum(c.sigma is not None for c in contexts)
    ref_delays = [c.ref_delay_s for c in contexts if c.ref_delay_s is not None]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    headline = (
        f"Windows discovered: {n_ctx}. With outcome: {n_res}. With volatility estimate: {n_sig}. "
        f"Strike and settlement feed: `{settings.ref_feed}`; state feed: `{settings.spot_feed}`; "
        f"settlement model: `{settings.settlement}`. Outcome sources: {sources}."
    )
    if ref_delays:
        headline += (
            f" Median reference-price delay after window start: {np.median(ref_delays):.2f} s."
        )
    parts = [
        f"# Up/Down desk report ({generated}, last {days} complete days)\n",
        "## Data coverage\n",
        headline + "\n",
    ]
    for src in ("rtds", "clob"):
        parts.append(f"### {src} messages per hour\n")
        cov = (
            coverage[coverage["source"] == src].drop(columns=["source"])
            if not coverage.empty
            else coverage
        )
        parts.append(_md(cov.tail(48), ".1f"))
    parts += [
        "## Which feed does the resolution follow\n",
        "Share of resolved windows where sign(last - first) on the feed matches the outcome.\n",
        _md(agreement),
        "## Replay grid: fair-value taker, hold to resolution\n",
        "Fill model: marketable limit at the observed best ask, filled only if the first snapshot "
        "received after `latency_ms` still shows an ask at or below it, size capped by displayed "
        "size and 100 shares. Fees from the market fee schedule.\n",
        _md(summary),
        "## Passive quoting replay: two-sided quotes on the Up token, inventory held to "
        "resolution\n",
        "Quotes are anchor +/- half_spread with linear inventory skew, "
        f"{PASSIVE_BASE.quote_size:.0f} shares a side, inventory capped at "
        f"{PASSIVE_BASE.max_inventory:.0f}, live from snapshot "
        "receive time + `latency_ms` until the next snapshot's quotes go live. A bid is filled "
        "by taker SELL prints at or below it, an ask by taker BUY prints at or above it, after "
        "the displayed size at the joined level has printed. Fills through the Down token are "
        f"not observed. Maker rebate: {PASSIVE_BASE.rebate_share:.0%} of the taker fee on the "
        f"fill. `tau_min_s` stops quoting inside the last N seconds. Prints in the tape: "
        f"{n_prints}, share carrying a size: {prints_sized:.1%}. Markouts are share-weighted "
        "per-share moves of the mid against the fill (negative = adverse).\n",
        _md(p_summary),
        "### Markout of passive fills by time to expiry (base cell: mid anchor, half spread "
        f"{PASSIVE_BASE.half_spread}, latency {PASSIVE_BASE.latency_ms} ms)\n",
        _md(p_markout),
        "## Model versus market at fixed checkpoints\n",
        _md(brier),
        "### Calibration by model probability decile\n",
        _md(cal),
    ]
    report = "\n".join(parts)

    if summary.empty:
        digest = f"updown-desk: {n_ctx} windows, {n_res} resolved, no trades in replay grid yet."
    else:
        best = summary.sort_values("t_stat", ascending=False).iloc[0]
        digest = (
            f"updown-desk: {n_ctx} windows, {n_res} resolved. Best cell th={best['threshold']} "
            f"lat={best['latency_ms']}ms: n={best['n']}, hit={best['hit_rate']:.2f}, "
            f"pnl={best['pnl_total']:.2f}, t={best['t_stat']:.2f}"
        )
    if not p_summary.empty:
        pb = p_summary.sort_values("t_stat", ascending=False).iloc[0]
        digest += (
            f" | passive best {pb['anchor']} h={pb['half_spread']} lat={pb['latency_ms']}ms "
            f"tau_min={pb['tau_min_s']:.0f}: n={pb['n_win']}, pnl={pb['pnl']:.2f} "
            f"(rebates {pb['rebates']:.2f}), t={pb['t_stat']:.2f}, mo30s={pb['mo_30s']:.4f}"
        )
    return report, digest


def write(settings: Settings, report: str) -> str:
    os.makedirs(settings.reports_dir, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(settings.reports_dir, f"{day}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(settings.reports_dir, "latest.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    return path


def derive_pending(settings: Settings, include_today: bool = False) -> list[str]:
    """Derive every complete UTC day not yet in data/derived. Returns the days processed."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    done = []
    for day in store.raw_days(settings.data_dir):
        if day > today or (day == today and not include_today):
            continue
        t0 = time.monotonic()
        counts = store.derive_day(settings.data_dir, day)
        if counts:
            log.info("derived %s in %.0f s: %s", day, time.monotonic() - t0, counts)
            done.append(day)
    return done


def _drop_derived_day(root: str, day: str) -> None:
    for table in store.DERIVED_TABLES:
        path = os.path.join(root, "derived", table, f"{day}.parquet")
        if os.path.exists(path):
            os.remove(path)


def run_once(settings: Settings, include_today: bool = False) -> str:
    # Raw compression belongs to the collector; two processes gzipping the same file at the
    # same time would corrupt it.
    derive_pending(settings, include_today)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    resolve.sweep_pending(
        settings.data_dir, [d for d in store.raw_days(settings.data_dir) if d < today]
    )
    try:
        report, digest = build(settings)
    finally:
        if include_today:  # a partial day must not be mistaken for a complete one tomorrow
            _drop_derived_day(settings.data_dir, datetime.now(timezone.utc).strftime("%Y%m%d"))
    path = write(settings, report)
    log.info("wrote %s", path)
    telegram.send(settings.telegram_token, settings.telegram_chat_id, digest)
    return path


async def _loop(settings: Settings, hour: int) -> None:
    while True:
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        log.info("next report at %s", nxt.isoformat())
        await asyncio.sleep((nxt - now).total_seconds())
        try:
            # run_once drives its own event loops (Gamma sweep); it must not run inside this one
            await asyncio.to_thread(run_once, settings)
        except Exception:  # keep the daemon alive, report next day
            log.exception("report failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the updown-desk report")
    parser.add_argument("--loop", action="store_true", help="run daily instead of once")
    parser.add_argument("--hour", type=int, default=6, help="UTC hour for --loop")
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="also use today's partial data (one-off runs only; the daily loop never does)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("UPDOWN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    if args.loop:
        asyncio.run(_loop(settings, args.hour))
    else:
        t0 = time.monotonic()
        run_once(settings, include_today=args.include_today)
        log.info("done in %.1f s", time.monotonic() - t0)


if __name__ == "__main__":
    main()
