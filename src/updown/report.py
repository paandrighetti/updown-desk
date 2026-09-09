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

from . import store, telegram
from .config import Settings
from .replay import build_contexts, checkpoints, feed_agreement, grid

log = logging.getLogger("updown.report")

THRESHOLDS = (0.01, 0.02, 0.05)
LATENCIES_MS = (0, 250, 1000, 3000)


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
    windows = store.load_windows(root)
    feeds = store.load_feeds(root)
    books = store.load_books(root)
    resolutions = store.load_resolutions(root)

    contexts = build_contexts(windows, feeds, books, resolutions, settings.ref_feed)
    trades = grid(contexts, THRESHOLDS, LATENCIES_MS)
    summary = summarize_trades(trades)
    agreement = feed_agreement(windows, feeds, resolutions)
    brier, cal = calibration(checkpoints(contexts))

    n_ctx = len(contexts)
    n_res = sum(c.outcome is not None for c in contexts)
    n_sig = sum(c.sigma is not None for c in contexts)
    ref_delays = [c.ref_delay_s for c in contexts if c.ref_delay_s is not None]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    coverage = (
        f"Windows discovered: {n_ctx}. With outcome: {n_res}. With volatility estimate: {n_sig}. "
        f"Reference feed: `{settings.ref_feed}`."
    )
    if ref_delays:
        coverage += (
            f" Median reference-price delay after window start: {np.median(ref_delays):.2f} s."
        )
    parts = [f"# Up/Down desk report ({generated})\n", "## Data coverage\n", coverage + "\n"]
    for src in ("rtds", "clob"):
        parts.append(f"### {src} messages per hour\n")
        parts.append(_md(store.message_counts(root, src).tail(48), ".1f"))
    parts += [
        "## Which feed does the resolution follow\n",
        "Share of resolved windows where sign(last - first) on the feed matches the outcome.\n",
        _md(agreement),
        "## Replay grid: fair-value taker, hold to resolution\n",
        "Fill model: marketable limit at the observed best ask, filled only if the first snapshot "
        "received after `latency_ms` still shows an ask at or below it, size capped by displayed "
        "size and 100 shares. Fees from the market fee schedule.\n",
        _md(summary),
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


def run_once(settings: Settings) -> str:
    n = store.compress_old_files(settings.data_dir)
    if n:
        log.info("compressed %d raw files", n)
    report, digest = build(settings)
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
            run_once(settings)
        except Exception:  # keep the daemon alive, report next day
            log.exception("report failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the updown-desk report")
    parser.add_argument("--loop", action="store_true", help="run daily instead of once")
    parser.add_argument("--hour", type=int, default=6, help="UTC hour for --loop")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("UPDOWN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    settings = Settings()
    if args.loop:
        asyncio.run(_loop(settings, args.hour))
    else:
        t0 = time.monotonic()
        run_once(settings)
        log.info("done in %.1f s", time.monotonic() - t0)


if __name__ == "__main__":
    main()
