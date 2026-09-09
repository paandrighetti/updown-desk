"""Synthetic end-to-end run. Also documents the exact on-disk wire formats the loaders expect."""

import json
import math
import os

import numpy as np
import pytest

from updown import store
from updown.config import Settings
from updown.fair_value import SECONDS_PER_YEAR, p_up
from updown.replay import Params, build_contexts, checkpoints, feed_agreement, grid, replay
from updown.report import build, calibration, summarize_trades

T0 = 1_771_210_800  # aligned window start
SIGMA = 0.6
N_WINDOWS = 3


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rx_ts, msg in rows:
            fh.write(json.dumps({"rx_ts": rx_ts, "msg": msg}, separators=(",", ":")) + "\n")


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(1)
    root = str(tmp_path / "data")
    t_begin, t_end = T0 - 3600, T0 + N_WINDOWS * 900 + 120
    n = t_end - t_begin
    r = rng.normal(0, SIGMA * math.sqrt(1 / SECONDS_PER_YEAR), n)
    px = 60_000.0 * np.exp(np.cumsum(r))
    ts = np.arange(t_begin, t_end)

    rtds = []
    for t, p in zip(ts, px, strict=True):
        ms = int(t) * 1000
        rtds.append(
            (
                ms + 40,
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                    "timestamp": ms + 40,
                    "payload": {"symbol": "btc/usd", "timestamp": ms, "value": float(p)},
                },
            )
        )
        rtds.append(
            (
                ms + 55,
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "timestamp": ms + 55,
                    "payload": {"symbol": "btcusdt", "timestamp": ms, "value": float(p) * 1.0002},
                },
            )
        )
    _write(os.path.join(root, "raw", "rtds", "a.jsonl"), rtds)

    clob, meta = [], []
    for k in range(N_WINDOWS):
        start, end = T0 + 900 * k, T0 + 900 * (k + 1)
        up, down, cid = f"u{k}", f"d{k}", f"0xc{k}"
        meta.append(
            (
                start * 1000 - 600_000,
                {
                    "symbol": "btc",
                    "start": start,
                    "end": end,
                    "slug": f"btc-updown-15m-{start}",
                    "condition_id": cid,
                    "up_token": up,
                    "down_token": down,
                    "up_label": "Up",
                    "down_label": "Down",
                    "tick_size": 0.01,
                    "fees_enabled": True,
                    "fee_rate": 0.25,
                    "fee_exponent": 2,
                },
            )
        )
        ref = float(px[start - t_begin])
        for t in range(start, end, 5):
            spot = float(px[t - t_begin])
            fair = p_up(spot, ref, SIGMA, end - t)
            mid = min(0.98, max(0.02, fair + rng.normal(0, 0.03)))
            ms = t * 1000 + 300
            for token, m in ((up, mid), (down, 1 - mid)):
                bid, ask = round(m - 0.01, 2), round(m + 0.01, 2)
                clob.append(
                    (
                        ms,
                        {
                            "event_type": "book",
                            "market": cid,
                            "asset_id": token,
                            "timestamp": str(ms),
                            "hash": "h",
                            "bids": [
                                {"price": f"{bid - 0.02:.2f}", "size": "900"},
                                {"price": f"{bid:.2f}", "size": "400"},
                            ],
                            "asks": [
                                {"price": f"{ask + 0.02:.2f}", "size": "900"},
                                {"price": f"{ask:.2f}", "size": "400"},
                            ],
                        },
                    )
                )
        winner = up if float(px[end - t_begin]) >= ref else down
        clob.append(
            (
                end * 1000 + 90_000,
                {
                    "event_type": "market_resolved",
                    "id": str(k),
                    "market": cid,
                    "assets_ids": [up, down],
                    "winning_asset_id": winner,
                    "winning_outcome": "Up" if winner == up else "Down",
                    "timestamp": str(end * 1000 + 90_000),
                },
            )
        )
    _write(os.path.join(root, "raw", "clob", "a.jsonl"), clob)
    _write(os.path.join(root, "raw", "windows", "a.jsonl"), meta)
    return root


def test_loaders_and_contexts(dataset):
    windows = store.load_windows(dataset)
    feeds = store.load_feeds(dataset)
    books = store.load_books(dataset)
    res = store.load_resolutions(dataset)
    assert len(windows) == N_WINDOWS
    assert set(feeds["topic"]) == {"crypto_prices_chainlink", "crypto_prices"}
    assert set(feeds["symbol"]) == {"btc"}
    assert books["ask"].min() > books["bid"].max() - 1.0  # sane top of book
    assert (books["ask_size"] == 400).all()  # best level, not the deeper one
    assert len(res) == N_WINDOWS

    ctx = build_contexts(windows, feeds, books, res, "crypto_prices_chainlink")
    assert len(ctx) == N_WINDOWS
    assert all(c.outcome_source == "market_resolved" for c in ctx)
    assert all(abs(c.sigma - SIGMA) / SIGMA < 0.25 for c in ctx)
    assert all(c.ref_delay_s == 0.0 for c in ctx)


def test_replay_is_causal_and_fills_within_displayed_size(dataset):
    windows, feeds = store.load_windows(dataset), store.load_feeds(dataset)
    books, res = store.load_books(dataset), store.load_resolutions(dataset)
    ctx = build_contexts(windows, feeds, books, res, "crypto_prices_chainlink")
    trades = replay(ctx, Params(threshold=0.01, latency_ms=0, max_shares=100))
    assert not trades.empty
    assert (trades["t_fill"] >= trades["t_signal"]).all()
    assert (trades["shares"] <= 100).all()
    assert (trades["fee"] > 0).all()
    # one trade per side per window at most
    assert trades.groupby(["start", "side"]).size().max() == 1
    slow = replay(ctx, Params(threshold=0.01, latency_ms=3000, max_shares=100))
    assert slow.empty or (slow["t_fill"] - slow["t_signal"] >= 3000).all()


def test_grid_checkpoints_agreement_and_report(dataset, tmp_path):
    windows, feeds = store.load_windows(dataset), store.load_feeds(dataset)
    books, res = store.load_books(dataset), store.load_resolutions(dataset)
    ctx = build_contexts(windows, feeds, books, res, "crypto_prices_chainlink")
    g = grid(ctx, (0.01, 0.05), (0, 1000))
    summary = summarize_trades(g)
    assert set(summary.columns) >= {"n", "hit_rate", "pnl_total", "t_stat", "max_drawdown"}
    cp = checkpoints(ctx)
    assert len(cp) == N_WINDOWS * 3
    brier, cal = calibration(cp)
    assert list(brier["offset_s"]) == [300, 600, 840]
    assert cal["n"].sum() == len(cp)
    agree = feed_agreement(windows, feeds, res)
    row = agree.set_index("topic").loc["crypto_prices_chainlink"]
    assert row["n"] == N_WINDOWS and row["agreement"] == 1.0

    settings = Settings(data_dir=dataset, reports_dir=str(tmp_path / "reports"))
    report, digest = build(settings)
    assert "## Replay grid" in report and "## Which feed" in report
    assert digest.startswith("updown-desk:")


def test_compress_old_files(dataset):
    raw = os.path.join(dataset, "raw", "rtds", "a.jsonl")
    old = os.path.getmtime(raw) - 10_000
    os.utime(raw, (old, old))
    assert store.compress_old_files(dataset) == 1
    assert os.path.exists(raw + ".gz") and not os.path.exists(raw)
    assert len(store.load_feeds(dataset)) > 0  # gz read transparently
