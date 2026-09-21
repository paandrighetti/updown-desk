"""Passive quoting replay: fill rules on hand-built tapes, then the synthetic pipeline."""

import json
import os

import numpy as np
import pytest

from updown import passive, store
from updown.config import Settings
from updown.passive import Cell, _Window, markout_by_tau, replay
from updown.replay import build_contexts
from updown.report import build, summarize_passive

START, END = 1_771_210_800, 1_771_211_700
FEE = (0.25, 2.0, True)


def _window(books, prints, outcome=1, tick=0.01):
    """books: (rx_ms, bid, ask, bid_size, ask_size); prints: (rx_ms, price, size, side)."""
    b = np.array(books, dtype=float)
    p = np.array(prints, dtype=object) if prints else np.empty((0, 4), dtype=object)
    mid = (b[:, 1] + b[:, 2]) / 2.0
    return _Window(
        symbol="btc",
        start=START,
        end=END,
        outcome=outcome,
        tick=tick,
        fee=FEE,
        rx=b[:, 0].astype(np.int64),
        bid=b[:, 1],
        ask=b[:, 2],
        bid_size=b[:, 3],
        ask_size=b[:, 4],
        mid=mid,
        fair=mid,
        t_rx=np.array([r[0] for r in p], dtype=np.int64),
        p=np.array([r[1] for r in p], dtype=float),
        sz=np.array([r[2] for r in p], dtype=float),
        side=np.array([1.0 if r[3] == "BUY" else -1.0 for r in p]),
    )


T = START * 1000
BASE = Cell(anchor="mid", half_spread=0.01, latency_ms=250)


def test_quote_placement_never_crosses_and_waits_for_latency():
    # mid 0.50, spread 2 ticks: bid_q = 0.49 joins the displayed bid, ask_q = 0.51 joins the ask
    w = _window([(T, 0.49, 0.51, 100, 100)], [(T + 100, 0.49, 500, "SELL")])
    fills, res = replay([w], BASE)
    assert fills.empty  # the print arrived before the quote went live (t + 250 ms)

    w = _window([(T, 0.49, 0.51, 100, 100)], [(T + 300, 0.49, 500, "SELL")])
    fills, res = replay([w], BASE)
    assert len(fills) == 1
    f = fills.iloc[0]
    assert f["price"] == pytest.approx(0.49) and f["dir"] == 1 and f["shares"] == 20
    # bought 20 at 0.49, outcome Up: cash -9.8, settles +20, rebate on 20 shares
    assert res.iloc[0]["pnl_ex_rebate"] == pytest.approx(20 - 9.8)
    assert res.iloc[0]["rebates"] == pytest.approx(0.20 * 0.25 * (0.49 * 0.51) ** 2 * 20)
    assert f["mo_res"] == pytest.approx(1 - 0.49)


def test_join_queue_rule_and_improve_rule():
    # joining the best bid (size 100): a 100-share print does not reach us, 130 fills 20 of us
    w = _window([(T, 0.49, 0.51, 100, 100)], [(T + 300, 0.49, 100, "SELL")])
    assert replay([w], BASE)[0].empty
    w = _window([(T, 0.49, 0.51, 100, 100)], [(T + 300, 0.49, 130, "SELL")])
    fills, _ = replay([w], BASE)
    assert len(fills) == 1 and fills.iloc[0]["shares"] == 20
    # partial: 110 printed -> 10 filled, then another print completes the quote
    w = _window(
        [(T, 0.49, 0.51, 100, 100)],
        [(T + 300, 0.49, 110, "SELL"), (T + 400, 0.48, 50, "SELL")],
    )
    fills, _ = replay([w], BASE)
    assert list(fills["shares"]) == [10, 10] and (fills["price"] == 0.49).all()
    # wide book: mid 0.50, best bid 0.45 -> bid_q 0.49 improves, first in line, a 5-lot fills 5
    w = _window([(T, 0.45, 0.55, 1000, 1000)], [(T + 300, 0.49, 5, "SELL")])
    fills, _ = replay([w], BASE)
    assert len(fills) == 1 and fills.iloc[0]["shares"] == 5
    # a BUY print at the bid price is not a bid fill; a SELL print above the bid is not either
    w = _window(
        [(T, 0.45, 0.55, 1000, 1000)],
        [(T + 300, 0.49, 50, "BUY"), (T + 310, 0.50, 50, "SELL")],
    )
    assert replay([w], BASE)[0].empty


def test_quotes_stay_one_tick_inside_opposite_best():
    # one-tick book 0.50/0.51, mid 0.505. With a zero half spread both raw quotes round to
    # 0.50; the ask is pushed to bid + tick so the pair never crosses or takes.
    w = _window(
        [(T, 0.50, 0.51, 10, 10)],
        [(T + 300, 0.50, 500, "SELL"), (T + 400, 0.51, 500, "BUY")],
    )
    fills, _ = replay([w], Cell(anchor="mid", half_spread=0.0, latency_ms=250))
    assert sorted(zip(fills["dir"], fills["price"], strict=True)) == [(-1.0, 0.51), (1.0, 0.50)]
    # a wide half spread quotes away from the book: neither print reaches the quotes
    assert replay([w], Cell(anchor="mid", half_spread=0.03, latency_ms=250))[0].empty


def test_stale_quote_is_picked_off_during_latency_and_requote_moves():
    # snapshot 1 mid 0.50; snapshot 2 (1 s later) mid 0.60. Our 0.51 ask from snapshot 1 is
    # live until the second snapshot's quotes go live at +1250 ms, so a BUY at 0.60 at +1100 ms
    # lifts it at 0.51: an adverse fill by construction.
    w = _window(
        [(T, 0.49, 0.51, 100, 100), (T + 1000, 0.59, 0.61, 100, 100)],
        [(T + 1100, 0.60, 200, "BUY")],
    )
    fills, res = replay([w], BASE)
    assert len(fills) == 1 and fills.iloc[0]["dir"] == -1 and fills.iloc[0]["price"] == 0.51
    assert fills.iloc[0]["mo_10s"] == pytest.approx(-(0.60 - 0.51))
    # sold 20 Up at 0.51, outcome Up: cash +10.2, settles -20
    assert res.iloc[0]["pnl_ex_rebate"] == pytest.approx(10.2 - 20)


def test_inventory_cap_skew_and_tau_min():
    books = [(T + k * 1000, 0.49, 0.51, 10, 10) for k in range(12)]
    prints = [(T + k * 1000 + 300, 0.49, 50, "SELL") for k in range(12)]
    cell = Cell(anchor="mid", half_spread=0.01, latency_ms=250, max_inventory=60, skew=0.0)
    fills, res = replay([_window(books, prints)], cell)
    assert res.iloc[0]["bought"] == 60  # three fills of 20, then the bid is withdrawn
    assert (fills["price"] == 0.49).all()
    # skew: after the first 20 the bid moves one tick down (skew 2 * 0.01 * 20/40) and the
    # 0.49 prints no longer reach it; a print through to 0.48 fills it behind the displayed size
    prints = [(T + k * 1000 + 300, 0.49 if k < 6 else 0.48, 50, "SELL") for k in range(12)]
    cell = Cell(anchor="mid", half_spread=0.01, latency_ms=250, max_inventory=40, skew=2.0)
    fills, res = replay([_window(books, prints)], cell)
    assert res.iloc[0]["bought"] == 40 and list(fills["price"]) == [0.49, 0.48]
    # tau_min: no quotes in the last 120 s
    late = [(END * 1000 - 100_000, 0.49, 0.51, 10, 10)]
    lp = [(END * 1000 - 99_000, 0.49, 50, "SELL")]
    assert not replay([_window(late, lp)], BASE)[0].empty
    assert replay([_window(late, lp)], Cell("mid", 0.01, 250, tau_min_s=120.0))[0].empty


def test_markout_table_and_summary():
    w = _window(
        [(T, 0.49, 0.51, 10, 10), (T + 20_000, 0.44, 0.46, 10, 10)],
        [(T + 300, 0.49, 50, "SELL")],
        outcome=0,
    )
    fills, res = replay([w], BASE)
    tbl = markout_by_tau(fills)
    assert len(tbl) == 1 and tbl.iloc[0]["n_fills"] == 1
    assert tbl.iloc[0]["mo_30s"] == pytest.approx(0.45 - 0.49)
    assert tbl.iloc[0]["mo_res"] == pytest.approx(-0.49)
    s = summarize_passive(res, fills)
    assert len(s) == 1 and s.iloc[0]["n_win"] == 1
    assert s.iloc[0]["pnl_ex_rebate"] == pytest.approx(-9.8)
    assert s.iloc[0]["mo_res"] == pytest.approx(-0.49)


# --- synthetic pipeline: trades derived from raw CLOB files and the report section ---


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rx_ts, msg in rows:
            fh.write(json.dumps({"rx_ts": rx_ts, "msg": msg}, separators=(",", ":")) + "\n")


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(3)
    root = str(tmp_path / "data")
    t_begin, t_end = START - 3600, END + 120
    px = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 0.6 * (1 / 31_536_000) ** 0.5, t_end - t_begin)))
    rtds = [
        (
            t * 1000 + 40,
            {
                "topic": topic,
                "type": "update",
                "timestamp": t * 1000 + 40,
                "payload": {"symbol": "btc/usd", "timestamp": t * 1000, "value": float(p)},
            },
        )
        for t, p in zip(range(t_begin, t_end), px, strict=True)
        for topic in ("crypto_prices_chainlink", "crypto_prices_twap_sixty")
    ]
    _write(os.path.join(root, "raw", "rtds", "20260216_03.jsonl"), rtds)
    meta = [
        (
            START * 1000 - 600_000,
            {
                "symbol": "btc",
                "start": START,
                "end": END,
                "slug": f"btc-updown-15m-{START}",
                "condition_id": "0xc0",
                "up_token": "u0",
                "down_token": "d0",
                "tick_size": 0.01,
                "fees_enabled": True,
                "fee_rate": 0.25,
                "fee_exponent": 2,
            },
        )
    ]
    _write(os.path.join(root, "raw", "windows", "20260216_03.jsonl"), meta)
    clob = []
    for t in range(START, END, 5):
        ms = t * 1000 + 300
        mid = 0.50 + 0.02 * np.sin(t / 60.0)
        bid, ask = round(mid - 0.01, 2), round(mid + 0.01, 2)
        clob.append(
            (
                ms,
                {
                    "event_type": "book",
                    "market": "0xc0",
                    "asset_id": "u0",
                    "bids": [{"price": f"{bid:.2f}", "size": "30"}],
                    "asks": [{"price": f"{ask:.2f}", "size": "30"}],
                },
            )
        )
        side = "SELL" if rng.random() < 0.5 else "BUY"
        clob.append(
            (
                ms + 2500,
                {
                    "event_type": "last_trade_price",
                    "market": "0xc0",
                    "asset_id": "u0",
                    "price": f"{bid if side == 'SELL' else ask:.2f}",
                    "size": "80",
                    "side": side,
                    "timestamp": str(ms + 2500),
                },
            )
        )
    clob.append(
        (
            END * 1000 + 90_000,
            {"event_type": "market_resolved", "market": "0xc0", "winning_asset_id": "u0"},
        )
    )
    _write(os.path.join(root, "raw", "clob", "20260216_03.jsonl"), clob)
    return root


def test_trades_are_derived_and_report_has_passive_section(dataset, tmp_path):
    counts = store.derive_day(dataset, "20260216")
    assert counts["trades"] == len(range(START, END, 5))
    trades = store.load_derived(dataset, "trades")
    assert set(trades["side"]) == {"BUY", "SELL"} and (trades["size"] == 80).all()
    ctx = build_contexts(
        store.load_derived(dataset, "windows"),
        store.load_derived(dataset, "feeds"),
        store.load_derived(dataset, "books"),
        store.load_derived(dataset, "resolutions"),
        "crypto_prices_chainlink",
    )
    assert ctx and ctx[0].tick_size == 0.01
    fills, res = passive.grid(ctx, trades, (BASE,))
    assert not fills.empty and (fills["shares"] == 20).all()
    assert (fills["price"] <= 0.99).all() and (fills["t_fill"] >= START * 1000 + 250).all()
    r = res.iloc[0]
    assert r["pnl"] == pytest.approx(r["pnl_ex_rebate"] + r["rebates"])

    settings = Settings(data_dir=dataset, reports_dir=str(tmp_path / "reports"))
    report, digest = build(settings)
    assert "## Passive quoting replay" in report and "passive best" in digest
    assert "| tau_bucket |" in report


def test_missing_size_counts_as_zero_volume():
    w = _window([(T, 0.45, 0.55, 10, 10)], [(T + 300, 0.49, float("nan"), "SELL")])
    w.sz = np.nan_to_num(w.sz, nan=0.0)  # what prepare() does to the tape
    assert replay([w], BASE)[0].empty
