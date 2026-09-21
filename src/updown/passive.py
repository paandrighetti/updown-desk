"""Replay a passive two-sided quoting strategy over recorded books and the trade tape.

The taker replay asked whether the market is mispriced against the model; it is not. This
module asks the question a liquidity provider would ask instead: is the flow that hits
resting quotes in these windows benign enough to earn the spread plus the maker rebate,
and how does that depend on time to expiry?

Everything is causal in receive time. At each book snapshot received at t the strategy
computes two quotes on the Up token; they go live at t + latency and stay live until the
quotes of the next snapshot replace them (again after latency) or the window ends, so a
stale quote can be picked off during the latency, as it would be in production. A live
bid is filled by taker SELL prints at or below it, a live ask by taker BUY prints at or
above it, on the same token. Queue rule: a quote that improves the displayed best level is
first in line; a quote that joins the best level, or sits behind it, waits for the
displayed best size to print at or through its price first (for a deeper level the
displayed best size stands in for the unknown size resting there). Matches arriving
through the complementary token (mint and merge against Down orders) are not observed and
are ignored. Every one of these choices can only under-count fills.

Quotes are anchor +/- half_spread, shifted against inventory by a linear skew (the
discrete analogue of the reservation price in Avellaneda and Stoikov, 2008), rounded away
from the anchor to the tick and kept one tick inside the opposite best so that a quote is
never tighter than intended and never takes. The anchor is either the market mid or the
model fair value. Inventory is held to resolution and settles at the outcome; a short Up
position is a long Down position and settles the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fair_value import fee_per_share
from .replay import WindowContext

MARKOUTS_S = (10, 30, 60)
CELL_KEYS = ("anchor", "half_spread", "latency_ms", "tau_min_s")


@dataclass(frozen=True)
class Cell:
    anchor: str  # "mid" or "model"
    half_spread: float  # price units
    latency_ms: int
    tau_min_s: float = 0.0  # no new quotes once time to expiry is below this
    quote_size: float = 20.0
    max_inventory: float = 100.0
    skew: float = 1.0  # quotes shift by skew * half_spread * inventory / max_inventory
    rebate_share: float = 0.20  # maker rebate as a share of the taker fee paid on the fill

    def key(self) -> dict:
        return {k: getattr(self, k) for k in CELL_KEYS}


@dataclass
class _Window:
    symbol: str
    start: int
    end: int
    outcome: int
    tick: float
    fee: tuple[float | None, float | None, bool | None]
    rx: np.ndarray  # book snapshots, receive ms
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    mid: np.ndarray
    fair: np.ndarray
    t_rx: np.ndarray  # prints, receive ms
    p: np.ndarray
    sz: np.ndarray
    side: np.ndarray  # +1 taker BUY, -1 taker SELL


def prepare(contexts: list[WindowContext], trades: pd.DataFrame) -> list[_Window]:
    """Per-window arrays for the Up token. Windows without outcome, volatility, book or
    prints are dropped; every cell of the grid replays the same set of windows."""
    by_token = (
        {t: g.reset_index(drop=True) for t, g in trades.groupby("token", observed=True)}
        if not trades.empty
        else {}
    )
    out: list[_Window] = []
    for c in contexts:
        if c.outcome is None or c.sigma is None or c.ref_price is None:
            continue
        book = c.books["up"]
        if book.empty:
            continue
        rx_all = book["rx_ts"].to_numpy(dtype=np.int64)
        lo = int(np.searchsorted(rx_all, c.start * 1000, side="left"))
        hi = int(np.searchsorted(rx_all, c.end * 1000, side="right"))
        if hi <= lo:
            continue
        rx = rx_all[lo:hi]
        bid = book["bid"].to_numpy(dtype=float)[lo:hi]
        ask = book["ask"].to_numpy(dtype=float)[lo:hi]
        fair = np.array([np.nan if (f := c.fair_up(int(t))) is None else f for t in rx])
        tape = by_token.get(c.up_token)
        if tape is None or tape.empty:
            continue  # no tape for this window (data gap or dead market): nothing to learn
        t_all = tape["rx_ts"].to_numpy(dtype=np.int64)
        a, b = np.searchsorted(t_all, [c.start * 1000, c.end * 1000 + 1], side="left")
        if b <= a:
            continue
        t_rx = t_all[a:b]
        p = tape["price"].to_numpy(dtype=float)[a:b]
        sz = np.nan_to_num(tape["size"].to_numpy(dtype=float)[a:b], nan=0.0)
        side = np.where(tape["side"].to_numpy()[a:b] == "BUY", 1.0, -1.0)
        out.append(
            _Window(
                symbol=c.symbol,
                start=c.start,
                end=c.end,
                outcome=int(c.outcome),
                tick=c.tick_size,
                fee=(c.fee_rate, c.fee_exponent, c.fees_enabled),
                rx=rx,
                bid=bid,
                ask=ask,
                bid_size=book["bid_size"].to_numpy(dtype=float)[lo:hi],
                ask_size=book["ask_size"].to_numpy(dtype=float)[lo:hi],
                mid=(bid + ask) / 2.0,
                fair=fair,
                t_rx=t_rx,
                p=p,
                sz=sz,
                side=side,
            )
        )
    return out


def _floor_tick(x: float, tick: float) -> float:
    """Round a bid down to the tick: never quote tighter than intended."""
    return round(math.floor(x / tick + 1e-9) * tick, 6)


def _ceil_tick(x: float, tick: float) -> float:
    return round(math.ceil(x / tick - 1e-9) * tick, 6)


def _mid_after(w: _Window, t_ms: int, offset_s: int) -> float:
    k = int(np.searchsorted(w.rx, t_ms + offset_s * 1000, side="right")) - 1
    return float(w.mid[k]) if k >= 0 else float("nan")


def _replay_window(w: _Window, cell: Cell, fills: list[dict]) -> dict:
    q = cash = rebates = bought = sold = 0.0
    n_quotes = 0
    anchor = w.mid if cell.anchor == "mid" else w.fair
    end_ms = w.end * 1000
    n = len(w.rx)
    for i in range(n):
        t_live = int(w.rx[i]) + cell.latency_ms
        t_dead = min(int(w.rx[i + 1]) + cell.latency_ms if i + 1 < n else end_ms, end_ms)
        if t_live >= t_dead:
            continue  # superseded before going live, or window over
        if w.end - w.rx[i] / 1000.0 < cell.tau_min_s:
            continue
        a, b, k = float(anchor[i]), float(w.bid[i]), float(w.ask[i])
        if np.isnan(a) or np.isnan(b) or np.isnan(k):
            continue
        shift = cell.skew * cell.half_spread * (q / cell.max_inventory)
        bid_q = min(
            _floor_tick(a - cell.half_spread - shift, w.tick), _floor_tick(k - w.tick, w.tick)
        )
        ask_q = max(
            _ceil_tick(a + cell.half_spread - shift, w.tick), _ceil_tick(b + w.tick, w.tick)
        )
        bid_q, ask_q = max(bid_q, w.tick), min(ask_q, 1.0 - w.tick)
        want_bid = q < cell.max_inventory and bid_q < ask_q
        want_ask = q > -cell.max_inventory and bid_q < ask_q
        if not (want_bid or want_ask):
            continue
        n_quotes += 1
        j0, j1 = np.searchsorted(w.t_rx, [t_live, t_dead], side="right")
        if j0 == j1:
            continue
        queue_bid = float(w.bid_size[i]) if bid_q <= b else 0.0
        queue_ask = float(w.ask_size[i]) if ask_q >= k else 0.0
        if np.isnan(queue_bid):
            queue_bid = 0.0
        if np.isnan(queue_ask):
            queue_ask = 0.0
        cum_sell = cum_buy = f_bid = f_ask = 0.0
        for j in range(int(j0), int(j1)):
            s, p, v = w.side[j], float(w.p[j]), float(w.sz[j])
            if s < 0 and want_bid and p <= bid_q and f_bid < cell.quote_size:
                cum_sell += v
                new = min(cell.quote_size, max(0.0, cum_sell - queue_bid)) - f_bid
                if new > 0:
                    f_bid += new
                    _record(w, cell, fills, int(w.t_rx[j]), 1.0, bid_q, new, a)
            elif s > 0 and want_ask and p >= ask_q and f_ask < cell.quote_size:
                cum_buy += v
                new = min(cell.quote_size, max(0.0, cum_buy - queue_ask)) - f_ask
                if new > 0:
                    f_ask += new
                    _record(w, cell, fills, int(w.t_rx[j]), -1.0, ask_q, new, a)
        if f_bid:
            rebates += cell.rebate_share * fee_per_share(bid_q, *w.fee) * f_bid
        if f_ask:
            rebates += cell.rebate_share * fee_per_share(ask_q, *w.fee) * f_ask
        q += f_bid - f_ask
        cash += ask_q * f_ask - bid_q * f_bid
        bought += f_bid
        sold += f_ask
    pnl_ex = cash + q * w.outcome
    return {
        "symbol": w.symbol,
        "start": w.start,
        "n_quotes": n_quotes,
        "bought": bought,
        "sold": sold,
        "q_final": q,
        "cash": cash,
        "rebates": rebates,
        "pnl_ex_rebate": pnl_ex,
        "pnl": pnl_ex + rebates,
        **cell.key(),
    }


def _record(
    w: _Window,
    cell: Cell,
    fills: list[dict],
    t_ms: int,
    direction: float,
    price: float,
    shares: float,
    anchor: float,
) -> None:
    row = {
        "symbol": w.symbol,
        "start": w.start,
        "t_fill": t_ms,
        "tau_s": w.end - t_ms / 1000.0,
        "dir": direction,  # +1 bought Up (bid filled), -1 sold Up (ask filled)
        "price": price,
        "shares": shares,
        "anchor_px": anchor,
        "rebate": cell.rebate_share * fee_per_share(price, *w.fee) * shares,
        "mo_res": direction * (w.outcome - price),
        **cell.key(),
    }
    for off in MARKOUTS_S:
        row[f"mo_{off}s"] = direction * (_mid_after(w, t_ms, off) - price)
    fills.append(row)


def replay(windows: list[_Window], cell: Cell) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fills, per-window results) for one cell."""
    fills: list[dict] = []
    rows = [_replay_window(w, cell, fills) for w in windows]
    return pd.DataFrame(fills), pd.DataFrame(rows)


def grid(
    contexts: list[WindowContext], trades: pd.DataFrame, cells: tuple[Cell, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = prepare(contexts, trades)
    if not windows:
        return pd.DataFrame(), pd.DataFrame()
    f_frames, w_frames = zip(*(replay(windows, c) for c in cells), strict=True)
    fills = [f for f in f_frames if not f.empty]
    return (
        pd.concat(fills, ignore_index=True) if fills else pd.DataFrame(),
        pd.concat(w_frames, ignore_index=True),
    )


def select(df: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    for k, v in cell.key().items():
        mask &= (df[k] == v).to_numpy()
    return df[mask]


def markout_by_tau(fills: pd.DataFrame, edges=(0, 60, 120, 300, 600, 900)) -> pd.DataFrame:
    """Share-weighted markouts of the fills of one cell by time-to-expiry bucket.

    `mo_res` is the realized per-share result of holding the fill to resolution; the
    shorter markouts measure how quickly the mid moves against the fill (adverse
    selection). Negative values mean the flow hitting the quotes was informed.
    """
    if fills.empty:
        return pd.DataFrame()
    df = fills.copy()
    df["tau_bucket"] = pd.cut(df["tau_s"], edges, right=False, include_lowest=True)
    cols = [f"mo_{o}s" for o in MARKOUTS_S] + ["mo_res"]
    rows = []
    for bucket, g in df.groupby("tau_bucket", observed=True):
        sh = g["shares"].to_numpy()
        row = {"tau_bucket": str(bucket), "n_fills": int(len(g)), "shares": float(sh.sum())}
        for col in cols:
            v = g[col].to_numpy()
            ok = ~np.isnan(v)
            row[col] = float((v[ok] * sh[ok]).sum() / sh[ok].sum()) if ok.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
