"""Replay a fair-value taker strategy over recorded books and reference prices.

Everything is evaluated in receive-time order. A decision at time t only uses messages
received at or before t; the fill is checked against the first book snapshot received at
or after t + latency. This is what makes the result falsifiable rather than flattering.

Strategy (deliberately simple, one trade per side per window, hold to resolution):
  edge = fair_value(side) - best_ask(side) - taker_fee_per_share(best_ask)
  if edge >= threshold: send a marketable limit at best_ask, size <= displayed size
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fair_value import fee_per_share, p_up, realized_vol_annualized


@dataclass(frozen=True)
class Params:
    threshold: float
    latency_ms: int
    max_shares: float = 100.0


@dataclass
class WindowContext:
    symbol: str
    start: int
    end: int
    up_token: str
    down_token: str
    fee_rate: float | None
    fee_exponent: float | None
    fees_enabled: bool | None
    feed_rx: np.ndarray  # receive time ms, sorted
    feed_px: np.ndarray
    ref_price: float | None
    ref_delay_s: float | None
    sigma: float | None
    outcome: int | None  # 1 = Up, 0 = Down
    outcome_source: str
    books: dict[str, pd.DataFrame] = field(default_factory=dict)  # side -> top of book rows

    def spot_at(self, t_ms: int) -> float | None:
        i = np.searchsorted(self.feed_rx, t_ms, side="right") - 1
        return float(self.feed_px[i]) if i >= 0 else None

    def fair_up(self, t_ms: int) -> float | None:
        spot = self.spot_at(t_ms)
        if spot is None or self.ref_price is None or self.sigma is None:
            return None
        return p_up(spot, self.ref_price, self.sigma, self.end - t_ms / 1000.0)


def build_contexts(
    windows: pd.DataFrame,
    feeds: pd.DataFrame,
    books: pd.DataFrame,
    resolutions: pd.DataFrame,
    ref_feed: str,
    vol_lookback_s: int = 3600,
) -> list[WindowContext]:
    out: list[WindowContext] = []
    if windows.empty or feeds.empty:
        return out
    ref = feeds[feeds["topic"] == ref_feed]
    res = (
        {}
        if resolutions.empty
        else dict(zip(resolutions["condition_id"], resolutions["winning_token"], strict=True))
    )
    books_by_token = (
        {t: g.reset_index(drop=True) for t, g in books.groupby("token")} if not books.empty else {}
    )

    for w in windows.itertuples(index=False):
        f = ref[ref["symbol"] == w.symbol]
        start_ms, end_ms = w.start * 1000, w.end * 1000
        in_win = f[(f["obs_ts"] >= start_ms) & (f["obs_ts"] <= end_ms + 60_000)]
        ref_price = ref_delay = None
        if not in_win.empty:
            first = in_win.iloc[0]
            ref_price = float(first["px"])
            ref_delay = (first["obs_ts"] - start_ms) / 1000.0
        hist = f[(f["obs_ts"] >= start_ms - vol_lookback_s * 1000) & (f["obs_ts"] < start_ms)]
        sigma = realized_vol_annualized(hist["obs_ts"].to_numpy() / 1000.0, hist["px"].to_numpy())

        outcome, source = None, "none"
        win_tok = res.get(w.condition_id)
        if win_tok == w.up_token:
            outcome, source = 1, "market_resolved"
        elif win_tok == w.down_token:
            outcome, source = 0, "market_resolved"
        elif ref_price is not None:
            at_end = f[(f["obs_ts"] <= end_ms) & (f["obs_ts"] >= start_ms)]
            if not at_end.empty and (end_ms - at_end.iloc[-1]["obs_ts"]) <= 5_000:
                outcome = int(float(at_end.iloc[-1]["px"]) >= ref_price)
                source = f"feed:{ref_feed}"

        live = f[(f["rx_ts"] >= start_ms - vol_lookback_s * 1000) & (f["rx_ts"] <= end_ms)]
        out.append(
            WindowContext(
                symbol=w.symbol,
                start=int(w.start),
                end=int(w.end),
                up_token=w.up_token,
                down_token=w.down_token,
                fee_rate=None if pd.isna(w.fee_rate) else float(w.fee_rate),
                fee_exponent=None if pd.isna(w.fee_exponent) else float(w.fee_exponent),
                fees_enabled=None if pd.isna(w.fees_enabled) else bool(w.fees_enabled),
                feed_rx=live["rx_ts"].to_numpy(dtype=np.int64),
                feed_px=live["px"].to_numpy(dtype=float),
                ref_price=ref_price,
                ref_delay_s=ref_delay,
                sigma=sigma,
                outcome=outcome,
                outcome_source=source,
                books={
                    "up": books_by_token.get(w.up_token, pd.DataFrame()),
                    "down": books_by_token.get(w.down_token, pd.DataFrame()),
                },
            )
        )
    return out


def _fill(book: pd.DataFrame, i: int, t_ms: int, latency_ms: int, limit: float, max_shares: float):
    """First snapshot received at or after t + latency; fill only if its ask is within the limit."""
    j = (
        int(np.searchsorted(book["rx_ts"].to_numpy(), t_ms + latency_ms, side="left"))
        if latency_ms
        else i
    )
    if j >= len(book):
        return None
    row = book.iloc[j]
    if pd.isna(row["ask"]) or row["ask"] > limit or not row["ask_size"] or row["ask_size"] <= 0:
        return None
    return int(row["rx_ts"]), float(row["ask"]), float(min(max_shares, row["ask_size"]))


def replay(contexts: list[WindowContext], params: Params) -> pd.DataFrame:
    rows: list[dict] = []
    for c in contexts:
        if c.outcome is None or c.sigma is None or c.ref_price is None:
            continue
        for side in ("up", "down"):
            book = c.books[side]
            if book.empty:
                continue
            arr_ts = book["rx_ts"].to_numpy()
            lo = int(np.searchsorted(arr_ts, c.start * 1000, side="left"))
            hi = int(np.searchsorted(arr_ts, c.end * 1000, side="right"))
            for i in range(lo, hi):
                row = book.iloc[i]
                if pd.isna(row["ask"]) or not row["ask_size"]:
                    continue
                t = int(row["rx_ts"])
                fu = c.fair_up(t)
                if fu is None:
                    continue
                fair = fu if side == "up" else 1.0 - fu
                ask = float(row["ask"])
                fee_ps = fee_per_share(ask, c.fee_rate, c.fee_exponent, c.fees_enabled)
                edge = fair - ask - fee_ps
                if edge < params.threshold:
                    continue
                filled = _fill(book, i, t, params.latency_ms, ask, params.max_shares)
                if filled is None:
                    break  # one attempt per side per window
                t_fill, price, shares = filled
                fee = fee_per_share(price, c.fee_rate, c.fee_exponent, c.fees_enabled) * shares
                won = (c.outcome == 1) == (side == "up")
                pnl = shares * (1.0 - price) - fee if won else -shares * price - fee
                rows.append(
                    {
                        "symbol": c.symbol,
                        "start": c.start,
                        "side": side,
                        "t_signal": t,
                        "t_fill": t_fill,
                        "tau_s": c.end - t_fill / 1000.0,
                        "price": price,
                        "shares": shares,
                        "fair": fair,
                        "edge": edge,
                        "fee": fee,
                        "won": won,
                        "pnl": pnl,
                        "outcome_source": c.outcome_source,
                        "threshold": params.threshold,
                        "latency_ms": params.latency_ms,
                    }
                )
                break
    return pd.DataFrame(rows)


def grid(
    contexts: list[WindowContext], thresholds, latencies_ms, max_shares: float = 100.0
) -> pd.DataFrame:
    frames = [
        replay(contexts, Params(threshold=th, latency_ms=lat, max_shares=max_shares))
        for th in thresholds
        for lat in latencies_ms
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def checkpoints(contexts: list[WindowContext], offsets_s=(300, 600, 840)) -> pd.DataFrame:
    """Model probability versus market mid at fixed times inside each window, with the outcome."""
    rows: list[dict] = []
    for c in contexts:
        if c.outcome is None:
            continue
        book = c.books["up"]
        if book.empty:
            continue
        arr_ts = book["rx_ts"].to_numpy()
        for off in offsets_s:
            t = (c.start + off) * 1000
            i = int(np.searchsorted(arr_ts, t, side="right")) - 1
            if i < 0:
                continue
            row = book.iloc[i]
            if pd.isna(row["bid"]) or pd.isna(row["ask"]):
                continue
            fu = c.fair_up(t)
            rows.append(
                {
                    "symbol": c.symbol,
                    "start": c.start,
                    "offset_s": off,
                    "model_p": fu,
                    "market_mid": (float(row["bid"]) + float(row["ask"])) / 2.0,
                    "spread": float(row["ask"]) - float(row["bid"]),
                    "outcome": c.outcome,
                }
            )
    return pd.DataFrame(rows)


def feed_agreement(
    windows: pd.DataFrame, feeds: pd.DataFrame, resolutions: pd.DataFrame
) -> pd.DataFrame:
    """For each RTDS topic: how often sign(end - start) matches the resolved outcome.

    This is the empirical answer to "which feed does the resolution actually follow".
    """
    if windows.empty or feeds.empty or resolutions.empty:
        return pd.DataFrame()
    res = dict(zip(resolutions["condition_id"], resolutions["winning_token"], strict=True))
    rows = []
    for w in windows.itertuples(index=False):
        win_tok = res.get(w.condition_id)
        if win_tok not in (w.up_token, w.down_token):
            continue
        outcome = int(win_tok == w.up_token)
        for topic, f in feeds[feeds["symbol"] == w.symbol].groupby("topic"):
            s = f[(f["obs_ts"] >= w.start * 1000) & (f["obs_ts"] <= w.end * 1000)]
            if len(s) < 2:
                continue
            pred = int(float(s.iloc[-1]["px"]) >= float(s.iloc[0]["px"]))
            rows.append(
                {"topic": topic, "symbol": w.symbol, "start": w.start, "agree": pred == outcome}
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("topic")["agree"].agg(n="count", agreement="mean").reset_index()
