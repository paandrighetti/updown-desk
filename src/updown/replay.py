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

import duckdb
import numpy as np
import pandas as pd

from .fair_value import fee_per_share, log_integral, p_up, p_up_twap, realized_vol_annualized


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
    settlement: str = "twap60"
    twap_window_s: float = 60.0

    def spot_at(self, t_ms: int) -> float | None:
        i = np.searchsorted(self.feed_rx, t_ms, side="right") - 1
        return float(self.feed_px[i]) if i >= 0 else None

    def fair_up(self, t_ms: int) -> float | None:
        spot = self.spot_at(t_ms)
        if spot is None or self.ref_price is None or self.sigma is None:
            return None
        tau = self.end - t_ms / 1000.0
        if self.settlement != "twap60":
            return p_up(spot, self.ref_price, self.sigma, tau)
        known = None
        if tau < self.twap_window_s:
            known = log_integral(
                self.feed_rx / 1000.0, self.feed_px, self.end - self.twap_window_s, t_ms / 1000.0
            )
        return p_up_twap(spot, self.ref_price, self.sigma, tau, self.twap_window_s, known)


class _FeedIndex:
    """One symbol's reference ticks as sorted numpy arrays, for cheap window slicing."""

    def __init__(self, g: pd.DataFrame) -> None:
        g = g.sort_values("obs_ts")
        self.obs = g["obs_ts"].to_numpy(dtype=np.int64)
        self.px = g["px"].to_numpy(dtype=float)
        rx = g[["rx_ts", "px"]].sort_values("rx_ts")
        self.rx = rx["rx_ts"].to_numpy(dtype=np.int64)
        self.rx_px = rx["px"].to_numpy(dtype=float)

    def by_obs(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        i, j = np.searchsorted(self.obs, [lo, hi], side="left")
        return self.obs[i:j], self.px[i:j]

    def by_rx(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        i, j = np.searchsorted(self.rx, [lo, hi], side="left")
        return self.rx[i:j], self.rx_px[i:j]


def resolution_map(resolutions: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """condition_id -> (winning_token, source). Stream events win over a Gamma sweep."""
    if resolutions.empty:
        return {}
    df = resolutions.copy()
    if "source" not in df:
        df["source"] = "market_resolved"
    df["source"] = df["source"].fillna("market_resolved")  # stream rows carry no source column
    df["rank"] = (df["source"] != "market_resolved").astype(int)
    df = df.sort_values(["rank", "rx_ts"]).drop_duplicates("condition_id", keep="first")
    return {
        c: (t, src)
        for c, t, src in zip(df["condition_id"], df["winning_token"], df["source"], strict=True)
    }


def build_contexts(
    windows: pd.DataFrame,
    feeds: pd.DataFrame,
    books: pd.DataFrame,
    resolutions: pd.DataFrame,
    ref_feed: str,
    vol_lookback_s: int = 3600,
    spot_feed: str | None = None,
    settlement: str = "twap60",
) -> list[WindowContext]:
    """Contexts per window. `ref_feed` gives the strike (value at start) and the settlement
    fallback (value at end); `spot_feed` (default: same topic) is the diffusing state used
    by the fair value and the volatility estimate."""
    out: list[WindowContext] = []
    if windows.empty or feeds.empty:
        return out
    spot_feed = spot_feed or ref_feed
    ref = feeds[feeds["topic"] == ref_feed]
    spot = feeds[feeds["topic"] == spot_feed]
    index = {sym: _FeedIndex(g) for sym, g in ref.groupby("symbol", observed=True)}
    spot_index = (
        index
        if spot_feed == ref_feed
        else {sym: _FeedIndex(g) for sym, g in spot.groupby("symbol", observed=True)}
    )
    res = resolution_map(resolutions)
    books_by_token = (
        {t: g.reset_index(drop=True) for t, g in books.groupby("token")} if not books.empty else {}
    )
    lookback_ms = vol_lookback_s * 1000

    for w in windows.itertuples(index=False):
        f = index.get(w.symbol)
        fs = spot_index.get(w.symbol)
        if f is None or fs is None:
            continue
        start_ms, end_ms = w.start * 1000, w.end * 1000
        obs, px = f.by_obs(start_ms, end_ms + 60_001)
        ref_price = ref_delay = None
        if len(obs):
            ref_price = float(px[0])
            ref_delay = (int(obs[0]) - start_ms) / 1000.0
        h_obs, h_px = fs.by_obs(start_ms - lookback_ms, start_ms)
        sigma = realized_vol_annualized(h_obs / 1000.0, h_px)

        outcome, source = None, "none"
        win_tok, src = res.get(w.condition_id, (None, None))
        if win_tok == w.up_token:
            outcome, source = 1, src
        elif win_tok == w.down_token:
            outcome, source = 0, src
        elif ref_price is not None:
            e_obs, e_px = f.by_obs(start_ms, end_ms + 1)
            if len(e_obs) and (end_ms - int(e_obs[-1])) <= 5_000:
                outcome = int(float(e_px[-1]) >= ref_price)
                source = f"feed:{ref_feed}"

        rx, rx_px = fs.by_rx(start_ms - lookback_ms, end_ms + 1)
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
                feed_rx=rx,
                feed_px=rx_px,
                ref_price=ref_price,
                ref_delay_s=ref_delay,
                sigma=sigma,
                outcome=outcome,
                outcome_source=source,
                books={
                    "up": books_by_token.get(w.up_token, pd.DataFrame()),
                    "down": books_by_token.get(w.down_token, pd.DataFrame()),
                },
                settlement=settlement,
            )
        )
    return out


def _fill(
    rx: np.ndarray,
    ask: np.ndarray,
    size: np.ndarray,
    i: int,
    t_ms: int,
    latency_ms: int,
    limit: float,
    max_shares: float,
):
    """First snapshot received at or after t + latency; fill only if its ask is within the limit."""
    j = int(np.searchsorted(rx, t_ms + latency_ms, side="left")) if latency_ms else i
    if j >= len(rx):
        return None
    a, sz = ask[j], size[j]
    if np.isnan(a) or a > limit or np.isnan(sz) or sz <= 0:
        return None
    return int(rx[j]), float(a), float(min(max_shares, sz))


def replay(contexts: list[WindowContext], params: Params) -> pd.DataFrame:
    rows: list[dict] = []
    for c in contexts:
        if c.outcome is None or c.sigma is None or c.ref_price is None:
            continue
        for side in ("up", "down"):
            book = c.books[side]
            if book.empty:
                continue
            rx = book["rx_ts"].to_numpy(dtype=np.int64)
            ask = book["ask"].to_numpy(dtype=float)
            size = book["ask_size"].to_numpy(dtype=float)
            lo = int(np.searchsorted(rx, c.start * 1000, side="left"))
            hi = int(np.searchsorted(rx, c.end * 1000, side="right"))
            for i in range(lo, hi):
                if np.isnan(ask[i]) or np.isnan(size[i]) or size[i] <= 0:
                    continue
                t = int(rx[i])
                fu = c.fair_up(t)
                if fu is None:
                    continue
                fair = fu if side == "up" else 1.0 - fu
                a = float(ask[i])
                fee_ps = fee_per_share(a, c.fee_rate, c.fee_exponent, c.fees_enabled)
                edge = fair - a - fee_ps
                if edge < params.threshold:
                    continue
                filled = _fill(rx, ask, size, i, t, params.latency_ms, a, params.max_shares)
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
    windows: pd.DataFrame, feeds: pd.DataFrame | list[str], resolutions: pd.DataFrame
) -> pd.DataFrame:
    """For each RTDS topic: how often sign(last - first) matches the resolved outcome.

    This is the empirical answer to "which feed does the resolution actually follow".
    `feeds` is either an in-memory frame or the list of derived parquet files; only the
    first and last tick of each (window, topic) pair leave DuckDB.
    """
    if windows.empty or resolutions.empty:
        return pd.DataFrame()
    if isinstance(feeds, pd.DataFrame):
        if feeds.empty:
            return pd.DataFrame()
        source = "feeds_df"
        feeds_df = feeds  # noqa: F841 - referenced by name in the DuckDB query below
    else:
        if not feeds:
            return pd.DataFrame()
        source = f"read_parquet({feeds!r}, union_by_name=true)"
    win = windows[["condition_id", "symbol", "start", "end"]]  # noqa: F841 - same
    endpoints = duckdb.sql(
        f"""
        SELECT w.condition_id, f.topic,
               arg_min(f.px, f.obs_ts) AS first_px, arg_max(f.px, f.obs_ts) AS last_px,
               count(*) AS n
        FROM win w
        JOIN {source} f
          ON f.symbol = w.symbol
         AND f.obs_ts >= w.start * 1000 AND f.obs_ts <= w."end" * 1000
        GROUP BY 1, 2
        """
    ).df()
    if endpoints.empty:
        return pd.DataFrame()
    res = resolution_map(resolutions)
    tokens = dict(
        zip(
            windows["condition_id"],
            zip(windows["up_token"], windows["down_token"], strict=True),
            strict=True,
        )
    )
    rows = []
    for e in endpoints.itertuples(index=False):
        if e.n < 2:
            continue
        win_tok, _src = res.get(e.condition_id, (None, None))
        up, down = tokens[e.condition_id]
        if win_tok not in (up, down):
            continue
        pred = int(float(e.last_px) >= float(e.first_px))
        rows.append({"topic": e.topic, "agree": pred == int(win_tok == up)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby("topic")["agree"]
        .agg(n="count", agreement="mean")
        .reset_index()
        .sort_values("topic")
        .reset_index(drop=True)
    )
