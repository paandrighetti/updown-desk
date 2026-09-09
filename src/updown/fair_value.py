"""Fair value of an Up/Down contract and the cost model, as pure functions.

Model: driftless geometric Brownian motion over the remaining life of the window.
The contract pays 1 if S_T >= S_ref. With s = sigma * sqrt(tau):

    P(S_T >= S_ref | S_t) = Phi( ln(S_t / S_ref) / s - s / 2 )

Zero drift is a deliberate simplification: over 15 minutes the drift term is orders of
magnitude below the diffusion term. Reference: Black and Scholes (1973), digital option
limit; the s/2 term is the Ito correction.
"""

from __future__ import annotations

import math

import numpy as np

SECONDS_PER_YEAR = 365 * 24 * 3600


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def p_up(spot: float, ref: float, sigma_ann: float, tau_s: float) -> float:
    """Probability that the price at expiry is at or above the reference price."""
    if spot <= 0 or ref <= 0:
        raise ValueError("prices must be positive")
    if tau_s <= 0 or sigma_ann <= 0:
        return 1.0 if spot >= ref else 0.0
    s = sigma_ann * math.sqrt(tau_s / SECONDS_PER_YEAR)
    return norm_cdf(math.log(spot / ref) / s - s / 2.0)


def realized_vol_annualized(ts_s: np.ndarray, px: np.ndarray, min_ticks: int = 30) -> float | None:
    """Realized variance estimator on irregular ticks: sum(r^2) / sum(dt), annualized.

    Using tick spacing directly avoids the zero-return bias of resampling a slow feed
    on a fixed grid. Duplicate timestamps are dropped.
    """
    ts_s = np.asarray(ts_s, dtype=float)
    px = np.asarray(px, dtype=float)
    keep = np.concatenate(([True], np.diff(ts_s) > 0))
    ts_s, px = ts_s[keep], px[keep]
    if len(px) < min_ticks:
        return None
    r2 = np.diff(np.log(px)) ** 2
    dt = np.diff(ts_s)
    var_per_s = r2.sum() / dt.sum()
    return float(math.sqrt(var_per_s * SECONDS_PER_YEAR)) if var_per_s > 0 else None


def taker_fee(shares: float, price: float, rate: float, exponent: float) -> float:
    """Polymarket taker fee: shares * rate * (p * (1 - p)) ** exponent, in collateral units.

    Parameters come from the market's fee_schedule on Gamma. With rate 0.25 and exponent 2
    a 100-share trade at 0.50 costs 1.5625, matching the figure quoted in Polymarket's
    documentation examples. Verify against the live fee_schedule before publishing numbers.
    """
    return shares * rate * (price * (1.0 - price)) ** exponent


def fee_per_share(
    price: float, rate: float | None, exponent: float | None, enabled: bool | None
) -> float:
    if not enabled or rate is None or exponent is None:
        return 0.0
    return taker_fee(1.0, price, rate, exponent)
