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
    if len(px) < min_ticks:
        return None
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


def p_up_twap(
    spot: float,
    ref: float,
    sigma_ann: float,
    tau_s: float,
    window_s: float = 60.0,
    known_log_integral: float | None = None,
) -> float:
    """Probability that a time-weighted average price over the last `window_s` seconds of the
    window ends at or above `ref`, given the current spot.

    With X the log price as a driftless Brownian motion and Y the average of X over
    [T - w, T], conditional on time t with tau = T - t:
      tau >= w : Y - X_t ~ N(0, sigma^2 (tau - w + w/3))
      tau <  w : Y = (K + tau X_t + noise) / w, K the realized part of the integral over
                 [T - w, t], noise ~ N(0, sigma^2 tau^3 / 3)
    `known_log_integral` is K (integral of log price over [T - w, t], in seconds); when the
    caller has no ticks it is approximated by (w - tau) * log(spot). Ito drift over these
    horizons shifts the argument by under 0.2 % and is omitted.
    """
    if spot <= 0 or ref <= 0:
        raise ValueError("prices must be positive")
    x_t, k = math.log(spot), math.log(ref)
    var_s = sigma_ann * sigma_ann / SECONDS_PER_YEAR  # variance per second
    if tau_s >= window_s:
        mean = x_t
        var = var_s * (tau_s - window_s + window_s / 3.0)
    else:
        tau = max(tau_s, 0.0)
        known = known_log_integral if known_log_integral is not None else (window_s - tau) * x_t
        mean = (known + tau * x_t) / window_s
        var = var_s * tau**3 / (3.0 * window_s * window_s)
    if var <= 0 or sigma_ann <= 0:
        return 1.0 if mean >= k else 0.0
    return norm_cdf((mean - k) / math.sqrt(var))


def log_integral(ts_s: np.ndarray, px: np.ndarray, t0: float, t1: float) -> float | None:
    """Integral of log(price) over [t0, t1] from irregular ticks, piecewise constant.

    Returns None when no tick is at or before t0 (the level at the start is unknown).
    """
    ts_s = np.asarray(ts_s, dtype=float)
    px = np.asarray(px, dtype=float)
    if t1 <= t0 or len(px) == 0:
        return 0.0 if t1 <= t0 else None
    i0 = int(np.searchsorted(ts_s, t0, side="right")) - 1
    if i0 < 0:
        return None
    i1 = int(np.searchsorted(ts_s, t1, side="right"))
    times = np.concatenate(([t0], ts_s[i0 + 1 : i1], [t1]))
    levels = np.log(px[i0:i1])
    return float(np.sum(np.diff(times) * levels))
