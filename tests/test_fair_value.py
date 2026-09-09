import math

import numpy as np
import pytest

from updown.fair_value import SECONDS_PER_YEAR, p_up, realized_vol_annualized, taker_fee


def test_at_the_money_is_slightly_below_half():
    # Ito correction: with zero drift, P(S_T >= S_0) = Phi(-s/2) < 0.5
    p = p_up(100.0, 100.0, 0.5, 900)
    assert 0.49 < p < 0.5


def test_limits_at_expiry():
    assert p_up(101.0, 100.0, 0.5, 0) == 1.0
    assert p_up(99.0, 100.0, 0.5, 0) == 0.0
    assert p_up(100.0, 100.0, 0.5, 0) == 1.0  # ties resolve Up


def test_monotone_in_spot_and_symmetric_in_log_space():
    ps = [p_up(s, 100.0, 0.6, 600) for s in (99.0, 99.5, 100.0, 100.5, 101.0)]
    assert ps == sorted(ps)
    s = 0.6 * math.sqrt(600 / SECONDS_PER_YEAR)
    up = p_up(100.0 * math.exp(s), 100.0, 0.6, 600)
    down = p_up(100.0 * math.exp(-s), 100.0, 0.6, 600)
    assert abs((up - 0.5) + (down - 0.5) + 0.0) < 0.02  # near-symmetric around 0.5


def test_higher_vol_pulls_toward_half():
    far = p_up(100.3, 100.0, 0.2, 600)
    near = p_up(100.3, 100.0, 1.0, 600)
    assert far > near > 0.5


def test_realized_vol_recovers_simulated_sigma():
    rng = np.random.default_rng(0)
    sigma = 0.8
    n = 20_000
    dt = 1.0
    ts = np.cumsum(np.full(n, dt))
    r = rng.normal(0, sigma * math.sqrt(dt / SECONDS_PER_YEAR), n)
    px = 100.0 * np.exp(np.cumsum(r))
    est = realized_vol_annualized(ts, px)
    assert est == pytest.approx(sigma, rel=0.05)


def test_realized_vol_needs_enough_ticks():
    assert realized_vol_annualized(np.arange(5.0), np.ones(5) * 100) is None


def test_taker_fee_matches_documented_example():
    # 100 shares at 0.50 with rate 0.25, exponent 2 -> 1.5625
    assert taker_fee(100, 0.5, 0.25, 2) == pytest.approx(1.5625)
    assert taker_fee(100, 0.01, 0.25, 2) < 0.01
