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


def test_realized_vol_handles_empty_and_single_inputs():
    assert realized_vol_annualized(np.array([]), np.array([])) is None
    assert realized_vol_annualized(np.array([1.0]), np.array([100.0])) is None


def test_twap_reduces_to_shorter_horizon_far_from_expiry():
    from updown.fair_value import p_up_twap

    # far from expiry the average over the last 60 s has less variance than the spot at expiry
    spot_like = p_up(100.3, 100.0, 0.6, 600)
    twap = p_up_twap(100.3, 100.0, 0.6, 600, window_s=60.0)
    assert twap > spot_like > 0.5


def test_twap_uses_realized_part_near_expiry():
    from updown.fair_value import p_up_twap

    # 20 s left, 40 s of the window already realized well above the strike
    k = 40 * math.log(101.0)
    high = p_up_twap(100.0, 100.0, 0.6, 20, window_s=60.0, known_log_integral=k)
    low = p_up_twap(100.0, 100.0, 0.6, 20, window_s=60.0, known_log_integral=40 * math.log(99.0))
    assert high > 0.99 and low < 0.01
    # at expiry the answer is the sign of the realized average
    assert (
        p_up_twap(100.0, 100.0, 0.6, 0, window_s=60.0, known_log_integral=60 * math.log(100.5))
        == 1.0
    )


def test_log_integral_piecewise_constant():
    from updown.fair_value import log_integral

    ts = np.array([0.0, 10.0, 20.0])
    px = np.array([100.0, 200.0, 400.0])
    # over [5, 25]: 5 s at 100, 10 s at 200, 5 s at 400
    expected = 5 * math.log(100) + 10 * math.log(200) + 5 * math.log(400)
    assert log_integral(ts, px, 5.0, 25.0) == pytest.approx(expected)
    assert log_integral(ts, px, -1.0, 5.0) is None  # no level known at the start
