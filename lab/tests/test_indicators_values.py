"""Hand-computed value checks for indicator math."""

import math

import pytest

from lab.indicators import (
    amihud_illiquidity,
    body_ratio,
    compute_atr,
    log_returns,
    lower_wick_ratio,
    momentum,
    parkinson_vol,
    rate_of_change,
    roll_spread_estimator,
    rolling_max,
    rolling_mean,
    rolling_min,
    rsi,
    simple_returns,
    typical_price,
    upper_wick_ratio,
    vwap,
)


def test_simple_returns():
    out = simple_returns([100.0, 110.0, 99.0])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(-0.10)


def test_log_returns():
    out = log_returns([100.0, 110.0])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(math.log(1.1))


def test_momentum_and_roc():
    out = momentum([10.0, 20.0, 30.0, 40.0], period=2)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(2.0)  # (30-10)/10
    assert out[3] == pytest.approx(1.0)  # (40-20)/20
    roc = rate_of_change([10.0, 20.0, 30.0, 40.0], period=2)
    assert roc[2] == pytest.approx(200.0)


def test_rolling_mean_max_min():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert rolling_mean(vals, 2)[1:] == pytest.approx([1.5, 2.5, 3.5])
    assert rolling_max(vals, 2)[1:] == pytest.approx([2.0, 3.0, 4.0])
    assert rolling_min(vals, 2)[1:] == pytest.approx([1.0, 2.0, 3.0])
    assert math.isnan(rolling_mean(vals, 2)[0])


def test_rsi_monotonic_up_is_100():
    prices = [float(x) for x in range(1, 40)]
    out = rsi(prices, period=14)
    assert out[-1] == pytest.approx(100.0)
    assert all(math.isnan(v) for v in out[:14])


def test_atr_flat_market_is_zero():
    flat = [100.0] * 30
    out = compute_atr(flat, flat, flat, period=14)
    assert out[14] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(0.0)
    assert all(math.isnan(v) for v in out[:14])


def test_atr_first_value_is_mean_true_range():
    highs = [10.0, 12.0, 11.0]
    lows = [9.0, 10.0, 10.0]
    closes = [9.5, 11.0, 10.5]
    out = compute_atr(highs, lows, closes, period=2)
    # TR[1] = max(2, |12-9.5|, |10-9.5|) = 2.5 ; TR[2] = max(1, 0, 1) = 1.0
    assert out[2] == pytest.approx((2.5 + 1.0) / 2)


def test_candle_ratios():
    # open=10 close=12 high=13 low=9 → range 4, body 2, upper wick 1, lower wick 1
    o, h, l, c = [10.0], [13.0], [9.0], [12.0]
    assert body_ratio(o, h, l, c)[0] == pytest.approx(0.5)
    assert upper_wick_ratio(o, h, l, c)[0] == pytest.approx(0.25)
    assert lower_wick_ratio(o, h, l, c)[0] == pytest.approx(0.25)


def test_candle_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        body_ratio([1.0], [1.0, 2.0], [1.0], [1.0])


def test_parkinson_vol_flat_is_zero():
    flat = [100.0] * 25
    out = parkinson_vol(flat, flat, period=20)
    assert out[-1] == pytest.approx(0.0)


def test_typical_price_and_vwap():
    h, l, c = [12.0], [8.0], [10.0]
    assert typical_price(h, l, c)[0] == pytest.approx(10.0)
    out = vwap([12.0, 12.0], [8.0, 8.0], [10.0, 10.0], [5.0, 5.0])
    assert out[1] == pytest.approx(10.0)


def test_amihud_constant_ratio():
    returns = [float("nan")] + [0.01] * 24
    volumes = [1.0] * 25
    out = amihud_illiquidity(returns, volumes, period=20)
    assert out[-1] == pytest.approx(0.01)


def test_roll_spread_alternating_prices():
    # Price alternates 100/101 → deltas alternate +1/-1 → serial cov = -1 → S = 2
    prices = [100.0 + (i % 2) for i in range(30)]
    out = roll_spread_estimator(prices, period=20)
    assert out[-1] == pytest.approx(2.0)
