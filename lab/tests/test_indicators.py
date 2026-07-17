"""Indicator checks: causality (no lookahead) + hand-computed values.

Causality is the load-bearing test (RULES §6): for every indicator, appending
future bars must not change any previously computed value.
"""

import math
import random

import pytest

from lab import indicators as ind

N_FULL = 120
N_PREFIX = 80


def _ohlcv(n: int, seed: int = 7):
    rng = random.Random(seed)
    o, h, l, c, v = [], [], [], [], []
    price = 100.0
    for _ in range(n):
        op = price
        cl = op * (1.0 + rng.uniform(-0.02, 0.02))
        hi = max(op, cl) * (1.0 + rng.uniform(0.0, 0.01))
        lo = min(op, cl) * (1.0 - rng.uniform(0.0, 0.01))
        o.append(op)
        h.append(hi)
        l.append(lo)
        c.append(cl)
        v.append(rng.uniform(10.0, 1000.0))
        price = cl
    return o, h, l, c, v


# Each entry maps a name to a call over (o, h, l, c, v).
INDICATORS = {
    "atr": lambda o, h, l, c, v: ind.compute_atr(h, l, c),
    "body_ratio": lambda o, h, l, c, v: ind.body_ratio(o, h, l, c),
    "upper_wick": lambda o, h, l, c, v: ind.upper_wick_ratio(o, h, l, c),
    "lower_wick": lambda o, h, l, c, v: ind.lower_wick_ratio(o, h, l, c),
    "log_returns": lambda o, h, l, c, v: ind.log_returns(c),
    "simple_returns": lambda o, h, l, c, v: ind.simple_returns(c),
    "momentum": lambda o, h, l, c, v: ind.momentum(c),
    "roc": lambda o, h, l, c, v: ind.rate_of_change(c),
    "rolling_max": lambda o, h, l, c, v: ind.rolling_max(c),
    "rolling_min": lambda o, h, l, c, v: ind.rolling_min(c),
    "rolling_mean": lambda o, h, l, c, v: ind.rolling_mean(c),
    "rolling_std": lambda o, h, l, c, v: ind.rolling_std(c),
    "rsi": lambda o, h, l, c, v: ind.rsi(c),
    "parkinson_vol": lambda o, h, l, c, v: ind.parkinson_vol(h, l),
    "parkinson_spread": lambda o, h, l, c, v: ind.parkinson_spread(h, l),
    "rolling_parkinson_spread": lambda o, h, l, c, v: ind.rolling_parkinson_spread(h, l),
    "corwin_schultz": lambda o, h, l, c, v: ind.corwin_schultz_spread(h, l),
    "amihud": lambda o, h, l, c, v: ind.amihud_illiquidity(ind.simple_returns(c), v),
    "roll_spread": lambda o, h, l, c, v: ind.roll_spread_estimator(c),
    "dollar_volume": lambda o, h, l, c, v: ind.dollar_volume(c, v),
    "typical_price": lambda o, h, l, c, v: ind.typical_price(h, l, c),
    "vwap": lambda o, h, l, c, v: ind.vwap(h, l, c, v),
    "rolling_vwap": lambda o, h, l, c, v: ind.rolling_vwap(h, l, c, v),
}


def _same(a: float, b: float) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_appending_future_bars_does_not_change_past(name):
    fn = INDICATORS[name]
    full = _ohlcv(N_FULL)
    prefix = tuple(series[:N_PREFIX] for series in full)

    out_prefix = fn(*prefix)
    out_full = fn(*full)

    assert len(out_prefix) == N_PREFIX and len(out_full) == N_FULL
    for i in range(N_PREFIX):
        assert _same(out_prefix[i], out_full[i]), (
            f"{name}: value at index {i} changed when future bars were appended "
            f"({out_prefix[i]!r} -> {out_full[i]!r}) — lookahead leak"
        )


# --- hand-computed values ----------------------------------------------------

def test_simple_and_log_returns():
    out = ind.simple_returns([100.0, 110.0, 99.0])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(-0.10)
    lg = ind.log_returns([100.0, 110.0])
    assert lg[1] == pytest.approx(math.log(1.1))


def test_momentum_and_roc():
    out = ind.momentum([10.0, 20.0, 30.0, 40.0], period=2)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(2.0)  # (30-10)/10
    assert out[3] == pytest.approx(1.0)  # (40-20)/20
    assert ind.rate_of_change([10.0, 20.0, 30.0, 40.0], period=2)[2] == pytest.approx(200.0)


def test_rolling_mean_max_min():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert ind.rolling_mean(vals, 2)[1:] == pytest.approx([1.5, 2.5, 3.5])
    assert ind.rolling_max(vals, 2)[1:] == pytest.approx([2.0, 3.0, 4.0])
    assert ind.rolling_min(vals, 2)[1:] == pytest.approx([1.0, 2.0, 3.0])
    assert math.isnan(ind.rolling_mean(vals, 2)[0])


def test_rsi_monotonic_up_is_100():
    prices = [float(x) for x in range(1, 40)]
    out = ind.rsi(prices, period=14)
    assert out[-1] == pytest.approx(100.0)
    assert all(math.isnan(v) for v in out[:14])


def test_atr_flat_market_is_zero():
    flat = [100.0] * 30
    out = ind.compute_atr(flat, flat, flat, period=14)
    assert out[14] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(0.0)
    assert all(math.isnan(v) for v in out[:14])


def test_atr_first_value_is_mean_true_range():
    highs = [10.0, 12.0, 11.0]
    lows = [9.0, 10.0, 10.0]
    closes = [9.5, 11.0, 10.5]
    # TR[1] = max(2, |12-9.5|, |10-9.5|) = 2.5 ; TR[2] = max(1, 0, 1) = 1.0
    assert ind.compute_atr(highs, lows, closes, period=2)[2] == pytest.approx((2.5 + 1.0) / 2)


def test_candle_ratios():
    # open=10 close=12 high=13 low=9 → range 4, body 2, wicks 1 each
    o, h, l, c = [10.0], [13.0], [9.0], [12.0]
    assert ind.body_ratio(o, h, l, c)[0] == pytest.approx(0.5)
    assert ind.upper_wick_ratio(o, h, l, c)[0] == pytest.approx(0.25)
    assert ind.lower_wick_ratio(o, h, l, c)[0] == pytest.approx(0.25)


def test_candle_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ind.body_ratio([1.0], [1.0, 2.0], [1.0], [1.0])


def test_parkinson_vol_flat_is_zero():
    flat = [100.0] * 25
    assert ind.parkinson_vol(flat, flat, period=20)[-1] == pytest.approx(0.0)


def test_typical_price_and_vwap():
    assert ind.typical_price([12.0], [8.0], [10.0])[0] == pytest.approx(10.0)
    out = ind.vwap([12.0, 12.0], [8.0, 8.0], [10.0, 10.0], [5.0, 5.0])
    assert out[1] == pytest.approx(10.0)


def test_amihud_constant_ratio():
    returns = [float("nan")] + [0.01] * 24
    volumes = [1.0] * 25
    assert ind.amihud_illiquidity(returns, volumes, period=20)[-1] == pytest.approx(0.01)


def test_roll_spread_alternating_prices():
    # Price alternates 100/101 → deltas ±1 → serial cov = -1 → S = 2
    prices = [100.0 + (i % 2) for i in range(30)]
    assert ind.roll_spread_estimator(prices, period=20)[-1] == pytest.approx(2.0)


def test_dollar_volume_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ind.dollar_volume([1.0, 2.0], [1.0])


# --- period validation (causality guard) -------------------------------------

_PERIOD_FUNCS = [
    ("momentum", lambda p: ind.momentum([1.0, 2.0, 3.0, 4.0], period=p)),
    ("rate_of_change", lambda p: ind.rate_of_change([1.0, 2.0, 3.0, 4.0], period=p)),
    ("rsi", lambda p: ind.rsi([1.0] * 20, period=p)),
    ("compute_atr", lambda p: ind.compute_atr([1.0] * 20, [1.0] * 20, [1.0] * 20, period=p)),
    ("rolling_std", lambda p: ind.rolling_std([1.0] * 20, period=p)),
    ("parkinson_vol", lambda p: ind.parkinson_vol([2.0] * 20, [1.0] * 20, period=p)),
    ("rolling_max", lambda p: ind.rolling_max([1.0] * 20, period=p)),
    ("rolling_min", lambda p: ind.rolling_min([1.0] * 20, period=p)),
    ("rolling_mean", lambda p: ind.rolling_mean([1.0] * 20, period=p)),
    ("rolling_parkinson_spread", lambda p: ind.rolling_parkinson_spread([2.0] * 20, [1.0] * 20, period=p)),
    ("amihud", lambda p: ind.amihud_illiquidity([0.01] * 20, [1.0] * 20, period=p)),
    ("roll_spread", lambda p: ind.roll_spread_estimator([1.0] * 20, period=p)),
    ("rolling_vwap", lambda p: ind.rolling_vwap([2.0] * 20, [1.0] * 20, [1.5] * 20, [1.0] * 20, period=p)),
    ("rolling_apply", lambda p: ind.rolling_apply([1.0] * 20, p, max)),
]


@pytest.mark.parametrize("name,fn", _PERIOD_FUNCS, ids=[n for n, _ in _PERIOD_FUNCS])
def test_negative_period_rejected(name, fn):
    # A negative period would index forward (prices[i+1]) — a lookahead leak.
    with pytest.raises(ValueError):
        fn(-1)


@pytest.mark.parametrize("name,fn", _PERIOD_FUNCS, ids=[n for n, _ in _PERIOD_FUNCS])
def test_zero_period_rejected(name, fn):
    with pytest.raises(ValueError):
        fn(0)


@pytest.mark.parametrize("name,fn", _PERIOD_FUNCS, ids=[n for n, _ in _PERIOD_FUNCS])
def test_bool_period_rejected(name, fn):
    # bool is a subclass of int; True/False must not sneak in as period 1/0.
    with pytest.raises(TypeError):
        fn(True)


def test_invalid_period_rejected_before_empty_check():
    # Validation must run before the len==0 early return, so a bad period is
    # caught even on empty input rather than slipping through.
    with pytest.raises(ValueError):
        ind.momentum([], 0)
    with pytest.raises(ValueError):
        ind.rolling_mean([], -1)
    with pytest.raises(TypeError):
        ind.rsi([], True)


# --- rsi edge cases ----------------------------------------------------------

def test_rsi_period_one_rejected():
    with pytest.raises(ValueError, match="RSI period must be >= 2"):
        ind.rsi([1.0, 2.0, 3.0], period=1)


def test_rsi_flat_market_is_neutral_50():
    flat = [100.0] * 20
    out = ind.rsi(flat, period=14)
    assert out[-1] == pytest.approx(50.0)  # no gains, no losses → neutral, not 100


def test_rsi_monotonic_down_is_0():
    prices = [float(x) for x in range(40, 1, -1)]
    assert ind.rsi(prices, period=14)[-1] == pytest.approx(0.0)


# --- non-finite price policy -------------------------------------------------

def test_returns_reject_non_finite_prices():
    assert math.isnan(ind.log_returns([100.0, float("inf")])[1])
    assert math.isnan(ind.simple_returns([100.0, float("inf")])[1])
    assert math.isnan(ind.log_returns([float("nan"), 100.0])[1])


def test_returns_reject_non_positive_prices():
    assert math.isnan(ind.simple_returns([100.0, -10.0])[1])  # bad data, not -110%
    assert math.isnan(ind.log_returns([-5.0, 100.0])[1])


def test_log_returns_matches_ratio_form():
    # difference-of-logs must equal log-of-ratio for well-behaved inputs
    out = ind.log_returns([100.0, 110.0])
    assert out[1] == pytest.approx(math.log(110.0 / 100.0))
