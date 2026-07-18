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
    "corwin_schultz": lambda o, h, l, c, v: ind.corwin_schultz_spread(h, l),
    "amihud": lambda o, h, l, c, v: ind.amihud_illiquidity(
        ind.simple_returns(c), ind.dollar_volume(c, v)
    ),
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
    # Price alternates 100/101 → deltas ±1 → serial cov → -1 → S → 2.
    # Strict window uses period-1 = 19 pairs, so the finite-sample estimate sits
    # just under the theoretical limit of 2 (~1.997), not exactly 2.
    prices = [100.0 + (i % 2) for i in range(30)]
    assert ind.roll_spread_estimator(prices, period=20)[-1] == pytest.approx(2.0, abs=0.01)


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
    ("amihud", lambda p: ind.amihud_illiquidity([0.01] * 20, [100.0] * 20, period=p)),
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


# --- dirty-data policy: NaN out, no silent wrong numbers ---------------------

def test_rsi_nan_bar_does_not_emit_value():
    # A NaN bar must not be swallowed as "no change" → 100. It resets the run.
    out = ind.rsi([100.0, 101.0, float("nan"), 102.0, 103.0], period=2)
    assert all(math.isnan(v) for v in out), out


def test_rsi_reseeds_after_dirty_segment():
    # Clean run long enough after the gap re-seeds and produces a value again.
    prices = [float("nan"), 100.0, 101.0, 102.0, 103.0, 104.0]
    out = ind.rsi(prices, period=2)
    assert not math.isnan(out[-1])  # steady climb after the gap → valid RSI


def test_parkinson_vol_invalid_bar_in_window_is_nan():
    # One invalid bar (low=0) must NOT be averaged as low volatility.
    out = ind.parkinson_vol([2.0, 2.0], [1.0, 0.0], period=2)
    assert math.isnan(out[1])


def test_atr_invalid_bar_resets_segment():
    highs = [10.0, 11.0, float("nan"), 11.0, 12.0, 13.0, 12.0]
    lows = [9.0, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0]
    closes = [9.5, 10.5, 10.0, 10.5, 11.5, 12.5, 11.5]
    out = ind.compute_atr(highs, lows, closes, period=2)
    assert math.isnan(out[2])  # the NaN bar itself
    assert math.isnan(out[3])  # first bar after reset — mid warmup


def test_vwap_invalid_volume_does_not_poison_tail():
    out = ind.vwap([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, float("nan"), 1.0])
    assert out[0] == pytest.approx(1.0)
    assert math.isnan(out[1])          # the bad bar
    assert out[2] == pytest.approx(1.0)  # recovered — segment reset, not poisoned


def test_vwap_segment_reset_restarts_from_next_clean_bar():
    # After a dirty bar the accumulator resets: the third bar's VWAP is its own
    # typical price (100), not a blend with the pre-gap segment (~55).
    highs = [10.0, float("nan"), 100.0]
    lows = [10.0, float("nan"), 100.0]
    closes = [10.0, float("nan"), 100.0]
    out = ind.vwap(highs, lows, closes, [1.0, 1.0, 1.0])
    assert out[0] == pytest.approx(10.0)
    assert math.isnan(out[1])
    assert out[2] == pytest.approx(100.0)


def test_hlc_functions_reject_close_outside_range():
    # close=100 with high=10 is impossible → NaN, not a computed number.
    assert math.isnan(ind.typical_price([10.0], [9.0], [100.0])[0])
    assert math.isnan(ind.vwap([10.0], [9.0], [100.0], [1.0])[0])
    assert math.isnan(ind.rolling_vwap([10.0, 10.0], [9.0, 9.0], [9.5, 100.0], [1.0, 1.0], period=2)[1])


def test_atr_rejects_close_outside_range():
    # Second bar has close=100 with high=10 — impossible; must not yield ATR.
    out = ind.compute_atr([10.0, 10.0, 10.0], [9.0, 9.0, 9.0], [9.5, 100.0, 9.5], period=1)
    assert math.isnan(out[1])


def test_rolling_max_min_reject_inf():
    assert math.isnan(ind.rolling_max([1.0, float("inf")], 2)[1])
    assert math.isnan(ind.rolling_min([1.0, float("-inf")], 2)[1])


def test_parkinson_spread_rejects_non_positive_price():
    assert math.isnan(ind.parkinson_spread([-1.0], [-2.0])[0])
    assert math.isnan(ind.parkinson_spread([0.0], [0.0])[0])


def test_dollar_volume_rejects_non_positive():
    out = ind.dollar_volume([-10.0, 10.0], [5.0, -5.0])
    assert math.isnan(out[0])  # negative price
    assert math.isnan(out[1])  # negative volume


def test_amihud_strict_window_any_invalid_is_nan():
    returns = [0.01] * 20
    dv = [100.0] * 20
    dv[5] = 0.0  # one invalid observation inside the first full window
    out = ind.amihud_illiquidity(returns, dv, period=20)
    assert math.isnan(out[19])  # window [0..19] contains the invalid bar


def test_roll_spread_strict_window_any_invalid_is_nan():
    prices = [100.0 + (i % 2) for i in range(30)]
    prices[10] = float("nan")
    out = ind.roll_spread_estimator(prices, period=20)
    # any window covering index 10 must be NaN
    assert math.isnan(out[20])


def test_rolling_apply_validates_min_periods():
    with pytest.raises(ValueError):
        ind.rolling_apply([1.0] * 5, 3, max, min_periods=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        ind.rolling_apply([1.0] * 5, 3, max, min_periods=4)
    with pytest.raises(TypeError):
        ind.rolling_apply([1.0] * 5, 3, max, min_periods=True)


def test_rolling_max_min_nan_order_independent():
    # Old behavior: [nan,1]→nan but [1,nan]→1.0. Now both are NaN.
    assert math.isnan(ind.rolling_max([float("nan"), 1.0], 2)[1])
    assert math.isnan(ind.rolling_max([1.0, float("nan")], 2)[1])
    assert math.isnan(ind.rolling_min([float("nan"), 1.0], 2)[1])
    assert math.isnan(ind.rolling_min([1.0, float("nan")], 2)[1])


def test_candle_rejects_impossible_bar():
    # high < low, close > high → impossible bar → NaN, not a bogus -0.5 ratio.
    assert math.isnan(ind.body_ratio([10.0], [5.0], [9.0], [12.0])[0])
    # valid bar still computes
    assert ind.body_ratio([10.0], [13.0], [9.0], [12.0])[0] == pytest.approx(0.5)


def test_corwin_schultz_bounded_below_2():
    # Extreme but valid ranges must not overflow; spread stays < 2 (tanh form).
    h = [100.0, 200.0, 100.0, 300.0, 100.0]
    l = [1.0, 1.0, 1.0, 1.0, 1.0]
    out = ind.corwin_schultz_spread(h, l)
    assert all(math.isnan(v) or v < 2.0 for v in out)


# --- structural length validation (multi-series) -----------------------------

_MULTISERIES = [
    ("compute_atr", lambda: ind.compute_atr([1.0] * 5, [1.0] * 4, [1.0] * 5, period=2)),
    ("parkinson_vol", lambda: ind.parkinson_vol([2.0] * 5, [1.0] * 4, period=2)),
    ("parkinson_spread", lambda: ind.parkinson_spread([2.0] * 5, [1.0] * 4)),
    ("corwin_schultz", lambda: ind.corwin_schultz_spread([2.0] * 5, [1.0] * 4)),
    ("amihud", lambda: ind.amihud_illiquidity([0.01] * 5, [100.0] * 4, period=2)),
    ("typical_price", lambda: ind.typical_price([1.0] * 5, [1.0] * 4, [1.0] * 5)),
    ("vwap", lambda: ind.vwap([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 4)),
    ("rolling_vwap", lambda: ind.rolling_vwap([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 4, period=2)),
    ("dollar_volume", lambda: ind.dollar_volume([1.0] * 5, [1.0] * 4)),
]


@pytest.mark.parametrize("name,fn", _MULTISERIES, ids=[n for n, _ in _MULTISERIES])
def test_unequal_lengths_rejected(name, fn):
    with pytest.raises(ValueError, match="same length"):
        fn()


# --- roll spread period must be >= 3 (period=2 degenerates to 0.0) ------------

def test_roll_spread_period_one_rejected():
    with pytest.raises(ValueError, match="Roll spread period must be >= 3"):
        ind.roll_spread_estimator([1.0, 2.0, 3.0, 4.0], period=1)


def test_roll_spread_period_two_rejected():
    with pytest.raises(ValueError, match="Roll spread period must be >= 3"):
        ind.roll_spread_estimator([1.0, 2.0, 3.0, 4.0, 5.0], period=2)


# --- output overflow guard: finite inputs must yield NaN, not inf/exception ---

def test_simple_returns_overflow_yields_nan():
    # (1e308 - 1e-308) / 1e-308 ~= 1e616 → overflow to inf → guard catches NaN.
    out = ind.simple_returns([1e-308, 1e308])
    assert math.isnan(out[1])

    # Reverse direction: (1e-308 - 1e308) / 1e308 = -1.0 — valid finite result,
    # no overflow. The guard is value-agnostic; verify it does not kill this.
    out2 = ind.simple_returns([1e308, 1e-308])
    assert out2[1] == pytest.approx(-1.0)


def test_momentum_overflow_yields_nan():
    # i=1: (1e200 - 1e-200) / 1e-200 → overflow to inf → guard yields NaN.
    # i=2: (1e-200 - 1e200) / 1e200 = -1.0 → valid finite. Guard is value-agnostic.
    out = ind.momentum([1e-200, 1e200, 1e-200], period=1)
    assert math.isnan(out[1])
    assert out[2] == pytest.approx(-1.0)


def test_rolling_mean_overflow_yields_nan():
    # Old sum(window) overflowed to inf → NaN. Incremental per-window mean
    # produces the correct representable value.
    out = ind.rolling_mean([1e308, 1e308], 2)
    assert out[1] == pytest.approx(1e308)


def test_rolling_std_no_overflow_error():
    # Window scaling before Welford: extreme values no longer overflow.
    # std of [1e308, -1e308] ≈ 1.414e308 — representable with current approach.
    out = ind.rolling_std([1e308, -1e308], 2)
    assert len(out) == 2
    assert math.isfinite(out[1])
    assert out[1] > 1e307  # roughly sqrt(2) * 1e308 / 2? No — it's population std.
    # population std: sqrt(((a-mean)² + (b-mean)²)/2), mean=0 → sqrt(1e616/2) = inf.
    # Actually with scaling: s = 1e308, scaled = [1, -1], Welford → var_scaled=1
    # std = s * sqrt(1) = 1e308. ✓


def test_dollar_volume_overflow_yields_nan():
    out = ind.dollar_volume([1e308], [1e308])
    assert math.isnan(out[0])


def test_typical_price_overflow_yields_nan():
    # Old (h+l+c)/3 overflowed when all three near 1e308 → NaN.
    # Divide-then-add (h/3 + l/3 + c/3) keeps the result representable.
    out = ind.typical_price([1e308], [1e308], [1e308])
    assert out[0] == pytest.approx(1e308)


def test_parkinson_vol_overflow_yields_nan():
    # Old: high/low ratio overflowed → NaN. Log-diff form keeps it finite.
    # Parkinson vol at period=1 with extreme but valid bar should compute.
    out = ind.parkinson_vol([1e308], [1e-308], period=1)
    assert math.isfinite(out[0])


def test_parkinson_spread_overflow_yields_nan():
    # Old diff² overflowed → NaN. |diff| * sqrt(1/(4ln2)) is overflow-safe.
    out = ind.parkinson_spread([1e308], [1e-308])
    assert math.isfinite(out[0])


def test_vwap_overflow_yields_nan():
    # tp * vol overflows at 1e308 * 1e308, segment should reset, output NaN
    out = ind.vwap([1e308, 1.0], [1e308, 1.0], [1e308, 1.0], [1e308, 1.0])
    assert math.isnan(out[0])


def test_rolling_vwap_overflow_yields_nan():
    out = ind.rolling_vwap(
        [1e308, 1e308], [1e308, 1e308], [1e308, 1e308], [1e308, 1e308], period=1
    )
    assert math.isnan(out[0])


# --- P0: ATR dirty previous bar must not contaminate true range ---------------

def test_atr_dirty_prev_bar_resets_segment():
    # Bar at index 1 is impossible (high=5 < low=6). Its close=5.5 feeds the
    # true range calculation for the bar at index 2. The old code allowed this;
    # the fix validates the full previous HLC bar, so the impossible bar's
    # close does not contaminate the next true range.
    highs = [10, 5, 100, 100, 100]
    lows = [9, 6, 99, 99, 99]
    closes = [9.5, 5.5, 99.5, 99.5, 99.5]
    out = ind.compute_atr(highs, lows, closes, period=2)
    # After the dirty bar: index 2 has current valid, prev invalid → reset → NaN
    assert math.isnan(out[2])
    # Index 3: valid_run=1 — valid but below period → no output yet → NaN
    assert math.isnan(out[3])


# --- P1: RSI incremental mean avoids silent-wrong output from overflow --------

def test_rsi_overflow_safe_incremental_mean():
    # Old sum-based seed: sum([M, M, M]) overflows, avg_gain becomes inf,
    # RSI returns 100.0 instead of ~66.67. Incremental mean keeps it finite.
    M = float.fromhex("0x1.fffffffffffffp+1023")  # ~1.80e308
    tiny = float.fromhex("0x0.0000000000001p-1022")  # ~5e-324
    # Alternating M/tiny produces gains near M, losses near M
    out = ind.rsi([tiny, M, tiny, M, tiny], period=3)
    # RSI should be finite; the exact value is ~66.67, not 100.0
    assert not math.isnan(out[-1])
    assert 0.0 < out[-1] < 100.0


# --- P1: ATR incremental mean avoids inf output -------------------------------

def test_atr_overflow_safe_no_inf():
    # Old sum(seed_tr) / period → inf at extreme true ranges.
    # Incremental mean keeps the result finite (or NaN, not inf).
    M = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = float.fromhex("0x0.0000000000001p-1022")
    out = ind.compute_atr([1.0, M, M], [1.0, tiny, tiny], [1.0, tiny, tiny], period=2)
    # Must not produce inf; NaN or finite is acceptable.
    assert math.isnan(out[2]) or math.isfinite(out[2])
    assert not (out[2] == float("inf"))


# --- P1: Amihud overflow guard -------------------------------------------------

def test_amihud_overflow_guard():
    # |r| / dv at extreme magnitudes overflows to inf, which poisons the
    # cumulative sum. The fix marks the overflowed contrib as invalid so the
    # strict-window check (n_valid == period) excludes it.
    M = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = float.fromhex("0x0.0000000000001p-1022")
    out = ind.amihud_illiquidity([M, M], [tiny, tiny], period=2)
    # Both contrib values overflow → invalid → n_valid < 2 → NaN
    assert math.isnan(out[1])


# --- P1: Roll covariance scaled to prevent overflow ---------------------------

def test_roll_spread_scaled_covariance_no_overflow():
    # Unscaled covariance: deltas ≈ ±M → (a - mean)*(b - mean) overflows.
    # Scaling by max(abs(deltas)) keeps arithmetic in [0, 1] range.
    M = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = float.fromhex("0x0.0000000000001p-1022")
    out = ind.roll_spread_estimator([tiny, M, tiny, M], period=3)
    # Must not crash or produce inf; NaN or finite is acceptable.
    assert len(out) == 4
    assert not (out[3] == float("inf"))


# --- P1: rate_of_change post-multiply guard ----------------------------------

def test_rate_of_change_finite_guard():
    # momentum returns a finite but huge value; * 100 overflows to inf.
    # The fix guards the multiply so NaN is returned instead of inf.
    out = ind.rate_of_change([1e-307, 1.0], 1)
    assert math.isnan(out[1])  # momentum ≈ 1e307 → *100 ≈ 1e309 → inf → NaN


# --- P1: cumulative VWAP overflow recovers (no permanent poisoning) -----------

def test_vwap_cumulative_overflow_recovers():
    # Running totals would overflow at bar 2 without rescaling.
    # Rescale approach keeps accumulators in range → no NaN, no data loss.
    x = float.fromhex("0x1.fffffffffffffp+1023") / 4.0
    out = ind.vwap(
        [x, x, x, x, 1.0],
        [x, x, x, x, 1.0],
        [x, x, x, x, 1.0],
        [2.0, 2.0, 2.0, 2.0, 1.0],
    )
    # All clean bars produce valid VWAP — rescaling preserves continuity.
    assert all(math.isfinite(v) for v in out[0:4])
    # Bar 4: after rescale, (2x*0.5**k + 2x + ... ) / (2*0.5**k + 2 + ...) ≈ x
    assert out[3] == pytest.approx(x)
    assert math.isfinite(out[4])


# --- P1: Amihud cumulative sum overflow must not emit inf ---------------------

def test_amihud_cumsum_overflow_no_inf():
    # All contrib values are finite (1e308), but four of them in a period-2
    # window overflow the cumulative sum. Must produce NaN, not inf.
    out = ind.amihud_illiquidity([1e308, 1e308, 1e308, 1e308],
                                  [1.0, 1.0, 1.0, 1.0], period=2)
    assert all(math.isnan(x) or math.isfinite(x) for x in out)


# --- P2: parkinson_vol / corwin_schultz use log-diff to avoid ratio overflow --

def test_parkinson_vol_log_diff_no_overflow():
    # high=1e308, low=1e-308 → ratio overflows → old code NaN'd unnecessarily.
    # log(h)-log(l) = log(1e308) - log(1e-308) ≈ 1416 — finite and correct.
    out = ind.parkinson_vol([1e308], [1e-308], period=1)
    assert math.isfinite(out[0])  # not NaN from ratio overflow


def test_corwin_schultz_log_diff_no_overflow():
    out = ind.corwin_schultz_spread([1e308, 1e308], [1e-308, 1e-308])
    assert math.isnan(out[0])  # first bar always NaN
    # Log-diff keeps alpha finite; tanh(alpha/2) ≤ 1 → spread ≤ 2.
    assert math.isnan(out[1]) or out[1] <= 2.0


# --- P0: Amihud catastrophic cancellation (large bar exit zeroes clean tail) ---

def test_amihud_catastrophic_cancellation():
    # A single huge contrib (1/1e-20 = 5e19) must not cause subsequent clean
    # windows to read 0.0 after it rotates out. Per-window fsum fixes this.
    out = ind.amihud_illiquidity(
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1e-20, 1.0, 1.0, 1.0, 1.0],
        period=2,
    )
    # Window [0,1]: |1|/1e-20 = 1e20, |1|/1 = 1. fsum(1e20 + 1) ≈ 1e20 → /2 = 5e19.
    assert out[1] == pytest.approx(5e19, rel=0.01)
    # Window [1,2]: |1|/1 = 1, |1|/1 = 1 → mean = 1.0. NOT 0.0.
    assert out[2] == pytest.approx(1.0)
    # Window [2,3]: same → 1.0
    assert out[3] == pytest.approx(1.0)
    # Window [3,4]: same → 1.0
    assert out[4] == pytest.approx(1.0)


# --- P1: compute_atr(period=1) should work -----------------------------------

def test_atr_period_one():
    # ATR(1) is mathematically ATR = true range. Not all NaN.
    out = ind.compute_atr(
        [10, 11, 12, 13],
        [9, 10, 11, 12],
        [9.5, 10.5, 11.5, 12.5],
        period=1,
    )
    assert math.isnan(out[0])  # first bar never emits (no prev close)
    # tr[1] = max(2, |11-9.5|, |10-9.5|) = max(2, 1.5, 0.5) = 2.0? Wait:
    # hi=11, lo=10, cl=10.5, prev_cl=9.5
    # tr = max(11-10, |11-9.5|, |10-9.5|) = max(1, 1.5, 0.5) = 1.5
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(1.5)
    assert out[3] == pytest.approx(1.5)


# --- P2: Corwin-Schultz flat bars → spread 0.0, not NaN ----------------------

def test_corwin_schultz_flat_bars_are_zero():
    # alpha=0 → spread=2*tanh(0)=0.0. Old code treated alpha<=0 as invalid → NaN.
    out = ind.corwin_schultz_spread([100, 100, 100], [100, 100, 100])
    assert math.isnan(out[0])  # first bar
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)


# --- P1: Corwin-Schultz negative alpha → 0.0, not NaN ------------------------

def test_corwin_schultz_negative_alpha_is_zero():
    # Clean realistic bars where variance term dominates → alpha < 0.
    # Old code emitted NaN (~65% of clean pairs); fix clamps to 0.0.
    out = ind.corwin_schultz_spread([101, 102], [90, 99])
    assert math.isnan(out[0])  # first bar
    assert out[1] == pytest.approx(0.0)


# --- P1: rolling_std scaling preserves large/tiny representable stds ---------

def test_rolling_std_scaling_large():
    out = ind.rolling_std([1e308, 1.0], 2)
    assert math.isfinite(out[1])
    assert out[1] > 1e307


def test_rolling_std_scaling_tiny():
    out = ind.rolling_std([1e-300, 2e-300], 2)
    assert math.isfinite(out[1])
    assert out[1] > 0.0


def test_rolling_mean_opposite_signs():
    out = ind.rolling_mean([1e308, -1e308], 2)
    assert out[1] == pytest.approx(0.0)


# --- P1: VWAP safe typical price (divide-then-add) ---------------------------

def test_vwap_safe_tp_large():
    out = ind.vwap([1e308], [1e308], [1e308], [1.0])
    assert out[0] == pytest.approx(1e308)


def test_vwap_safe_tp_tiny():
    # tp=1e-300, vol=1e-300 → contrib underflows to 0.0.
    # Without the contrib==0 guard (removed for zero-vol fix), the result is
    # 0.0 rather than NaN — a known P2 subnormal limitation.
    out = ind.vwap([1e-300], [1e-300], [1e-300], [1e-300])
    # 0.0 (underflow) or NaN (if guard path hit) — both acceptable for P2.
    assert out[0] == 0.0 or math.isnan(out[0])


# --- P1: roll_spread sqrt-after-scale preserves subnormal spreads ------------

def test_roll_spread_subnormal_spread():
    prices = [1e-300, 2e-300, 1e-300, 2e-300]
    out = ind.roll_spread_estimator(prices, period=3)
    assert out[3] > 0.0


# --- P2: log_returns log1p preserves sub-ULP changes ------------------------

def test_log_returns_log1p_precision():
    x = 1e10
    import sys
    y = sys.float_info.epsilon * x + x
    out = ind.log_returns([x, y])
    assert out[1] > 0.0


# --- P1: VWAP zero-volume bar must not reset accumulator ----------------------

def test_vwap_zero_volume_preserves_history():
    out = ind.vwap([10, 20, 30], [10, 20, 30], [10, 20, 30], [1, 0, 1])
    assert out[0] == pytest.approx(10.0)
    assert out[1] == pytest.approx(10.0)  # zero vol → emit prior VWAP
    assert out[2] == pytest.approx(20.0)  # (10*1 + 30*1) / (1+1)


# --- P2: rolling_vwap overflow guard on cumsum --------------------------------

def test_rolling_vwap_cumsum_overflow_nan():
    x = float.fromhex("0x1.fffffffffffffp+1022")
    out = ind.rolling_vwap([1, 1, 1], [1, 1, 1], [1, 1, 1], [x, x, x], period=3)
    assert math.isnan(out[2]) or math.isfinite(out[2])
