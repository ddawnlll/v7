"""Causality: for every indicator, appending future bars must not change
any previously computed value (RULES §6)."""

import math
import random

import pytest

from lab.indicators import (
    amihud_illiquidity,
    body_ratio,
    compute_atr,
    corwin_schultz_spread,
    dollar_volume,
    log_returns,
    lower_wick_ratio,
    momentum,
    parkinson_spread,
    parkinson_vol,
    rate_of_change,
    roll_spread_estimator,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_parkinson_spread,
    rolling_std,
    rolling_vwap,
    rsi,
    simple_returns,
    typical_price,
    upper_wick_ratio,
    vwap,
)

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


INDICATORS = {
    "atr": lambda o, h, l, c, v: compute_atr(h, l, c),
    "body_ratio": lambda o, h, l, c, v: body_ratio(o, h, l, c),
    "upper_wick": lambda o, h, l, c, v: upper_wick_ratio(o, h, l, c),
    "lower_wick": lambda o, h, l, c, v: lower_wick_ratio(o, h, l, c),
    "log_returns": lambda o, h, l, c, v: log_returns(c),
    "simple_returns": lambda o, h, l, c, v: simple_returns(c),
    "momentum": lambda o, h, l, c, v: momentum(c),
    "roc": lambda o, h, l, c, v: rate_of_change(c),
    "rolling_max": lambda o, h, l, c, v: rolling_max(c),
    "rolling_min": lambda o, h, l, c, v: rolling_min(c),
    "rolling_mean": lambda o, h, l, c, v: rolling_mean(c),
    "rolling_std": lambda o, h, l, c, v: rolling_std(c),
    "rsi": lambda o, h, l, c, v: rsi(c),
    "parkinson_vol": lambda o, h, l, c, v: parkinson_vol(h, l),
    "parkinson_spread": lambda o, h, l, c, v: parkinson_spread(h, l),
    "rolling_parkinson_spread": lambda o, h, l, c, v: rolling_parkinson_spread(h, l),
    "corwin_schultz": lambda o, h, l, c, v: corwin_schultz_spread(h, l),
    "amihud": lambda o, h, l, c, v: amihud_illiquidity(simple_returns(c), v),
    "roll_spread": lambda o, h, l, c, v: roll_spread_estimator(c),
    "dollar_volume": lambda o, h, l, c, v: dollar_volume(c, v),
    "typical_price": lambda o, h, l, c, v: typical_price(h, l, c),
    "vwap": lambda o, h, l, c, v: vwap(h, l, c, v),
    "rolling_vwap": lambda o, h, l, c, v: rolling_vwap(h, l, c, v),
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

    assert len(out_prefix) == N_PREFIX
    assert len(out_full) == N_FULL
    for i in range(N_PREFIX):
        assert _same(out_prefix[i], out_full[i]), (
            f"{name}: value at index {i} changed when future bars were appended "
            f"({out_prefix[i]!r} -> {out_full[i]!r}) — lookahead leak"
        )
