"""Pure, causal indicator primitives — one audit unit.

Every function is a pure function of past-and-present bars only: appending
future bars must never change an already-computed value (RULES §6, enforced by
lab/tests/test_indicators.py). No state, no adapters, no business logic.

This module is never imported by lab/sim.py — research primitives and the
economic truth core stay on opposite sides of that line (RULES §4).

Data-validity contract (fixed):
- Structural errors raise: unequal series lengths, non-positive/non-int period.
  These are programming errors, not data.
- Dirty cell data yields NaN, never a silently wrong number: a non-finite value,
  a non-positive price, or an impossible OHLC bar (not low <= open,close <= high)
  produces NaN at the affected output positions; clean regions stay valid.
- Stateful indicators segment-reset on dirty data: rsi and atr discard their
  running averages and re-seed after `period` consecutive clean observations;
  cumulative vwap discards its running totals and restarts from the next clean
  bar. One bad bar never permanently poisons the rest of the series.

Contents:
  returns        log_returns, simple_returns
  momentum       momentum, rate_of_change
  rsi            rsi
  atr            compute_atr
  volatility     rolling_std, parkinson_vol
  rolling        rolling_apply, rolling_max, rolling_min, rolling_mean
  candle         body_ratio, upper_wick_ratio, lower_wick_ratio
  spread         parkinson_spread, corwin_schultz_spread
  microstructure dollar_volume, amihud_illiquidity, roll_spread_estimator
  volume_profile typical_price, vwap, rolling_vwap
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

_NAN = float("nan")


def _validate_period(period: int) -> None:
    """Reject non-positive or non-integer periods.

    Load-bearing for the causality guarantee: a negative period turns
    ``values[i - period]`` into a forward reference (lookahead), so this is not
    mere input validation — it is what keeps these primitives causal (RULES §6).
    ``bool`` is rejected explicitly because ``True``/``False`` are ints in
    Python and would silently pass as period 1/0.
    """
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError(f"period must be an int, got {type(period).__name__}")
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")


def _validate_equal_lengths(*series: Sequence[float]) -> None:
    """Structural check: all input series must have identical length. Raises so
    a length bug surfaces deterministically instead of an IndexError deep in a
    loop or a silent NumPy broadcast."""
    if not series:
        return
    n = len(series[0])
    if any(len(s) != n for s in series):
        raise ValueError("all input sequences must have the same length")


def _valid_price(x: float) -> bool:
    """A usable price is finite and strictly positive. Consistent policy across
    all return primitives (an inf or a non-positive price is bad data)."""
    return math.isfinite(x) and x > 0.0


def _valid_ohlc(o: float, h: float, l: float, c: float) -> bool:
    """A usable bar is finite, positive, and self-consistent:
    low <= min(open, close) <= max(open, close) <= high. Impossible bars
    (high < low, close > high, ...) are dirty data → excluded."""
    if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
        return False
    if l <= 0.0:
        return False
    return l <= min(o, c) and max(o, c) <= h


def _valid_hlc(h: float, l: float, c: float) -> bool:
    """A usable HLC bar (no open): finite, positive, and 0 < low <= close <= high.
    Rejects impossible bars where close falls outside the [low, high] range."""
    if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
        return False
    return 0.0 < l <= c <= h


# --- returns -----------------------------------------------------------------

def log_returns(prices: Sequence[float]) -> list[float]:
    """Log returns: ln(P_t) - ln(P_{t-1}). First element is NaN.

    Uses difference of logs for wide ratios (overflow-safe) and ``log1p`` of the
    simple return for small changes (preserves sub-ULP precision; e.g. 1e10 →
    nextafter gives ~2e-16 rather than 0.0). NaN where either price is non-finite
    or non-positive.
    """
    n = len(prices)
    if n < 2:
        return [_NAN] * n
    result: list[float] = [_NAN] * n
    for i in range(1, n):
        if not (_valid_price(prices[i - 1]) and _valid_price(prices[i])):
            continue
        prev, curr = prices[i - 1], prices[i]
        ratio = curr / prev
        if 0.9 <= ratio <= 1.1:
            # Small change: log1p preserves precision better than log diff.
            sr = (curr - prev) / prev
            result[i] = math.log1p(sr) if math.isfinite(sr) else _NAN
        else:
            result[i] = math.log(curr) - math.log(prev)
    return result


def simple_returns(prices: Sequence[float]) -> list[float]:
    """Simple returns: (P_t - P_{t-1}) / P_{t-1}. First element is NaN.

    Same price policy as log_returns (both prices finite and positive) so the
    two primitives agree on what counts as bad data.
    """
    n = len(prices)
    if n < 2:
        return [_NAN] * n
    result: list[float] = [_NAN] * n
    for i in range(1, n):
        if _valid_price(prices[i - 1]) and _valid_price(prices[i]):
            val = (prices[i] - prices[i - 1]) / prices[i - 1]
            if math.isfinite(val):
                result[i] = val
    return result


# --- momentum ----------------------------------------------------------------

def momentum(prices: Sequence[float], period: int = 10) -> list[float]:
    """Fractional rate of change: (P_t - P_{t-period}) / P_{t-period}.

    Despite the classical name, this is a *fractional* momentum (a ratio), not
    the absolute price difference P_t - P_{t-period}. `rate_of_change` below is
    the identical signal scaled by 100 — do NOT treat the two as independent
    features; they are perfectly correlated.

    First `period` values are NaN. Non-finite/non-positive prices yield NaN.
    """
    _validate_period(period)
    n = len(prices)
    if n == 0:
        return []
    result: list[float] = [_NAN] * n
    for i in range(period, n):
        base, curr = prices[i - period], prices[i]
        if _valid_price(base) and _valid_price(curr):
            val = (curr - base) / base
            if math.isfinite(val):
                result[i] = val
    return result


def rate_of_change(prices: Sequence[float], period: int = 10) -> list[float]:
    """Rate of Change: momentum * 100. Same signal as `momentum`, percentage
    units — not an independent feature. First `period` values are NaN.

    The post-multiply by 100 can overflow even when momentum itself is finite
    (e.g. momentum ≈ 1e307 → 1e309 → inf). Finite-guarded so the output contract
    (NaN for dirty / unrepresentable) holds.
    """
    raw = momentum(prices, period=period)
    return [
        (v * 100.0) if (v == v and math.isfinite(v * 100.0)) else _NAN
        for v in raw
    ]


# --- rsi ---------------------------------------------------------------------

def rsi(prices: Sequence[float], period: int = 14) -> list[float]:
    """Relative Strength Index using Wilder's smoothed EMA.

    First `period` values are NaN (period changes need period+1 prices). Valid
    outputs are in [0, 100]. A flat window (no gains and no losses) is 50 —
    neutral — not 100. period must be >= 2.

    Dirty data segment-resets: an invalid price transition discards the running
    averages; RSI re-seeds only after `period` consecutive valid transitions, so
    a single bad bar does not emit a misleading value or poison the tail.

    Uses an incremental mean (Welford-style) for the seed phase so that large
    gains/losses cannot overflow `sum(seed_gains)` before the division; Wilder
    updating uses the numerically equivalent `avg += (val - avg) / period`
    recurrence for the same reason.
    """
    _validate_period(period)
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}")
    n = len(prices)
    if n == 0:
        return []
    result: list[float] = [_NAN] * n

    valid_run = 0
    avg_gain = avg_loss = _NAN

    for i in range(1, n):
        if not (_valid_price(prices[i - 1]) and _valid_price(prices[i])):
            valid_run = 0
            avg_gain = avg_loss = _NAN
            continue

        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        valid_run += 1

        if valid_run < period:
            # Incremental mean: numerically stable, overflow-resistant
            if valid_run == 1:
                avg_gain = gain
                avg_loss = loss
            else:
                avg_gain += (gain - avg_gain) / valid_run
                avg_loss += (loss - avg_loss) / valid_run
            continue
        if valid_run == period:
            # Transition from seed to Wilder: fold last sample into the seed mean
            avg_gain += (gain - avg_gain) / period
            avg_loss += (loss - avg_loss) / period
        else:
            # Wilder EMA: avg += (val - avg) / period
            avg_gain += (gain - avg_gain) / period
            avg_loss += (loss - avg_loss) / period

        # Guard: the incremental recurrence can still overflow at extreme inputs.
        if not (math.isfinite(avg_gain) and math.isfinite(avg_loss)):
            valid_run = 0
            avg_gain = avg_loss = _NAN
            continue

        if avg_gain == 0.0 and avg_loss == 0.0:
            result[i] = 50.0
        elif avg_loss == 0.0:
            result[i] = 100.0
        elif avg_gain == 0.0:
            result[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


# --- atr ---------------------------------------------------------------------

def compute_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float]:
    """Average True Range (Wilder's smoothing).

    First `period` values are NaN. atr[period] is the simple average of the
    first `period` true ranges; later values use Wilder's smoothed EMA.

    Dirty data segment-resets like rsi: an invalid bar (non-finite/non-positive
    high/low/close, or high < low) discards the running ATR and re-seeds after
    `period` consecutive valid bars.
    """
    _validate_period(period)
    _validate_equal_lengths(highs, lows, closes)
    n = len(highs)
    result: list[float] = [_NAN] * n

    valid_run = 0
    atr = _NAN

    for i in range(1, n):
        hi, lo, cl = highs[i], lows[i], closes[i]
        prev_hi, prev_lo, prev_cl = highs[i - 1], lows[i - 1], closes[i - 1]
        # Both the current bar AND the previous bar must be valid HLC bars,
        # since the previous bar's close feeds the true range calculation.
        # A prior bar that was impossible (high < low, etc.) is dirty data
        # and must not contaminate the current true range.
        if not (_valid_hlc(hi, lo, cl) and _valid_hlc(prev_hi, prev_lo, prev_cl)):
            valid_run = 0
            atr = _NAN
            continue

        tr = max(hi - lo, abs(hi - prev_cl), abs(lo - prev_cl))
        valid_run += 1

        if valid_run == 1:
            # First clean bar after reset: seed ATR directly.
            atr = tr
            if period == 1:
                # Period 1: first bar is both seed and output. Falling through to
                # the `valid_run >= period` check below emits the value.
                pass
            else:
                # Period > 1: building seed mean; not enough bars yet to emit.
                continue
        elif valid_run < period:
            # Incremental mean for remaining seed bars.
            atr += (tr - atr) / valid_run
            continue
        elif valid_run == period:
            # Transition: fold the last seed observation, then switch to EMA.
            atr += (tr - atr) / period
        else:
            # Wilder EMA: avg += (val - avg) / period
            atr += (tr - atr) / period

        if math.isfinite(atr):
            result[i] = atr
        else:
            valid_run = 0
            atr = _NAN
    return result


# --- volatility --------------------------------------------------------------

def rolling_std(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling population std (ddof=0). First `period-1` values are NaN.
    A window containing any non-finite value yields NaN.

    Each window is scaled by its max absolute value before computing variance
    via Welford's algorithm, then the std is unscaled. This prevents both
    overflow and underflow in the ``delta * (x - mean)`` intermediate product,
    so that large but representable stds (e.g. 5e307) and tiny but non-zero
    stds (e.g. 5e-301) are preserved rather than silently NaN'd or zeroed.
    """
    _validate_period(period)
    n = len(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        s = max(abs(x) for x in window)
        if s == 0.0:
            result[i] = 0.0
            continue
        scaled = [x / s for x in window]
        mean = 0.0
        m2 = 0.0
        for k, x in enumerate(scaled):
            delta = x - mean
            mean += delta / (k + 1)
            m2 += delta * (x - mean)
        try:
            val = s * math.sqrt(m2 / period)
            if math.isfinite(val):
                result[i] = val
        except (OverflowError, ValueError):
            continue
    return result


def parkinson_vol(
    highs: Sequence[float],
    lows: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Parkinson (range-based) volatility: sqrt(1/(4 ln2) * mean(ln(H/L)^2)).

    First `period-1` values are NaN. A window containing any invalid bar
    (non-finite/non-positive high/low, or high < low) yields NaN — a partial
    window is not divided by the full period (which would understate vol).
    """
    _validate_period(period)
    _validate_equal_lengths(highs, lows)
    n = len(highs)
    result: list[float] = [_NAN] * n
    c = 1.0 / (4.0 * math.log(2.0))
    for i in range(period - 1, n):
        sum_sq = 0.0
        ok = True
        for j in range(i - period + 1, i + 1):
            if not (_valid_price(highs[j]) and _valid_price(lows[j]) and highs[j] >= lows[j]):
                ok = False
                break
            # log(H) - log(L) = log(H/L) — mathematically identical but avoids
            # forming the intermediate ratio which can overflow at extreme prices.
            log_ratio = math.log(highs[j]) - math.log(lows[j])
            if not math.isfinite(log_ratio):
                ok = False
                break
            sq = log_ratio * log_ratio
            if not math.isfinite(sq):
                ok = False
                break
            sum_sq += sq
        if not ok:
            continue
        val = math.sqrt(c * sum_sq / period)
        if math.isfinite(val):
            result[i] = val
    return result


# --- rolling -----------------------------------------------------------------

def rolling_apply(
    values: Sequence[T],
    window: int,
    func: Callable[[list[T]], R],
    min_periods: int | None = None,
) -> list[R | None]:
    """Apply `func` to each rolling window. Before min_periods, returns None.

    Generic combinator: missing-data policy is delegated to `func`. The module's
    purity/causality guarantee holds only if `func` is itself pure and causal —
    that is the caller's precondition, not something this function can enforce.
    """
    _validate_period(window)
    if min_periods is None:
        min_periods = window
    else:
        _validate_period(min_periods)
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})"
            )
    n = len(values)
    result: list[R | None] = [None] * n
    for i in range(n):
        start = max(0, i - window + 1)
        window_values = list(values[start : i + 1])
        if len(window_values) >= min_periods:
            result[i] = func(window_values)
    return result


def rolling_max(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling max. First `period-1` values NaN. A window with any non-finite
    value (NaN or inf) yields NaN — deterministic regardless of position (plain
    max is order-dependent on NaN, and would let inf through)."""
    _validate_period(period)
    n = len(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        result[i] = max(window)
    return result


def rolling_min(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling min. First `period-1` values NaN. A window with any non-finite
    value (NaN or inf) yields NaN (deterministic)."""
    _validate_period(period)
    n = len(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        result[i] = min(window)
    return result


def rolling_mean(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling mean. First `period-1` values NaN. A window with any non-finite
    value yields NaN.

    Each window is scaled by its max absolute value so that opposite-sign
    extremes (e.g. [-MAX, +MAX] → 0.0) do not cause the incremental-mean
    subtraction to overflow.
    """
    _validate_period(period)
    n = len(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        s = max(abs(x) for x in window)
        if s == 0.0:
            result[i] = 0.0
            continue
        scaled = [x / s for x in window]
        mean = 0.0
        for k, x in enumerate(scaled):
            mean += (x - mean) / (k + 1)
        val = mean * s
        if math.isfinite(val):
            result[i] = val
    return result


# --- candle geometry ---------------------------------------------------------

def body_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """|close - open| / (high - low). NaN for an impossible bar or high == low."""
    _validate_equal_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [_NAN] * n
    for i in range(n):
        if not _valid_ohlc(opens[i], highs[i], lows[i], closes[i]):
            continue
        denom = highs[i] - lows[i]
        result[i] = _NAN if denom == 0.0 else abs(closes[i] - opens[i]) / denom
    return result


def upper_wick_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """(high - max(open, close)) / (high - low). NaN for impossible bar/high==low."""
    _validate_equal_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [_NAN] * n
    for i in range(n):
        if not _valid_ohlc(opens[i], highs[i], lows[i], closes[i]):
            continue
        denom = highs[i] - lows[i]
        if denom != 0.0:
            result[i] = (highs[i] - max(opens[i], closes[i])) / denom
    return result


def lower_wick_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """(min(open, close) - low) / (high - low). NaN for impossible bar/high==low."""
    _validate_equal_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [_NAN] * n
    for i in range(n):
        if not _valid_ohlc(opens[i], highs[i], lows[i], closes[i]):
            continue
        denom = highs[i] - lows[i]
        if denom != 0.0:
            result[i] = (min(opens[i], closes[i]) - lows[i]) / denom
    return result


# --- spread estimators -------------------------------------------------------

def parkinson_spread(highs: Sequence[float], lows: Sequence[float]) -> list[float]:
    """Per-bar Parkinson high-low spread proxy, in ABSOLUTE price units:
    sqrt((H-L)^2 / (4 ln2)). NaN for an invalid bar.

    Note the unit: this is a price-level spread, unlike parkinson_vol which is a
    log-return volatility. They are different physical quantities; do not mix.
    """
    _validate_equal_lengths(highs, lows)
    n = len(highs)
    result: list[float] = [_NAN] * n
    sqrt_inv_4ln2 = math.sqrt(1.0 / (4.0 * math.log(2.0)))
    for i in range(n):
        hi, lo = highs[i], lows[i]
        if not (_valid_price(hi) and _valid_price(lo) and hi >= lo):
            continue
        # |H-L| * sqrt(1/(4ln2)) = sqrt((H-L)²/(4ln2)) — mathematically identical
        # but avoids the intermediate diff² overflow when diff ≈ 1e154.
        diff = abs(hi - lo)
        val = diff * sqrt_inv_4ln2
        if math.isfinite(val):
            result[i] = val
    return result


def corwin_schultz_spread(highs: Sequence[float], lows: Sequence[float]) -> list[float]:
    """Corwin-Schultz (2012) two-day high-low spread estimator.

    First value is NaN (needs two bars). Negative estimates (variance term
    dominates) are clamped to zero. Uses 2*tanh(alpha/2) — the overflow-safe
    form of 2*(e^alpha - 1)/(1 + e^alpha) — so large alpha can't overflow;
    the spread stays bounded below 2 as the theory requires.
    """
    _validate_equal_lengths(highs, lows)
    n = len(highs)
    result: list[float] = [_NAN] * n
    inv_denom = 1.0 / (3.0 - 2.0 * math.sqrt(2.0))
    for i in range(1, n):
        h1, l1, h2, l2 = highs[i - 1], lows[i - 1], highs[i], lows[i]
        if not all(math.isfinite(v) and v > 0 for v in (h1, l1, h2, l2)):
            continue
        if h1 < l1 or h2 < l2:
            continue
        beta = (math.log(h1) - math.log(l1)) ** 2 + (math.log(h2) - math.log(l2)) ** 2
        gamma = (math.log(max(h1, h2)) - math.log(min(l1, l2))) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) * inv_denom - math.sqrt(
            gamma * inv_denom
        )
        # Negative-alpha estimates are theoretically a negative spread (variance
        # dominates). Clamp to zero: the spread cannot be negative, and emitting
        # NaN for clean data violates the "clean regions stay valid" contract.
        alpha = max(0.0, alpha)
        result[i] = 2.0 * math.tanh(alpha / 2.0)
    return result


# --- microstructure ----------------------------------------------------------

def dollar_volume(prices: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """Dollar volume per bar: price * volume. Raises on length mismatch; NaN
    where price is not positive or volume is negative or either is non-finite."""
    _validate_equal_lengths(prices, volumes)
    n = len(prices)
    if n == 0:
        return []
    p = np.asarray(prices, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(v) & (p > 0.0) & (v >= 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        product = p * v
    result = np.where(valid & np.isfinite(product), product, np.nan)
    return result.tolist()


def amihud_illiquidity(
    returns: Sequence[float],
    dollar_volumes: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Amihud (2002) illiquidity: mean(|r| / dollar_volume) over the window.

    The denominator must be DOLLAR volume (price * base volume), not raw
    base-asset volume — otherwise the ratio is not comparable across symbols.
    Build it with `dollar_volume(...)`. Higher = less liquid. First `period-1`
    values NaN. Strict window: if any observation in the window is invalid
    (non-finite return, non-positive dollar volume), the result is NaN — the
    sample size is never silently shrunk.

    Each window is summed independently via ``math.fsum`` (Kahan-style
    compensated summation). The old global ``np.cumsum`` approach suffered
    catastrophic cancellation: a large early contribution left the accumulator
    at a magnitude that swallowed later small entries even after the large
    value rotated out of the window, silently zeroing clean regions.
    """
    _validate_period(period)
    _validate_equal_lengths(returns, dollar_volumes)
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result.tolist()
    r = np.asarray(returns, dtype=np.float64)
    dv = np.asarray(dollar_volumes, dtype=np.float64)
    base_valid = (dv > 0) & np.isfinite(r) & np.isfinite(dv)
    with np.errstate(over="ignore", invalid="ignore"):
        contrib_raw = np.abs(r) / np.where(base_valid, dv, 1.0)
    contrib_valid = base_valid & np.isfinite(contrib_raw)

    for i in range(period - 1, n):
        start = i - period + 1
        window_valid = contrib_valid[start : i + 1]
        if not np.all(window_valid):
            continue
        try:
            total = math.fsum(float(x) for x in contrib_raw[start : i + 1])
        except OverflowError:
            continue
        val = total / period
        if math.isfinite(val):
            result[i] = val
    return result.tolist()


def roll_spread_estimator(prices: Sequence[float], period: int = 20) -> list[float]:
    """Roll (1984) spread from serial covariance of price changes.

    S = 2 * sqrt(max(0, -cov(dp_t, dp_{t-1}))). First `period` values are NaN.
    Strict window: each estimate uses the `period+1` prices prices[i-period..i];
    if any is invalid the result is NaN, so every estimate uses the full,
    equal-size sample of `period` deltas (no first-window off-by-one, no silently
    shrunk covariance).
    """
    _validate_period(period)
    if period < 3:
        raise ValueError(f"Roll spread period must be >= 3, got {period}")
    n = len(prices)
    result: list[float] = [_NAN] * n
    if n < period + 1:
        return result

    for i in range(period, n):
        window = prices[i - period : i + 1]  # period + 1 prices
        if any(not _valid_price(x) for x in window):
            continue
        deltas = [window[k] - window[k - 1] for k in range(1, len(window))]  # period
        lead = deltas[1:]   # dp_t
        lag = deltas[:-1]   # dp_{t-1}
        m = len(lead)       # period - 1 pairs

        # Scale deltas by max absolute delta to prevent covariance overflow.
        # Rescaling AFTER the sqrt (cov_scaled is in [-1,1]) keeps subnormal
        # spreads representable: cov_scaled * scale^2 can underflow to 0.0
        # while scale * sqrt(-cov_scaled) stays non-zero.
        scale = max(abs(x) for x in deltas)
        if scale != 0.0:
            scaled_deltas = [x / scale for x in deltas]
            scaled_lead = scaled_deltas[1:]
            scaled_lag = scaled_deltas[:-1]
            mean_l = sum(scaled_lead) / m
            mean_g = sum(scaled_lag) / m
            cov_scaled = sum(
                (a - mean_l) * (b - mean_g)
                for a, b in zip(scaled_lead, scaled_lag)
            ) / m
            if cov_scaled < 0:
                val = scale * (2.0 * math.sqrt(-cov_scaled))
                if math.isfinite(val):
                    result[i] = val
            elif math.isfinite(cov_scaled):
                result[i] = 0.0
        else:
            result[i] = 0.0
    return result


# --- volume profile ----------------------------------------------------------

def typical_price(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Typical price: (high + low + close) / 3. NaN for an invalid HLC bar.

    Computed as h/3 + l/3 + c/3 rather than (h+l+c)/3 so the intermediate sum
    does not overflow when all three prices are near the float ceiling — the
    final result is representable even when h+l+c is not.
    """
    _validate_equal_lengths(highs, lows, closes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    inv3 = 1.0 / 3.0
    for i in range(n):
        if _valid_hlc(highs[i], lows[i], closes[i]):
            val = highs[i] * inv3 + lows[i] * inv3 + closes[i] * inv3
            if math.isfinite(val):
                result[i] = val
    return result


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Cumulative VWAP from the start. An invalid bar (bad HLC, non-finite or
    negative volume) segment-resets the accumulator: its output is NaN and the
    running totals are discarded, so VWAP restarts fresh from the next clean bar
    (consistent with rsi/atr reset; matches the module data-validity contract)."""
    _validate_equal_lengths(highs, lows, closes, volumes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    cum_pv = cum_v = 0.0
    inv3 = 1.0 / 3.0
    for i in range(n):
        hi, lo, cl, vol = highs[i], lows[i], closes[i], volumes[i]
        if not (_valid_hlc(hi, lo, cl) and math.isfinite(vol) and vol >= 0.0):
            cum_pv = cum_v = 0.0  # segment reset
            continue
        # Safe TP: divide-then-add, same as typical_price.
        tp = hi * inv3 + lo * inv3 + cl * inv3
        if not math.isfinite(tp):
            cum_pv = cum_v = 0.0
            continue
        # Zero volume: emit current VWAP but do not affect accumulator.
        # Zero-volume bars are valid data (no trades), not a reason to reset.
        if vol == 0.0:
            if cum_v > 0:
                result[i] = cum_pv / cum_v
            continue
        # Rescale accumulators before adding when either is near max float.
        # Halving both preserves VWAP = cum_pv / cum_v exactly, and the new
        # bar is also halved so the relative weight is unchanged.
        thresh = float.fromhex("0x1.fffffffffffffp+1022")  # ~MAX/2
        if abs(cum_pv) > thresh or abs(cum_v) > thresh:
            cum_pv *= 0.5
            cum_v *= 0.5
            contrib = tp * vol * 0.5
            if not math.isfinite(contrib):
                cum_pv = cum_v = 0.0
                continue
            cum_pv += contrib
            cum_v += vol * 0.5
        else:
            contrib = tp * vol
            if not math.isfinite(contrib):
                cum_pv = cum_v = 0.0
                continue
            cum_pv += contrib
            cum_v += vol
        if cum_v > 0:
            val = cum_pv / cum_v
            if math.isfinite(val):
                result[i] = val
    return result


def rolling_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Rolling VWAP over a fixed window. First `period-1` values are NaN. A window
    containing any invalid bar yields NaN.

    Uses safe TP (divide-then-add) to avoid intermediate overflow in the price
    sum; per-bar contribution and window totals are likewise overflow-guarded.
    """
    _validate_period(period)
    _validate_equal_lengths(highs, lows, closes, volumes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    inv3 = 1.0 / 3.0
    for i in range(period - 1, n):
        cum_pv = cum_v = 0.0
        ok = True
        for j in range(i - period + 1, i + 1):
            hi, lo, cl, vol = highs[j], lows[j], closes[j], volumes[j]
            if not (_valid_hlc(hi, lo, cl) and math.isfinite(vol) and vol >= 0.0):
                ok = False
                break
            tp = hi * inv3 + lo * inv3 + cl * inv3
            contrib = tp * vol
            if not (math.isfinite(tp) and math.isfinite(contrib)):
                ok = False
                break
            cum_pv += contrib
            cum_v += vol
        if ok and cum_v > 0 and math.isfinite(cum_pv) and math.isfinite(cum_v):
            val = cum_pv / cum_v
            if math.isfinite(val):
                result[i] = val
    return result
