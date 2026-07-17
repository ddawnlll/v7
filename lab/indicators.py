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
- Stateful indicators (rsi, atr, cumulative vwap) segment-reset on dirty data:
  they discard their running state and re-seed after `period` consecutive clean
  observations, so one bad bar does not permanently poison the rest of the series.

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


# --- returns -----------------------------------------------------------------

def log_returns(prices: Sequence[float]) -> list[float]:
    """Log returns: ln(P_t) - ln(P_{t-1}). First element is NaN.

    Uses the difference of logs rather than log of the ratio: mathematically
    identical but avoids forming an intermediate ratio that could over/underflow
    at extreme magnitudes. NaN where either price is non-finite or non-positive.
    """
    n = len(prices)
    if n < 2:
        return [_NAN] * n
    result: list[float] = [_NAN] * n
    for i in range(1, n):
        if _valid_price(prices[i - 1]) and _valid_price(prices[i]):
            result[i] = math.log(prices[i]) - math.log(prices[i - 1])
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
            result[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
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
            result[i] = (curr - base) / base
    return result


def rate_of_change(prices: Sequence[float], period: int = 10) -> list[float]:
    """Rate of Change: momentum * 100. Same signal as `momentum`, percentage
    units — not an independent feature. First `period` values are NaN."""
    raw = momentum(prices, period=period)
    return [v * 100 if v == v else v for v in raw]  # v == v is False for NaN


# --- rsi ---------------------------------------------------------------------

def rsi(prices: Sequence[float], period: int = 14) -> list[float]:
    """Relative Strength Index using Wilder's smoothed EMA.

    First `period` values are NaN (period changes need period+1 prices). Valid
    outputs are in [0, 100]. A flat window (no gains and no losses) is 50 —
    neutral — not 100. period must be >= 2.

    Dirty data segment-resets: an invalid price transition discards the running
    averages; RSI re-seeds only after `period` consecutive valid transitions, so
    a single bad bar does not emit a misleading value or poison the tail.
    """
    _validate_period(period)
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}")
    n = len(prices)
    if n == 0:
        return []
    result: list[float] = [_NAN] * n

    valid_run = 0
    seed_gains: list[float] = []
    seed_losses: list[float] = []
    avg_gain = avg_loss = _NAN

    for i in range(1, n):
        if not (_valid_price(prices[i - 1]) and _valid_price(prices[i])):
            valid_run = 0
            seed_gains.clear()
            seed_losses.clear()
            avg_gain = avg_loss = _NAN
            continue

        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        valid_run += 1

        if valid_run < period:
            seed_gains.append(gain)
            seed_losses.append(loss)
            continue
        if valid_run == period:
            seed_gains.append(gain)
            seed_losses.append(loss)
            avg_gain = sum(seed_gains) / period
            avg_loss = sum(seed_losses) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

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
    seed_tr: list[float] = []
    atr = _NAN

    for i in range(1, n):
        hi, lo, cl, prev_cl = highs[i], lows[i], closes[i], closes[i - 1]
        if not (
            _valid_price(hi) and _valid_price(lo) and _valid_price(cl)
            and _valid_price(prev_cl) and hi >= lo
        ):
            valid_run = 0
            seed_tr.clear()
            atr = _NAN
            continue

        tr = max(hi - lo, abs(hi - prev_cl), abs(lo - prev_cl))
        valid_run += 1

        if valid_run < period:
            seed_tr.append(tr)
            continue
        if valid_run == period:
            seed_tr.append(tr)
            atr = sum(seed_tr) / period
        else:
            atr = (atr * (period - 1) + tr) / period
        result[i] = atr
    return result


# --- volatility --------------------------------------------------------------

def rolling_std(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling population std (ddof=0). First `period-1` values are NaN.
    A window containing any non-finite value yields NaN."""
    _validate_period(period)
    n = len(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        result[i] = math.sqrt(variance)
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
            sum_sq += math.log(highs[j] / lows[j]) ** 2
        if ok:
            result[i] = math.sqrt(c * sum_sq / period)
    return result


# --- rolling -----------------------------------------------------------------

def rolling_apply(
    values: Sequence[T],
    window: int,
    func: Callable[[list[T]], R],
    min_periods: int | None = None,
) -> list[R | None]:
    """Apply `func` to each rolling window. Before min_periods, returns None.
    Missing-data policy is delegated to `func` (this is a generic combinator)."""
    _validate_period(window)
    if min_periods is None:
        min_periods = window
    n = len(values)
    result: list[R | None] = [None] * n
    for i in range(n):
        start = max(0, i - window + 1)
        window_values = list(values[start : i + 1])
        if len(window_values) >= min_periods:
            result[i] = func(window_values)
    return result


def rolling_max(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling max. First `period-1` values NaN. A window with any NaN yields
    NaN — deterministic regardless of NaN position (plain max is order-dependent
    on NaN). period=1 returns values unchanged."""
    _validate_period(period)
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(math.isnan(x) for x in window):
            continue
        result[i] = max(window)
    return result


def rolling_min(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling min. First `period-1` values NaN. A window with any NaN yields
    NaN (deterministic). period=1 returns values unchanged."""
    _validate_period(period)
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(math.isnan(x) for x in window):
            continue
        result[i] = min(window)
    return result


def rolling_mean(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling mean. First `period-1` values NaN. A window with any non-finite
    value yields NaN. period=1 returns values unchanged."""
    _validate_period(period)
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(not math.isfinite(x) for x in window):
            continue
        result[i] = sum(window) / period
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
    inv_4ln2 = 1.0 / (4.0 * math.log(2.0))
    for i in range(n):
        hi, lo = highs[i], lows[i]
        if not (math.isfinite(hi) and math.isfinite(lo) and hi >= lo):
            continue
        diff = hi - lo
        result[i] = math.sqrt(diff * diff * inv_4ln2)
    return result


def corwin_schultz_spread(highs: Sequence[float], lows: Sequence[float]) -> list[float]:
    """Corwin-Schultz (2012) two-day high-low spread estimator.

    First value is NaN (needs two bars). Negative estimates (variance term
    dominates) are NaN. Uses 2*tanh(alpha/2) — the overflow-safe form of
    2*(e^alpha - 1)/(1 + e^alpha) — so large alpha can't overflow; the spread
    stays bounded below 2 as the theory requires.
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
        beta = math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2
        gamma = math.log(max(h1, h2) / min(l1, l2)) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) * inv_denom - math.sqrt(
            gamma * inv_denom
        )
        if alpha <= 0:
            continue
        result[i] = 2.0 * math.tanh(alpha / 2.0)
    return result


# --- microstructure ----------------------------------------------------------

def dollar_volume(prices: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """Dollar volume per bar: price * volume. Raises on length mismatch; NaN
    where either input is non-finite."""
    _validate_equal_lengths(prices, volumes)
    n = len(prices)
    if n == 0:
        return []
    p = np.asarray(prices, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(v)
    return np.where(valid, p * v, np.nan).tolist()


def amihud_illiquidity(
    returns: Sequence[float],
    dollar_volumes: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Amihud (2002) illiquidity: mean(|r| / dollar_volume) over the window.

    The denominator must be DOLLAR volume (price * base volume), not raw
    base-asset volume — otherwise the ratio is not comparable across symbols.
    Build it with `dollar_volume(...)`. Higher = less liquid. First `period-1`
    values NaN; windows with no valid observation are NaN.
    """
    _validate_period(period)
    _validate_equal_lengths(returns, dollar_volumes)
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result.tolist()
    r = np.asarray(returns, dtype=np.float64)
    dv = np.asarray(dollar_volumes, dtype=np.float64)
    valid = (dv > 0) & np.isfinite(r) & np.isfinite(dv)
    safe_dv = np.where(valid, dv, 1.0)
    contrib = np.where(valid, np.abs(r) / safe_dv, 0.0)
    csum = np.cumsum(np.insert(contrib, 0, 0.0))
    ccount = np.cumsum(np.insert(valid.astype(np.float64), 0, 0.0))
    idx = np.arange(period - 1, n)
    start = idx - period + 1
    total = csum[idx + 1] - csum[start]
    count = ccount[idx + 1] - ccount[start]
    ok = count > 0
    result[idx[ok]] = total[ok] / count[ok]
    return result.tolist()


def roll_spread_estimator(prices: Sequence[float], period: int = 20) -> list[float]:
    """Roll (1984) spread from serial covariance of price changes.

    S = 2 * sqrt(max(0, -cov(dp_t, dp_{t-1}))). First `period` values are NaN.
    Non-finite prices are excluded from the covariance window.
    """
    _validate_period(period)
    n = len(prices)
    if n < period + 1:
        return [_NAN] * n

    deltas: list[float] = [_NAN] * n
    for i in range(1, n):
        if math.isfinite(prices[i]) and math.isfinite(prices[i - 1]):
            deltas[i] = prices[i] - prices[i - 1]

    result: list[float] = [_NAN] * n
    for i in range(period, n):
        start = i - period + 1
        count = 0
        sum_d = sum_d_lag = 0.0
        for j in range(start, i + 1):
            if not math.isnan(deltas[j]) and not math.isnan(deltas[j - 1]):
                sum_d += deltas[j]
                sum_d_lag += deltas[j - 1]
                count += 1
        if count < 2:
            continue
        mean_d = sum_d / count
        mean_d_lag = sum_d_lag / count
        cov = 0.0
        valid = 0
        for j in range(start, i + 1):
            if not math.isnan(deltas[j]) and not math.isnan(deltas[j - 1]):
                cov += (deltas[j] - mean_d) * (deltas[j - 1] - mean_d_lag)
                valid += 1
        if valid >= 2:
            cov /= valid
            result[i] = 2.0 * math.sqrt(-cov) if cov < 0 else 0.0
    return result


# --- volume profile ----------------------------------------------------------

def typical_price(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Typical price: (high + low + close) / 3. NaN where any input is non-finite."""
    _validate_equal_lengths(highs, lows, closes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    for i in range(n):
        if math.isfinite(highs[i]) and math.isfinite(lows[i]) and math.isfinite(closes[i]):
            result[i] = (highs[i] + lows[i] + closes[i]) / 3.0
    return result


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Cumulative VWAP from the start. An invalid bar (non-finite price/volume or
    negative volume) is skipped: its output is NaN but the cumulative state is
    preserved, so one bad bar does not poison every later value."""
    _validate_equal_lengths(highs, lows, closes, volumes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    cum_pv = cum_v = 0.0
    for i in range(n):
        hi, lo, cl, vol = highs[i], lows[i], closes[i], volumes[i]
        if not (
            math.isfinite(hi) and math.isfinite(lo) and math.isfinite(cl)
            and math.isfinite(vol) and vol >= 0.0
        ):
            continue
        tp = (hi + lo + cl) / 3.0
        cum_pv += tp * vol
        cum_v += vol
        if cum_v > 0:
            result[i] = cum_pv / cum_v
    return result


def rolling_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Rolling VWAP over a fixed window. First `period-1` values are NaN. A window
    containing any invalid bar yields NaN."""
    _validate_period(period)
    _validate_equal_lengths(highs, lows, closes, volumes)
    n = len(highs)
    result: list[float] = [_NAN] * n
    for i in range(period - 1, n):
        cum_pv = cum_v = 0.0
        ok = True
        for j in range(i - period + 1, i + 1):
            hi, lo, cl, vol = highs[j], lows[j], closes[j], volumes[j]
            if not (
                math.isfinite(hi) and math.isfinite(lo) and math.isfinite(cl)
                and math.isfinite(vol) and vol >= 0.0
            ):
                ok = False
                break
            tp = (hi + lo + cl) / 3.0
            cum_pv += tp * vol
            cum_v += vol
        if ok and cum_v > 0:
            result[i] = cum_pv / cum_v
    return result
