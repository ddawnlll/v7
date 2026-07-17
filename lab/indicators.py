"""Pure, causal indicator primitives — one audit unit.

Every function is a pure function of past-and-present bars only: appending
future bars must never change an already-computed value (RULES §6, enforced by
lab/tests/test_indicators.py). No state, no adapters, no business logic.

This module is never imported by lab/sim.py — research primitives and the
economic truth core stay on opposite sides of that line (RULES §4).

Contents:
  returns        log_returns, simple_returns
  momentum       momentum, rate_of_change
  rsi            rsi
  atr            compute_atr
  volatility     rolling_std, parkinson_vol
  rolling        rolling_apply, rolling_max, rolling_min, rolling_mean
  candle         body_ratio, upper_wick_ratio, lower_wick_ratio
  spread         parkinson_spread, rolling_parkinson_spread, corwin_schultz_spread
  microstructure dollar_volume, amihud_illiquidity, roll_spread_estimator
  volume_profile typical_price, vwap, rolling_vwap
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


# --- returns -----------------------------------------------------------------

def log_returns(prices: Sequence[float]) -> list[float]:
    """Log returns: ln(P_t / P_{t-1}). First element is NaN (no prior price)."""
    n = len(prices)
    if n < 2:
        return [float("nan")] * n
    result: list[float] = [float("nan")] * n
    for i in range(1, n):
        if prices[i - 1] > 0 and prices[i] > 0:
            result[i] = math.log(prices[i] / prices[i - 1])
        else:
            result[i] = float("nan")
    return result


def simple_returns(prices: Sequence[float]) -> list[float]:
    """Simple returns: (P_t - P_{t-1}) / P_{t-1}. First element is NaN."""
    n = len(prices)
    if n < 2:
        return [float("nan")] * n
    result: list[float] = [float("nan")] * n
    for i in range(1, n):
        if prices[i - 1] > 0:
            result[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
        else:
            result[i] = float("nan")
    return result


# --- momentum ----------------------------------------------------------------

def momentum(prices: Sequence[float], period: int = 10) -> list[float]:
    """Momentum: (P_t - P_{t-period}) / P_{t-period}.

    First `period` values are NaN. Zero/negative base price returns NaN.
    """
    n = len(prices)
    if n == 0:
        return []
    result: list[float] = [float("nan")] * n
    for i in range(period, n):
        if prices[i - period] <= 0:
            continue
        result[i] = (prices[i] - prices[i - period]) / prices[i - period]
    return result


def rate_of_change(prices: Sequence[float], period: int = 10) -> list[float]:
    """Rate of Change: momentum * 100. First `period` values are NaN."""
    raw = momentum(prices, period=period)
    return [v * 100 if v == v else v for v in raw]  # v == v is False for NaN


# --- rsi ---------------------------------------------------------------------

def rsi(prices: Sequence[float], period: int = 14) -> list[float]:
    """Relative Strength Index using Wilder's smoothed EMA.

    First `period` values are NaN. Valid outputs are in [0, 100].
    """
    n = len(prices)
    if n == 0:
        return []
    if period == 1:
        return [float("nan")] * n
    result: list[float] = [float("nan")] * n
    if n < period + 1:
        return result

    gains: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(1, n):
        if prices[i - 1] > 0 and prices[i] > 0:
            delta = prices[i] - prices[i - 1]
            if delta > 0:
                gains[i] = delta
            else:
                losses[i] = -delta

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if prices[i] <= 0:
            continue
        if avg_loss == 0.0:
            result[i] = 100.0
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
    """
    n = len(highs)
    if n < period + 1:
        return [float("nan")] * n

    tr: list[float] = [float("nan")] * n
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr: list[float] = [float("nan")] * n
    atr[period] = sum(tr[1 : period + 1]) / period
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# --- volatility --------------------------------------------------------------

def rolling_std(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling population std (ddof=0). First `period-1` values are NaN."""
    n = len(values)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
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

    First `period-1` values are NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    c = 1.0 / (4.0 * math.log(2.0))
    for i in range(period - 1, n):
        sum_sq = 0.0
        for j in range(i - period + 1, i + 1):
            if highs[j] > 0 and lows[j] > 0:
                sum_sq += math.log(highs[j] / lows[j]) ** 2
        result[i] = math.sqrt(c * sum_sq / period)
    return result


# --- rolling -----------------------------------------------------------------

def rolling_apply(
    values: Sequence[T],
    window: int,
    func: Callable[[list[T]], R],
    min_periods: int | None = None,
) -> list[R | None]:
    """Apply `func` to each rolling window. Before min_periods, returns None."""
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
    """Rolling max. First `period-1` values NaN. period=1 returns values."""
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        result[i] = max(values[i - period + 1 : i + 1])
    return result


def rolling_min(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling min. First `period-1` values NaN. period=1 returns values."""
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        result[i] = min(values[i - period + 1 : i + 1])
    return result


def rolling_mean(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling mean. First `period-1` values NaN. period=1 returns values."""
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        result[i] = sum(values[i - period + 1 : i + 1]) / period
    return result


# --- candle geometry ---------------------------------------------------------

def _validate_ohlc_lengths(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> None:
    n = len(opens)
    if len(highs) != n or len(lows) != n or len(closes) != n:
        raise ValueError("all input sequences must have the same length")


def body_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """|close - open| / (high - low). NaN when high == low; doji → 0.0."""
    _validate_ohlc_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = highs[i] - lows[i]
        result[i] = float("nan") if denom == 0.0 else abs(closes[i] - opens[i]) / denom
    return result


def upper_wick_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """(high - max(open, close)) / (high - low). NaN when high == low."""
    _validate_ohlc_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = highs[i] - lows[i]
        if denom == 0.0:
            result[i] = float("nan")
        else:
            result[i] = (highs[i] - max(opens[i], closes[i])) / denom
    return result


def lower_wick_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """(min(open, close) - low) / (high - low). NaN when high == low."""
    _validate_ohlc_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = highs[i] - lows[i]
        if denom == 0.0:
            result[i] = float("nan")
        else:
            result[i] = (min(opens[i], closes[i]) - lows[i]) / denom
    return result


# --- spread estimators -------------------------------------------------------

def parkinson_spread(highs: Sequence[float], lows: Sequence[float]) -> list[float]:
    """Per-bar Parkinson high-low spread proxy: sqrt((H-L)^2 / (4 ln2))."""
    n = len(highs)
    result: list[float] = [0.0] * n
    inv_4ln2 = 1.0 / (4.0 * math.log(2.0))
    for i in range(n):
        diff = highs[i] - lows[i]
        result[i] = math.sqrt(diff * diff * inv_4ln2)
    return result


def rolling_parkinson_spread(
    highs: Sequence[float],
    lows: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Rolling Parkinson spread: sqrt(1/(4 ln2) * mean(ln(H/L)^2)).

    First `period-1` values are NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    c = 1.0 / (4.0 * math.log(2.0))
    for i in range(period - 1, n):
        sum_sq = 0.0
        for j in range(i - period + 1, i + 1):
            if highs[j] > 0 and lows[j] > 0:
                sum_sq += math.log(highs[j] / lows[j]) ** 2
        result[i] = math.sqrt(c * sum_sq / period)
    return result


def corwin_schultz_spread(highs: Sequence[float], lows: Sequence[float]) -> list[float]:
    """Corwin-Schultz (2012) two-day high-low spread estimator.

    First value is NaN (needs two bars). Negative estimates (variance term
    dominates) are clamped to NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    inv_denom = 1.0 / (3.0 - 2.0 * math.sqrt(2.0))
    for i in range(1, n):
        h1, l1 = highs[i - 1], lows[i - 1]
        h2, l2 = highs[i], lows[i]
        if any(v <= 0 for v in (h1, l1, h2, l2)):
            continue
        beta = math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2
        gamma = math.log(max(h1, h2) / min(l1, l2)) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) * inv_denom - math.sqrt(
            gamma * inv_denom
        )
        if alpha <= 0:
            continue
        result[i] = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return result


# --- microstructure ----------------------------------------------------------

def dollar_volume(prices: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """Dollar volume per bar: price * volume. Fails closed on length mismatch."""
    if len(prices) != len(volumes):
        raise ValueError("prices and volumes must have the same length")
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n > 0:
        p = np.asarray(prices, dtype=np.float64)
        v = np.asarray(volumes, dtype=np.float64)
        valid = ~(np.isnan(p) | np.isnan(v))
        result = np.where(valid, p * v, np.nan)
    return result.tolist()


def amihud_illiquidity(
    returns: Sequence[float],
    volumes: Sequence[float],
    period: int = 20,
) -> list[float]:
    """Amihud (2002) illiquidity: mean(|r| / V) over the window.

    Higher = less liquid. First `period-1` values NaN; zero-volume windows NaN.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return result.tolist()
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    valid = (v > 0) & ~np.isnan(r)
    safe_v = np.where(valid, v, 1.0)
    contrib = np.where(valid, np.abs(r) / safe_v, 0.0)
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
    """
    n = len(prices)
    if n < period + 1:
        return [float("nan")] * n

    deltas: list[float] = [float("nan")] * n
    for i in range(1, n):
        if not math.isnan(prices[i]) and not math.isnan(prices[i - 1]):
            deltas[i] = prices[i] - prices[i - 1]

    result: list[float] = [float("nan")] * n
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
    """Typical price: (high + low + close) / 3."""
    n = len(highs)
    return [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Cumulative VWAP from the start. Zero total-volume entries are NaN."""
    n = len(highs)
    result: list[float] = [float("nan")] * n
    cum_pv = cum_v = 0.0
    for i in range(n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_pv += tp * volumes[i]
        cum_v += volumes[i]
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
    """Rolling VWAP over a fixed window. First `period-1` values are NaN."""
    n = len(highs)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        cum_pv = cum_v = 0.0
        for j in range(i - period + 1, i + 1):
            tp = (highs[j] + lows[j] + closes[j]) / 3.0
            cum_pv += tp * volumes[j]
            cum_v += volumes[j]
        if cum_v > 0:
            result[i] = cum_pv / cum_v
    return result
