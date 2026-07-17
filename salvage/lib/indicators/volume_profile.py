"""
Volume-weighted average price (VWAP) and volume profile.

Pure math -- no state, no adapters.
"""

from typing import Sequence


def typical_price(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Typical price: (high + low + close) / 3.

    Returns:
        List of typical prices, same length as inputs.
    """
    n = len(highs)
    result: list[float] = [0.0] * n
    for i in range(n):
        result[i] = (highs[i] + lows[i] + closes[i]) / 3.0
    return result


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Cumulative volume-weighted average price from the start.

    VWAP = sum(typical_price * volume) / sum(volume)

    Args:
        highs: High prices.
        lows: Low prices.
        closes: Close prices.
        volumes: Volume values.

    Returns:
        List of VWAP values, same length as inputs.
        First entry equals the first typical price.
        Entries with zero total-volume are NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    cum_pv = 0.0
    cum_v = 0.0
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
    """Rolling VWAP over a fixed lookback window.

    Args:
        highs: High prices.
        lows: Low prices.
        closes: Close prices.
        volumes: Volume values.
        period: Lookback window (default 20).

    Returns:
        List of rolling VWAP values, same length as inputs.
        First ``period - 1`` values are NaN.
        Entries with zero total-volume over the window are NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        cum_pv = 0.0
        cum_v = 0.0
        for j in range(i - period + 1, i + 1):
            tp = (highs[j] + lows[j] + closes[j]) / 3.0
            cum_pv += tp * volumes[j]
            cum_v += volumes[j]
        if cum_v > 0:
            result[i] = cum_pv / cum_v
    return result
