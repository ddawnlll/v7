"""
RSI (Relative Strength Index) using Wilder's smoothed EMA.

Pure math — no state, no adapters, no business logic.
"""

from typing import Sequence


def rsi(prices: Sequence[float], period: int = 14) -> list[float]:
    """Compute Relative Strength Index using Wilder's smoothed EMA.

    Args:
        prices: Price sequence (length >= period + 1 for first valid value).
        period: Lookback window (default 14).

    Returns:
        List of RSI values (same length). First `period` values are NaN.
        RSI values are in [0, 100] for valid inputs.
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
