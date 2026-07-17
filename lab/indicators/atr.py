"""
Average True Range (ATR) calculation.

Pure math — no state, no adapters, no business logic.
"""

from typing import Sequence


def compute_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float]:
    """Compute Average True Range over the given price series.

    Args:
        highs: High prices (length >= period + 1).
        lows: Low prices (same length).
        closes: Close prices (same length).
        period: Lookback window (default 14).

    Returns:
        List of ATR values, one per bar starting at index `period`.
        First `period` values are NaN (no ATR available yet).
    """
    n = len(highs)
    if n < period + 1:
        return [float("nan")] * n

    # True Range for each bar
    tr: list[float] = [float("nan")] * n
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # First ATR is simple average of first `period` TR values
    atr: list[float] = [float("nan")] * n
    atr[period] = sum(tr[1 : period + 1]) / period

    # Subsequent ATR uses Wilder's smoothed EMA
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr
