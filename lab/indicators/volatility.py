"""
Volatility calculations.

Pure math — no state, no adapters.
"""

from typing import Sequence
import math


def rolling_std(values: Sequence[float], period: int = 20) -> list[float]:
    """Rolling standard deviation.

    Returns list of same length as values. First `period-1` values are NaN.
    Uses population std (ddof=0).
    """
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
    """Parkinson (range-based) volatility estimator.

    vol = sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))

    Returns list of same length. First `period-1` values are NaN.
    """
    n = len(highs)
    result: list[float] = [float("nan")] * n
    c = 1.0 / (4.0 * math.log(2.0))

    for i in range(period - 1, n):
        sum_sq = 0.0
        for j in range(i - period + 1, i + 1):
            if highs[j] > 0 and lows[j] > 0:
                hl_ratio = highs[j] / lows[j]
                sum_sq += math.log(hl_ratio) ** 2
        result[i] = math.sqrt(c * sum_sq / period)
    return result
