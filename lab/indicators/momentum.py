"""
Momentum and Rate of Change indicators.

Pure math — no state, no adapters, no business logic.
"""

from typing import Sequence


def momentum(prices: Sequence[float], period: int = 10) -> list[float]:
    """Compute momentum: (P_t - P_{t-period}) / P_{t-period}.

    Args:
        prices: Price sequence.
        period: Lookback period (default 10).

    Returns:
        List of momentum values (same length).
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
    """Compute Rate of Change as momentum * 100.

    Args:
        prices: Price sequence.
        period: Lookback period (default 10).

    Returns:
        List of ROC values (same length).
        First `period` values are NaN. Zero/negative base price returns NaN.
    """
    raw = momentum(prices, period=period)
    return [v * 100 if not _is_nan(v) else v for v in raw]


def _is_nan(x: float) -> bool:
    return x != x
