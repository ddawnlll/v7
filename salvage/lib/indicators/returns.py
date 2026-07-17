"""
Return calculations (log and simple).

Pure math — no state, no adapters.
"""

from typing import Sequence
import math


def log_returns(prices: Sequence[float]) -> list[float]:
    """Compute log returns: ln(P_t / P_{t-1}).

    First element is NaN (no prior price).
    """
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
    """Compute simple returns: (P_t - P_{t-1}) / P_{t-1}.

    First element is NaN.
    """
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
