"""
Bid-ask spread estimators from OHLCV data.

Pure math -- no state, no adapters.
"""

from typing import Sequence
import math


def parkinson_spread(
    highs: Sequence[float],
    lows: Sequence[float],
) -> list[float]:
    """Per-bar Parkinson high-low spread proxy.

    Uses the high-low ratio scaled by the Parkinson (1980) constant
    to produce an effective-spread estimate for each bar:

        spread_i = sqrt((high_i - low_i)^2 / (4 * ln(2)))

    Returns:
        List of spread estimates, same length as inputs.
    """
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
    """Rolling Parkinson high-low spread estimator.

    Averages the squared log-ratio over a rolling window:

        S = sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))

    Args:
        highs: High prices.
        lows: Low prices.
        period: Lookback window (default 20).

    Returns:
        List of spread estimates, same length as inputs.
        First ``period - 1`` values are NaN.
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


def corwin_schultz_spread(
    highs: Sequence[float],
    lows: Sequence[float],
) -> list[float]:
    """Corwin-Schultz (2012) two-day high-low spread estimator.

    Separates variance from the bid-ask spread by comparing one-day
    and two-day high-low ratios.  Negative spread estimates (which
    occur when the variance term dominates) are clamped to NaN.

    Reference:
        Corwin & Schultz (2012) -- "A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices"

    Returns:
        List of spread estimates, same length as inputs.
        First value is NaN (requires two consecutive bars).
        NaN entries where the estimate is negative.
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

        sqrt_2beta = math.sqrt(2.0 * beta)
        sqrt_beta = math.sqrt(beta)

        alpha = (sqrt_2beta - sqrt_beta) * inv_denom - math.sqrt(gamma * inv_denom)

        if alpha <= 0:
            continue

        spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        result[i] = spread

    return result
