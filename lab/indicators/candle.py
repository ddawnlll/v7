"""
Candle geometry ratios.

Pure math — no state, no adapters, no business logic.
"""

from typing import Sequence


def body_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Compute the body-to-total-range ratio for each candle.

    body_ratio = |close - open| / (high - low)

    Args:
        opens: Open prices.
        highs: High prices.
        lows: Low prices.
        closes: Close prices.

    Returns:
        List of body ratios (same length). NaN when high == low.
        Doji (open == close) returns 0.0, not NaN.

    Raises:
        ValueError: If input sequences have different lengths.
    """
    _validate_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = highs[i] - lows[i]
        if denom == 0.0:
            result[i] = float("nan")
        else:
            result[i] = abs(closes[i] - opens[i]) / denom
    return result


def upper_wick_ratio(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Compute the upper wick ratio for each candle.

    upper_wick_ratio = (high - max(open, close)) / (high - low)

    Args:
        opens: Open prices.
        highs: High prices.
        lows: Low prices.
        closes: Close prices.

    Returns:
        List of upper wick ratios (same length). NaN when high == low.

    Raises:
        ValueError: If input sequences have different lengths.
    """
    _validate_lengths(opens, highs, lows, closes)
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
    """Compute the lower wick ratio for each candle.

    lower_wick_ratio = (min(open, close) - low) / (high - low)

    Args:
        opens: Open prices.
        highs: High prices.
        lows: Low prices.
        closes: Close prices.

    Returns:
        List of lower wick ratios (same length). NaN when high == low.

    Raises:
        ValueError: If input sequences have different lengths.
    """
    _validate_lengths(opens, highs, lows, closes)
    n = len(opens)
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = highs[i] - lows[i]
        if denom == 0.0:
            result[i] = float("nan")
        else:
            result[i] = (min(opens[i], closes[i]) - lows[i]) / denom
    return result


def _validate_lengths(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> None:
    n = len(opens)
    if len(highs) != n or len(lows) != n or len(closes) != n:
        raise ValueError("all input sequences must have the same length")
