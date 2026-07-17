"""
Generic rolling window utilities.

Pure math — no state, no adapters.
"""

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def rolling_apply(
    values: Sequence[T],
    window: int,
    func: Callable[[list[T]], R],
    min_periods: int | None = None,
) -> list[R | None]:
    """Apply a function to each rolling window.

    Args:
        values: Input sequence.
        window: Window size.
        func: Function that takes a list of window values and returns a result.
        min_periods: Minimum number of observations required (default = window).

    Returns:
        List of results (same length). Prior to min_periods, returns None.
    """
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
    """Rolling maximum.

    Args:
        values: Input sequence.
        period: Window size (default 20).

    Returns:
        List of rolling max values (same length).
        First `period - 1` values are NaN. period=1 returns values unchanged.
    """
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
    """Rolling minimum.

    Args:
        values: Input sequence.
        period: Window size (default 20).

    Returns:
        List of rolling min values (same length).
        First `period - 1` values are NaN. period=1 returns values unchanged.
    """
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
    """Rolling mean.

    Args:
        values: Input sequence.
        period: Window size (default 20).

    Returns:
        List of rolling mean values (same length).
        First `period - 1` values are NaN. period=1 returns values unchanged.
    """
    n = len(values)
    if n == 0:
        return []
    if period == 1:
        return list(values)

    result: list[float] = [float("nan")] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        result[i] = sum(window) / period
    return result
