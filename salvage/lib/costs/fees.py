"""
Fee estimation.

Basic formulas — shared by both v7/ (simulation costs) and
alphaforge/ (label costs).
"""

from typing import Literal

FeeTier = Literal["maker", "taker"]

# Default fee rates (can be overridden by config)
_DEFAULT_MAKER_FEE = 0.0001   # 0.01%
_DEFAULT_TAKER_FEE = 0.0004   # 0.04%


def estimate_fee(
    notional: float,
    tier: FeeTier,
    maker_rate: float = _DEFAULT_MAKER_FEE,
    taker_rate: float = _DEFAULT_TAKER_FEE,
) -> float:
    """Estimate fee for a trade of `notional` size.

    Args:
        notional: Trade size in quote currency.
        tier: 'maker' or 'taker'.
        maker_rate: Maker fee rate (default 0.01%).
        taker_rate: Taker fee rate (default 0.04%).

    Returns:
        Fee amount in quote currency.
    """
    rate = maker_rate if tier == "maker" else taker_rate
    return notional * rate


def estimate_maker_fee(notional: float, rate: float = _DEFAULT_MAKER_FEE) -> float:
    """Shortcut for maker fee."""
    return estimate_fee(notional, "maker", rate)


def estimate_taker_fee(notional: float, rate: float = _DEFAULT_TAKER_FEE) -> float:
    """Shortcut for taker fee."""
    return estimate_fee(notional, "taker", rate)
