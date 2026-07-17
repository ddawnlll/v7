"""
Slippage estimation.

Basic formulas shared by both v7/ and alphaforge/.
"""

from typing import Optional


def get_slippage(
    notional: float,
    avg_liquidity: float,
    slippage_pct: Optional[float] = None,
) -> float:
    """Estimate slippage for a trade.

    If `slippage_pct` is provided, use it directly.
    Otherwise estimate as a function of trade size relative to liquidity.

    Args:
        notional: Trade size in quote currency.
        avg_liquidity: Average liquidity depth (e.g. 2% order book depth).
        slippage_pct: Explicit slippage percentage (overrides estimation).

    Returns:
        Slippage amount in quote currency (positive value).
    """
    if slippage_pct is not None:
        return notional * (slippage_pct / 100.0)

    # Simple model: slippage scales with trade size relative to liquidity
    if avg_liquidity <= 0:
        return 0.0
    ratio = notional / avg_liquidity
    # Base slippage 0.01%, scaling linearly with size ratio
    estimated_pct = 0.01 * max(1.0, ratio)
    return notional * (estimated_pct / 100.0)
