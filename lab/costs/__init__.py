"""Trade cost primitives (quote-currency units only).

R-multiple conversions are deliberately absent: they were the broken zone in
the predecessor repo. net_R is computed in one place — the outcome engine.
"""

from lab.costs.fees import FeeTier, estimate_fee, estimate_maker_fee, estimate_taker_fee
from lab.costs.slippage import get_slippage

__all__ = [
    "FeeTier",
    "estimate_fee",
    "estimate_maker_fee",
    "estimate_taker_fee",
    "get_slippage",
]
