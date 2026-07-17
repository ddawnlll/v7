"""
lib/costs — Basic cost estimation shared by both systems.
"""

from lib.costs.fees import estimate_fee, estimate_maker_fee, estimate_taker_fee
from lib.costs.r_costs import fee_cost_r, slippage_cost_r, total_cost_r as _r_total_cost_r
from lib.costs.slippage import get_slippage
from lib.costs.funding_impact import funding_cost_r, max_funding_intervals, Mode
from lib.costs.combined import total_cost_r

__all__ = [
    "estimate_fee", "estimate_maker_fee", "estimate_taker_fee",
    "fee_cost_r", "slippage_cost_r", "total_cost_r",
    "get_slippage",
    "funding_cost_r", "max_funding_intervals", "Mode",
]
