"""Simulation truth core. The single source of economic truth (RULES §14).

net_R, labels, and outcomes are defined here and nowhere else.
"""

from lab.sim.contracts import (
    CostBreakdown,
    ExitReason,
    Side,
    TradeOutcome,
    TradeSpec,
)
from lab.sim.costs import (
    fee_fraction,
    funding_fraction,
    round_trip_cost,
    slippage_fraction,
)
from lab.sim.engine import simulate

__all__ = [
    "CostBreakdown",
    "ExitReason",
    "Side",
    "TradeOutcome",
    "TradeSpec",
    "fee_fraction",
    "funding_fraction",
    "round_trip_cost",
    "slippage_fraction",
    "simulate",
]
