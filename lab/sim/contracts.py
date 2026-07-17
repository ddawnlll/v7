"""Trade contracts for the simulation truth core.

Frozen, explicit, unit-tagged. Every field carries its unit in the name or
docstring. There is no config registry and no mode object: a trade is fully
described by its TradeSpec, nothing is read from ambient state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["LONG", "SHORT"]
ExitReason = Literal["stop", "target", "time"]


@dataclass(frozen=True, slots=True)
class TradeSpec:
    """A single trade to simulate, fully self-contained.

    Prices are in quote currency per base unit. Costs are fractional
    (fraction of notional), applied per side unless noted. The simulation
    walks bars strictly after ``entry_index`` — the entry bar itself is never
    inspected for exits (no lookahead).

    Fields
    ------
    side : "LONG" | "SHORT"
    entry_index : int
        Index into the bar arrays where the position opens.
    entry_price : float
        Assumed fill price (quote). Caller decides its meaning (e.g. next
        bar open); the engine treats it as given.
    stop_price : float
        Protective stop (quote). Must be on the losing side of entry.
    target_price : float
        Profit target (quote). Must be on the winning side of entry.
    max_holding_bars : int
        Hard cap on bars held. If neither barrier is hit within this many
        bars after entry, the trade exits at that bar's close ("time").
    fee_rate : float
        Per-side taker/maker fee as a fraction of notional (e.g. 0.0004).
    slippage_bps : float
        Per-side slippage in basis points of notional (e.g. 1.0 = 0.01%).
    funding_rate : float
        Per-interval funding rate as a fraction (e.g. 0.0001).
    funding_intervals : float
        Number of funding intervals the position is assumed to cross.
    """

    side: Side
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    max_holding_bars: int
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_intervals: float = 0.0


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """All costs for a round-trip trade, as fractions of notional.

    Every component is non-negative except ``funding``, which is signed
    (a short can receive funding when the rate is positive). ``total`` is the
    sum and is subtracted from the gross fractional return.
    """

    fee: float          # entry_fee + exit_fee
    slippage: float     # entry_slip + exit_slip
    funding: float      # signed: +cost, -credit
    total: float


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """Result of simulating one TradeSpec.

    ``net_r`` is the only performance number that matters downstream:
        net_r = net_return / risk_fraction
    where risk_fraction = |entry_price - stop_price| / entry_price (the 1R
    distance). All returns are fractional (fraction of entry price).
    """

    side: Side
    entry_index: int
    exit_index: int
    exit_reason: ExitReason
    entry_price: float
    exit_price: float
    risk_fraction: float
    gross_return: float
    net_return: float
    net_r: float
    mae_r: float          # max adverse excursion, in R (>= 0)
    mfe_r: float          # max favorable excursion, in R (>= 0)
    costs: CostBreakdown
