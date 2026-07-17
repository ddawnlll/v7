"""Fractional trade costs — the only place in the repo that computes money.

Everything here is a fraction of notional, so it composes directly with
fractional returns in the engine. Convert to quote currency by multiplying by
notional at the call site if ever needed; the engine never needs to.

Deliberately NOT salvaged from the predecessor's cost stack: its R-multiple
conversions divided quote-currency costs by a price-unit risk, dropping a
notional/entry factor. Here costs are fractional from the start, so net_R is a
pure fractional/fractional ratio in engine.py with no unit juggling.
"""

from __future__ import annotations

from lab.sim.contracts import CostBreakdown, Side

_BPS = 1e-4


def fee_fraction(fee_rate: float) -> float:
    """Per-side fee as a fraction of notional. Fails closed on bad input."""
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")
    return fee_rate


def slippage_fraction(slippage_bps: float) -> float:
    """Per-side slippage (bps of notional) as a fraction. Fails closed."""
    if slippage_bps < 0.0:
        raise ValueError(f"slippage_bps must be >= 0, got {slippage_bps}")
    return slippage_bps * _BPS


def funding_fraction(funding_rate: float, intervals: float, side: Side) -> float:
    """Signed funding as a fraction of notional over the holding period.

    Positive result = net cost (reduces net return). A LONG pays funding when
    the rate is positive; a SHORT receives it (negative cost). Intervals must
    be >= 0; the rate may be negative (funding regimes flip sign).
    """
    if intervals < 0.0:
        raise ValueError(f"funding_intervals must be >= 0, got {intervals}")
    direction = 1.0 if side == "LONG" else -1.0
    return direction * funding_rate * intervals


def round_trip_cost(
    fee_rate: float,
    slippage_bps: float,
    funding_rate: float,
    funding_intervals: float,
    side: Side,
) -> CostBreakdown:
    """Total round-trip cost (entry + exit) as fractions of notional.

    Fee and slippage are charged on both entry and exit (2x per side).
    Funding is charged once over the whole holding period.
    """
    fee = 2.0 * fee_fraction(fee_rate)
    slippage = 2.0 * slippage_fraction(slippage_bps)
    funding = funding_fraction(funding_rate, funding_intervals, side)
    total = fee + slippage + funding
    return CostBreakdown(fee=fee, slippage=slippage, funding=funding, total=total)
