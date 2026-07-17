"""
Combined cost function — fee + slippage + funding in R-multiples.

Wraps the individual cost primitives so callers get a single
``total_cost_r`` that accounts for all relevant costs.

Formula
-------
total_cost_r = fee_cost_r + slippage_cost_r + funding_cost_r

Reference
---------
simulation/docs/cost_model.md — Core Formula
"""
from typing import Literal

from lib.costs.fees import FeeTier
from lib.costs.r_costs import fee_cost_r, slippage_cost_r
from lib.costs.funding_impact import funding_cost_r, Mode


def total_cost_r(
    notional: float,
    entry_price: float,
    atr: float,
    stop_multiplier: float,
    tier: FeeTier = "taker",
    avg_liquidity: float = 0.0,
    mode: Mode = "SWING",
    funding_rate: float = 0.0001,
    holding_hours: float | None = None,
    direction: str = "LONG",
) -> float:
    """Compute total cost (fee + slippage + funding) in R-multiples.

    Parameters
    ----------
    notional : float
        Trade size in quote currency.
    entry_price : float
        Entry price in quote per base unit.
    atr : float
        Average True Range (price units).
    stop_multiplier : float
        Stop-loss multiplier (1R = atr * stop_multiplier).
    tier : str
        ``"maker"`` or ``"taker"`` (default ``"taker"``).
    avg_liquidity : float
        Average liquidity depth for slippage estimation (default 0.0).
    mode : Mode
        Trading mode for funding sensitivity (default ``"SWING"``).
    funding_rate : float
        Per-interval funding rate (default 0.0001 = 0.01 %).
    holding_hours : float | None
        Estimated holding time in hours.  *None* uses the mode-specific
        conservative default.
    direction : str
        ``"LONG"`` or ``"SHORT"`` (default ``"LONG"``).

    Returns
    -------
    float
        Total cost expressed in R-multiples (positive = net cost).
        Returns 0.0 when ``atr <= 0`` or ``stop_multiplier <= 0``.
    """
    if atr <= 0 or stop_multiplier <= 0:
        return 0.0

    fee = fee_cost_r(notional, entry_price, atr, stop_multiplier, tier)
    slip = slippage_cost_r(notional, entry_price, atr, stop_multiplier, avg_liquidity)
    fund = funding_cost_r(
        notional, entry_price, atr, stop_multiplier,
        mode=mode, funding_rate=funding_rate,
        holding_hours=holding_hours, direction=direction,
    )

    return fee + slip + fund
