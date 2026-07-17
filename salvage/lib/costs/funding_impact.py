"""
Funding cost impact estimation for perpetual swap positions.

Provides a per-mode ``funding_cost_r`` function that estimates funding
cost in R-multiples for SWING, SCALP, and AGGRESSIVE_SCALP modes.

Funding is only applicable to perpetual (perpetual swap) positions.
Spot trading is unaffected.

Formula (LOCK_CANDIDATE per simulation/docs/cost_model.md):

    funding_cost_r = direction_sign * num_intervals * funding_rate * notional
                     / (atr * stop_multiplier)

    direction_sign = +1 for LONG, -1 for SHORT

Reference
---------
simulation/docs/cost_model.md — Funding Cost Model for Perpetuals (LOCK_CANDIDATE)
"""
from typing import Literal

Mode = Literal["SWING", "SCALP", "AGGRESSIVE_SCALP"]

# Default funding interval for most perpetual exchanges.
FUNDING_INTERVAL_HOURS: float = 8.0

# Per-interval funding rate default (0.01 %).
_DEFAULT_FUNDING_RATE: float = 0.0001

# Mode-specific max funding intervals crossed (from cost_model.md table).
# These are conservative estimates used when holding_hours is not provided.
_MODE_MAX_FUNDING_INTERVALS: dict[Mode, float] = {
    "SWING": 15.0,              # 30 bars x 4h / 8h interval
    "SCALP": 2.0,               # 12 bars x 1h / 8h interval  (rounds up)
    "AGGRESSIVE_SCALP": 0.0,    # 5 bars x 15m / 8h interval -> negligible
}


def max_funding_intervals(
    mode: Mode,
    holding_hours: float | None = None,
    funding_interval_hours: float = FUNDING_INTERVAL_HOURS,
) -> float:
    """Return the number of funding intervals a position could cross.

    When ``holding_hours`` is provided the estimate is based on actual
    holding time; otherwise the mode-specific conservative default is
    used.

    Parameters
    ----------
    mode : Mode
        Trading mode (SWING / SCALP / AGGRESSIVE_SCALP).
    holding_hours : float | None
        Actual (or worst-case) holding time in hours.  When *None* the
        mode-specific default is used.
    funding_interval_hours : float
        Funding settlement interval in hours (default 8h).

    Returns
    -------
    float
        Number of funding intervals crossed (may be fractional for pro-rata
        overlap).  Returns 0.0 when ``holding_hours`` or
        ``funding_interval_hours`` is non-positive.
    """
    if funding_interval_hours <= 0:
        return 0.0

    if holding_hours is not None:
        if holding_hours <= 0:
            return 0.0
        return holding_hours / funding_interval_hours

    return _MODE_MAX_FUNDING_INTERVALS[mode]


def funding_cost_r(
    notional: float,
    entry_price: float,
    atr: float,
    stop_multiplier: float,
    mode: Mode = "SWING",
    funding_rate: float = _DEFAULT_FUNDING_RATE,
    holding_hours: float | None = None,
    direction: str = "LONG",
) -> float:
    """Estimate funding cost in R-multiples for a perpetual swap position.

    Parameters
    ----------
    notional : float
        Position size in quote currency.
    entry_price : float
        Entry price (unused, maintained for API consistency).
    atr : float
        Average True Range in price units.
    stop_multiplier : float
        Stop-loss multiplier (1R = atr * stop_multiplier).
    mode : Mode
        Trading mode.
    funding_rate : float
        Per-interval funding rate as a decimal (default 0.0001 = 0.01 %).
    holding_hours : float | None
        Estimated holding time in hours.  When *None* the mode-specific
        conservative default is used.
    direction : str
        ``"LONG"`` (pays funding when rate > 0) or ``"SHORT"`` (receives
        funding when rate > 0).

    Returns
    -------
    float
        Funding cost expressed in R-multiples.

        * Positive value = net cost (detracts from gross R).
        * Negative value = net credit (adds to gross R).
        * Returns 0.0 if ``atr <= 0`` or ``stop_multiplier <= 0``.
        * Returns 0.0 for AGGRESSIVE_SCALP when no explicit
          ``holding_hours`` is provided (funding impact is negligible per
          cost_model.md).
    """
    if atr <= 0 or stop_multiplier <= 0:
        return 0.0

    # AGGRESSIVE_SCALP: funding is negligible per cost_model.md table,
    # unless the caller explicitly provides holding_hours.
    if mode == "AGGRESSIVE_SCALP" and holding_hours is None:
        return 0.0

    direction_sign = 1.0 if direction == "LONG" else -1.0
    num_intervals = max_funding_intervals(mode, holding_hours)
    r_value = atr * stop_multiplier

    raw_funding = direction_sign * num_intervals * funding_rate * notional
    return raw_funding / r_value
