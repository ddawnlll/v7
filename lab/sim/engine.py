"""Reference simulation engine — scalar, readable, hand-verifiable.

This defines economic truth. It is intentionally a plain bar-by-bar loop with
no vectorization: correctness and auditability come first. Any faster path
(vectorized tape, CUDA) must reproduce this engine's output exactly, checked by
a parity test — the fast path is never the sole path (RULES §14).

Conventions (fixed decisions, not conventions to be overridden):
- The entry bar is never inspected for exits. The walk starts at entry_index+1,
  so an outcome can never use information from the bar the decision was made on.
- Barriers are checked against bar extremes: a LONG stops out when a bar's low
  touches the stop, and targets out when a bar's high touches the target
  (mirror for SHORT).
- If a single bar touches BOTH stop and target, the stop wins. This is the
  conservative assumption: we cannot know intrabar ordering, so we assume the
  adverse fill.
- If neither barrier is hit within max_holding_bars, the trade exits at the
  close of bar (entry_index + max_holding_bars), reason "time".
- Not enough forward bars to cover max_holding_bars is an error, not a silent
  truncation (RULES fail-closed).
"""

from __future__ import annotations

from typing import Sequence

from lab.sim.contracts import TradeOutcome, TradeSpec
from lab.sim.costs import round_trip_cost


def simulate(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    spec: TradeSpec,
) -> TradeOutcome:
    """Simulate one trade against OHLC bars and return its TradeOutcome."""
    n = len(highs)
    if not (len(lows) == n and len(closes) == n):
        raise ValueError("highs, lows, closes must have equal length")
    _validate(spec, n)

    entry = spec.entry_price
    is_long = spec.side == "LONG"
    sign = 1.0 if is_long else -1.0
    risk_fraction = abs(entry - spec.stop_price) / entry

    last_idx = spec.entry_index + spec.max_holding_bars
    exit_index = last_idx
    exit_reason = "time"
    exit_price = closes[last_idx]

    worst_adverse = 0.0    # fractional, >= 0
    best_favorable = 0.0   # fractional, >= 0

    for i in range(spec.entry_index + 1, last_idx + 1):
        hi, lo = highs[i], lows[i]

        # Excursions on this bar (before deciding exit), clamped at 0.
        if is_long:
            adverse = (entry - lo) / entry
            favorable = (hi - entry) / entry
        else:
            adverse = (hi - entry) / entry
            favorable = (entry - lo) / entry
        if adverse > worst_adverse:
            worst_adverse = adverse
        if favorable > best_favorable:
            best_favorable = favorable

        if is_long:
            stop_hit = lo <= spec.stop_price
            target_hit = hi >= spec.target_price
        else:
            stop_hit = hi >= spec.stop_price
            target_hit = lo <= spec.target_price

        # Stop wins ties (conservative — unknown intrabar order).
        if stop_hit:
            exit_index, exit_reason, exit_price = i, "stop", spec.stop_price
            break
        if target_hit:
            exit_index, exit_reason, exit_price = i, "target", spec.target_price
            break

    gross_return = sign * (exit_price - entry) / entry
    costs = round_trip_cost(
        fee_rate=spec.fee_rate,
        slippage_bps=spec.slippage_bps,
        funding_rate=spec.funding_rate,
        funding_intervals=spec.funding_intervals,
        side=spec.side,
    )
    net_return = gross_return - costs.total
    net_r = net_return / risk_fraction

    return TradeOutcome(
        side=spec.side,
        entry_index=spec.entry_index,
        exit_index=exit_index,
        exit_reason=exit_reason,
        entry_price=entry,
        exit_price=exit_price,
        risk_fraction=risk_fraction,
        gross_return=gross_return,
        net_return=net_return,
        net_r=net_r,
        mae_r=worst_adverse / risk_fraction,
        mfe_r=best_favorable / risk_fraction,
        costs=costs,
    )


def _validate(spec: TradeSpec, n: int) -> None:
    if spec.max_holding_bars < 1:
        raise ValueError(f"max_holding_bars must be >= 1, got {spec.max_holding_bars}")
    if spec.entry_index < 0:
        raise ValueError(f"entry_index must be >= 0, got {spec.entry_index}")
    if spec.entry_index + spec.max_holding_bars > n - 1:
        raise ValueError(
            "not enough forward bars: entry_index + max_holding_bars = "
            f"{spec.entry_index + spec.max_holding_bars} exceeds last index {n - 1}"
        )
    if spec.entry_price <= 0 or spec.stop_price <= 0 or spec.target_price <= 0:
        raise ValueError("entry, stop, and target prices must all be > 0")
    if spec.side == "LONG":
        if not (spec.stop_price < spec.entry_price < spec.target_price):
            raise ValueError("LONG requires stop < entry < target")
    else:
        if not (spec.target_price < spec.entry_price < spec.stop_price):
            raise ValueError("SHORT requires target < entry < stop")
