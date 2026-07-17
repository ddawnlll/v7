"""Simulation truth core — the single source of economic truth (RULES §14).

net_R, labels, and outcomes are defined here and nowhere else. This is one
audit unit on purpose: the entire economic contract — trade shape, cost math,
and the bar-walk that turns them into a net_R — reads top to bottom in one pass.

Determinism contract (RULES §14): no wall-clock, no global RNG, no network, no
env reads. Every function is a pure function of its inputs. Same inputs →
byte-identical output on any machine.

Reference engine only: this scalar loop *defines* truth. Any faster path
(vectorized tape, CUDA) must reproduce it exactly under a parity test and is
never the sole path.

Conventions (fixed decisions):
- The entry bar is never inspected for exits. The walk starts at entry_index+1,
  so an outcome can never use information from the decision bar (no lookahead).
- Barriers use bar extremes: a LONG stops when a bar's low touches the stop and
  targets when its high touches the target (mirror for SHORT).
- If one bar touches BOTH stop and target, the stop wins — the conservative
  assumption, since intrabar order is unknown.
- If neither barrier is hit within max_holding_bars, exit at the close of bar
  (entry_index + max_holding_bars), reason "time".
- Too few forward bars is an error, not a silent truncation (fail-closed).

All returns are fractional (fraction of entry price); all costs are fractional
(fraction of notional). net_R is therefore a pure fractional/fractional ratio,
with no unit conversion — the class of bug that plagued the predecessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Side = Literal["LONG", "SHORT"]
ExitReason = Literal["stop", "target", "time"]

_BPS = 1e-4


# --- contracts ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TradeSpec:
    """A single trade to simulate, fully self-contained.

    Prices are quote currency per base unit. Costs are fractional, applied per
    side unless noted. No config registry, no mode object: a trade is described
    entirely by this spec, nothing is read from ambient state.

    side              : "LONG" | "SHORT"
    entry_index       : index into the bar arrays where the position opens
    entry_price       : assumed fill price (quote); the engine takes it as given
    stop_price        : protective stop (quote); on the losing side of entry
    target_price      : profit target (quote); on the winning side of entry
    max_holding_bars  : hard cap on bars held; else exit at that bar's close
    fee_rate          : per-side fee as a fraction of notional (e.g. 0.0004)
    slippage_bps      : per-side slippage in bps of notional (1.0 = 0.01%)
    funding_rate      : per-interval funding rate as a fraction (e.g. 0.0001)
    funding_intervals : number of funding intervals the position crosses
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
    """Round-trip costs as fractions of notional. `funding` is signed (a short
    can receive funding); `total` is subtracted from the gross return."""

    fee: float          # entry_fee + exit_fee
    slippage: float     # entry_slip + exit_slip
    funding: float      # signed: +cost, -credit
    total: float


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """Result of simulating one TradeSpec. `net_r` is the only performance
    number that matters downstream: net_return / risk_fraction, where
    risk_fraction = |entry - stop| / entry (the 1R distance)."""

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


# --- costs (the only place in the repo that computes money) ------------------

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

    Positive = net cost (reduces net return). A LONG pays funding when the rate
    is positive; a SHORT receives it. Intervals must be >= 0; the rate may be
    negative (funding regimes flip sign).
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
    """Total round-trip cost (entry + exit) as fractions of notional. Fee and
    slippage are charged on both sides (2x); funding once over the hold."""
    fee = 2.0 * fee_fraction(fee_rate)
    slippage = 2.0 * slippage_fraction(slippage_bps)
    funding = funding_fraction(funding_rate, funding_intervals, side)
    return CostBreakdown(fee=fee, slippage=slippage, funding=funding,
                         total=fee + slippage + funding)


# --- reference engine --------------------------------------------------------

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
    exit_reason: ExitReason = "time"
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
        net_r=net_return / risk_fraction,
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
