"""Candidate-event authority (ROADMAP Phase 3 + Phase 4).

Pure: no I/O, no network, no wall-clock. Two public faces:

  observe()      — Phase 3: descriptive outcome statistics (MAE/MFE, cost
                   sensitivity, ambiguity). Trains no model, presumes no edge.
  build_events() — Phase 4: locked candidate event rows with splits and
                   purge. Deterministic, hash-verifiable.

Both share a single candidate-generation iterator (_candidate_decisions) so
the stop/target/timeout geometry is defined exactly once.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lab import indicators, sim, tape
from lab.tape import Bar

# ═══════════════════════════════════════════════════════════════════════════════
# constants
# ═══════════════════════════════════════════════════════════════════════════════

ATR_PERIOD = 14
BASE_INTERVAL_MS = 300_000  # 5m

_SIDES: tuple[sim.Side, ...] = ("LONG", "SHORT")


# ═══════════════════════════════════════════════════════════════════════════════
# setup geometry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Setup:
    """One candidate stop/target/timeout geometry, ATR-scaled.

    k_stop      : stop distance, in multiples of ATR(ATR_PERIOD) at the
                  decision bar.
    reward_risk : target distance = reward_risk * (k_stop * ATR).
    max_holding_bars   : timeout, in 5m bars.
    decision_interval_factor : how many 5m bars make up one decision
                  interval (1=5m, 3=15m, 12=1h, 48=4h).
    decision_interval_label : human-readable label.
    """

    label: str
    k_stop: float
    reward_risk: float
    max_holding_bars: int
    decision_interval_factor: int = 1
    decision_interval_label: str = "5m"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EventInput:
    """Per-symbol input for candidate event generation.

    Callers (tools/data.py) load bars from disk and construct these.
    """

    symbol: str
    trade_bars: list[tape.Bar]
    funding_events: list[sim.FundingEvent]


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    """One candidate trade event, directional (LONG/SHORT), with simulated outcome."""

    event_id: str
    symbol: str
    side: str
    feature_cutoff_ts: int
    decision_ts: int
    planned_entry_ts: int
    fill_ts: int
    outcome_end_ts: int
    locked_outcome: sim.TradeOutcome
    split: str


# ═══════════════════════════════════════════════════════════════════════════════
# shared candidate-generation iterator (single authority)
# ═══════════════════════════════════════════════════════════════════════════════

def _candidate_decisions(
    setup: Setup,
    trade_bars: Sequence[Bar],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    bar_open_ts: Sequence[int],
    n: int,
):
    """Yield ``(entry_index, atr_value)`` for every candidate decision point.

    ``entry_index`` is None when the decision doesn't map onto the 5m tape's
    covered range or the horizon wouldn't fit.
    """
    if setup.decision_interval_factor == 1:
        atr = indicators.compute_atr(highs, lows, closes, period=ATR_PERIOD)
        last_entry_i = n - setup.max_holding_bars - 2
        for i in range(ATR_PERIOD, max(ATR_PERIOD, last_entry_i + 1)):
            yield i + 1, atr[i]
        return

    factor = setup.decision_interval_factor
    derived = tape.aggregate(trade_bars, factor, BASE_INTERVAL_MS)
    if len(derived) <= ATR_PERIOD:
        return
    d_atr = indicators.compute_atr(
        [b.high for b in derived], [b.low for b in derived],
        [b.close for b in derived], period=ATR_PERIOD,
    )
    for di in range(ATR_PERIOD, len(derived)):
        decision_close_ts = derived[di].open_ts + factor * BASE_INTERVAL_MS
        idx = bisect.bisect_left(bar_open_ts, decision_close_ts)
        if (
            idx >= n
            or bar_open_ts[idx] != decision_close_ts
            or idx + setup.max_holding_bars >= n
        ):
            yield None, d_atr[di]
            continue
        yield idx, d_atr[di]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: observe
# ═══════════════════════════════════════════════════════════════════════════════

def _percentiles(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _touches(
    is_long: bool, hi: float, lo: float,
    stop_price: float, target_price: float,
) -> tuple[bool, bool]:
    if is_long:
        return lo <= stop_price, hi >= target_price
    return hi >= stop_price, lo <= target_price


def observe(
    trade_bars: Sequence[Bar],
    funding_events: Sequence[sim.FundingEvent] = (),
    *,
    setups: Sequence[Setup],
) -> dict:
    """Measure observable outcome properties for each setup.

    Returns one JSON-able dict keyed by setup label.
    """
    n = len(trade_bars)
    highs = [b.high for b in trade_bars]
    lows = [b.low for b in trade_bars]
    opens = [b.open for b in trade_bars]
    closes = [b.close for b in trade_bars]
    bar_open_ts = [b.open_ts for b in trade_bars]

    _default_fee_rate = sim.TradeSpec.__dataclass_fields__["fee_rate"].default
    _default_slippage_bps = sim.TradeSpec.__dataclass_fields__["slippage_bps"].default
    cost_assumptions = {
        "fee_rate_per_side": _default_fee_rate,
        "slippage_bps_per_side": _default_slippage_bps,
        "round_trip_fee_bps": 2 * _default_fee_rate * 10_000,
        "round_trip_slippage_bps": 2 * _default_slippage_bps,
        "source": "lab.sim.TradeSpec defaults (single authority, RULES §4)",
    }

    report: dict = {"cost_assumptions": cost_assumptions}
    for setup in setups:
        n_candidates = 0
        n_atr_unavailable = 0
        n_out_of_range = 0
        n_simulated = 0
        exit_reasons = {"stop": 0, "target": 0, "time": 0}
        n_ambiguous = 0
        mae_r_values: list[float] = []
        mfe_r_values: list[float] = []
        net_r_values: list[float] = []
        zero_cost_net_r_values: list[float] = []
        hold_bars_values: list[int] = []
        fee_r_values: list[float] = []
        slippage_r_values: list[float] = []
        funding_r_values: list[float] = []

        for entry_index, a in _candidate_decisions(
            setup, trade_bars, highs, lows, closes, bar_open_ts, n,
        ):
            n_candidates += len(_SIDES)
            if entry_index is None:
                n_out_of_range += len(_SIDES)
                continue
            if not math.isfinite(a):
                n_atr_unavailable += len(_SIDES)
                continue

            entry_price = opens[entry_index]
            stop_dist = setup.k_stop * a
            target_dist = setup.reward_risk * stop_dist

            for side in _SIDES:
                is_long = side == "LONG"
                if is_long:
                    stop_price = entry_price - stop_dist
                    target_price = entry_price + target_dist
                else:
                    stop_price = entry_price + stop_dist
                    target_price = entry_price - target_dist

                spec = sim.TradeSpec(
                    side=side, entry_index=entry_index, entry_price=entry_price,
                    stop_price=stop_price, target_price=target_price,
                    max_holding_bars=setup.max_holding_bars,
                )
                outcome = sim.simulate(opens, highs, lows, closes, spec, funding_events)

                n_simulated += 1
                exit_reasons[outcome.exit_reason] += 1
                mae_r_values.append(outcome.mae_r)
                mfe_r_values.append(outcome.mfe_r)
                net_r_values.append(outcome.net_r)
                zero_cost_net_r_values.append(outcome.gross_return / outcome.risk_fraction)
                hold_bars_values.append(outcome.exit_index - outcome.entry_index)
                fee_r_values.append(outcome.costs.fee / outcome.risk_fraction)
                slippage_r_values.append(outcome.costs.slippage / outcome.risk_fraction)
                funding_r_values.append(outcome.costs.funding / outcome.risk_fraction)

                if outcome.exit_reason in ("stop", "target"):
                    stop_touched, target_touched = _touches(
                        is_long, highs[outcome.exit_index], lows[outcome.exit_index],
                        stop_price, target_price,
                    )
                    if stop_touched and target_touched:
                        n_ambiguous += 1

        report[setup.label] = {
            "k_stop": setup.k_stop,
            "reward_risk": setup.reward_risk,
            "max_holding_bars": setup.max_holding_bars,
            "decision_interval_factor": setup.decision_interval_factor,
            "decision_interval_label": setup.decision_interval_label,
            "overlap_fraction": max(
                0.0, 1.0 - setup.decision_interval_factor / setup.max_holding_bars
            ),
            "n_candidates": n_candidates,
            "n_atr_unavailable": n_atr_unavailable,
            "n_out_of_range": n_out_of_range,
            "n_simulated": n_simulated,
            "coverage": n_simulated / n_candidates if n_candidates else None,
            "exit_reason_rate": {
                k: (v / n_simulated if n_simulated else None)
                for k, v in exit_reasons.items()
            },
            "ambiguous_bar_count": n_ambiguous,
            "ambiguous_bar_rate": n_ambiguous / n_simulated if n_simulated else None,
            "mae_r": _percentiles(mae_r_values),
            "mfe_r": _percentiles(mfe_r_values),
            "net_r_all_taker_conservative": _percentiles(net_r_values),
            "net_r_zero_cost": _percentiles(zero_cost_net_r_values),
            "fee_r": _percentiles(fee_r_values),
            "slippage_r": _percentiles(slippage_r_values),
            "funding_r": _percentiles(funding_r_values),
            "hold_bars": _percentiles(hold_bars_values),
        }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: build events
# ═══════════════════════════════════════════════════════════════════════════════

def build_events(
    inputs: Sequence[EventInput],
    split_ts: int,
    setup: Setup,
) -> list[CandidateEvent]:
    """Build candidate events from EventInput records, assigning splits and purging.

    Uses the shared _candidate_decisions() iterator — stop/target/timeout
    geometry is defined exactly once across Phase 3 and Phase 4.
    """
    events: list[CandidateEvent] = []

    for inp in inputs:
        symbol = inp.symbol
        trade_bars = inp.trade_bars
        funding_events = inp.funding_events

        n = len(trade_bars)
        if n == 0:
            continue

        highs = [b.high for b in trade_bars]
        lows = [b.low for b in trade_bars]
        opens = [b.open for b in trade_bars]
        closes = [b.close for b in trade_bars]
        bar_open_ts = [b.open_ts for b in trade_bars]

        for entry_index, a in _candidate_decisions(
            setup, trade_bars, highs, lows, closes, bar_open_ts, n,
        ):
            if entry_index is None:
                continue
            if not math.isfinite(a):
                continue

            entry_price = opens[entry_index]
            stop_dist = setup.k_stop * a
            target_dist = setup.reward_risk * stop_dist

            decision_ts = bar_open_ts[entry_index]

            for side in ("LONG", "SHORT"):
                is_long = side == "LONG"
                if is_long:
                    stop_price = entry_price - stop_dist
                    target_price = entry_price + target_dist
                else:
                    stop_price = entry_price + stop_dist
                    target_price = entry_price - target_dist

                spec = sim.TradeSpec(
                    side=side, entry_index=entry_index, entry_price=entry_price,
                    stop_price=stop_price, target_price=target_price,
                    max_holding_bars=setup.max_holding_bars,
                )
                outcome = sim.simulate(opens, highs, lows, closes, spec, funding_events)
                outcome_end_ts = bar_open_ts[outcome.exit_index] + BASE_INTERVAL_MS

                if decision_ts < split_ts:
                    if outcome_end_ts >= split_ts:
                        continue  # purge train events that leak into test
                    split = "train"
                else:
                    split = "test"

                event_id = hashlib.sha256(
                    f"{symbol}_{decision_ts}_{side}".encode("utf-8")
                ).hexdigest()

                events.append(CandidateEvent(
                    event_id=event_id, symbol=symbol, side=side,
                    feature_cutoff_ts=decision_ts, decision_ts=decision_ts,
                    planned_entry_ts=decision_ts, fill_ts=decision_ts,
                    outcome_end_ts=outcome_end_ts, locked_outcome=outcome,
                    split=split,
                ))

    events.sort(key=lambda e: (e.decision_ts, e.symbol, e.side))
    return events


def canonical_bytes(events: Sequence[CandidateEvent]) -> bytes:
    """Deterministic, round-trippable serialization of the candidate events."""
    lines = ["candidate-event-v0"]
    for e in events:
        lines.append(
            f"{e.event_id} {e.symbol} {e.side} {e.feature_cutoff_ts} "
            f"{e.decision_ts} {e.planned_entry_ts} {e.fill_ts} "
            f"{e.outcome_end_ts} {e.split} "
            f"{e.locked_outcome.exit_index} {e.locked_outcome.exit_reason} "
            f"{e.locked_outcome.net_r!r}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def events_hash(events: Sequence[CandidateEvent]) -> str:
    """SHA-256 hex digest of the canonical events serialization."""
    return hashlib.sha256(canonical_bytes(events)).hexdigest()
