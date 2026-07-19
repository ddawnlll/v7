"""Phase 3 — outcome observation (ROADMAP Phase 3).

Pure: no I/O, no network, no wall-clock. Given already-verified trade bars
and funding events (tools/load_snapshot.py hands those over), measures what
outcome structures actually exist in the data using the locked `lab.sim`
simulation authority — MAE/MFE, time-to-outcome, target-before-stop base
rates, cost sensitivity, same-bar ambiguity. Trains no model and presumes no
profitable geometry: this produces descriptive statistics only, never a
label or a decision.

Event definition follows ARCHITECTURE §9.1/§9.3: every completed decision
bar with sufficient valid history (ATR warmed up) is a candidate event,
decided at that bar's close, entered at the next bar's open.

Stop/target distance is ATR-based (`lab.indicators.compute_atr`), scaled by
a small, explicit, deliberately bounded grid of three named setups — not a
parameter sweep. A sweep this early would start to look like hypothesis
mining, which ROADMAP gates behind Phase 6, not Phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lab import indicators, sim
from lab.data import Bar

ATR_PERIOD = 14


@dataclass(frozen=True, slots=True)
class Setup:
    """One candidate stop/target/timeout geometry, ATR-scaled.

    k_stop      : stop distance, in multiples of ATR(ATR_PERIOD) at the
                  decision bar.
    reward_risk : target distance = reward_risk * (k_stop * ATR) — i.e. the
                  R-multiple the target sits at if stop is exactly 1R away.
    max_holding_bars : timeout, in 5m bars.
    """

    label: str
    k_stop: float
    reward_risk: float
    max_holding_bars: int


DEFAULT_SETUPS: tuple[Setup, ...] = (
    Setup("tight", k_stop=1.0, reward_risk=1.5, max_holding_bars=12),    # ~1h
    Setup("medium", k_stop=1.5, reward_risk=2.0, max_holding_bars=48),   # ~4h
    Setup("wide", k_stop=2.0, reward_risk=3.0, max_holding_bars=288),    # ~1d
)

_SIDES: tuple[sim.Side, ...] = ("LONG", "SHORT")


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
    is_long: bool, hi: float, lo: float, stop_price: float, target_price: float,
) -> tuple[bool, bool]:
    """Same touch condition `sim.simulate()` uses internally — a read-only
    diagnostic check for same-bar ambiguity. Produces no label and no net_R,
    so this does not become a second money-computing authority (RULES §4);
    it only re-asks a question `simulate()` already answered once, for a
    bar we already know is the exit bar."""
    if is_long:
        return lo <= stop_price, hi >= target_price
    return hi >= stop_price, lo <= target_price


def observe(
    trade_bars: Sequence[Bar],
    funding_events: Sequence[sim.FundingEvent] = (),
    setups: Sequence[Setup] = DEFAULT_SETUPS,
) -> dict:
    """Measure observable outcome properties on ``trade_bars`` for each
    setup in ``setups``. Returns one JSON-able dict keyed by setup label."""
    n = len(trade_bars)
    highs = [b.high for b in trade_bars]
    lows = [b.low for b in trade_bars]
    opens = [b.open for b in trade_bars]
    closes = [b.close for b in trade_bars]

    atr = indicators.compute_atr(highs, lows, closes, period=ATR_PERIOD)

    report: dict = {}
    for setup in setups:
        n_candidates = 0
        n_atr_unavailable = 0
        n_simulated = 0
        exit_reasons = {"stop": 0, "target": 0, "time": 0}
        n_ambiguous = 0
        mae_r_values: list[float] = []
        mfe_r_values: list[float] = []
        net_r_values: list[float] = []
        zero_cost_net_r_values: list[float] = []
        hold_bars_values: list[int] = []

        last_entry_i = n - setup.max_holding_bars - 2
        for i in range(ATR_PERIOD, max(ATR_PERIOD, last_entry_i + 1)):
            n_candidates += len(_SIDES)
            a = atr[i]
            if not math.isfinite(a):
                n_atr_unavailable += len(_SIDES)
                continue

            entry_index = i + 1
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
                    side=side,
                    entry_index=entry_index,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    max_holding_bars=setup.max_holding_bars,
                )
                outcome = sim.simulate(opens, highs, lows, closes, spec, funding_events)

                n_simulated += 1
                exit_reasons[outcome.exit_reason] += 1
                mae_r_values.append(outcome.mae_r)
                mfe_r_values.append(outcome.mfe_r)
                net_r_values.append(outcome.net_r)
                # Zero-cost net_r for free from the same simulate() call —
                # gross_return is execution_return before costs (RULES-
                # required cost-sensitivity evidence, no second simulation).
                zero_cost_net_r_values.append(outcome.gross_return / outcome.risk_fraction)
                hold_bars_values.append(outcome.exit_index - outcome.entry_index)

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
            "n_candidates": n_candidates,
            "n_atr_unavailable": n_atr_unavailable,
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
            "net_r_realistic_cost": _percentiles(net_r_values),
            "net_r_zero_cost": _percentiles(zero_cost_net_r_values),
            "hold_bars": _percentiles(hold_bars_values),
        }

    return report
