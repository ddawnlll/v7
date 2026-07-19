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
decided at that bar's close, entered at the next bar's open. "Decision bar"
defaults to the raw 5m authority (`decision_interval_factor=1`) but can be a
derived 15m/1h/4h bar instead (ARCHITECTURE §8.3's `lab.data.aggregate`,
§8.1's "primary decision candidate: 1h") — the stop/target/timeout WALK
always runs on the 5m tape regardless (§8.2: trade bars are what the walk
uses), only which bar counts as a decision moment changes.

Stop/target distance is ATR-based (`lab.indicators.compute_atr`), scaled by
a small, explicit, deliberately bounded grid of named setups — not a
parameter sweep. A sweep this early would start to look like hypothesis
mining, which ROADMAP gates behind Phase 6, not Phase 3.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lab import data, indicators, sim
from lab.data import Bar

ATR_PERIOD = 14
BASE_INTERVAL_MS = 300_000  # 5m — ARCHITECTURE §8.1 raw authority interval


@dataclass(frozen=True, slots=True)
class Setup:
    """One candidate stop/target/timeout geometry, ATR-scaled.

    k_stop      : stop distance, in multiples of ATR(ATR_PERIOD) at the
                  decision bar.
    reward_risk : target distance = reward_risk * (k_stop * ATR) — i.e. the
                  R-multiple the target sits at if stop is exactly 1R away.
    max_holding_bars   : timeout, in 5m bars — the economic horizon. Stays
                  fixed regardless of decision_interval_factor: only
                  decision *frequency* changes across setups, never how
                  long a trade is allowed to run.
    decision_interval_factor : how many 5m bars make up one decision
                  interval (1 = decide on every 5m bar; 3=15m; 12=1h;
                  48=4h). ATR is computed on bars aggregated to this
                  interval (`lab.data.aggregate`), not always on 5m.
    decision_interval_label : human-readable label for the above, purely
                  for report readability.
    """

    label: str
    k_stop: float
    reward_risk: float
    max_holding_bars: int
    decision_interval_factor: int = 1
    decision_interval_label: str = "5m"


DEFAULT_SETUPS: tuple[Setup, ...] = (
    Setup("tight", k_stop=1.0, reward_risk=1.5, max_holding_bars=12),    # ~1h
    Setup("medium", k_stop=1.5, reward_risk=2.0, max_holding_bars=48),   # ~4h
    Setup("wide", k_stop=2.0, reward_risk=3.0, max_holding_bars=288),    # ~1d
)

# Stage B — ARCHITECTURE §8.1's actual candidate geometry (15m/1h/4h decision
# intervals, 1h primary), same three economic horizons as DEFAULT_SETUPS.
# Exploratory: not a locked HunterSpec (see tools/run_observation.py).
_HORIZONS: tuple[tuple[str, float, float, int], ...] = (
    ("tight", 1.0, 1.5, 12),
    ("medium", 1.5, 2.0, 48),
    ("wide", 2.0, 3.0, 288),
)
_STAGE_B_INTERVALS: tuple[tuple[str, int], ...] = (
    ("15m", 3),
    ("1h", 12),
    ("4h", 48),
)
STAGE_B_SETUPS: tuple[Setup, ...] = tuple(
    Setup(
        f"{h_label}_{i_label}", k_stop=k, reward_risk=rr, max_holding_bars=mh,
        decision_interval_factor=factor, decision_interval_label=i_label,
    )
    for h_label, k, rr, mh in _HORIZONS
    for i_label, factor in _STAGE_B_INTERVALS
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


def _candidate_decisions(
    setup: Setup,
    trade_bars: Sequence[Bar],
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    bar_open_ts: Sequence[int], n: int,
):
    """Yield ``(entry_index, atr_value)`` for every raw candidate decision
    point of this setup, in range or not — the caller decides what counts
    as usable (NaN ATR, out-of-range) so coverage bookkeeping stays uniform
    across both decision-interval paths below.

    ``decision_interval_factor == 1``: decide on every 5m bar, ATR on the
    5m tape — today's plumbing-baseline behavior, unchanged.

    ``> 1``: decide on every derived bar (``lab.data.aggregate`` — complete,
    gap-free buckets only), ATR on the *derived* bars. A derived bar's
    decision closes exactly when its last constituent 5m bar closes, which
    is the same moment the next 5m bar opens (candles are contiguous) — so
    the entry is found by locating that 5m bar's ``open_ts`` via bisect,
    same pattern ``tools/load_snapshot.py`` uses for funding-event mapping.
    ``entry_index`` is ``None`` when a decision doesn't map onto the 5m
    tape's covered range or the horizon wouldn't fit.
    """
    if setup.decision_interval_factor == 1:
        atr = indicators.compute_atr(highs, lows, closes, period=ATR_PERIOD)
        last_entry_i = n - setup.max_holding_bars - 2
        for i in range(ATR_PERIOD, max(ATR_PERIOD, last_entry_i + 1)):
            yield i + 1, atr[i]
        return

    factor = setup.decision_interval_factor
    derived = data.aggregate(trade_bars, factor, BASE_INTERVAL_MS)
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
    bar_open_ts = [b.open_ts for b in trade_bars]

    # Cost assumptions come from sim.TradeSpec's own declared defaults
    # (source of truth is sim.py, not duplicated here) — recorded explicitly
    # so a reader never has to reverse-engineer which fee/slippage was used.
    _default_fee_rate = sim.TradeSpec.__dataclass_fields__["fee_rate"].default
    _default_slippage_bps = sim.TradeSpec.__dataclass_fields__["slippage_bps"].default
    cost_assumptions = {
        "fee_rate_per_side": _default_fee_rate,
        "slippage_bps_per_side": _default_slippage_bps,
        "round_trip_fee_bps": 2 * _default_fee_rate * 10_000,
        "round_trip_slippage_bps": 2 * _default_slippage_bps,
        "funding": "real historical OKX funding events from the snapshot "
                   "(variable, not a fixed assumption — applies only when "
                   "the trade's holding period crosses a settlement)",
        "source": "lab.sim.TradeSpec defaults (single authority, RULES §4) "
                  "— not venue-specific; the snapshot's price data is OKX, "
                  "these are sim.py's own declared per-side cost constants",
        "note": "all-taker on both legs (entry and exit) is a conservative "
                "default, not a claim of venue-calibrated realism — a limit "
                "target exit filling as maker would cost less than this. "
                "See net_r_all_taker_conservative below, not "
                "net_r_realistic_cost.",
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
                # Cost components in R units (same normalization as net_r),
                # so fee_R + slippage_R is directly "round-trip cost as a
                # fraction of the ATR stop distance" the reviewer asked for.
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
            # Fraction of one trade's holding window shared with the very
            # next candidate's window: 0 = fully independent consecutive
            # decisions, near 1 = almost total overlap (e.g. every-5m-bar
            # decisioning on a 12-bar horizon). Raw ratio, not a derived
            # "effective N" statistic — decision spacing vs. horizon is
            # already visible from the two fields above; this just names
            # the relationship explicitly.
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
            # Conservative all-taker default (both legs at TradeSpec's
            # per-side taker fee), NOT a venue-calibrated realism claim —
            # see cost_assumptions above and the module docstring.
            "net_r_all_taker_conservative": _percentiles(net_r_values),
            "net_r_zero_cost": _percentiles(zero_cost_net_r_values),
            "fee_r": _percentiles(fee_r_values),
            "slippage_r": _percentiles(slippage_r_values),
            "funding_r": _percentiles(funding_r_values),
            "hold_bars": _percentiles(hold_bars_values),
        }

    return report
