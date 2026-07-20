"""Hand-verifiable tests for lab/events.py observe() — Phase 3 outcome observation.

Bars 0..15 are a flat "warmup" region (open=close=100, high=101, low=99):
true range is 2.0 on every bar, so Wilder's ATR(14) is exactly 2.0 at every
index >= 14 (a constant TR series smooths to itself, no approximation).
That makes stop/target distances hand-computable: with k_stop=1.0,
reward_risk=2.0, decision bar i=14 gives entry_index=15, entry_price=100,
LONG stop=98/target=104, SHORT stop=102/target=96.

Each test builds exactly one testable decision bar (n = ATR_PERIOD +
max_holding_bars + 2) so there is exactly one (LONG, SHORT) pair to reason
about by hand, then engineers the single outcome-scan bar to produce a
specific, predictable exit.
"""
import pytest

from lab.market import Bar
from lab.events import ATR_PERIOD, Setup, observe

_SETUP = Setup("test", k_stop=1.0, reward_risk=2.0, max_holding_bars=3)
_N = ATR_PERIOD + _SETUP.max_holding_bars + 2  # = 19, decision bar i=14 only


def _flat_bars(n: int) -> list[Bar]:
    return [
        Bar(open_ts=i * 300_000, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0)
        for i in range(n)
    ]


def _with_outcome_bars(outcome_bars: dict[int, tuple[float, float]]) -> list[Bar]:
    """Flat background through index 15 (entry bar), then override the
    outcome-scan bars (16, 17, 18) with the given (high, low), keeping
    open=close=100 throughout so entry_price stays exactly 100."""
    bars = _flat_bars(_N)
    for idx, (hi, lo) in outcome_bars.items():
        bars[idx] = Bar(open_ts=bars[idx].open_ts, open=100.0, high=hi, low=lo, close=100.0, volume=1.0)
    return bars


def test_target_then_stop_unambiguous():
    # bar16: high=105 (>=104, LONG target) low=99 (>98, no LONG stop touch)
    #        for SHORT: stop=102 -> 105>=102 hit; target=96 -> 99<=96 false.
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))["test"]

    assert report["n_candidates"] == 2
    assert report["n_atr_unavailable"] == 0
    assert report["n_simulated"] == 2
    assert report["coverage"] == 1.0
    assert report["exit_reason_rate"]["target"] == 0.5  # LONG only
    assert report["exit_reason_rate"]["stop"] == 0.5     # SHORT only
    assert report["exit_reason_rate"]["time"] == 0.0
    assert report["ambiguous_bar_count"] == 0
    assert report["hold_bars"]["mean"] == 1  # bar16 is 1 bar after entry (15)


def test_same_bar_ambiguous_for_long_only():
    # bar16: high=105 (LONG target, and SHORT stop) low=97 (LONG stop, but
    #        NOT SHORT target: SHORT target=96, 97<=96 is false).
    bars = _with_outcome_bars({16: (105.0, 97.0)})
    report = observe(bars, setups=(_SETUP,))["test"]

    assert report["n_simulated"] == 2
    # Both sides exit via "stop" (sim.py's tie-break), but only LONG's exit
    # bar also touched its target -> only LONG counts as ambiguous.
    assert report["exit_reason_rate"]["stop"] == 1.0
    assert report["ambiguous_bar_count"] == 1
    assert report["ambiguous_bar_rate"] == 0.5


def test_timeout_when_neither_barrier_touched():
    # LONG's band is (98, 104), SHORT's is (96, 102) -> the intersection
    # (98, 102) is safe for both; bars16-18 stay strictly inside it.
    bars = _with_outcome_bars({
        16: (101.0, 99.0), 17: (101.0, 99.0), 18: (101.0, 99.0),
    })
    report = observe(bars, setups=(_SETUP,))["test"]

    assert report["n_simulated"] == 2
    assert report["exit_reason_rate"]["time"] == 1.0
    assert report["ambiguous_bar_count"] == 0
    assert report["hold_bars"]["mean"] == _SETUP.max_holding_bars


def test_zero_cost_net_r_never_worse_than_all_taker_conservative():
    # Costs only ever subtract from execution_return (RULES: fee_rate,
    # slippage_bps > 0 by default) -> zero-cost net_r >= realistic net_r,
    # for every setup, always.
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))["test"]
    assert report["net_r_zero_cost"]["mean"] >= report["net_r_all_taker_conservative"]["mean"]


def test_cost_components_sum_to_the_zero_vs_all_taker_delta():
    # fee_R + slippage_R + funding_R must exactly account for the gap
    # between zero-cost and all-taker-conservative net_r (net_return =
    # execution_return - costs.total, both normalized by the same
    # risk_fraction) — an accounting identity, not a statistical
    # approximation.
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))["test"]
    delta = report["net_r_zero_cost"]["mean"] - report["net_r_all_taker_conservative"]["mean"]
    cost_sum = (
        report["fee_r"]["mean"] + report["slippage_r"]["mean"] + report["funding_r"]["mean"]
    )
    assert cost_sum == pytest.approx(delta, abs=1e-9)


def test_cost_assumptions_match_sim_tradespec_defaults():
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))
    ca = report["cost_assumptions"]
    assert ca["fee_rate_per_side"] == 0.0004
    assert ca["slippage_bps_per_side"] == 1.0
    assert ca["round_trip_fee_bps"] == pytest.approx(8.0)
    assert ca["round_trip_slippage_bps"] == pytest.approx(2.0)


def test_too_short_series_yields_no_candidates_not_a_crash():
    bars = _flat_bars(ATR_PERIOD)  # shorter than any setup needs
    report = observe(bars, setups=(_SETUP,))["test"]
    assert report["n_candidates"] == 0
    assert report["n_simulated"] == 0
    assert report["coverage"] is None
    assert report["mae_r"]["mean"] is None


# --- multi-interval decisioning (Stage B: decision_interval_factor > 1) -------
#
# Flat background bars aggregate (lab.tape.aggregate) into flat 15m derived
# bars with the same constant-TR trick as above: every derived bar's true
# range is 2.0, so ATR(14) on the *derived* series is exactly 2.0 at
# derived-index >= 14, hand-computable exactly like the 5m case.

_STAGE_B_SETUP = Setup(
    "test_15m", k_stop=1.0, reward_risk=2.0, max_holding_bars=3,
    decision_interval_factor=3, decision_interval_label="15m",
)
# 45 flat 5m bars (0..44) = 15 complete 15m buckets -> derived ATR(14) valid
# at derived-index 14 (bucket [42,43,44]), entry_index=45. Bars 45-47 also
# happen to complete a 16th bucket (45 is itself bucket-aligned: 45 % 3 ==
# 0) giving a *second* derived candidate at derived-index 15, entry_index=48
# — deliberately left out of range (needs bars 49-51, which don't exist) so
# this one array exercises both the mapping-succeeds and the
# mapping-produces-an-out-of-range-candidate paths in the same hand-worked
# example, rather than trying to (unavoidably) dodge the second bucket.
_STAGE_B_N = 45 + 1 + _STAGE_B_SETUP.max_holding_bars  # = 49


def test_multi_interval_decision_maps_to_correct_5m_entry_bar():
    # Same dual-sided target/stop pattern as test_target_then_stop_
    # unambiguous, just at a 15m decision interval: bar46 (the first
    # outcome-scan bar after the mapped 5m entry) engineered identically.
    bars = _flat_bars(_STAGE_B_N)
    bars[46] = Bar(open_ts=bars[46].open_ts, open=100.0, high=105.0, low=99.0,
                    close=100.0, volume=1.0)

    report = observe(bars, setups=(_STAGE_B_SETUP,))["test_15m"]

    assert report["decision_interval_factor"] == 3
    assert report["decision_interval_label"] == "15m"
    # Two derived decision points exist (derived-index 14 and 15); only the
    # first maps to a usable 5m entry (45) whose 3-bar horizon fits in the
    # 49-bar array — the second (entry=48) doesn't, see comment above.
    assert report["n_candidates"] == 4
    assert report["n_out_of_range"] == 2
    assert report["n_atr_unavailable"] == 0
    assert report["n_simulated"] == 2
    assert report["coverage"] == 0.5
    # Same engineered bar as the 5m case: LONG hits target, SHORT hits stop.
    assert report["exit_reason_rate"]["target"] == 0.5
    assert report["exit_reason_rate"]["stop"] == 0.5
    assert report["hold_bars"]["mean"] == 1  # bar46 is 1 bar after entry (45)


def test_multi_interval_decision_out_of_range_when_horizon_does_not_fit():
    # Truncate right after the mapped entry bar (index 45) so bars 46-48
    # (the 3-bar horizon) don't exist -> the one decision candidate must be
    # reported as out-of-range, not silently dropped or crashed on.
    bars = _flat_bars(46)  # indices 0..45 only
    report = observe(bars, setups=(_STAGE_B_SETUP,))["test_15m"]

    assert report["n_candidates"] == 2
    assert report["n_out_of_range"] == 2
    assert report["n_simulated"] == 0
    assert report["coverage"] == 0.0


def test_overlap_fraction_matches_decision_spacing_vs_horizon():
    always_short = _flat_bars(ATR_PERIOD)  # no candidates either way; only
    # overlap_fraction (a pure function of the Setup) is under test here.

    every_bar = Setup("s", k_stop=1.0, reward_risk=1.0, max_holding_bars=12)
    assert observe(always_short, setups=(every_bar,))["s"]["overlap_fraction"] \
        == pytest.approx(1 - 1 / 12)

    exactly_independent = Setup(
        "s", k_stop=1.0, reward_risk=1.0, max_holding_bars=12,
        decision_interval_factor=12, decision_interval_label="1h",
    )
    assert observe(always_short, setups=(exactly_independent,))["s"]["overlap_fraction"] == 0.0

    spaced_wider_than_horizon = Setup(
        "s", k_stop=1.0, reward_risk=1.0, max_holding_bars=12,
        decision_interval_factor=48, decision_interval_label="4h",
    )
    # factor > max_holding_bars -> negative raw value, clamped to 0.
    assert observe(always_short, setups=(spaced_wider_than_horizon,))["s"]["overlap_fraction"] == 0.0
