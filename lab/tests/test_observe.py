"""Hand-verifiable tests for lab/observe.py.

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

from lab.data import Bar
from lab.observe import ATR_PERIOD, Setup, observe

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


def test_zero_cost_net_r_never_worse_than_realistic_cost():
    # Costs only ever subtract from execution_return (RULES: fee_rate,
    # slippage_bps > 0 by default) -> zero-cost net_r >= realistic net_r,
    # for every setup, always.
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))["test"]
    assert report["net_r_zero_cost"]["mean"] >= report["net_r_realistic_cost"]["mean"]


def test_cost_components_sum_to_the_zero_vs_realistic_delta():
    # fee_R + slippage_R + funding_R must exactly account for the gap
    # between zero-cost and realistic-cost net_r (net_return = execution_
    # return - costs.total, both normalized by the same risk_fraction) —
    # an accounting identity, not a statistical approximation.
    bars = _with_outcome_bars({16: (105.0, 99.0)})
    report = observe(bars, setups=(_SETUP,))["test"]
    delta = report["net_r_zero_cost"]["mean"] - report["net_r_realistic_cost"]["mean"]
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
