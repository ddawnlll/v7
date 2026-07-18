"""Truth-core checks. Every trade here is hand-computable from the bars.

Bar layout convention in these tests: index 0 is the entry bar (never
inspected), the walk starts at index 1.
"""

import math

import pytest

from lab.sim import (
    CostBreakdown,
    FundingEvent,
    TradeOutcome,
    TradeSpec,
    _finite_number,
    _funding_return,
    _strict_int,
    _validate_bar,
    round_trip_cost,
    simulate,
)

ZERO = dict(fee_rate=0.0, slippage_bps=0.0)


def _bars(opens, highs, lows, closes):
    return opens, highs, lows, closes


def _long(**kw):
    base = dict(
        side="LONG", entry_index=0, entry_price=100.0,
        stop_price=90.0, target_price=110.0, max_holding_bars=5,
    )
    base.update(ZERO)
    base.update(kw)
    return TradeSpec(**base)


def _flat6():
    return [100.0] * 6


# --- validation helpers -------------------------------------------------------

def test_strict_int():
    assert _strict_int(5)
    assert not _strict_int(True)
    assert not _strict_int(5.0)
    assert not _strict_int("5")
    assert _strict_int(True, allow_bool=True)


def test_finite_number():
    assert _finite_number(1.0)
    assert _finite_number(0)
    assert not _finite_number(float("nan"))
    assert not _finite_number(float("inf"))
    assert not _finite_number(True)


def test_bar_validation():
    _validate_bar(10, 9, 9.5, 9.8, 0)  # valid
    with pytest.raises(ValueError, match="not finite"):
        _validate_bar(float("nan"), 9, 9.5, 9.8, 0)
    with pytest.raises(ValueError, match="low.*> 0"):
        _validate_bar(10, 0, 9.5, 9.8, 0)
    with pytest.raises(ValueError, match="high.*< low"):
        _validate_bar(5, 9, 9.5, 9.8, 0)
    with pytest.raises(ValueError, match="close.*outside"):
        _validate_bar(10, 9, 9.5, 11, 0)
    with pytest.raises(ValueError, match="open.*outside"):
        _validate_bar(10, 9, 11, 9.8, 0)


# --- barrier logic -----------------------------------------------------------

def test_long_target_hit():
    opens = [100, 100, 109, 100, 100, 100]
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "target"
    assert out.exit_index == 2
    # Gap: open=109, target=110 → max(109,110)=110 for LONG target
    assert out.exit_price == 110.0
    assert out.nominal_return == pytest.approx(0.10)
    assert out.net_r == pytest.approx(1.0)


def test_long_stop_hit():
    opens = [100, 95, 100, 100, 100, 100]
    highs = [100, 105, 100, 100, 100, 100]
    lows = [100, 89, 100, 100, 100, 100]
    closes = [100, 95, 100, 100, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "stop"
    assert out.exit_index == 1
    # Gap: open=95, stop=90 → min(95,90)=90 for LONG stop
    assert out.exit_price == 90.0
    assert out.net_r == pytest.approx(-1.0)


def test_long_stop_gap_fill():
    # Bar opens at 88 — gapped through stop. Fill at min(88, 90) = 88.
    opens = [100, 88, 100, 100, 100, 100]
    highs = [100, 90, 100, 100, 100, 100]
    lows = [100, 85, 100, 100, 100, 100]
    closes = [100, 87, 100, 100, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "stop"
    assert out.exit_price == 88.0
    assert out.nominal_return == pytest.approx(-0.10)  # at stop=90
    assert out.gross_return == pytest.approx(-0.12)    # at fill=88
    assert out.net_r < -1.0  # worse than -1R due to gap


def test_long_target_gap_fill():
    # Bar opens at 112 — gapped through target. Fill at max(112, 110) = 112.
    opens = [100, 112, 100, 100, 100, 100]
    highs = [100, 115, 100, 100, 100, 100]
    lows = [100, 110, 100, 100, 100, 100]
    closes = [100, 113, 100, 100, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "target"
    assert out.exit_price == 112.0
    assert out.nominal_return == pytest.approx(0.10)
    assert out.gross_return == pytest.approx(0.12)
    assert out.net_r > 1.0


def test_time_exit_at_last_bar_close():
    flat = _flat6()
    out = simulate(flat, flat, flat, flat, _long(max_holding_bars=3))
    assert out.exit_reason == "time"
    assert out.exit_index == 3
    assert out.exit_price == 100.0
    assert out.gross_return == pytest.approx(0.0)


def test_stop_wins_tie_within_bar():
    opens = [100, 100, 100, 100, 100, 100]
    highs = [100, 111, 100, 100, 100, 100]
    lows = [100, 89, 100, 100, 100, 100]
    closes = [100, 100, 100, 100, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "stop"
    assert out.exit_price == 90.0


def test_no_lookahead_on_entry_bar():
    opens = [100, 999, 100, 100, 100, 100]
    highs = [100, 999, 100, 100, 100, 100]
    lows = [100, 100, 100, 100, 100, 100]
    closes = [100, 100, 100, 100, 100, 100]
    out = simulate(opens, highs, lows, closes,
                   _long(entry_index=1, max_holding_bars=3))
    assert out.exit_reason == "time"
    assert out.exit_index == 4


def test_short_target_symmetric():
    opens = [100, 100, 91, 100, 100, 100]
    highs = [100, 105, 100, 100, 100, 100]
    lows = [100, 95, 89, 100, 100, 100]
    closes = [100, 100, 92, 100, 100, 100]
    spec = TradeSpec(
        side="SHORT", entry_index=0, entry_price=100.0,
        stop_price=110.0, target_price=90.0, max_holding_bars=5, **ZERO,
    )
    out = simulate(opens, highs, lows, closes, spec)
    assert out.exit_reason == "target"
    assert out.exit_price == 90.0
    assert out.gross_return == pytest.approx(0.10)
    assert out.net_r == pytest.approx(1.0)


def test_short_stop_gap_fill():
    # SHORT stop at 110, bar opens at 115 → gap fill at max(115,110)=115.
    opens = [100, 115, 100, 100, 100, 100]
    highs = [100, 118, 100, 100, 100, 100]
    lows = [100, 112, 100, 100, 100, 100]
    closes = [100, 114, 100, 100, 100, 100]
    spec = TradeSpec(
        side="SHORT", entry_index=0, entry_price=100.0,
        stop_price=110.0, target_price=90.0, max_holding_bars=5, **ZERO,
    )
    out = simulate(opens, highs, lows, closes, spec)
    assert out.exit_reason == "stop"
    assert out.exit_price == 115.0


# --- excursions --------------------------------------------------------------

def test_mae_mfe_in_r():
    opens = [100, 99, 104, 109, 100, 100]
    highs = [100, 104, 108, 110, 100, 100]
    lows = [100, 95, 102, 100, 100, 100]
    closes = [100, 100, 105, 110, 100, 100]
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_reason == "target"
    assert out.mae_r == pytest.approx(0.5)
    assert out.mfe_r == pytest.approx(1.0)


# --- costs -------------------------------------------------------------------

def test_costs_reduce_net_monotonically():
    opens = [100, 100, 109, 100, 100, 100]
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    free = simulate(opens, highs, lows, closes, _long())
    costed = simulate(
        opens, highs, lows, closes,
        _long(fee_rate=0.0004, slippage_bps=1.0),
    )
    assert costed.net_return < free.net_return
    assert free.net_return - costed.net_return == pytest.approx(0.001)
    assert costed.net_r == pytest.approx((0.10 - 0.001) / 0.10)


def test_round_trip_cost_components():
    c = round_trip_cost(
        fee_rate=0.0004, slippage_bps=1.0,
        side="LONG", funding_return=0.001,
    )
    assert c.fee == pytest.approx(0.0008)
    assert c.slippage == pytest.approx(0.0002)
    assert c.funding == pytest.approx(0.001)
    assert c.total == pytest.approx(0.0008 + 0.0002 + 0.001)


# --- funding tape ------------------------------------------------------------

def test_funding_basic():
    events = [
        FundingEvent(1, 0.0001, 100.0),
        FundingEvent(2, 0.0001, 100.0),
    ]
    fr = _funding_return(events, entry_index=0, exit_index=10,
                         side="LONG", entry_fill_price=100.0)
    # Both events after entry: 2 × 0.0001 × 1.0 = 0.0002
    assert fr == pytest.approx(0.0002)


def test_funding_excludes_entry_bar_event():
    events = [FundingEvent(0, 0.0001, 100.0)]
    fr = _funding_return(events, entry_index=0, exit_index=10,
                         side="LONG", entry_fill_price=100.0)
    # bar_index=0 is NOT > entry_index → excluded
    assert fr == pytest.approx(0.0)


def test_funding_post_exit_excluded():
    events = [
        FundingEvent(1, 0.0001, 100.0),
        FundingEvent(5, 0.0001, 100.0),  # after exit_index=3
    ]
    fr = _funding_return(events, entry_index=0, exit_index=3,
                         side="LONG", entry_fill_price=100.0)
    assert fr == pytest.approx(0.0001)


def test_unsorted_funding_tape_fails_closed():
    events = [
        FundingEvent(5, 0.01, 100.0),
        FundingEvent(1, 0.01, 100.0),  # out of order
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        _funding_return(events, 0, 10, "LONG", 100.0)


def test_post_exit_funding_values_are_not_read():
    # Future event with NaN values: must pass (values not read).
    events = [FundingEvent(99, float("nan"), float("nan"))]
    fr = _funding_return(events, entry_index=0, exit_index=3,
                         side="LONG", entry_fill_price=100.0)
    assert fr == pytest.approx(0.0)


def test_funding_short_sign():
    events = [FundingEvent(1, 0.0001, 100.0)]
    fr = _funding_return(events, entry_index=0, exit_index=10,
                         side="SHORT", entry_fill_price=100.0)
    assert fr == pytest.approx(-0.0001)


def test_funding_mark_price_scaling():
    events = [FundingEvent(1, 0.0001, 200.0)]
    fr = _funding_return(events, entry_index=0, exit_index=10,
                         side="LONG", entry_fill_price=100.0)
    assert fr == pytest.approx(0.0002)  # 0.0001 × 200/100


def test_full_simulate_with_funding():
    opens = [100, 100, 109, 100, 100, 100]
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    events = [
        FundingEvent(1, 0.0001, 100.0),
        FundingEvent(2, 0.0001, 100.0),
    ]
    out = simulate(opens, highs, lows, closes, _long(fee_rate=0.0004, slippage_bps=1.0),
                   funding_events=events)
    # Gross return = 0.10, costs: fee 0.0008 + slip 0.0002 + funding 0.0002 = 0.0012
    assert out.costs.funding == pytest.approx(0.0002)
    assert out.net_return == pytest.approx(0.10 - 0.0012)
    assert out.net_r == pytest.approx((0.10 - 0.0012) / 0.10)


def test_funding_bad_rate_fails():
    events = [FundingEvent(1, 10.0, 100.0)]  # |rate| >= 1
    with pytest.raises(ValueError, match="rate"):
        _funding_return(events, 0, 10, "LONG", 100.0)


def test_funding_bad_bar_index_fails():
    events = [FundingEvent(-1, 0.01, 100.0)]
    with pytest.raises(ValueError, match="bar_index"):
        _funding_return(events, 0, 10, "LONG", 100.0)


# --- bar validation (lazy, at event end) -------------------------------------

def test_lazy_bar_validation():
    opens = [100, 100, 109, 100, 100, float("nan")]
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    # Bar 5 is NaN but never inspected (exit at bar 2) → OK
    out = simulate(opens, highs, lows, closes, _long())
    assert out.exit_index == 2

    # But bar 1 is inspected and must be valid.
    opens_bad = [100, float("inf"), 109, 100, 100, 100]
    with pytest.raises(ValueError, match="not finite"):
        simulate(opens_bad, highs, lows, closes, _long())


# --- fail-closed validation --------------------------------------------------

def test_rejects_insufficient_forward_bars():
    flat = _flat6()
    with pytest.raises(ValueError, match="not enough forward bars"):
        simulate(flat, flat, flat, flat, _long(max_holding_bars=500))


def test_rejects_bad_barrier_ordering():
    flat = _flat6()
    with pytest.raises(ValueError, match="LONG requires"):
        simulate(flat, flat, flat, flat, _long(stop_price=110.0, target_price=120.0))


def test_rejects_mismatched_bar_lengths():
    with pytest.raises(ValueError, match="equal length"):
        simulate([100.0]*6, [100.0]*6, [100.0]*5, [100.0]*6, _long())


def test_negative_fee_fails_closed():
    flat = _flat6()
    with pytest.raises(ValueError, match="fee_rate"):
        simulate(flat, flat, flat, flat, _long(fee_rate=-0.1))


def test_non_finite_entry_price_fails():
    flat = _flat6()
    with pytest.raises(ValueError, match="entry_price"):
        simulate(flat, flat, flat, flat, _long(entry_price=float("nan")))


def test_non_finite_output_fails():
    # Extreme gap: entry=1e-300, target fills at 1e308 → fractional return overflows.
    opens = [1e-300, 1e308, 1, 1, 1, 1]
    highs = [1e-300, 1e308, 1, 1, 1, 1]
    lows  = [1e-300, 1e308, 1, 1, 1, 1]
    closes = [1e-300, 1e308, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="non-finite output"):
        simulate(opens, highs, lows, closes,
                 _long(entry_price=1e-300, stop_price=1e-301, target_price=1e308))


# --- determinism + outcome hash -----------------------------------------------

def test_deterministic_repeat():
    opens = [100, 100, 109, 100, 100, 100]
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    a = simulate(opens, highs, lows, closes, _long())
    b = simulate(opens, highs, lows, closes, _long())
    assert a == b
    assert a.outcome_hash == b.outcome_hash
    assert len(a.outcome_hash) == 64  # SHA-256 hex digest


def test_outcome_hash_changes_with_exit():
    opens = [100, 100, 109, 100, 100, 100]
    highs = [100, 105, 111, 100, 100, 100]
    lows_hit = [100, 89, 100, 100, 100, 100]
    lows_miss = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    a = simulate(opens, highs, lows_hit, closes, _long())
    b = simulate(opens, highs, lows_miss, closes, _long())
    assert a.outcome_hash != b.outcome_hash
    assert a.exit_reason == "stop"
    assert b.exit_reason == "target"


# --- frozen hash test (cross-machine determinism anchor) ----------------------

def test_frozen_canonical_outcome_hash():
    """The hash of a specific, hand-verified trade. Must never change — this
    is the cross-machine determinism anchor (RULES §14)."""
    opens = [100.0, 101.0, 102.0, 100.0, 100.0, 100.0]
    highs = [100.0, 105.0, 111.0, 100.0, 100.0, 100.0]
    lows = [100.0, 95.0, 101.0, 100.0, 100.0, 100.0]
    closes = [100.0, 102.0, 110.0, 100.0, 100.0, 100.0]
    out = simulate(opens, highs, lows, closes, _long(
        fee_rate=0.0004, slippage_bps=1.0,
    ))
    # Hand-verified: LONG target hit at bar 2, exit_price=110, net_r=0.99
    assert out.exit_reason == "target"
    assert out.exit_index == 2
    assert out.net_r == pytest.approx(0.99)
    # Frozen canonical hash — change = engine contract change = heavy audit.
    assert out.outcome_hash == (
        "17ca5e2c7b99ba36d5edc1eff7afdcd85f06b77023252b11953a23559b029364"
    )
