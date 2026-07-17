"""Truth-core checks. Every trade here is hand-computable from the bars.

Bar layout convention in these tests: index 0 is the entry bar (never
inspected), the walk starts at index 1.
"""

import math

import pytest

from lab.sim import (
    TradeSpec,
    round_trip_cost,
    simulate,
)

# Zero-cost spec knobs, so gross == net and net_r is exact.
ZERO = dict(fee_rate=0.0, slippage_bps=0.0, funding_rate=0.0, funding_intervals=0.0)


def _long(**kw):
    base = dict(
        side="LONG", entry_index=0, entry_price=100.0,
        stop_price=90.0, target_price=110.0, max_holding_bars=5,
    )
    base.update(ZERO)
    base.update(kw)
    return TradeSpec(**base)


# --- barrier logic -----------------------------------------------------------

def test_long_target_hit():
    # bar1 no touch; bar2 high reaches 110 → target, exit at 110
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    out = simulate(highs, lows, closes, _long())
    assert out.exit_reason == "target"
    assert out.exit_index == 2
    assert out.exit_price == 110.0
    assert out.gross_return == pytest.approx(0.10)
    assert out.net_r == pytest.approx(1.0)  # reward 10 / risk 10


def test_long_stop_hit():
    highs = [100, 105, 100, 100, 100, 100]
    lows = [100, 89, 100, 100, 100, 100]  # bar1 low pierces 90
    closes = [100, 95, 100, 100, 100, 100]
    out = simulate(highs, lows, closes, _long())
    assert out.exit_reason == "stop"
    assert out.exit_index == 1
    assert out.exit_price == 90.0
    assert out.net_r == pytest.approx(-1.0)


def test_time_exit_at_last_bar_close():
    flat = [100.0] * 6
    out = simulate(flat, flat, flat, _long(max_holding_bars=3))
    assert out.exit_reason == "time"
    assert out.exit_index == 3
    assert out.exit_price == 100.0
    assert out.gross_return == pytest.approx(0.0)


def test_stop_wins_tie_within_bar():
    # bar1 touches BOTH target (111) and stop (89) → stop must win
    highs = [100, 111, 100, 100, 100, 100]
    lows = [100, 89, 100, 100, 100, 100]
    closes = [100, 100, 100, 100, 100, 100]
    out = simulate(highs, lows, closes, _long())
    assert out.exit_reason == "stop"
    assert out.exit_price == 90.0


def test_no_lookahead_on_entry_bar():
    # entry bar (index 1) blows through the target, but the walk starts at
    # index 2, so that move must be invisible.
    highs = [100, 999, 100, 100, 100, 100]
    lows = [100, 100, 100, 100, 100, 100]
    closes = [100, 100, 100, 100, 100, 100]
    out = simulate(highs, lows, closes, _long(entry_index=1, max_holding_bars=3))
    assert out.exit_reason == "time"
    assert out.exit_index == 4


def test_short_target_symmetric():
    # SHORT entry 100, stop 110, target 90; bar2 low reaches 90 → target
    highs = [100, 105, 100, 100, 100, 100]
    lows = [100, 95, 89, 100, 100, 100]
    closes = [100, 100, 92, 100, 100, 100]
    spec = TradeSpec(
        side="SHORT", entry_index=0, entry_price=100.0,
        stop_price=110.0, target_price=90.0, max_holding_bars=5, **ZERO,
    )
    out = simulate(highs, lows, closes, spec)
    assert out.exit_reason == "target"
    assert out.exit_price == 90.0
    assert out.gross_return == pytest.approx(0.10)
    assert out.net_r == pytest.approx(1.0)


# --- excursions --------------------------------------------------------------

def test_mae_mfe_in_r():
    # LONG risk = 10. Before targeting out at bar3, low dips to 95 (adverse 5
    # → 0.5R) and high peaks at 108 (favorable 8 → 0.8R).
    highs = [100, 104, 108, 110, 100, 100]
    lows = [100, 95, 102, 100, 100, 100]
    closes = [100, 100, 105, 110, 100, 100]
    out = simulate(highs, lows, closes, _long())
    assert out.exit_reason == "target"
    assert out.mae_r == pytest.approx(0.5)
    assert out.mfe_r == pytest.approx(1.0)  # target bar high 110 → 1.0R


# --- costs -------------------------------------------------------------------

def test_costs_reduce_net_monotonically():
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    free = simulate(highs, lows, closes, _long())
    costed = simulate(
        highs, lows, closes,
        _long(fee_rate=0.0004, slippage_bps=1.0),
    )
    assert costed.net_return < free.net_return
    # fee 2*0.0004 + slip 2*0.0001 = 0.001 fractional
    assert free.net_return - costed.net_return == pytest.approx(0.001)
    assert costed.net_r == pytest.approx((0.10 - 0.001) / 0.10)


def test_round_trip_cost_components():
    c = round_trip_cost(
        fee_rate=0.0004, slippage_bps=1.0,
        funding_rate=0.0001, funding_intervals=10.0, side="LONG",
    )
    assert c.fee == pytest.approx(0.0008)
    assert c.slippage == pytest.approx(0.0002)
    assert c.funding == pytest.approx(0.001)   # LONG pays
    assert c.total == pytest.approx(0.0008 + 0.0002 + 0.001)


def test_funding_sign_long_pays_short_receives():
    long_c = round_trip_cost(0.0, 0.0, 0.0001, 10.0, "LONG")
    short_c = round_trip_cost(0.0, 0.0, 0.0001, 10.0, "SHORT")
    assert long_c.funding == pytest.approx(0.001)
    assert short_c.funding == pytest.approx(-0.001)


# --- fail-closed validation --------------------------------------------------

def test_rejects_insufficient_forward_bars():
    flat = [100.0] * 3
    with pytest.raises(ValueError, match="not enough forward bars"):
        simulate(flat, flat, flat, _long(max_holding_bars=5))


def test_rejects_bad_barrier_ordering():
    flat = [100.0] * 6
    with pytest.raises(ValueError, match="LONG requires"):
        simulate(flat, flat, flat, _long(stop_price=110.0, target_price=120.0))


def test_rejects_mismatched_bar_lengths():
    with pytest.raises(ValueError, match="equal length"):
        simulate([100.0] * 6, [100.0] * 5, [100.0] * 6, _long())


def test_negative_fee_fails_closed():
    flat = [100.0] * 6
    with pytest.raises(ValueError, match="fee_rate"):
        simulate(flat, flat, flat, _long(fee_rate=-0.1))


# --- determinism -------------------------------------------------------------

def test_deterministic_repeat():
    highs = [100, 105, 111, 100, 100, 100]
    lows = [100, 95, 108, 100, 100, 100]
    closes = [100, 102, 110, 100, 100, 100]
    a = simulate(highs, lows, closes, _long())
    b = simulate(highs, lows, closes, _long())
    assert a == b  # frozen dataclass structural equality
