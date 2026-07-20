"""Tests for lab/features.py — Phase 6 feature extraction authority.

Covers: directional mirror, future-proofness, insufficient history,
unknown timestamps, finite values, feature order stability, O(1) lookup.
"""

from __future__ import annotations

import pytest
import numpy as np

from lab.features import (
    FEATURE_NAMES,
    FEATURE_DIM,
    precompute_features,
    build_event_features,
)
from lab.events import CandidateEvent
from lab.sim import TradeOutcome, CostBreakdown
from lab.market import Bar


def _flat_bars(n: int = 30) -> list[Bar]:
    return [
        Bar(open_ts=i * 300_000, open=100.0, high=101.0, low=99.0,
            close=100.0, volume=1.0)
        for i in range(n)
    ]


def _event(event_id: str, symbol: str, side: str, decision_ts: int) -> CandidateEvent:
    return CandidateEvent(
        event_id=event_id, symbol=symbol, side=side,
        feature_cutoff_ts=decision_ts, decision_ts=decision_ts,
        planned_entry_ts=decision_ts, fill_ts=decision_ts,
        outcome_end_ts=decision_ts + 300_000,
        locked_outcome=TradeOutcome(
            side=side, entry_index=0, exit_index=1, exit_reason="stop",
            entry_price=100.0, exit_price=101.0,
            nominal_return=1.0, risk_fraction=1.0,
            gross_return=0.0, net_return=0.0, net_r=0.0,
            mae_r=0.0, mfe_r=0.0,
            costs=CostBreakdown(fee=0.0, slippage=0.0, funding=0.0, total=0.0),
        ),
        split="train",
    )


# --- directional mirror ---

def test_long_short_features_are_mirrored():
    """Same timestamp, LONG vs SHORT: returns mirrored, ATR identical."""
    bars = _flat_bars(30)
    ts = 15 * 300_000  # after ATR warmup
    events = [
        _event("e1", "TEST", "LONG", ts),
        _event("e2", "TEST", "SHORT", ts),
    ]
    feats = build_event_features(events, {"TEST": bars})

    f_long = feats["e1"]
    f_short = feats["e2"]

    # Momentum returns mirrored
    assert f_long[0] == pytest.approx(-f_short[0])
    assert f_long[1] == pytest.approx(-f_short[1])
    assert f_long[2] == pytest.approx(-f_short[2])
    # ATR identical (unsigned)
    assert f_long[3] == pytest.approx(f_short[3])


# --- future-proofness ---

def test_future_append_does_not_change_features():
    """Adding bars AFTER decision_ts must not change computed features."""
    bars = _flat_bars(30)
    ts = 15 * 300_000
    events = [_event("e1", "TEST", "LONG", ts)]

    feats_before = build_event_features(events, {"TEST": bars})

    # Append bars after decision_ts
    bars_after = bars + _flat_bars(50)
    feats_after = build_event_features(events, {"TEST": bars_after})

    np.testing.assert_array_equal(feats_before["e1"], feats_after["e1"])


def test_bars_after_cutoff_do_not_change_features():
    """Modifying bars AFTER decision_ts must not affect features."""
    bars = _flat_bars(30)
    ts = 15 * 300_000
    events = [_event("e1", "TEST", "LONG", ts)]

    feats_before = build_event_features(events, {"TEST": bars})

    # Modify bars at index 20+ (after decision bar 14)
    for i in range(20, 30):
        bars[i] = Bar(open_ts=i * 300_000, open=999.0, high=999.0,
                      low=999.0, close=999.0, volume=1.0)

    feats_after = build_event_features(events, {"TEST": bars})

    np.testing.assert_array_equal(feats_before["e1"], feats_after["e1"])


# --- fail-closed ---

def test_unknown_timestamp_raises():
    bars = _flat_bars(30)
    with pytest.raises(KeyError, match="not found"):
        precompute_features(bars, [999_999_999])


def test_insufficient_history_raises():
    bars = _flat_bars(30)
    # decision at bar index 5 (only 5 bars of history, need >=14)
    with pytest.raises(ValueError, match="insufficient history"):
        precompute_features(bars, [5 * 300_000])


def test_empty_bars_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        precompute_features([], [0])


# --- finite values ---

def test_all_values_are_finite():
    bars = _flat_bars(30)
    # Make one bar have a different close
    bars[14] = Bar(open_ts=14 * 300_000, open=100.0, high=105.0,
                   low=99.0, close=103.0, volume=1.0)
    feats = precompute_features(bars, [15 * 300_000])
    f = feats[15 * 300_000]
    assert np.all(np.isfinite(f))
    assert len(f) == FEATURE_DIM


# --- feature order ---

def test_feature_order_is_stable():
    """FEATURE_NAMES order must match array index order."""
    assert FEATURE_NAMES == ("return_5m", "return_15m", "return_1h", "atr_pct")
    bars = _flat_bars(30)
    feats = precompute_features(bars, [15 * 300_000])
    f = feats[15 * 300_000]
    assert f[0] == pytest.approx(0.0)  # flat bars, no return
    assert f[3] > 0  # ATR positive


# --- O(1) lookup ---

def test_precompute_o1_lookup():
    bars = _flat_bars(100)
    tss = [i * 300_000 for i in range(15, 98)]  # need idx >= ATR_PERIOD
    feats = precompute_features(bars, tss)
    for ts in tss:
        assert ts in feats
        assert len(feats[ts]) == FEATURE_DIM


# --- determinism ---

def test_features_are_deterministic():
    bars = _flat_bars(30)
    ts = 15 * 300_000
    f1 = precompute_features(bars, [ts])[ts]
    f2 = precompute_features(bars, [ts])[ts]
    np.testing.assert_array_equal(f1, f2)
