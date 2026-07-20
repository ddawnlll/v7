"""Hand-verifiable tests for lab/events.py."""

from __future__ import annotations

import pytest

from lab import events as events_module, sim
from lab.tape import Bar, MarkBar
from lab.observe import Setup
from tools.snapshot import LoadedSnapshot

_SETUP = Setup("test", k_stop=1.0, reward_risk=2.0, max_holding_bars=3)
_N = 20  # Sufficient for ATR warmup + 3-bar holding horizon


def _flat_snapshot() -> LoadedSnapshot:
    """Create a basic flat snapshot with constant TR=2.0 (ATR=2.0)."""
    trade_bars = [
        Bar(
            open_ts=i * 300_000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
        for i in range(_N)
    ]
    mark_bars = [
        MarkBar(
            open_ts=i * 300_000, open=100.0, high=101.0, low=99.0, close=100.0
        )
        for i in range(_N)
    ]
    manifest = {
        "inst_id": "TEST-SWAP",
        "bar": "5m",
        "trade": {"dataset_hash": "dummy"},
        "mark": {"dataset_hash": "dummy"},
        "funding": {"dataset_hash": "dummy"},
    }
    return LoadedSnapshot(
        trade_bars=trade_bars,
        mark_bars=mark_bars,
        funding_events=[],
        manifest=manifest,
    )


def test_build_events_split_boundaries_and_purging():
    # Setup exit bar at index 16: high=105 (LONG target), low=99
    # Decision at close of 14 -> entry index 15.
    # Exit bar index = 16 -> outcome_end_ts = 16 * 300_000 + 300_000 = 17 * 300_000
    snap = _flat_snapshot()
    snap.trade_bars[16] = Bar(
        open_ts=16 * 300_000,
        open=100.0,
        high=105.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )

    # 1. split_ts is very large: no test split, no train events purged
    events_all_train = events_module.build_events(
        [snap], split_ts=100 * 300_000, setup=_SETUP
    )
    # n_candidates = 4 (LONG and SHORT for two decision bars i=14 and i=15)
    assert len(events_all_train) == 4
    assert all(e.split == "train" for e in events_all_train)

    # 2. split_ts is 16 * 300_000:
    # i=14: decision_ts = 15 * 300_000 < split_ts -> Candidate is train.
    # exit at index 16 -> outcome_end_ts = 17 * 300_000 >= split_ts. Purged.
    # i=15: decision_ts = 16 * 300_000 >= split_ts -> Candidate is test.
    # exit at index 19 (timeout) -> outcome_end_ts = 20 * 300_000.
    # Test events are never purged.
    events_purged = events_module.build_events([snap], split_ts=16 * 300_000, setup=_SETUP)
    assert len(events_purged) == 2
    assert all(e.split == "test" for e in events_purged)

    # 3. split_ts is 15 * 300_000:
    # decision_ts = 15 * 300_000 >= split_ts -> Candidate is test.
    # Test events are never purged.
    events_test = events_module.build_events([snap], split_ts=15 * 300_000, setup=_SETUP)
    assert len(events_test) == 4
    assert all(e.split == "test" for e in events_test)


def test_events_determinism_and_hashing():
    snap = _flat_snapshot()
    events1 = events_module.build_events([snap], split_ts=100 * 300_000, setup=_SETUP)
    events2 = events_module.build_events([snap], split_ts=100 * 300_000, setup=_SETUP)

    assert len(events1) == 4
    h1 = events_module.events_hash(events1)
    h2 = events_module.events_hash(events2)

    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_events_order_is_consistent():
    snap = _flat_snapshot()
    events = events_module.build_events([snap], split_ts=100 * 300_000, setup=_SETUP)
    assert len(events) == 4
    # Verification: Sorted by decision_ts, then symbol, then side.
    # Symbol is TEST-SWAP, decision_ts is 15 * 300_000 for first pair.
    # Side order: "LONG" before "SHORT" alphabetically.
    assert events[0].side == "LONG"
    assert events[1].side == "SHORT"
    assert events[2].side == "LONG"
    assert events[3].side == "SHORT"
