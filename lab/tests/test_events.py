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


# --- Phase 4 golden matches: hand-verified event outputs --------------------

class TestGoldenMatches:
    """Each candidate event's locked_outcome must match what sim.simulate()
    produces independently for the same trade — this is the golden-match
    gate from ROADMAP Phase 4.

    We use a known bar sequence where every outcome is hand-computable:
    - Flat TR=2.0 background → ATR(14) = 2.0 exactly
    - k_stop=1.0, reward_risk=2.0 → stop=2.0, target=4.0 from entry=100.0
    - One engineered outcome bar (index 16) that triggers unambiguous exits
    """

    def test_locked_outcome_matches_direct_simulation(self):
        snap = _flat_snapshot()
        # bar16: high=105 (LONG target at 104), low=99 (SHORT stop at 102)
        snap.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )

        opens = [b.open for b in snap.trade_bars]
        highs = [b.high for b in snap.trade_bars]
        lows = [b.low for b in snap.trade_bars]
        closes = [b.close for b in snap.trade_bars]

        for event in events:
            # Reconstruct the exact same TradeSpec that build_events used
            entry_index = event.planned_entry_ts // 300_000
            # entry_price from the snapshot at the entry bar
            entry_price = opens[entry_index]
            # ATR = 2.0, k_stop=1.0 → stop_dist=2.0, reward_risk=2.0 → target_dist=4.0
            stop_dist = 2.0
            target_dist = 4.0

            if event.side == "LONG":
                stop_price = entry_price - stop_dist
                target_price = entry_price + target_dist
            else:
                stop_price = entry_price + stop_dist
                target_price = entry_price - target_dist

            spec = sim.TradeSpec(
                side=event.side,
                entry_index=entry_index,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                max_holding_bars=_SETUP.max_holding_bars,
            )
            expected = sim.simulate(opens, highs, lows, closes, spec)

            assert event.locked_outcome.exit_index == expected.exit_index
            assert event.locked_outcome.exit_reason == expected.exit_reason
            assert event.locked_outcome.net_r == expected.net_r
            assert event.locked_outcome.mae_r == expected.mae_r
            assert event.locked_outcome.mfe_r == expected.mfe_r

    def test_canonical_bytes_hash_is_stable(self):
        """The hash of canonical_bytes must be deterministic and stable —
        changing event ordering or fields must change the hash."""
        snap = _flat_snapshot()
        snap.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )
        h1 = events_module.events_hash(events)
        h2 = events_module.events_hash(events)
        assert h1 == h2

        # Build again — must produce the same hash
        events2 = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )
        h3 = events_module.events_hash(events2)
        assert h1 == h3

        # Changing split_ts changes which events exist → different hash
        events_diff = events_module.build_events(
            [snap], split_ts=16 * 300_000, setup=_SETUP,
        )
        h_diff = events_module.events_hash(events_diff)
        assert h1 != h_diff

    def test_event_id_is_deterministic(self):
        """event_id is a SHA-256 of (symbol, decision_ts, side) — must be
        reproducible across builds."""
        snap = _flat_snapshot()
        snap.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events1 = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )
        events2 = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )

        for e1, e2 in zip(events1, events2):
            assert e1.event_id == e2.event_id
            assert len(e1.event_id) == 64

    def test_split_assignment_is_time_based(self):
        """Events with decision_ts < split_ts are train, >= are test.
        Train events whose outcome spans into test window are purged."""
        snap = _flat_snapshot()
        # bar16: high=105 triggers LONG target at index 16
        snap.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        # Split at 16 * 300_000:
        # decision i=14: ts=15*300K < split → train, outcome_end_ts=17*300K >= split → purged
        # decision i=15: ts=16*300K >= split → test, never purged
        events = events_module.build_events(
            [snap], split_ts=16 * 300_000, setup=_SETUP,
        )
        assert len(events) == 2
        assert all(e.split == "test" for e in events)

        # Split at 100 * 300_000 (far future): all train, none purged
        events2 = events_module.build_events(
            [snap], split_ts=100 * 300_000, setup=_SETUP,
        )
        assert len(events2) == 4
        assert all(e.split == "train" for e in events2)
