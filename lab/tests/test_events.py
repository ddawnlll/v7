"""Hand-verifiable tests for lab/events.py."""

from __future__ import annotations

import pytest

from lab import events as events_module, sim
from lab.events import CandidateEvent, EventInput
from lab.tape import Bar
from lab.events import Setup

_SETUP = Setup("test", k_stop=1.0, reward_risk=2.0, max_holding_bars=3)
_N = 20  # Sufficient for ATR warmup + 3-bar holding horizon


def _flat_input() -> EventInput:
    """Create an EventInput with constant TR=2.0 (ATR=2.0)."""
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
    return EventInput(
        symbol="TEST-SWAP",
        trade_bars=trade_bars,
        funding_events=[],
    )


def test_build_events_split_boundaries_and_purging():
    inp = _flat_input()
    inp.trade_bars[16] = Bar(
        open_ts=16 * 300_000,
        open=100.0,
        high=105.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )

    # 1. split_ts is very large: no test split, no train events purged
    events_all_train = events_module.build_events(
        [inp], split_ts=100 * 300_000, setup=_SETUP
    )
    assert len(events_all_train) == 4
    assert all(e.split == "train" for e in events_all_train)

    # 2. split_ts is 16 * 300_000: first decision purged, second is test
    events_purged = events_module.build_events([inp], split_ts=16 * 300_000, setup=_SETUP)
    assert len(events_purged) == 2
    assert all(e.split == "test" for e in events_purged)

    # 3. split_ts is 15 * 300_000: all test
    events_test = events_module.build_events([inp], split_ts=15 * 300_000, setup=_SETUP)
    assert len(events_test) == 4
    assert all(e.split == "test" for e in events_test)


def test_events_determinism_and_hashing():
    inp = _flat_input()
    events1 = events_module.build_events([inp], split_ts=100 * 300_000, setup=_SETUP)
    events2 = events_module.build_events([inp], split_ts=100 * 300_000, setup=_SETUP)

    assert len(events1) == 4
    h1 = events_module.events_hash(events1)
    h2 = events_module.events_hash(events2)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_events_order_is_consistent():
    inp = _flat_input()
    events = events_module.build_events([inp], split_ts=100 * 300_000, setup=_SETUP)
    assert len(events) == 4
    assert events[0].side == "LONG"
    assert events[1].side == "SHORT"
    assert events[2].side == "LONG"
    assert events[3].side == "SHORT"


# --- Phase 4 golden matches: hand-verified event outputs --------------------

class TestGoldenMatches:
    def test_locked_outcome_matches_direct_simulation(self):
        inp = _flat_input()
        inp.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )

        opens = [b.open for b in inp.trade_bars]
        highs = [b.high for b in inp.trade_bars]
        lows = [b.low for b in inp.trade_bars]
        closes = [b.close for b in inp.trade_bars]

        for event in events:
            entry_index = event.planned_entry_ts // 300_000
            entry_price = opens[entry_index]
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
        inp = _flat_input()
        inp.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )
        h1 = events_module.events_hash(events)
        h2 = events_module.events_hash(events)
        assert h1 == h2

        events2 = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )
        h3 = events_module.events_hash(events2)
        assert h1 == h3

        events_diff = events_module.build_events(
            [inp], split_ts=16 * 300_000, setup=_SETUP,
        )
        h_diff = events_module.events_hash(events_diff)
        assert h1 != h_diff

    def test_event_id_is_deterministic(self):
        inp = _flat_input()
        inp.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events1 = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )
        events2 = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )

        for e1, e2 in zip(events1, events2):
            assert e1.event_id == e2.event_id
            assert len(e1.event_id) == 64

    def test_split_assignment_is_time_based(self):
        inp = _flat_input()
        inp.trade_bars[16] = Bar(
            open_ts=16 * 300_000, open=100.0, high=105.0, low=99.0,
            close=100.0, volume=1.0,
        )

        events = events_module.build_events(
            [inp], split_ts=16 * 300_000, setup=_SETUP,
        )
        assert len(events) == 2
        assert all(e.split == "test" for e in events)

        events2 = events_module.build_events(
            [inp], split_ts=100 * 300_000, setup=_SETUP,
        )
        assert len(events2) == 4
        assert all(e.split == "train" for e in events2)
