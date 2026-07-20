"""Tests for tools/snapshot.py — build validation, pagination guards, and
load round-trip integrity.

Deliberately does NOT test the network fetch functions themselves — that
would mean either hitting live OKX from CI (flaky, slow, rate-limited) or
mocking urllib deeply enough that the test proves nothing about the real
endpoint.
"""
import json
from unittest import mock

import pytest

from lab import market
from tools import data as sn


# ═══════════════════════════════════════════════════════════════════════════════
# build() window validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_partial_bound_start_only_rejected(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        sn.build(start_ts=1776698400000, end_ts=None, out_dir=tmp_path / "snap")


def test_partial_bound_end_only_rejected(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        sn.build(start_ts=None, end_ts=1784474400000, out_dir=tmp_path / "snap")


def test_unsupported_bar_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported bar"):
        sn.build(bar="1m", out_dir=tmp_path / "snap")


def test_inverted_window_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be before"):
        sn.build(start_ts=1784474400000, end_ts=1776698400000, out_dir=tmp_path / "snap")


def test_misaligned_window_rejected(tmp_path):
    with pytest.raises(ValueError, match="align to the"):
        sn.build(start_ts=1776698400001, end_ts=1784474400000, out_dir=tmp_path / "snap")


def test_nonpositive_days_rejected(tmp_path):
    with pytest.raises(ValueError, match="days must be positive"):
        sn.build(days=0, out_dir=tmp_path / "snap")


def test_existing_out_dir_rejected(tmp_path):
    existing = tmp_path / "snap"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        sn.build(start_ts=1776698400000, end_ts=1776698700000, out_dir=existing)


# ═══════════════════════════════════════════════════════════════════════════════
# _paginate_bounded() guards
# ═══════════════════════════════════════════════════════════════════════════════

def test_pagination_stuck_cursor_raises():
    with mock.patch.object(sn, "_get_retry") as get_retry:
        get_retry.return_value = [[str(1000), "1", "1", "1", "1", "1"]]
        with pytest.raises(RuntimeError, match="did not advance"):
            sn._paginate_bounded(
                "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=3
            )


def test_pagination_max_pages_exhausted_raises():
    counter = {"n": 2000}

    def descending_page(_path, _params):
        counter["n"] -= 1
        return [[str(counter["n"]), "1", "1", "1", "1", "1"]]

    with mock.patch.object(sn, "_get_retry", side_effect=descending_page):
        with pytest.raises(RuntimeError, match="exhausted max_pages"):
            sn._paginate_bounded(
                "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=3
            )


def test_pagination_normal_termination_still_works():
    """Regression guard: the two new checks above must not break the
    ordinary case of a cursor that strictly advances to stop_ts."""
    pages = [
        [[str(500), "1", "1", "1", "1", "1"]],
        [[str(0), "1", "1", "1", "1", "1"]],
    ]
    with mock.patch.object(sn, "_get_retry", side_effect=lambda _path, _params: pages.pop(0)):
        rows = sn._paginate_bounded(
            "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=5
        )
    assert len(rows) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# load() round-trip — fabricated snapshot on disk
# ═══════════════════════════════════════════════════════════════════════════════

_TRADE_BARS = [
    market.Bar(open_ts=0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0),
    market.Bar(open_ts=300_000, open=100.0, high=102.0, low=98.0, close=101.0, volume=2.0),
    market.Bar(open_ts=600_000, open=101.0, high=103.0, low=100.0, close=102.0, volume=1.5),
]
_MARK_BARS = [
    market.MarkBar(open_ts=0, open=100.0, high=101.0, low=99.0, close=100.0),
    market.MarkBar(open_ts=300_000, open=100.0, high=102.0, low=98.0, close=101.5),
    market.MarkBar(open_ts=600_000, open=101.0, high=103.0, low=100.0, close=102.5),
]
_FUNDING_RECORDS = [market.FundingRecord(funding_time=300_000, rate=0.0001)]


def _write_snapshot(tmp_path, *, trade_complete=True, mark_complete=True, tamper_trade=False):
    sn.write_trade_parquet(_TRADE_BARS, tmp_path / "trade_bars_5m.parquet")
    sn.write_mark_parquet(_MARK_BARS, tmp_path / "mark_bars_5m.parquet")
    sn.write_funding_parquet(_FUNDING_RECORDS, tmp_path / "funding_events.parquet")
    (tmp_path / "instrument.json").write_text(json.dumps({"instId": "TEST-SWAP"}))

    manifest = {
        "instrument_id": "TEST-SWAP",
        "source": "okx",
        "bar": "5m",
        "requested_start_ts": 0,
        "requested_end_ts": 900_000,
        "trade": {
            "coverage_complete": trade_complete,
            "dataset_hash": market.trade_tape_hash(_TRADE_BARS),
        },
        "mark": {
            "coverage_complete": mark_complete,
            "dataset_hash": market.mark_tape_hash(_MARK_BARS),
        },
        "funding": {"dataset_hash": market.funding_tape_hash(_FUNDING_RECORDS)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    if tamper_trade:
        tampered = [
            market.Bar(open_ts=b.open_ts, open=b.open, high=b.high, low=b.low,
                     close=b.close + 999.0, volume=b.volume)
            for b in _TRADE_BARS
        ]
        sn.write_trade_parquet(tampered, tmp_path / "trade_bars_5m.parquet")


def test_load_valid_snapshot_succeeds(tmp_path):
    _write_snapshot(tmp_path)
    loaded = sn.load(tmp_path)

    assert [b.open_ts for b in loaded.trade_bars] == [0, 300_000, 600_000]
    assert [b.open_ts for b in loaded.mark_bars] == [0, 300_000, 600_000]
    assert len(loaded.funding_events) == 1
    assert loaded.funding_events[0].bar_index == 1
    assert loaded.funding_events[0].mark_price == _MARK_BARS[1].close
    assert loaded.funding_events[0].rate == 0.0001


def test_tampered_trade_tape_rejected(tmp_path):
    _write_snapshot(tmp_path, tamper_trade=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        sn.load(tmp_path)


def test_incomplete_trade_coverage_rejected(tmp_path):
    _write_snapshot(tmp_path, trade_complete=False)
    with pytest.raises(ValueError, match="coverage_complete=false"):
        sn.load(tmp_path)


def test_incomplete_mark_coverage_rejected(tmp_path):
    _write_snapshot(tmp_path, mark_complete=False)
    with pytest.raises(ValueError, match="coverage_complete=false"):
        sn.load(tmp_path)


def test_bar_count_short_of_expected_rejected_even_if_flagged_complete(tmp_path):
    """coverage_complete=true is manifest-authored, not re-derived — a
    truncated tape mislabeled complete must not silently pass."""
    short_trade = _TRADE_BARS[:2]  # window (0, 900_000) implies 3 bars
    sn.write_trade_parquet(short_trade, tmp_path / "trade_bars_5m.parquet")
    sn.write_mark_parquet(_MARK_BARS, tmp_path / "mark_bars_5m.parquet")
    sn.write_funding_parquet(_FUNDING_RECORDS, tmp_path / "funding_events.parquet")
    (tmp_path / "instrument.json").write_text(json.dumps({"instId": "TEST-SWAP"}))

    manifest = {
        "instrument_id": "TEST-SWAP",
        "source": "okx",
        "bar": "5m",
        "requested_start_ts": 0,
        "requested_end_ts": 900_000,
        "trade": {
            "coverage_complete": True,
            "dataset_hash": market.trade_tape_hash(short_trade),
        },
        "mark": {
            "coverage_complete": True,
            "dataset_hash": market.mark_tape_hash(_MARK_BARS),
        },
        "funding": {"dataset_hash": market.funding_tape_hash(_FUNDING_RECORDS)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="implies 3"):
        sn.load(tmp_path)


def test_trade_mark_misalignment_rejected(tmp_path):
    """Trade and mark tapes must share the same open_ts sequence — funding
    events are mapped onto both by index, so a silently misaligned mark
    tape would attach the wrong settlement price to a funding event."""
    shifted_mark = [
        market.MarkBar(open_ts=b.open_ts + 300_000, open=b.open, high=b.high,
                     low=b.low, close=b.close)
        for b in _MARK_BARS
    ]
    sn.write_trade_parquet(_TRADE_BARS, tmp_path / "trade_bars_5m.parquet")
    sn.write_mark_parquet(shifted_mark, tmp_path / "mark_bars_5m.parquet")
    sn.write_funding_parquet(_FUNDING_RECORDS, tmp_path / "funding_events.parquet")
    (tmp_path / "instrument.json").write_text(json.dumps({"instId": "TEST-SWAP"}))

    manifest = {
        "instrument_id": "TEST-SWAP",
        "source": "okx",
        "bar": "5m",
        "requested_start_ts": 0,
        "requested_end_ts": 900_000,
        "trade": {
            "coverage_complete": True,
            "dataset_hash": market.trade_tape_hash(_TRADE_BARS),
        },
        "mark": {
            "coverage_complete": True,
            "dataset_hash": market.mark_tape_hash(shifted_mark),
        },
        "funding": {"dataset_hash": market.funding_tape_hash(_FUNDING_RECORDS)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not index-aligned"):
        sn.load(tmp_path)
