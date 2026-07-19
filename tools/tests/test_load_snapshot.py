"""Round-trip tests for tools/load_snapshot.py against a tiny fabricated
snapshot directory — reuses build_snapshot.py's own parquet writers so the
on-disk shape is exactly what a real build produces, without touching
network."""
import json

import pytest

from lab import data
from tools.build_snapshot import write_funding_parquet, write_mark_parquet, write_trade_parquet
from tools.load_snapshot import load

_TRADE_BARS = [
    data.Bar(open_ts=0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0),
    data.Bar(open_ts=300_000, open=100.0, high=102.0, low=98.0, close=101.0, volume=2.0),
    data.Bar(open_ts=600_000, open=101.0, high=103.0, low=100.0, close=102.0, volume=1.5),
]
_MARK_BARS = [
    data.MarkBar(open_ts=0, open=100.0, high=101.0, low=99.0, close=100.0),
    data.MarkBar(open_ts=300_000, open=100.0, high=102.0, low=98.0, close=101.5),
    data.MarkBar(open_ts=600_000, open=101.0, high=103.0, low=100.0, close=102.5),
]
_FUNDING_RECORDS = [data.FundingRecord(funding_time=300_000, rate=0.0001)]


def _write_snapshot(tmp_path, *, trade_complete=True, mark_complete=True, tamper_trade=False):
    write_trade_parquet(_TRADE_BARS, tmp_path / "trade_bars_5m.parquet")
    write_mark_parquet(_MARK_BARS, tmp_path / "mark_bars_5m.parquet")
    write_funding_parquet(_FUNDING_RECORDS, tmp_path / "funding_events.parquet")
    (tmp_path / "instrument.json").write_text(json.dumps({"instId": "TEST-SWAP"}))

    manifest = {
        "inst_id": "TEST-SWAP",
        "bar": "5m",
        "requested_start_ts": 0,
        "requested_end_ts": 900_000,
        "trade": {
            "coverage_complete": trade_complete,
            "dataset_hash": data.dataset_hash(_TRADE_BARS),
        },
        "mark": {
            "coverage_complete": mark_complete,
            "dataset_hash": data.mark_dataset_hash(_MARK_BARS),
        },
        "funding": {"dataset_hash": data.funding_dataset_hash(_FUNDING_RECORDS)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    if tamper_trade:
        # Overwrite the trade tape with different values AFTER the manifest
        # (which still records the original, now-stale hash) was written.
        tampered = [
            data.Bar(open_ts=b.open_ts, open=b.open, high=b.high, low=b.low,
                     close=b.close + 999.0, volume=b.volume)
            for b in _TRADE_BARS
        ]
        write_trade_parquet(tampered, tmp_path / "trade_bars_5m.parquet")


def test_load_valid_snapshot_succeeds(tmp_path):
    _write_snapshot(tmp_path)
    loaded = load(tmp_path)

    assert [b.open_ts for b in loaded.trade_bars] == [0, 300_000, 600_000]
    assert [b.open_ts for b in loaded.mark_bars] == [0, 300_000, 600_000]
    assert len(loaded.funding_events) == 1
    # funding_time=300_000 maps to trade_bars[1] (open_ts <= 300_000, latest).
    assert loaded.funding_events[0].bar_index == 1
    assert loaded.funding_events[0].mark_price == _MARK_BARS[1].close
    assert loaded.funding_events[0].rate == 0.0001


def test_tampered_trade_tape_rejected(tmp_path):
    _write_snapshot(tmp_path, tamper_trade=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        load(tmp_path)


def test_incomplete_trade_coverage_rejected(tmp_path):
    _write_snapshot(tmp_path, trade_complete=False)
    with pytest.raises(ValueError, match="coverage_complete=false"):
        load(tmp_path)


def test_incomplete_mark_coverage_rejected(tmp_path):
    _write_snapshot(tmp_path, mark_complete=False)
    with pytest.raises(ValueError, match="coverage_complete=false"):
        load(tmp_path)
