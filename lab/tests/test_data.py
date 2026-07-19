"""Dataset builder checks: structural validation, gap detection, aggregation,
and hash determinism for all three raw tapes (trade, mark, funding).

Every rejection test asserts a specific record index is named, matching the
fail-closed contract (RULES §1): no record is silently dropped or repaired.
"""

import math

import pytest

from lab import data

I = 300_000  # 5m in ms


def _trade(n: int, start: float = 100.0):
    return [
        (k * I, start + k, start + k + 1.0, start + k - 1.0, start + k + 0.5, 10.0 + k)
        for k in range(n)
    ]


def _mark(n: int, start: float = 100.0):
    return [
        (k * I, start + k, start + k + 1.0, start + k - 1.0, start + k + 0.5)
        for k in range(n)
    ]


# --- to_bars: happy path -------------------------------------------------------

def test_to_bars_parses_valid_records():
    bars = data.to_bars(_trade(3), I)
    assert len(bars) == 3
    assert bars[0] == data.Bar(0, 100.0, 101.0, 99.0, 100.5, 10.0)
    assert bars[1].open_ts == I


def test_to_bars_empty_input():
    assert data.to_bars([], I) == []


# --- to_bars: rejection branches (each names the offending index) -------------

def test_to_bars_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 6 fields"):
        data.to_bars([(0, 100.0, 101.0, 99.0, 100.0)], I)


def test_to_bars_rejects_bool_timestamp():
    with pytest.raises(TypeError, match=r"record\[0\]: open_ts must be an int"):
        data.to_bars([(True, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_non_int_timestamp():
    with pytest.raises(TypeError, match=r"record\[0\]: open_ts must be an int"):
        data.to_bars([(0.5, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_negative_timestamp():
    with pytest.raises(ValueError, match=r"open_ts -1 must be >= 0"):
        data.to_bars([(-1, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_unaligned_timestamp():
    with pytest.raises(ValueError, match="not aligned to interval"):
        data.to_bars([(150_000, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_high_below_low():
    with pytest.raises(ValueError, match="invalid OHLC"):
        data.to_bars([(0, 100.0, 90.0, 95.0, 92.0, 1.0)], I)


def test_to_bars_rejects_close_outside_range():
    with pytest.raises(ValueError, match="invalid OHLC"):
        data.to_bars([(0, 100.0, 101.0, 99.0, 110.0, 1.0)], I)


def test_to_bars_rejects_non_finite_price():
    with pytest.raises(ValueError, match="invalid OHLC"):
        data.to_bars([(0, float("nan"), 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_non_positive_low():
    with pytest.raises(ValueError, match="invalid OHLC"):
        data.to_bars([(0, 0.0, 0.0, 0.0, 0.0, 1.0)], I)


def test_to_bars_rejects_negative_volume():
    with pytest.raises(ValueError, match="volume -1.0 must be finite"):
        data.to_bars([(0, 100.0, 101.0, 99.0, 100.0, -1.0)], I)


def test_to_bars_rejects_non_finite_volume():
    with pytest.raises(ValueError, match="volume .* must be finite"):
        data.to_bars([(0, 100.0, 101.0, 99.0, 100.0, float("inf"))], I)


def test_to_bars_rejects_duplicate_timestamp():
    recs = [(0, 100.0, 101.0, 99.0, 100.0, 1.0), (0, 100.0, 101.0, 99.0, 100.0, 1.0)]
    with pytest.raises(ValueError, match=r"record\[1\].*duplicate or out of order"):
        data.to_bars(recs, I)


def test_to_bars_rejects_out_of_order_timestamp():
    recs = [(I, 100.0, 101.0, 99.0, 100.0, 1.0), (0, 100.0, 101.0, 99.0, 100.0, 1.0)]
    with pytest.raises(ValueError, match=r"record\[1\].*duplicate or out of order"):
        data.to_bars(recs, I)


def test_to_bars_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval_ms must be > 0"):
        data.to_bars(_trade(1), 0)
    with pytest.raises(TypeError, match="interval_ms must be an int"):
        data.to_bars(_trade(1), True)


# --- to_mark_bars: shares _check_ts_ohlc, so verify wiring + no-volume shape --

def test_to_mark_bars_parses_valid_records():
    bars = data.to_mark_bars(_mark(2), I)
    assert bars[0] == data.MarkBar(0, 100.0, 101.0, 99.0, 100.5)


def test_to_mark_bars_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 5 fields"):
        data.to_mark_bars([(0, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_mark_bars_rejects_unaligned_timestamp():
    with pytest.raises(ValueError, match="not aligned to interval"):
        data.to_mark_bars([(150_000, 100.0, 101.0, 99.0, 100.0)], I)


def test_to_mark_bars_rejects_invalid_ohlc():
    with pytest.raises(ValueError, match="invalid OHLC"):
        data.to_mark_bars([(0, 100.0, 90.0, 95.0, 92.0)], I)


def test_to_mark_bars_rejects_duplicate_timestamp():
    recs = [(0, 100.0, 101.0, 99.0, 100.0), (0, 100.0, 101.0, 99.0, 100.0)]
    with pytest.raises(ValueError, match="duplicate or out of order"):
        data.to_mark_bars(recs, I)


# --- to_funding_records --------------------------------------------------------

def test_to_funding_records_parses_valid_records():
    events = data.to_funding_records([(0, 0.0001), (28_800_000, 0.00015)])
    assert events[0] == data.FundingRecord(0, 0.0001)
    assert events[1].funding_time == 28_800_000


def test_to_funding_records_not_grid_aligned():
    # Funding has no interval-alignment requirement (unlike bars).
    events = data.to_funding_records([(1, 0.0001), (12345, 0.0002)])
    assert len(events) == 2


def test_to_funding_records_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 2 fields"):
        data.to_funding_records([(0, 0.0001, 100.0)])


def test_to_funding_records_rejects_bool_timestamp():
    with pytest.raises(TypeError, match="funding_time must be an int"):
        data.to_funding_records([(True, 0.0001)])


def test_to_funding_records_rejects_negative_timestamp():
    with pytest.raises(ValueError, match="funding_time -1 must be >= 0"):
        data.to_funding_records([(-1, 0.0001)])


def test_to_funding_records_rejects_duplicate_timestamp():
    with pytest.raises(ValueError, match="duplicate or out of order"):
        data.to_funding_records([(0, 0.0001), (0, 0.0002)])


def test_to_funding_records_rejects_out_of_order_timestamp():
    with pytest.raises(ValueError, match="duplicate or out of order"):
        data.to_funding_records([(100, 0.0001), (0, 0.0002)])


def test_to_funding_records_rejects_rate_at_or_above_bound():
    with pytest.raises(ValueError, match="abs < 1.0"):
        data.to_funding_records([(0, 1.0)])
    with pytest.raises(ValueError, match="abs < 1.0"):
        data.to_funding_records([(0, -1.5)])


def test_to_funding_records_rejects_non_finite_rate():
    with pytest.raises(ValueError, match="abs < 1.0"):
        data.to_funding_records([(0, float("nan"))])


# --- detect_gaps -----------------------------------------------------------

def test_detect_gaps_empty_when_contiguous():
    bars = data.to_bars(_trade(5), I)
    assert data.detect_gaps(bars, I) == []


def test_detect_gaps_finds_single_gap():
    recs = [_trade(5)[0], _trade(5)[1], _trade(5)[4]]  # keep idx 0,1,4 -> gap of 2
    bars = data.to_bars(recs, I)
    gaps = data.detect_gaps(bars, I)
    assert gaps == [data.Gap(prev_open_ts=I, next_open_ts=4 * I, missing=2)]


def test_detect_gaps_finds_multiple_gaps():
    full = _trade(7)
    recs = [full[0], full[2], full[3], full[6]]  # gaps at (0->2) and (3->6)
    bars = data.to_bars(recs, I)
    gaps = data.detect_gaps(bars, I)
    assert len(gaps) == 2
    assert gaps[0].missing == 1
    assert gaps[1].missing == 2


def test_detect_gaps_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval_ms must be > 0"):
        data.detect_gaps([], -1)


# --- aggregate ---------------------------------------------------------------

def test_aggregate_combines_complete_bucket():
    bars = data.to_bars(_trade(3), I)  # opens 100,101,102; highs 101,102,103
    agg = data.aggregate(bars, 3, I)
    assert len(agg) == 1
    b = agg[0]
    assert b.open_ts == 0
    assert b.open == 100.0          # first bar's open
    assert b.close == 102.5         # last bar's close
    assert b.high == pytest.approx(103.0)   # max high
    assert b.low == pytest.approx(99.0)     # min low
    assert b.volume == pytest.approx(10.0 + 11.0 + 12.0)


def test_aggregate_skips_incomplete_trailing_bucket():
    bars = data.to_bars(_trade(2), I)  # only 2 of 3 needed for factor=3
    assert data.aggregate(bars, 3, I) == []


def test_aggregate_skips_bucket_with_internal_gap():
    full = _trade(6)
    recs = [full[0], full[2], full[3], full[4], full[5]]  # bar 1 missing
    bars = data.to_bars(recs, I)
    agg = data.aggregate(bars, 3, I)  # bucket [0,1,2) incomplete, [3,4,5) complete
    assert len(agg) == 1
    assert agg[0].open_ts == 3 * I


def test_aggregate_realigns_after_gap_to_grid_boundary():
    # 6 bars per 3x bucket grid: bars at idx 0 and 4..8 (bar 0 alone can't form
    # a bucket with 4..6; the next complete bucket must start on a 3x boundary).
    full = _trade(9)
    recs = [full[0]] + full[4:9]
    bars = data.to_bars(recs, I)
    agg = data.aggregate(bars, 3, I)
    assert len(agg) == 1
    assert agg[0].open_ts == 6 * I  # bucket [6,7,8) is the first complete one


def test_aggregate_rejects_factor_below_two():
    bars = data.to_bars(_trade(2), I)
    with pytest.raises(ValueError, match="factor must be >= 2"):
        data.aggregate(bars, 1, I)


def test_aggregate_rejects_non_int_factor():
    bars = data.to_bars(_trade(2), I)
    with pytest.raises(TypeError, match="factor must be an int"):
        data.aggregate(bars, 2.0, I)


def test_aggregate_empty_input():
    assert data.aggregate([], 3, I) == []


# --- canonical serialization + hash: round-trip and determinism --------------

def test_canonical_bytes_is_human_readable_and_stable():
    bars = data.to_bars(_trade(2), I)
    raw = data.canonical_bytes(bars)
    assert raw == (
        b"market-v0\n"
        b"0 100.0 101.0 99.0 100.5 10.0\n"
        b"300000 101.0 102.0 100.0 101.5 11.0\n"
    )
    assert data.canonical_bytes(bars) == raw  # deterministic repeat


def test_dataset_hash_stable_and_sensitive():
    bars = data.to_bars(_trade(3), I)
    h1 = data.dataset_hash(bars)
    h2 = data.dataset_hash(bars)
    assert h1 == h2
    assert len(h1) == 64

    mutated = data.to_bars(_trade(3, start=100.0001), I)
    assert data.dataset_hash(mutated) != h1


def test_dataset_hash_frozen_golden_value():
    """Same tape used in canonical-bytes test. Must never change silently — a
    change means the serialization contract changed (RULES §8 anchor)."""
    bars = data.to_bars(
        [(0, 100.0, 101.0, 99.0, 100.5, 10.0), (I, 100.5, 102.0, 100.0, 101.5, 12.0)],
        I,
    )
    assert data.dataset_hash(bars) == (
        "3778984891e6ceb8dff309cc1b6df0c62c7818affd917da9f8094c8d8e92608f"
    )


def test_mark_dataset_hash_frozen_golden_value():
    bars = data.to_mark_bars(
        [(0, 100.0, 101.0, 99.0, 100.5), (I, 100.5, 102.0, 100.0, 101.5)], I
    )
    assert data.mark_dataset_hash(bars) == (
        "1cc54122f10c2dd558877ab307076bd285d5746b884740c4184c827bd492173a"
    )


def test_funding_dataset_hash_frozen_golden_value():
    records = data.to_funding_records([(0, 0.0001), (28_800_000, 0.00015)])
    assert data.funding_dataset_hash(records) == (
        "71518633185e0ad8d27301ec71ad89e8bceb23fbf81617d6449296ed9387c06d"
    )


def test_empty_tapes_hash_to_header_only():
    assert data.canonical_bytes([]) == b"market-v0\n"
    assert data.canonical_mark_bytes([]) == b"market-v0\n"
    assert data.canonical_funding_bytes([]) == b"market-v0\n"
    # All three empty-tape hashes agree (identical canonical bytes) — expected,
    # not a collision: same header, zero records, nothing to distinguish.
    assert data.dataset_hash([]) == data.mark_dataset_hash([]) == data.funding_dataset_hash([])
