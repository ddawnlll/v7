"""Market tape builder checks: structural validation, gap detection, aggregation,
and hash determinism for all three raw tapes (trade, mark, funding).

Every rejection test asserts a specific record index is named, matching the
fail-closed contract (RULES §1): no record is silently dropped or repaired.
"""

from __future__ import annotations

import math

import pytest

from lab import market as tape

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
    bars = tape.to_bars(_trade(3), I)
    assert len(bars) == 3
    assert bars[0] == tape.Bar(0, 100.0, 101.0, 99.0, 100.5, 10.0)
    assert bars[1].open_ts == I


def test_to_bars_empty_input():
    assert tape.to_bars([], I) == []


# --- to_bars: rejection branches (each names the offending index) -------------

def test_to_bars_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 6 or 10 fields"):
        tape.to_bars([(0, 100.0, 101.0, 99.0, 100.0)], I)


def test_to_bars_rejects_bool_timestamp():
    with pytest.raises(TypeError, match=r"record\[0\]: open_ts must be an int"):
        tape.to_bars([(True, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_non_int_timestamp():
    with pytest.raises(TypeError, match=r"record\[0\]: open_ts must be an int"):
        tape.to_bars([(0.5, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_negative_timestamp():
    with pytest.raises(ValueError, match=r"open_ts -1 must be >= 0"):
        tape.to_bars([(-1, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_unaligned_timestamp():
    with pytest.raises(ValueError, match="not aligned to interval"):
        tape.to_bars([(150_000, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_high_below_low():
    with pytest.raises(ValueError, match="invalid OHLC"):
        tape.to_bars([(0, 100.0, 90.0, 95.0, 92.0, 1.0)], I)


def test_to_bars_rejects_close_outside_range():
    with pytest.raises(ValueError, match="invalid OHLC"):
        tape.to_bars([(0, 100.0, 101.0, 99.0, 110.0, 1.0)], I)


def test_to_bars_rejects_non_finite_price():
    with pytest.raises(ValueError, match="invalid OHLC"):
        tape.to_bars([(0, float("nan"), 101.0, 99.0, 100.0, 1.0)], I)


def test_to_bars_rejects_non_positive_low():
    with pytest.raises(ValueError, match="invalid OHLC"):
        tape.to_bars([(0, 0.0, 0.0, 0.0, 0.0, 1.0)], I)


def test_to_bars_rejects_negative_volume():
    with pytest.raises(ValueError, match="volume -1.0 must be finite"):
        tape.to_bars([(0, 100.0, 101.0, 99.0, 100.0, -1.0)], I)


def test_to_bars_rejects_non_finite_volume():
    with pytest.raises(ValueError, match="volume .* must be finite"):
        tape.to_bars([(0, 100.0, 101.0, 99.0, 100.0, float("inf"))], I)


def test_to_bars_rejects_duplicate_timestamp():
    recs = [(0, 100.0, 101.0, 99.0, 100.0, 1.0), (0, 100.0, 101.0, 99.0, 100.0, 1.0)]
    with pytest.raises(ValueError, match=r"record\[1\].*duplicate or out of order"):
        tape.to_bars(recs, I)


def test_to_bars_rejects_out_of_order_timestamp():
    recs = [(I, 100.0, 101.0, 99.0, 100.0, 1.0), (0, 100.0, 101.0, 99.0, 100.0, 1.0)]
    with pytest.raises(ValueError, match=r"record\[1\].*duplicate or out of order"):
        tape.to_bars(recs, I)


def test_to_bars_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval_ms must be > 0"):
        tape.to_bars(_trade(1), 0)
    with pytest.raises(TypeError, match="interval_ms must be an int"):
        tape.to_bars(_trade(1), True)


# --- to_mark_bars: shares _check_ts_ohlc, so verify wiring + no-volume shape --

def test_to_mark_bars_parses_valid_records():
    bars = tape.to_mark_bars(_mark(2), I)
    assert bars[0] == tape.MarkBar(0, 100.0, 101.0, 99.0, 100.5)


def test_to_mark_bars_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 5 fields"):
        tape.to_mark_bars([(0, 100.0, 101.0, 99.0, 100.0, 1.0)], I)


def test_to_mark_bars_rejects_unaligned_timestamp():
    with pytest.raises(ValueError, match="not aligned to interval"):
        tape.to_mark_bars([(150_000, 100.0, 101.0, 99.0, 100.0)], I)


def test_to_mark_bars_rejects_invalid_ohlc():
    with pytest.raises(ValueError, match="invalid OHLC"):
        tape.to_mark_bars([(0, 100.0, 90.0, 95.0, 92.0)], I)


def test_to_mark_bars_rejects_duplicate_timestamp():
    recs = [(0, 100.0, 101.0, 99.0, 100.0), (0, 100.0, 101.0, 99.0, 100.0)]
    with pytest.raises(ValueError, match="duplicate or out of order"):
        tape.to_mark_bars(recs, I)


# --- to_funding_records --------------------------------------------------------

def test_to_funding_records_parses_valid_records():
    events = tape.to_funding_records([(0, 0.0001), (28_800_000, 0.00015)])
    assert events[0] == tape.FundingRecord(0, 0.0001)
    assert events[1].funding_time == 28_800_000


def test_to_funding_records_not_grid_aligned():
    # Funding has no interval-alignment requirement (unlike bars).
    events = tape.to_funding_records([(1, 0.0001), (12345, 0.0002)])
    assert len(events) == 2


def test_to_funding_records_rejects_wrong_field_count():
    with pytest.raises(ValueError, match=r"record\[0\]: expected 2 fields"):
        tape.to_funding_records([(0, 0.0001, 100.0)])


def test_to_funding_records_rejects_bool_timestamp():
    with pytest.raises(TypeError, match="funding_time must be an int"):
        tape.to_funding_records([(True, 0.0001)])


def test_to_funding_records_rejects_negative_timestamp():
    with pytest.raises(ValueError, match="funding_time -1 must be >= 0"):
        tape.to_funding_records([(-1, 0.0001)])


def test_to_funding_records_rejects_duplicate_timestamp():
    with pytest.raises(ValueError, match="duplicate or out of order"):
        tape.to_funding_records([(0, 0.0001), (0, 0.0002)])


def test_to_funding_records_rejects_out_of_order_timestamp():
    with pytest.raises(ValueError, match="duplicate or out of order"):
        tape.to_funding_records([(100, 0.0001), (0, 0.0002)])


def test_to_funding_records_rejects_rate_at_or_above_bound():
    with pytest.raises(ValueError, match="abs < 1.0"):
        tape.to_funding_records([(0, 1.0)])
    with pytest.raises(ValueError, match="abs < 1.0"):
        tape.to_funding_records([(0, -1.5)])


def test_to_funding_records_rejects_non_finite_rate():
    with pytest.raises(ValueError, match="abs < 1.0"):
        tape.to_funding_records([(0, float("nan"))])


# --- detect_gaps -----------------------------------------------------------

def test_detect_gaps_empty_when_contiguous():
    bars = tape.to_bars(_trade(5), I)
    assert tape.detect_gaps(bars, I) == []


def test_detect_gaps_finds_single_gap():
    recs = [_trade(5)[0], _trade(5)[1], _trade(5)[4]]  # keep idx 0,1,4 -> gap of 2
    bars = tape.to_bars(recs, I)
    gaps = tape.detect_gaps(bars, I)
    assert gaps == [tape.Gap(prev_open_ts=I, next_open_ts=4 * I, missing=2)]


def test_detect_gaps_finds_multiple_gaps():
    full = _trade(7)
    recs = [full[0], full[2], full[3], full[6]]  # gaps at (0->2) and (3->6)
    bars = tape.to_bars(recs, I)
    gaps = tape.detect_gaps(bars, I)
    assert len(gaps) == 2
    assert gaps[0].missing == 1
    assert gaps[1].missing == 2


def test_detect_gaps_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval_ms must be > 0"):
        tape.detect_gaps([], -1)


# --- aggregate ---------------------------------------------------------------

def test_aggregate_combines_complete_bucket():
    bars = tape.to_bars(_trade(3), I)  # opens 100,101,102; highs 101,102,103
    agg = tape.aggregate(bars, 3, I)
    assert len(agg) == 1
    b = agg[0]
    assert b.open_ts == 0
    assert b.open == 100.0          # first bar's open
    assert b.close == 102.5         # last bar's close
    assert b.high == pytest.approx(103.0)   # max high
    assert b.low == pytest.approx(99.0)     # min low
    assert b.volume == pytest.approx(10.0 + 11.0 + 12.0)


def test_aggregate_skips_incomplete_trailing_bucket():
    bars = tape.to_bars(_trade(2), I)  # only 2 of 3 needed for factor=3
    assert tape.aggregate(bars, 3, I) == []


def test_aggregate_skips_bucket_with_internal_gap():
    full = _trade(6)
    recs = [full[0], full[2], full[3], full[4], full[5]]  # bar 1 missing
    bars = tape.to_bars(recs, I)
    agg = tape.aggregate(bars, 3, I)  # bucket [0,1,2) incomplete, [3,4,5) complete
    assert len(agg) == 1
    assert agg[0].open_ts == 3 * I


def test_aggregate_realigns_after_gap_to_grid_boundary():
    # 6 bars per 3x bucket grid: bars at idx 0 and 4..8 (bar 0 alone can't form
    # a bucket with 4..6; the next complete bucket must start on a 3x boundary).
    full = _trade(9)
    recs = [full[0]] + full[4:9]
    bars = tape.to_bars(recs, I)
    agg = tape.aggregate(bars, 3, I)
    assert len(agg) == 1
    assert agg[0].open_ts == 6 * I  # bucket [6,7,8) is the first complete one


def test_aggregate_rejects_factor_below_two():
    bars = tape.to_bars(_trade(2), I)
    with pytest.raises(ValueError, match="factor must be >= 2"):
        tape.aggregate(bars, 1, I)


def test_aggregate_rejects_non_int_factor():
    bars = tape.to_bars(_trade(2), I)
    with pytest.raises(TypeError, match="factor must be an int"):
        tape.aggregate(bars, 2.0, I)


def test_aggregate_empty_input():
    assert tape.aggregate([], 3, I) == []


# --- canonical serialization + hash: round-trip and determinism --------------

def test_canonical_bytes_is_human_readable_and_stable():
    bars = tape.to_bars(_trade(2), I)
    raw = tape.canonical_bytes(bars)
    assert raw == (
        b"market-v0\n"
        b"0 100.0 101.0 99.0 100.5 10.0\n"
        b"300000 101.0 102.0 100.0 101.5 11.0\n"
    )
    assert tape.canonical_bytes(bars) == raw  # deterministic repeat


def test_tape_hash_stable_and_sensitive():
    bars = tape.to_bars(_trade(3), I)
    h1 = tape.trade_tape_hash(bars)
    h2 = tape.trade_tape_hash(bars)
    assert h1 == h2
    assert len(h1) == 64

    mutated = tape.to_bars(_trade(3, start=100.0001), I)
    assert tape.trade_tape_hash(mutated) != h1


def test_trade_tape_hash_frozen_golden_value():
    """Same tape used in canonical-bytes test. Must never change silently — a
    change means the serialization contract changed (RULES §8 anchor)."""
    bars = tape.to_bars(
        [(0, 100.0, 101.0, 99.0, 100.5, 10.0), (I, 100.5, 102.0, 100.0, 101.5, 12.0)],
        I,
    )
    assert tape.trade_tape_hash(bars) == (
        "3778984891e6ceb8dff309cc1b6df0c62c7818affd917da9f8094c8d8e92608f"
    )


def test_mark_tape_hash_frozen_golden_value():
    bars = tape.to_mark_bars(
        [(0, 100.0, 101.0, 99.0, 100.5), (I, 100.5, 102.0, 100.0, 101.5)], I
    )
    assert tape.mark_tape_hash(bars) == (
        "1cc54122f10c2dd558877ab307076bd285d5746b884740c4184c827bd492173a"
    )


def test_funding_tape_hash_frozen_golden_value():
    records = tape.to_funding_records([(0, 0.0001), (28_800_000, 0.00015)])
    assert tape.funding_tape_hash(records) == (
        "71518633185e0ad8d27301ec71ad89e8bceb23fbf81617d6449296ed9387c06d"
    )


def test_empty_tapes_hash_to_header_only():
    assert tape.canonical_bytes([]) == b"market-v0\n"
    assert tape.canonical_mark_bytes([]) == b"market-v0\n"
    assert tape.canonical_funding_bytes([]) == b"market-v0\n"
    # All three empty-tape hashes agree (identical canonical bytes) — expected,
    # not a collision: same header, zero records, nothing to distinguish.
    assert tape.trade_tape_hash([]) == tape.mark_tape_hash([]) == tape.funding_tape_hash([])


# --- extended (order-flow) tapes ----------------------------------------------

def _trade_ext(n: int, start: float = 100.0):
    """Extended records whose flow fields satisfy the containment invariants:
    quote volume is priced at the bar's midpoint, taker buys are 60% of volume."""
    out = []
    for k in range(n):
        low, high = start + k - 1.0, start + k + 1.0
        mid = start + k + 0.5
        volume = 10.0 + k
        tbb = volume * 0.6
        out.append((
            k * I, start + k, high, low, mid, volume,
            volume * mid, 100 + k, tbb, tbb * mid,
        ))
    return out


def test_to_bars_parses_extended_records():
    bars = tape.to_bars(_trade_ext(2), I)
    assert bars[0].quote_volume == pytest.approx(10.0 * 100.5)
    assert bars[0].trade_count == 100
    assert bars[0].taker_buy_base_volume == pytest.approx(6.0)
    assert bars[1].trade_count == 101


def test_to_bars_rejects_mixed_width_tape():
    records = _trade_ext(1) + _trade(2)[1:]
    with pytest.raises(ValueError, match=r"record\[1\]: width 6 differs from 10"):
        tape.to_bars(records, I)


def test_to_bars_rejects_close_time_parsed_as_quote_volume():
    """Regression: Binance kline column 6 is close_time, column 7 is quote
    volume. Parsing 6 as quote volume put a timestamp in the field and nothing
    caught it, because flow fields were neither validated nor hashed. The
    containment bound rejects it by orders of magnitude."""
    ts, close_time = 1689866400000, 1689866699999.0
    rec = (ts, 29805.8, 29824.5, 29771.4, 29822.7, 1450.864,
           close_time, 14054, 824.24, 24562734.1732)
    with pytest.raises(ValueError, match=r"record\[0\]: quote_volume .* outside"):
        tape.to_bars([rec], I)


def test_to_bars_accepts_the_corrected_quote_volume():
    """Same bar with column 7 — the true quote volume — passes."""
    ts = 1689866400000
    rec = (ts, 29805.8, 29824.5, 29771.4, 29822.7, 1450.864,
           43_248_000.0, 14054, 824.24, 24562734.1732)
    bars = tape.to_bars([rec], I)
    assert bars[0].quote_volume == 43_248_000.0


def test_to_bars_rejects_taker_buy_exceeding_volume():
    rec = list(_trade_ext(1)[0])
    rec[8] = rec[5] * 1.5
    with pytest.raises(ValueError, match=r"record\[0\]: taker_buy_base_volume"):
        tape.to_bars([tuple(rec)], I)


def test_to_bars_rejects_quote_volume_without_base_volume():
    rec = (0, 100.0, 101.0, 99.0, 100.0, 0.0, 5.0, 0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"quote_volume 5.0 must be 0"):
        tape.to_bars([rec], I)


def test_to_bars_rejects_non_int_trade_count():
    rec = list(_trade_ext(1)[0])
    rec[7] = 100.5
    with pytest.raises(TypeError, match=r"record\[0\]: trade_count must be an int"):
        tape.to_bars([tuple(rec)], I)


def test_extended_canonical_bytes_marks_header_and_carries_flow():
    bars = tape.to_bars(_trade_ext(1), I)
    raw = tape.canonical_bytes(bars)
    assert raw.startswith(b"market-v0 extended\n")
    assert raw.split(b"\n")[1].split(b" ") == [
        b"0", b"100.0", b"101.0", b"99.0", b"100.5", b"10.0",
        b"1005.0", b"100", b"6.0", b"603.0",
    ]


def test_extended_tape_never_collides_with_base_tape():
    """Same OHLCV, different tape identity — the header alone guarantees it."""
    ext = tape.to_bars(_trade_ext(2), I)
    base = tape.to_bars([r[:6] for r in _trade_ext(2)], I)
    assert tape.trade_tape_hash(ext) != tape.trade_tape_hash(base)


def test_flow_field_corruption_changes_the_tape_hash():
    """The point of hashing flow fields: silently altering one is detectable."""
    bars = tape.to_bars(_trade_ext(2), I)
    h1 = tape.trade_tape_hash(bars)
    records = [list(r) for r in _trade_ext(2)]
    records[0][7] += 1  # one extra trade in the count
    assert tape.trade_tape_hash(tape.to_bars([tuple(r) for r in records], I)) != h1


def test_canonical_bytes_rejects_partially_populated_bar():
    bars = [
        tape.Bar(0, 100.0, 101.0, 99.0, 100.5, 10.0, quote_volume=1005.0),
    ]
    with pytest.raises(ValueError, match=r"bar\[0\]: extended tape with a partially"):
        tape.canonical_bytes(bars)


def test_canonical_bytes_rejects_mixed_tape():
    bars = tape.to_bars(_trade_ext(1), I) + tape.to_bars([_trade(2)[1]], I)
    with pytest.raises(ValueError, match="mixed tape"):
        tape.canonical_bytes(bars)


def test_aggregate_sums_flow_fields():
    bars = tape.to_bars(_trade_ext(4), I)
    agg = tape.aggregate(bars, 2, I)
    assert len(agg) == 2
    assert agg[0].volume == pytest.approx(10.0 + 11.0)
    assert agg[0].quote_volume == pytest.approx(10.0 * 100.5 + 11.0 * 101.5)
    assert agg[0].trade_count == 100 + 101
    assert agg[0].taker_buy_base_volume == pytest.approx(6.0 + 6.6)


def test_aggregate_leaves_base_tape_flow_unset():
    agg = tape.aggregate(tape.to_bars(_trade(4), I), 2, I)
    assert agg[0].quote_volume is None
    assert agg[0].trade_count is None


# --- flow_violation: the reporting face of the same rule ----------------------

def test_flow_violation_none_for_a_sound_bar():
    _, _, high, low, _, volume, qv, _, tbb, tbq = _trade_ext(1)[0]
    assert tape.flow_violation(volume, low, high, qv, tbb, tbq) is None


def test_flow_violation_names_the_broken_invariant():
    reason = tape.flow_violation(37811.0, 5.335, 5.356, 963232.0673, 104350.4, 558032.054)
    assert reason is not None
    assert "taker_buy_base_volume" in reason and "exceeds volume" in reason


def test_flow_violation_and_to_bars_agree():
    """Single authority (RULES §4): every bar ``flow_violation`` flags is a bar
    ``to_bars`` refuses, and every bar it clears is one ``to_bars`` accepts."""
    ok = _trade_ext(1)[0]
    cases = [ok]
    for pos, value in ((6, 1689866699999.0), (8, 999.0), (9, 1e12), (6, float("nan"))):
        bad = list(ok)
        bad[pos] = value
        cases.append(tuple(bad))

    for rec in cases:
        _, _, high, low, _, volume, qv, _, tbb, tbq = rec
        flagged = tape.flow_violation(volume, low, high, qv, tbb, tbq) is not None
        try:
            tape.to_bars([rec], I)
            rejected = False
        except ValueError:
            rejected = True
        assert flagged == rejected, f"disagreement on {rec}"
