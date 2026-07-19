"""Dataset builder — the single authority that turns raw candle and funding
records into a validated, hashable market tape (ROADMAP Phase 2).

One row = one completed candle = one potential decision point. A decision at bar
t may look only at bars <= t; the raw tapes themselves carry no features, labels
or outcomes — those are later phases, downstream of the locked simulation.

Purity (mirrors lab/sim.py, RULES §14): no network, no wall-clock, no file or
env access. This module consumes already-fetched, already-completed records and
is a pure function of them. Acquisition — the exchange HTTP fetch — lives in
tools/, never here: the non-deterministic capture stays out of the deterministic
builder, so the same records in produce the same tape hashes on any machine.

Fail-closed (RULES §1): structural defects raise. A gap is recorded explicitly,
never silently filled. Prices that are not finite, positive and OHLC-consistent
are rejected — the same validity contract lab/sim.py and lab/indicators.py
enforce at their boundaries; this module is where it is first established for
the raw tapes.

Determinism (RULES §8): each tape's identity is a SHA-256 over a canonical,
round-tripping text serialization of its records — NOT over container bytes
(parquet is not byte-stable across writers, so hashing its bytes would break
"same input -> same hash"). The canonical form is plain text and hand-verifiable.

Scope (v0): three separate immutable tapes for one instrument/interval, per
ARCHITECTURE.md §8.2 — last-price OHLCV bars, mark-price OHLC bars (no volume:
mark price has no traded size), and funding-rate events. Each tape is validated
and hashed independently; a snapshot is the tuple of the three hashes plus the
instrument/range identity recorded in its manifest (tools/, not this module).

``FundingRecord`` here is a *raw, timestamped* funding-rate observation — the
tape as fetched from the exchange. It is distinct from ``lab.sim.FundingEvent``,
which is *bar-index-relative* and consumed directly by ``simulate()``. Aligning
a funding timestamp to a bar index and looking up the concurrent mark price is
event-construction work for a later phase, not this module's concern.

Contents:
  Bar                  one completed OHLCV candle (frozen)
  MarkBar              one completed mark-price OHLC candle, no volume (frozen)
  Gap                  one explicit run of missing candles (frozen)
  FundingRecord        one raw funding-rate observation (frozen)
  to_bars              parse + validate raw trade-candle records, fail-closed
  to_mark_bars         parse + validate raw mark-candle records, fail-closed
  to_funding_records   parse + validate raw funding records, fail-closed
  detect_gaps          find and report missing candles (never fill)
  aggregate            base interval -> higher interval, complete buckets only
  canonical_bytes / dataset_hash                   trade tape identity
  canonical_mark_bytes / mark_dataset_hash         mark tape identity
  canonical_funding_bytes / funding_dataset_hash   funding tape identity
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

SCHEMA_VERSION = "market-v0"


@dataclass(frozen=True, slots=True)
class Bar:
    """One completed candle. ``open_ts`` is the bar's open time in milliseconds
    since the Unix epoch, aligned to the interval grid. Prices are quote per
    base unit; ``volume`` is in base units."""

    open_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarkBar:
    """One completed mark-price candle. Same shape as ``Bar`` minus volume —
    mark price is a valuation curve, not a traded quantity."""

    open_ts: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of missing candles between two present bars. ``missing`` is the
    count of absent candles strictly between ``prev_open_ts`` and
    ``next_open_ts`` — recorded, never filled."""

    prev_open_ts: int
    next_open_ts: int
    missing: int


@dataclass(frozen=True, slots=True)
class FundingRecord:
    """One raw funding-rate observation from the exchange's funding tape.
    ``funding_time`` is milliseconds since the Unix epoch; ``rate`` is the
    funding rate as a fraction (e.g. 0.0001 = 0.01%). Not aligned to the price
    bar grid — funding settles on its own cadence (typically every 8h).

    This is the *raw, timestamped* record as fetched. See the module docstring
    for how this differs from ``lab.sim.FundingEvent``."""

    funding_time: int
    rate: float


# --- validation helpers ------------------------------------------------------

def _finite(x: float) -> bool:
    """True if x is a finite real number (not bool, NaN or inf)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return math.isfinite(float(x))


def _valid_ohlc(o: float, h: float, l: float, c: float) -> bool:
    """A usable bar is finite, positive, and self-consistent:
    ``low <= min(open, close) <= max(open, close) <= high``. Same contract as
    lab/sim.py's ``_validate_bar`` and lab/indicators.py's ``_valid_ohlc``."""
    if not (_finite(o) and _finite(h) and _finite(l) and _finite(c)):
        return False
    if l <= 0.0:
        return False
    return l <= min(o, c) and max(o, c) <= h


def _check_interval(interval_ms: int) -> None:
    """Interval must be a positive int of milliseconds (structural — RULES §1).
    ``bool`` is rejected because ``True``/``False`` are ints in Python."""
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
        raise TypeError(
            f"interval_ms must be an int, got {type(interval_ms).__name__}"
        )
    if interval_ms <= 0:
        raise ValueError(f"interval_ms must be > 0, got {interval_ms}")


def _check_ts_ohlc(
    ts: object, o: float, h: float, l: float, c: float,
    prev_ts: int | None, idx: int, interval_ms: int,
) -> int:
    """Shared structural checks for one OHLC bar record: integer timestamp
    aligned to the interval grid, OHLC validity, and strictly-increasing order
    vs. the previous record. Returns ``ts`` (now known to be int) so callers
    can track it as their next ``prev_ts``. Raises on any violation, naming the
    offending index — shared by ``to_bars`` and ``to_mark_bars`` so the trade
    and mark tapes enforce identically the same integrity contract."""
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise TypeError(f"record[{idx}]: open_ts must be an int")
    if ts < 0:
        raise ValueError(f"record[{idx}]: open_ts {ts} must be >= 0")
    if ts % interval_ms != 0:
        raise ValueError(
            f"record[{idx}]: open_ts {ts} not aligned to interval {interval_ms}"
        )
    if not _valid_ohlc(o, h, l, c):
        raise ValueError(
            f"record[{idx}]: invalid OHLC (o={o}, h={h}, l={l}, c={c})"
        )
    if prev_ts is not None and ts <= prev_ts:
        raise ValueError(
            f"record[{idx}]: open_ts {ts} not strictly after previous "
            f"{prev_ts} (duplicate or out of order)"
        )
    return ts


# --- raw records -> validated bars -------------------------------------------

def to_bars(records: Sequence[Sequence[float]], interval_ms: int) -> list[Bar]:
    """Parse and structurally validate raw trade-candle records, fail-closed.

    Each record is ``(open_ts, open, high, low, close, volume)``. Every record
    must have an integer ``open_ts`` aligned to the interval grid, finite and
    OHLC-consistent prices, and a finite non-negative volume. Across the
    sequence, ``open_ts`` must be strictly increasing (this rejects duplicates
    and out-of-order candles). Any violation raises ``ValueError``/``TypeError``
    naming the offending index — no record is silently dropped or repaired.

    Returns the validated bars in input order. Detecting *missing* candles is a
    separate, non-raising concern (``detect_gaps``): a hole in the tape is data
    to record, not a structural error.
    """
    _check_interval(interval_ms)
    bars: list[Bar] = []
    prev_ts: int | None = None
    for idx, rec in enumerate(records):
        if len(rec) != 6:
            raise ValueError(f"record[{idx}]: expected 6 fields, got {len(rec)}")
        ts, o, h, l, c, v = rec
        ts = _check_ts_ohlc(ts, o, h, l, c, prev_ts, idx, interval_ms)
        if not _finite(v) or v < 0.0:
            raise ValueError(f"record[{idx}]: volume {v} must be finite and >= 0")
        prev_ts = ts
        bars.append(Bar(ts, float(o), float(h), float(l), float(c), float(v)))
    return bars


def to_mark_bars(
    records: Sequence[Sequence[float]], interval_ms: int
) -> list[MarkBar]:
    """Parse and structurally validate raw mark-candle records, fail-closed.

    Each record is ``(open_ts, open, high, low, close)`` — mark price carries
    no traded volume. Same integrity contract as ``to_bars`` (alignment, OHLC
    validity, strictly increasing timestamps), enforced by the same shared
    check so the two tapes cannot silently diverge in what counts as valid.
    """
    _check_interval(interval_ms)
    bars: list[MarkBar] = []
    prev_ts: int | None = None
    for idx, rec in enumerate(records):
        if len(rec) != 5:
            raise ValueError(f"record[{idx}]: expected 5 fields, got {len(rec)}")
        ts, o, h, l, c = rec
        ts = _check_ts_ohlc(ts, o, h, l, c, prev_ts, idx, interval_ms)
        prev_ts = ts
        bars.append(MarkBar(ts, float(o), float(h), float(l), float(c)))
    return bars


def to_funding_records(
    records: Sequence[Sequence[float]],
) -> list[FundingRecord]:
    """Parse and structurally validate raw funding records, fail-closed.

    Each record is ``(funding_time, rate)``. ``funding_time`` must be a
    non-negative int, strictly increasing across the sequence (rejects
    duplicates and out-of-order events). ``rate`` must be finite with
    ``abs(rate) < 1.0`` — the same bound ``lab.sim._funding_return`` enforces,
    so a tape that fails here would also fail closed inside the simulator.
    Funding is not aligned to the price-bar grid, so no interval check applies.
    """
    events: list[FundingRecord] = []
    prev_ts: int | None = None
    for idx, rec in enumerate(records):
        if len(rec) != 2:
            raise ValueError(f"record[{idx}]: expected 2 fields, got {len(rec)}")
        ts, rate = rec
        if isinstance(ts, bool) or not isinstance(ts, int):
            raise TypeError(f"record[{idx}]: funding_time must be an int")
        if ts < 0:
            raise ValueError(f"record[{idx}]: funding_time {ts} must be >= 0")
        if prev_ts is not None and ts <= prev_ts:
            raise ValueError(
                f"record[{idx}]: funding_time {ts} not strictly after previous "
                f"{prev_ts} (duplicate or out of order)"
            )
        if not _finite(rate) or abs(rate) >= 1.0:
            raise ValueError(
                f"record[{idx}]: rate {rate} must be finite with abs < 1.0"
            )
        prev_ts = ts
        events.append(FundingRecord(ts, float(rate)))
    return events


# --- gap detection -----------------------------------------------------------

def detect_gaps(bars: Sequence[Bar], interval_ms: int) -> list[Gap]:
    """Report every run of missing candles in ``bars`` (output of ``to_bars``).

    Consecutive bars should differ by exactly one interval; any larger step is a
    gap whose ``missing`` count is recorded. Gaps are never filled. Because
    ``to_bars`` guarantees aligned, strictly increasing timestamps, every step
    is a positive multiple of the interval; a non-multiple would mean the input
    was not produced by ``to_bars`` and raises.
    """
    _check_interval(interval_ms)
    gaps: list[Gap] = []
    for a, b in zip(bars, bars[1:]):
        delta = b.open_ts - a.open_ts
        if delta % interval_ms != 0:
            raise ValueError(
                f"unaligned step between {a.open_ts} and {b.open_ts} "
                f"(interval {interval_ms}) — bars not from to_bars"
            )
        missing = delta // interval_ms - 1
        if missing > 0:
            gaps.append(Gap(a.open_ts, b.open_ts, missing))
    return gaps


# --- aggregation -------------------------------------------------------------

def aggregate(bars: Sequence[Bar], factor: int, interval_ms: int) -> list[Bar]:
    """Aggregate base-interval bars into ``factor``-times-longer bars.

    A higher-interval bar is emitted only for a *complete* bucket: exactly
    ``factor`` contiguous base bars aligned to the higher-interval grid. A
    bucket touched by a gap (a missing base bar) is skipped, never fabricated —
    consistent with the fail-closed, no-silent-fill contract. Open is the first
    bar's open, close the last bar's close, high/low the extremes, volume the
    (compensated) sum.

    ``bars`` must be ``to_bars`` output (aligned, strictly increasing).
    """
    _check_interval(interval_ms)
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError(f"factor must be an int, got {type(factor).__name__}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")

    higher = interval_ms * factor
    result: list[Bar] = []
    n = len(bars)
    i = 0
    while i < n:
        bucket_start = bars[i].open_ts - (bars[i].open_ts % higher)
        window: list[Bar] = []
        for k in range(factor):
            j = i + k
            if j >= n or bars[j].open_ts != bucket_start + k * interval_ms:
                break
            window.append(bars[j])
        if len(window) == factor:
            result.append(Bar(
                open_ts=bucket_start,
                open=window[0].open,
                high=max(b.high for b in window),
                low=min(b.low for b in window),
                close=window[-1].close,
                volume=math.fsum(b.volume for b in window),
            ))
            i += factor
        else:
            # Incomplete bucket (gap or partial leading bucket): skip, do not
            # fabricate. Advance one bar and re-align on the next boundary.
            i += 1
    return result


# --- canonical serialization + hash ------------------------------------------

def _sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes — shared by all three tape hashers."""
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(bars: Sequence[Bar]) -> bytes:
    """Deterministic, round-tripping serialization of the trade tape.

    One header line (schema version) then one line per bar. Floats use ``repr``,
    which since CPython 3.1 is the shortest string that round-trips to the exact
    value and is stable across platforms — so identical bars hash identically
    everywhere, and a human can read the file and check a candle by hand.
    """
    lines = [SCHEMA_VERSION]
    for b in bars:
        lines.append(
            f"{b.open_ts} {b.open!r} {b.high!r} {b.low!r} {b.close!r} {b.volume!r}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def dataset_hash(bars: Sequence[Bar]) -> str:
    """SHA-256 hex digest of ``canonical_bytes`` — the trade tape identity."""
    return _sha256_hex(canonical_bytes(bars))


def canonical_mark_bytes(bars: Sequence[MarkBar]) -> bytes:
    """Deterministic, round-tripping serialization of the mark tape. Same
    format contract as ``canonical_bytes`` (see there), one line per bar,
    minus the volume field mark bars do not have."""
    lines = [SCHEMA_VERSION]
    for b in bars:
        lines.append(f"{b.open_ts} {b.open!r} {b.high!r} {b.low!r} {b.close!r}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def mark_dataset_hash(bars: Sequence[MarkBar]) -> str:
    """SHA-256 hex digest of ``canonical_mark_bytes`` — the mark tape identity."""
    return _sha256_hex(canonical_mark_bytes(bars))


def canonical_funding_bytes(records: Sequence[FundingRecord]) -> bytes:
    """Deterministic, round-tripping serialization of the funding tape."""
    lines = [SCHEMA_VERSION]
    for r in records:
        lines.append(f"{r.funding_time} {r.rate!r}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def funding_dataset_hash(records: Sequence[FundingRecord]) -> str:
    """SHA-256 hex digest of ``canonical_funding_bytes`` — funding identity."""
    return _sha256_hex(canonical_funding_bytes(records))
