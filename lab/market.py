"""Market tape builder — the single authority that turns raw candle and funding
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
  canonical_bytes / trade_tape_hash                trade tape identity
  canonical_mark_bytes / mark_tape_hash            mark tape identity
  canonical_funding_bytes / funding_tape_hash      funding tape identity
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

# Schema version for verified tapes on disk
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
    quote_volume: float | None = None
    trade_count: int | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None


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


# Relative slack for the quote-volume containment bounds below. The bounds are
# exact in real arithmetic; this only absorbs the exchange's own rounding of
# ``volume`` and ``quote_volume`` to fixed decimal places.
_FLOW_REL_TOL = 1e-6


def flow_violation(
    volume: float, low: float, high: float,
    quote_volume: float, taker_buy_base: float, taker_buy_quote: float,
) -> str | None:
    """Describe the first order-flow invariant this bar violates, or ``None``.

    The invariants are identities, not heuristics: quote volume is
    ``sum(price * qty)`` over the bar's trades and every price lies in
    ``[low, high]``, so

        volume * low <= quote_volume <= volume * high

    and the same holds for the taker-buy subset against its own base volume,
    which is itself a subset of volume. A field parsed from the wrong CSV
    column (a timestamp, say) breaks these by orders of magnitude.

    This is the single authority for what makes flow fields economically
    impossible (RULES §4). It reports rather than raises so a *builder* can
    apply the dirty-cell policy of RULES §1 — drop the bar and let it surface
    as a gap — while ``to_bars`` keeps its strict, no-silent-repair contract.
    """
    for name, x in (
        ("quote_volume", quote_volume),
        ("taker_buy_base_volume", taker_buy_base),
        ("taker_buy_quote_volume", taker_buy_quote),
    ):
        if not _finite(x) or x < 0.0:
            return f"{name} {x} must be finite and >= 0"

    if taker_buy_base > volume * (1.0 + _FLOW_REL_TOL):
        return f"taker_buy_base_volume {taker_buy_base} exceeds volume {volume}"

    return (
        _containment_violation(quote_volume, volume, low, high, "quote_volume")
        or _containment_violation(
            taker_buy_quote, taker_buy_base, low, high, "taker_buy_quote_volume"
        )
    )


def _containment_violation(
    quote: float, base: float, low: float, high: float, name: str,
) -> str | None:
    """``base * low <= quote <= base * high`` within rounding slack. With no
    base volume there are no trades, so the quote side must be zero."""
    if base == 0.0:
        if quote != 0.0:
            return f"{name} {quote} must be 0 when its base volume is 0"
        return None
    lo, hi = base * low, base * high
    tol = _FLOW_REL_TOL * max(abs(hi), 1.0)
    if not (lo - tol <= quote <= hi + tol):
        return (
            f"{name} {quote} outside [{lo}, {hi}] implied by base volume "
            f"{base} and price range [{low}, {high}]"
        )
    return None


def _check_flow(
    qv: object, tc: object, tbb: object, tbq: object,
    volume: float, low: float, high: float, idx: int,
) -> tuple[float, int, float, float]:
    """Fail-closed gate for one bar's order-flow fields. Type errors mean a
    broken parser and raise on their own; everything else defers to
    ``flow_violation``. Returns the validated fields."""
    if isinstance(tc, bool) or not isinstance(tc, int):
        raise TypeError(f"record[{idx}]: trade_count must be an int")
    if tc < 0:
        raise ValueError(f"record[{idx}]: trade_count {tc} must be >= 0")
    for name, x in (
        ("quote_volume", qv), ("taker_buy_base_volume", tbb),
        ("taker_buy_quote_volume", tbq),
    ):
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(f"record[{idx}]: {name} must be a number")

    qv, tbb, tbq = float(qv), float(tbb), float(tbq)
    reason = flow_violation(volume, low, high, qv, tbb, tbq)
    if reason is not None:
        raise ValueError(f"record[{idx}]: {reason}")
    return qv, tc, tbb, tbq


# --- raw records -> validated bars -------------------------------------------

def to_bars(records: Sequence[Sequence[float]], interval_ms: int) -> list[Bar]:
    """Parse and structurally validate raw trade-candle records, fail-closed.

    Each record is either the base form ``(open_ts, open, high, low, close,
    volume)`` or the extended form, which appends ``(quote_volume, trade_count,
    taker_buy_base_volume, taker_buy_quote_volume)``. Every record must have an
    integer ``open_ts`` aligned to the interval grid, finite and OHLC-consistent
    prices, and a finite non-negative volume. Across the sequence, ``open_ts``
    must be strictly increasing (this rejects duplicates and out-of-order
    candles). Any violation raises ``ValueError``/``TypeError`` naming the
    offending index — no record is silently dropped or repaired.

    Record width must be uniform across the sequence: a tape is either wholly
    base or wholly extended. A mixed tape is a structural error, because a bar
    silently missing its flow fields would otherwise hash as if the exchange
    never reported them.

    Returns the validated bars in input order. Detecting *missing* candles is a
    separate, non-raising concern (``detect_gaps``): a hole in the tape is data
    to record, not a structural error.
    """
    _check_interval(interval_ms)
    bars: list[Bar] = []
    prev_ts: int | None = None
    width: int | None = None
    for idx, rec in enumerate(records):
        if len(rec) not in (6, 10):
            raise ValueError(
                f"record[{idx}]: expected 6 or 10 fields, got {len(rec)}"
            )
        if width is None:
            width = len(rec)
        elif len(rec) != width:
            raise ValueError(
                f"record[{idx}]: width {len(rec)} differs from {width} earlier "
                f"in the tape — a tape is wholly base or wholly extended"
            )
        ts, o, h, l, c, v = rec[:6]
        ts = _check_ts_ohlc(ts, o, h, l, c, prev_ts, idx, interval_ms)
        if not _finite(v) or v < 0.0:
            raise ValueError(f"record[{idx}]: volume {v} must be finite and >= 0")
        v = float(v)
        prev_ts = ts
        if width == 6:
            bars.append(Bar(ts, float(o), float(h), float(l), float(c), v))
            continue
        qv, tc, tbb, tbq = _check_flow(
            rec[6], rec[7], rec[8], rec[9], v, float(l), float(h), idx
        )
        bars.append(Bar(
            ts, float(o), float(h), float(l), float(c), v,
            quote_volume=qv, trade_count=tc,
            taker_buy_base_volume=tbb, taker_buy_quote_volume=tbq,
        ))
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

    Order-flow fields, when the tape carries them, are additive over the bucket
    exactly as volume is, and are carried through — an aggregated bar that
    dropped them would silently hand a 1h consumer a bar the exchange never
    reported flow for.

    ``bars`` must be ``to_bars`` output (aligned, strictly increasing).
    """
    _check_interval(interval_ms)
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError(f"factor must be an int, got {type(factor).__name__}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")

    extended = _tape_is_extended(bars)
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
            flow: dict[str, float | int] = {}
            if extended:
                flow = {
                    "quote_volume": math.fsum(b.quote_volume for b in window),
                    "trade_count": sum(b.trade_count for b in window),
                    "taker_buy_base_volume": math.fsum(
                        b.taker_buy_base_volume for b in window
                    ),
                    "taker_buy_quote_volume": math.fsum(
                        b.taker_buy_quote_volume for b in window
                    ),
                }
            result.append(Bar(
                open_ts=bucket_start,
                open=window[0].open,
                high=max(b.high for b in window),
                low=min(b.low for b in window),
                close=window[-1].close,
                volume=math.fsum(b.volume for b in window),
                **flow,
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


def _tape_is_extended(bars: Sequence[Bar]) -> bool:
    """True if the tape carries order-flow fields, False if none do.

    All-or-nothing: a tape where only some bars carry flow fields raises. This
    mirrors the uniform-width contract in ``to_bars`` and keeps the tape
    identity unambiguous — see ``canonical_bytes``.
    """
    flagged = [
        b.quote_volume is not None or b.trade_count is not None
        or b.taker_buy_base_volume is not None
        or b.taker_buy_quote_volume is not None
        for b in bars
    ]
    if not flagged or not any(flagged):
        return False
    if not all(flagged):
        raise ValueError(
            "mixed tape: some bars carry order-flow fields and some do not"
        )
    for i, b in enumerate(bars):
        if (b.quote_volume is None or b.trade_count is None
                or b.taker_buy_base_volume is None
                or b.taker_buy_quote_volume is None):
            raise ValueError(
                f"bar[{i}]: extended tape with a partially populated bar"
            )
    return True


def canonical_bytes(bars: Sequence[Bar]) -> bytes:
    """Deterministic, round-tripping serialization of the trade tape.

    One header line (schema version) then one line per bar. Floats use ``repr``,
    which since CPython 3.1 is the shortest string that round-trips to the exact
    value and is stable across platforms — so identical bars hash identically
    everywhere, and a human can read the file and check a candle by hand.

    A tape carrying order-flow fields serializes them too, and says so in its
    header. Two consequences, both deliberate: the flow fields are inside the
    tape identity (silently corrupting one changes the hash), and a base tape
    can never collide with an extended tape that happens to share its OHLCV.
    A base tape's bytes are unchanged from ``market-v0``, so hashes recorded
    before flow fields existed stay valid.
    """
    extended = _tape_is_extended(bars)
    lines = [f"{SCHEMA_VERSION} extended" if extended else SCHEMA_VERSION]
    for b in bars:
        line = (
            f"{b.open_ts} {b.open!r} {b.high!r} {b.low!r} {b.close!r} {b.volume!r}"
        )
        if extended:
            line += (
                f" {b.quote_volume!r} {b.trade_count!r} "
                f"{b.taker_buy_base_volume!r} {b.taker_buy_quote_volume!r}"
            )
        lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8")


def trade_tape_hash(bars: Sequence[Bar]) -> str:
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


def mark_tape_hash(bars: Sequence[MarkBar]) -> str:
    """SHA-256 hex digest of ``canonical_mark_bytes`` — the mark tape identity."""
    return _sha256_hex(canonical_mark_bytes(bars))


def canonical_funding_bytes(records: Sequence[FundingRecord]) -> bytes:
    """Deterministic, round-tripping serialization of the funding tape."""
    lines = [SCHEMA_VERSION]
    for r in records:
        lines.append(f"{r.funding_time} {r.rate!r}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def funding_tape_hash(records: Sequence[FundingRecord]) -> str:
    """SHA-256 hex digest of ``canonical_funding_bytes`` — funding identity."""
    return _sha256_hex(canonical_funding_bytes(records))
