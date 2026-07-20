"""Market snapshot: build, load, verify, observe (ARCHITECTURE §6.3).

This lives OUTSIDE the deterministic core: it is the ONLY file that touches
the network, the wall clock, pyarrow, or the disk. It fetches raw OKX
candles/funding for an explicit, bounded ``[start_ts, end_ts)`` window,
validates them, persists three immutable parquet tapes plus a manifest, then
can reload and re-verify those tapes for downstream consumption — including
running Phase 3 outcome observation against a loaded snapshot.

Three public entry points, one file, one concern (snapshot lifecycle):

  build()  — fetch + validate + persist one bounded, reproducible snapshot
  load()   — load, re-verify and return a LoadedSnapshot from disk
  main()   — CLI: dispatch between ``build`` and ``observe`` subcommands

BUILD
  Four fetches (trade, mark, funding, instrument) hit independent OKX
  endpoints and share no state, so build() runs them concurrently.
  Pagination *within* one fetch stays sequential — OKX's ``after`` cursor is
  stateful. Fail-closed persistence (RULES §1): every file is written into a
  hidden staging directory first and published with a single atomic rename.

  Reproducibility (ROADMAP Phase 2 exit): the window is explicit and
  recorded, never "however many pages as of now" — refetching with the SAME
  start_ts/end_ts must reproduce the same tapes and the same hashes.

  parquet files: trade_bars_5m.parquet  mark_bars_5m.parquet  funding_events.parquet
                  instrument.json       manifest.json

LOAD
  A snapshot on disk is untrusted bytes until re-verified: load() re-derives
  the trade/mark/funding dataset hashes from the loaded rows via lab.data's
  own hashers and compares them against the manifest's recorded hashes, and
  refuses a snapshot whose manifest says its tape coverage was incomplete.
  Nothing downstream (lab/observe.py) ever sees unverified bars.

OBSERVE (CLI subcommand)
  Thin wrapper: load a verified snapshot, run lab.observe (Phase 3 outcome
  observation), persist the report next to the manifest.
  Two ``--setups`` modes:
    ``baseline`` — every 5m bar, plumbing/sanity baseline (observations.json)
    ``stage_b``  — 15m/1h/4h decision intervals (observations_stage_b.json)

CLI:  python3 tools/snapshot.py build [--inst-id BTC-USDT-SWAP] [--days 90]
      python3 tools/snapshot.py observe --snapshot-dir data/snapshots/okx-btc-usdt-swap-5m-...
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import observe as observe_module, sim, tape  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# constants
# ═══════════════════════════════════════════════════════════════════════════════

OKX = "https://www.okx.com"
INTERVAL_MS = {"5m": 300_000}
DAY_MS = 86_400_000

_TRADE_SCHEMA = pa.schema([
    ("open_ts", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
])

_MARK_SCHEMA = pa.schema([
    ("open_ts", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
])

_FUNDING_SCHEMA = pa.schema([
    ("funding_time", pa.int64()),
    ("rate", pa.float64()),
])


# ═══════════════════════════════════════════════════════════════════════════════
# build — network fetch
# ═══════════════════════════════════════════════════════════════════════════════

def _get(path: str, params: dict) -> list:
    """One OKX GET, fail-closed on any non-zero API code."""
    url = f"{OKX}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "v7-lab/0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    if body.get("code") != "0":
        raise RuntimeError(f"OKX error {body.get('code')}: {body.get('msg')} — {url}")
    return body["data"]


def _get_retry(path: str, params: dict, attempts: int = 3) -> list:
    """Retry only transient failures — not OKX-reported API errors, and not
    client-side (4xx) HTTP errors, which are real findings (bad instId, bad
    params) that retrying can never fix."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _get(path, params)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if attempt < attempts - 1:
            time.sleep(0.5 * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def _paginate_bounded(
    path: str, params: dict, initial_after: str, ts_of, stop_ts: int,
    limit: int = 100, max_pages: int = 5000,
) -> list:
    """Page backward via OKX's ``after`` cursor until the oldest fetched row's
    timestamp reaches (or passes) ``stop_ts``, or the exchange runs out of
    history first (an empty page — reported honestly by the caller, not
    papered over). Shared pagination convention for trade, mark and funding.

    Fails closed (RULES §1) on two conditions that would otherwise silently
    under- or over-report data: the cursor not strictly advancing (a stuck or
    repeated page — accumulating it would duplicate rows without ever
    finishing), and exhausting ``max_pages`` before reaching ``stop_ts`` (the
    window is larger than this call expected to page through).
    """
    rows: list = []
    after = initial_after
    previous_oldest: int | None = None
    for page_num in range(max_pages):
        p = dict(params, limit=str(limit))
        if after is not None:
            p["after"] = after
        page = _get_retry(path, p)
        if not page:
            return rows
        rows.extend(page)
        oldest = ts_of(page[-1])
        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(
                f"{path}: pagination cursor did not advance ({oldest} >= "
                f"{previous_oldest}) after {page_num + 1} pages — refusing to "
                "loop on what would be duplicated data"
            )
        previous_oldest = oldest
        after = str(oldest)
        if oldest <= stop_ts:
            return rows
        time.sleep(0.3)  # be polite to the public endpoint
    raise RuntimeError(
        f"{path}: exhausted max_pages={max_pages} before reaching "
        f"stop_ts={stop_ts} (last oldest={previous_oldest}) — window is "
        "larger than this call expected to page through"
    )


def _fetch_trade_candles(inst_id: str, bar: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]."""
    return _paginate_bounded(
        "/api/v5/market/history-candles", {"instId": inst_id, "bar": bar},
        str(end_ts), lambda row: int(row[0]), start_ts,
    )


def _fetch_mark_candles(inst_id: str, bar: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: [ts,o,h,l,c,confirm] — mark price has no traded volume."""
    return _paginate_bounded(
        "/api/v5/market/history-mark-price-candles", {"instId": inst_id, "bar": bar},
        str(end_ts), lambda row: int(row[0]), start_ts,
    )


def _fetch_funding_history(inst_id: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: dicts with fundingTime, fundingRate (+ vendor extras)."""
    return _paginate_bounded(
        "/api/v5/public/funding-rate-history", {"instId": inst_id},
        str(end_ts), lambda row: int(row["fundingTime"]), start_ts,
    )


def _fetch_instrument(inst_id: str) -> dict:
    """Single call: contract/tick/precision metadata for one instrument."""
    rows = _get_retry("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst_id})
    if not rows:
        raise RuntimeError(f"no instrument metadata for {inst_id}")
    return rows[0]


# ═══════════════════════════════════════════════════════════════════════════════
# build — raw rows → validated records
# ═══════════════════════════════════════════════════════════════════════════════

def _to_trade_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    """Keep only completed candles (confirm==\"1\") inside [start_ts, end_ts).
    Volume is base ccy. Ascending order."""
    recs = [
        (int(ts), float(o), float(h), float(l), float(c), float(volccy))
        for ts, o, h, l, c, _vol, volccy, _volq, confirm in rows
        if confirm == "1" and start_ts <= int(ts) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


def _to_mark_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    recs = [
        (int(ts), float(o), float(h), float(l), float(c))
        for ts, o, h, l, c, confirm in rows
        if confirm == "1" and start_ts <= int(ts) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


def _to_funding_input_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    recs = [
        (int(r["fundingTime"]), float(r["fundingRate"]))
        for r in rows
        if start_ts <= int(r["fundingTime"]) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# build — persist
# ═══════════════════════════════════════════════════════════════════════════════

def write_trade_parquet(bars: list, path: Path) -> None:
    table = pa.table({
        "open_ts": [b.open_ts for b in bars],
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
    }, schema=_TRADE_SCHEMA)
    pq.write_table(table, path)


def write_mark_parquet(bars: list, path: Path) -> None:
    table = pa.table({
        "open_ts": [b.open_ts for b in bars],
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
    }, schema=_MARK_SCHEMA)
    pq.write_table(table, path)


def write_funding_parquet(records: list, path: Path) -> None:
    table = pa.table({
        "funding_time": [r.funding_time for r in records],
        "rate": [r.rate for r in records],
    }, schema=_FUNDING_SCHEMA)
    pq.write_table(table, path)


def _default_window(bar: str, days: int) -> tuple[int, int]:
    """end_ts = now floored to the interval grid; start_ts = end_ts - days.
    Only used when the caller doesn't pin an explicit window — the resulting
    concrete ms values are what actually get recorded and must be reused to
    reproduce this exact snapshot later."""
    interval_ms = INTERVAL_MS[bar]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ts = now_ms - (now_ms % interval_ms)
    start_ts = end_ts - days * DAY_MS
    return start_ts, end_ts


def build(
    inst_id: str = "BTC-USDT-SWAP",
    bar: str = "5m",
    start_ts: int | None = None,
    end_ts: int | None = None,
    days: int = 90,
    out_dir: Path | None = None,
    strict: bool = True,
) -> dict:
    """Fetch, validate, persist and hash one bounded, reproducible snapshot.
    Returns the manifest dict (also written to <out_dir>/manifest.json).

    Fails closed (RULES §1): rejects a half-specified window (``start_ts``
    given without ``end_ts`` or vice versa — silently falling back to
    ``--days`` for the missing one would discard what the caller actually
    asked for), refuses to overwrite an existing snapshot directory, stages
    every file and publishes them with one atomic rename (a crash mid-build
    leaves either nothing or a complete snapshot, never a mix), and — when
    ``strict`` (default) — refuses to persist a snapshot whose trade or mark
    tape is short of ``expected_bars`` rather than writing a partial one
    silently. Pass ``strict=False`` (CLI: ``--allow-incomplete``) to inspect
    a partial fetch anyway.
    """
    if bar not in INTERVAL_MS:
        raise ValueError(f"unsupported bar {bar!r}; supported: {sorted(INTERVAL_MS)}")
    interval_ms = INTERVAL_MS[bar]

    if (start_ts is None) != (end_ts is None):
        raise ValueError(
            "start_ts and end_ts must be supplied together — got only one, "
            "which would otherwise silently fall back to --days for the "
            "other and discard the value actually given"
        )
    if start_ts is None:
        if days <= 0:
            raise ValueError(f"days must be positive, got {days}")
        start_ts, end_ts = _default_window(bar, days)
    else:
        assert end_ts is not None  # narrowed by the XOR check above
        if start_ts >= end_ts:
            raise ValueError(f"start_ts ({start_ts}) must be before end_ts ({end_ts})")
        if start_ts % interval_ms != 0 or end_ts % interval_ms != 0:
            raise ValueError(
                f"start_ts/end_ts must align to the {bar} grid ({interval_ms}ms)"
            )

    out_dir = out_dir or Path(
        f"data/snapshots/okx-{inst_id.lower()}-{bar}-{start_ts}-{end_ts}"
    )
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists — snapshots are immutable; remove it "
            "first if you intend to rebuild it"
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".build-", dir=out_dir.parent))

    try:
        # Four independent OKX endpoints, no shared state — fetch concurrently.
        with ThreadPoolExecutor(max_workers=4) as pool:
            trade_future = pool.submit(_fetch_trade_candles, inst_id, bar, start_ts, end_ts)
            mark_future = pool.submit(_fetch_mark_candles, inst_id, bar, start_ts, end_ts)
            funding_future = pool.submit(_fetch_funding_history, inst_id, start_ts, end_ts)
            instrument_future = pool.submit(_fetch_instrument, inst_id)

            trade_rows = trade_future.result()
            mark_rows = mark_future.result()
            funding_rows = funding_future.result()
            instrument = instrument_future.result()

        trade_bars = tape.to_bars(_to_trade_records(trade_rows, start_ts, end_ts), interval_ms)
        mark_bars = tape.to_mark_bars(_to_mark_records(mark_rows, start_ts, end_ts), interval_ms)
        funding_records = tape.to_funding_records(
            _to_funding_input_records(funding_rows, start_ts, end_ts)
        )

        trade_gaps = tape.detect_gaps(trade_bars, interval_ms)
        mark_gaps = tape.detect_gaps(mark_bars, interval_ms)

        expected_bars = (end_ts - start_ts) // interval_ms
        trade_complete = len(trade_bars) == expected_bars
        mark_complete = len(mark_bars) == expected_bars
        if strict and not (trade_complete and mark_complete):
            raise RuntimeError(
                f"incomplete snapshot: trade bars {len(trade_bars)}/{expected_bars}, "
                f"mark bars {len(mark_bars)}/{expected_bars} — pass strict=False "
                "(CLI: --allow-incomplete) to persist a partial fetch anyway"
            )

        write_trade_parquet(trade_bars, staging_dir / "trade_bars_5m.parquet")
        write_mark_parquet(mark_bars, staging_dir / "mark_bars_5m.parquet")
        write_funding_parquet(funding_records, staging_dir / "funding_events.parquet")
        (staging_dir / "instrument.json").write_text(
            json.dumps(instrument, indent=2, sort_keys=True)
        )
        instrument_hash = hashlib.sha256(
            json.dumps(instrument, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        manifest = {
            "schema_version": "market-v0",
            "source": "okx",
            "inst_id": inst_id,
            "bar": bar,
            "requested_start_ts": start_ts,
            "requested_end_ts": end_ts,
            "requested_start_utc": datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat(),
            "requested_end_utc": datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "instrument_hash": instrument_hash,
            "trade": {
                "rows_fetched": len(trade_rows),
                "bars_completed": len(trade_bars),
                "expected_bars": expected_bars,
                "coverage_complete": trade_complete,
                "start_ts": trade_bars[0].open_ts if trade_bars else None,
                "end_ts": trade_bars[-1].open_ts if trade_bars else None,
                "gap_count": len(trade_gaps),
                "gaps": [g.__dict__ for g in trade_gaps],
                "dataset_hash": tape.trade_tape_hash(trade_bars),
            },
            "mark": {
                "rows_fetched": len(mark_rows),
                "bars_completed": len(mark_bars),
                "expected_bars": expected_bars,
                "coverage_complete": mark_complete,
                "start_ts": mark_bars[0].open_ts if mark_bars else None,
                "end_ts": mark_bars[-1].open_ts if mark_bars else None,
                "gap_count": len(mark_gaps),
                "gaps": [g.__dict__ for g in mark_gaps],
                "dataset_hash": tape.mark_tape_hash(mark_bars),
            },
            "funding": {
                "records_fetched": len(funding_rows),
                "records_valid": len(funding_records),
                "dataset_hash": tape.funding_tape_hash(funding_records),
            },
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

        os.rename(staging_dir, out_dir)  # atomic publish, same filesystem
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# load — re-verify and return verified tapes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class LoadedSnapshot:
    """A snapshot's contents, already hash-verified against its manifest and
    ready to hand to lab/observe.py (or any other pure consumer)."""

    trade_bars: list[tape.Bar]
    mark_bars: list[tape.MarkBar]
    funding_events: list[sim.FundingEvent]
    manifest: dict
    index_bars: list[tape.MarkBar] | None = None
    premium_bars: list[tape.MarkBar] | None = None


def _read_trade_bars(path: Path) -> list[tape.Bar]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    if "quote_volume" in cols:
        return [
            tape.Bar(
                open_ts=o, open=op, high=h, low=lo, close=c, volume=v,
                quote_volume=qv, trade_count=tc,
                taker_buy_base_volume=tbb, taker_buy_quote_volume=tbq
            )
            for o, op, h, lo, c, v, qv, tc, tbb, tbq in zip(
                cols["open_ts"], cols["open"], cols["high"], cols["low"],
                cols["close"], cols["volume"], cols["quote_volume"],
                cols["trade_count"], cols["taker_buy_base_volume"],
                cols["taker_buy_quote_volume"]
            )
        ]
    return [
        tape.Bar(open_ts=o, open=op, high=h, low=lo, close=c, volume=v)
        for o, op, h, lo, c, v in zip(
            cols["open_ts"], cols["open"], cols["high"], cols["low"],
            cols["close"], cols["volume"],
        )
    ]


def _read_mark_bars(path: Path) -> list[tape.MarkBar]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    return [
        tape.MarkBar(open_ts=o, open=op, high=h, low=lo, close=c)
        for o, op, h, lo, c in zip(
            cols["open_ts"], cols["open"], cols["high"], cols["low"], cols["close"],
        )
    ]


def _read_funding_records(path: Path) -> list[tape.FundingRecord]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    return [
        tape.FundingRecord(funding_time=t, rate=r)
        for t, r in zip(cols["funding_time"], cols["rate"])
    ]


def _funding_events(
    funding_records: Sequence[tape.FundingRecord],
    trade_bars: Sequence[tape.Bar],
    mark_bars: Sequence[tape.MarkBar],
) -> list[sim.FundingEvent]:
    """Map each raw funding record onto a trade-bar index and a settlement
    mark price (ARCHITECTURE §8.2: mark bars are used only where the
    simulation contract requires mark price, initially funding valuation).

    A funding record at ``funding_time`` maps to the last trade bar whose
    ``open_ts <= funding_time`` (bisect on the sorted open_ts grid) — the
    bar during which the funding event settles. A record that falls before
    the first bar or after the last is out of the snapshot's covered range
    and is dropped, not guessed at.
    """
    open_ts = [b.open_ts for b in trade_bars]
    events: list[sim.FundingEvent] = []
    for rec in funding_records:
        idx = bisect.bisect_right(open_ts, rec.funding_time) - 1
        if idx < 0 or idx >= len(trade_bars):
            continue
        events.append(
            sim.FundingEvent(
                bar_index=idx, rate=rec.rate, mark_price=mark_bars[idx].close
            )
        )
    for prev, cur in zip(events, events[1:]):
        if cur.bar_index <= prev.bar_index:
            raise ValueError(
                f"funding events did not map to strictly increasing bar "
                f"indices ({prev.bar_index} -> {cur.bar_index}) — funding "
                "cadence finer than the bar grid, or a duplicate record"
            )
    return events


def load(snapshot_dir: Path) -> LoadedSnapshot:
    """Load, re-verify and return one snapshot's contents. Fails closed:
    raises if any tape's re-derived hash disagrees with the manifest, if the
    manifest records incomplete coverage for trade or mark, if the on-disk
    bar count disagrees with what the requested window implies (the
    coverage_complete flag is manifest-authored, not re-derived — a stale or
    hand-edited manifest must not be trusted on its word alone), or if the
    trade and mark tapes are not index-aligned (``_funding_events`` below
    reads ``mark_bars[idx]`` for the trade bar at the same ``idx`` and would
    silently attach the wrong settlement price if the two tapes drifted)."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())

    if not manifest["trade"]["coverage_complete"]:
        raise ValueError(
            f"{snapshot_dir}: trade tape coverage_complete=false — refusing "
            "to observe outcomes on an incomplete snapshot"
        )
    if not manifest["mark"]["coverage_complete"]:
        raise ValueError(
            f"{snapshot_dir}: mark tape coverage_complete=false — refusing "
            "to observe outcomes on an incomplete snapshot"
        )

    trade_bars = _read_trade_bars(snapshot_dir / "trade_bars_5m.parquet")
    mark_bars = _read_mark_bars(snapshot_dir / "mark_bars_5m.parquet")
    funding_records = _read_funding_records(snapshot_dir / "funding_events.parquet")

    interval_ms = INTERVAL_MS[manifest["bar"]]
    expected_bars = (
        manifest["requested_end_ts"] - manifest["requested_start_ts"]
    ) // interval_ms
    if len(trade_bars) != expected_bars:
        raise ValueError(
            f"{snapshot_dir}: trade tape has {len(trade_bars)} bars but the "
            f"requested window implies {expected_bars} — manifest claims "
            "coverage_complete=true; refusing to trust that flag alone"
        )
    if len(mark_bars) != expected_bars:
        raise ValueError(
            f"{snapshot_dir}: mark tape has {len(mark_bars)} bars but the "
            f"requested window implies {expected_bars} — manifest claims "
            "coverage_complete=true; refusing to trust that flag alone"
        )
    if any(t.open_ts != m.open_ts for t, m in zip(trade_bars, mark_bars)):
        raise ValueError(
            f"{snapshot_dir}: trade and mark tapes are not index-aligned "
            "(different open_ts at the same position) — funding-event "
            "mapping reads mark_bars[idx] for trade_bars[idx] and requires "
            "identical timestamps at every index"
        )

    checks = [
        ("trade", tape.trade_tape_hash(trade_bars), manifest["trade"]["dataset_hash"]),
        ("mark", tape.mark_tape_hash(mark_bars), manifest["mark"]["dataset_hash"]),
        ("funding", tape.funding_tape_hash(funding_records),
         manifest["funding"]["dataset_hash"]),
    ]

    index_bars = None
    if "index" in manifest:
        index_bars = _read_mark_bars(snapshot_dir / "index_bars_5m.parquet")
        if len(index_bars) != expected_bars:
            raise ValueError(
                f"{snapshot_dir}: index tape has {len(index_bars)} bars but requested window implies {expected_bars}"
            )
        if any(t.open_ts != idx_b.open_ts for t, idx_b in zip(trade_bars, index_bars)):
            raise ValueError(
                f"{snapshot_dir}: trade and index tapes are not index-aligned"
            )
        checks.append(("index", tape.mark_tape_hash(index_bars), manifest["index"]["dataset_hash"]))

    premium_bars = None
    if "premium" in manifest:
        premium_bars = _read_mark_bars(snapshot_dir / "premium_bars_5m.parquet")
        if len(premium_bars) != expected_bars:
            raise ValueError(
                f"{snapshot_dir}: premium tape has {len(premium_bars)} bars but requested window implies {expected_bars}"
            )
        if any(t.open_ts != p_b.open_ts for t, p_b in zip(trade_bars, premium_bars)):
            raise ValueError(
                f"{snapshot_dir}: trade and premium tapes are not index-aligned"
            )
        checks.append(("premium", tape.mark_tape_hash(premium_bars), manifest["premium"]["dataset_hash"]))

    for tape_name, recomputed, recorded in checks:
        if recomputed != recorded:
            raise ValueError(
                f"{snapshot_dir}: {tape_name} tape hash mismatch — recomputed "
                f"{recomputed}, manifest says {recorded}. The on-disk data "
                "does not match what was recorded; refusing to use it."
            )

    funding_events = _funding_events(funding_records, trade_bars, mark_bars)

    return LoadedSnapshot(
        trade_bars=trade_bars,
        mark_bars=mark_bars,
        funding_events=funding_events,
        manifest=manifest,
        index_bars=index_bars,
        premium_bars=premium_bars,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — dispatch between build and observe subcommands
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 3 plumbing setups — hardcoded here because this is the CLI tool,
# not a pure consumer. observe() itself takes setups as a parameter with
# no defaults (Phase 4 decoupling: ROADMAP Phase 4).
_BASELINE_SETUPS: tuple[observe_module.Setup, ...] = (
    observe_module.Setup("tight", k_stop=1.0, reward_risk=1.5, max_holding_bars=12),
    observe_module.Setup("medium", k_stop=1.5, reward_risk=2.0, max_holding_bars=48),
    observe_module.Setup("wide", k_stop=2.0, reward_risk=3.0, max_holding_bars=288),
)

_STAGE_B_HORIZONS: tuple[tuple[str, float, float, int], ...] = (
    ("tight", 1.0, 1.5, 12),
    ("medium", 1.5, 2.0, 48),
    ("wide", 2.0, 3.0, 288),
)
_STAGE_B_INTERVALS: tuple[tuple[str, int], ...] = (
    ("15m", 3),
    ("1h", 12),
    ("4h", 48),
)
_STAGE_B_SETUPS: tuple[observe_module.Setup, ...] = tuple(
    observe_module.Setup(
        f"{h_label}_{i_label}", k_stop=k, reward_risk=rr, max_holding_bars=mh,
        decision_interval_factor=factor, decision_interval_label=i_label,
    )
    for h_label, k, rr, mh in _STAGE_B_HORIZONS
    for i_label, factor in _STAGE_B_INTERVALS
)

_OBSERVE_MODES = {
    "baseline": {
        "setups": _BASELINE_SETUPS,
        "out_filename": "observations.json",
        "decision_interval": "5m",
        "observation_purpose": "plumbing_sanity_baseline",
    },
    "stage_b": {
        "setups": _STAGE_B_SETUPS,
        "out_filename": "observations_stage_b.json",
        "decision_interval": "15m/1h/4h",
        "observation_purpose": "stage_b_interval_geometry",
    },
}


def _cmd_build(args: argparse.Namespace) -> None:
    manifest = build(
        inst_id=args.inst_id, bar=args.bar,
        start_ts=args.start_ts, end_ts=args.end_ts, days=args.days,
        out_dir=args.out_dir, strict=not args.allow_incomplete,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _cmd_observe(args: argparse.Namespace) -> None:
    mode = _OBSERVE_MODES[args.setups]
    loaded = load(args.snapshot_dir)
    report = observe_module.observe(
        loaded.trade_bars, loaded.funding_events, setups=mode["setups"]
    )

    output = {
        "phase": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_interval": mode["decision_interval"],
        "simulation_interval": "5m",
        "observation_purpose": mode["observation_purpose"],
        "official_hunter_geometry": False,
        "source_manifest": {
            "inst_id": loaded.manifest["inst_id"],
            "bar": loaded.manifest["bar"],
            "requested_start_ts": loaded.manifest["requested_start_ts"],
            "requested_end_ts": loaded.manifest["requested_end_ts"],
            "trade_dataset_hash": loaded.manifest["trade"]["dataset_hash"],
            "mark_dataset_hash": loaded.manifest["mark"]["dataset_hash"],
            "funding_dataset_hash": loaded.manifest["funding"]["dataset_hash"],
        },
        "n_funding_events_mapped": len(loaded.funding_events),
        "setups": report,
    }

    out_path = args.snapshot_dir / mode["out_filename"]
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))

    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")

    # --- build subcommand ---
    p_build = sub.add_parser("build", help="fetch, validate and persist a snapshot")
    p_build.add_argument("--inst-id", default="BTC-USDT-SWAP")
    p_build.add_argument("--bar", default="5m", choices=list(INTERVAL_MS))
    p_build.add_argument("--days", type=int, default=90, help="window size when --start/--end omitted")
    p_build.add_argument("--start-ts", type=int, default=None, help="explicit window start, ms epoch")
    p_build.add_argument("--end-ts", type=int, default=None, help="explicit window end, ms epoch")
    p_build.add_argument("--out-dir", type=Path, default=None)
    p_build.add_argument(
        "--allow-incomplete", action="store_true",
        help="persist even if trade/mark bar coverage is short of expected_bars",
    )

    # --- observe subcommand ---
    p_observe = sub.add_parser("observe", help="load a snapshot and run Phase 3 outcome observation")
    p_observe.add_argument("--snapshot-dir", type=Path, required=True)
    p_observe.add_argument("--setups", choices=sorted(_OBSERVE_MODES), default="baseline")

    args = p.parse_args()

    if args.command == "build":
        _cmd_build(args)
    elif args.command == "observe":
        _cmd_observe(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
