"""Market snapshot builder — network + disk orchestration (ARCHITECTURE §6.3).

This lives OUTSIDE the deterministic core: it is the ONLY place that touches
the network or the wall clock. It fetches raw OKX candles/funding for an
explicit, bounded ``[start_ts, end_ts)`` window, drops incomplete bars, hands
clean records to lab/data.py (pure) for validation, gap detection and hashing,
then persists three separate immutable tapes plus instrument metadata and a
manifest (ARCHITECTURE §8.2):

  trade_bars_5m.parquet   mark_bars_5m.parquet   funding_events.parquet
  instrument.json         manifest.json

Reproducibility (ROADMAP Phase 2 exit): the window is explicit and recorded,
never "however many pages as of now" — refetching with the SAME start_ts/end_ts
must reproduce the same tapes and the same hashes (barring an upstream
retroactive correction, which is a cross-audit finding, not a bug here). The
parquet files are a storage container only; every dataset hash in the manifest
is computed by lab/data.py over the CANONICAL TEXT form of the validated
records, never over parquet bytes (not writer-stable — RULES §8).

The four fetches (trade, mark, funding, instrument) hit independent OKX
endpoints and share no state, so `build()` runs them concurrently. Pagination
*within* one fetch stays sequential — OKX's `after` cursor is stateful, each
page's request depends on the previous page's oldest timestamp; a cursor that
fails to advance, or a window too large to page through in max_pages, raises
rather than silently returning duplicated or partial rows.

Fail-closed persistence (RULES §1): every file is written into a hidden
staging directory first and published with a single atomic `os.rename` — a
crash mid-build leaves either nothing or a complete snapshot, never a mix. An
existing `out_dir` is refused outright (snapshots are immutable), and a
snapshot whose trade or mark tape is short of `expected_bars` is refused too
unless the caller explicitly passes `strict=False` (CLI: --allow-incomplete).

Run:  python3 tools/build_snapshot.py
      python3 tools/build_snapshot.py --inst-id ETH-USDT-SWAP --days 30
"""

from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import data  # noqa: E402  (path set above)

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
    """Page backward via OKX's `after` cursor until the oldest fetched row's
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
        time.sleep(0.15)  # be polite to the public endpoint
    raise RuntimeError(
        f"{path}: exhausted max_pages={max_pages} before reaching "
        f"stop_ts={stop_ts} (last oldest={previous_oldest}) — window is "
        "larger than this call expected to page through"
    )


# --- fetch (non-deterministic: network + time) --------------------------------

def fetch_trade_candles(inst_id: str, bar: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]."""
    return _paginate_bounded(
        "/api/v5/market/history-candles", {"instId": inst_id, "bar": bar},
        str(end_ts), lambda row: int(row[0]), start_ts,
    )


def fetch_mark_candles(inst_id: str, bar: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: [ts,o,h,l,c,confirm] — mark price has no traded volume."""
    return _paginate_bounded(
        "/api/v5/market/history-mark-price-candles", {"instId": inst_id, "bar": bar},
        str(end_ts), lambda row: int(row[0]), start_ts,
    )


def fetch_funding_history(inst_id: str, start_ts: int, end_ts: int) -> list:
    """Raw rows: dicts with fundingTime, fundingRate (+ vendor extras)."""
    return _paginate_bounded(
        "/api/v5/public/funding-rate-history", {"instId": inst_id},
        str(end_ts), lambda row: int(row["fundingTime"]), start_ts,
    )


def fetch_instrument(inst_id: str) -> dict:
    """Single call: contract/tick/precision metadata for one instrument."""
    rows = _get_retry("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst_id})
    if not rows:
        raise RuntimeError(f"no instrument metadata for {inst_id}")
    return rows[0]


# --- raw rows -> validated records (bounded + shape-only, still non-deterministic input) -

def to_trade_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    """Keep only completed candles (confirm=="1") inside [start_ts, end_ts).
    Volume is base ccy. Ascending order."""
    recs = [
        (int(ts), float(o), float(h), float(l), float(c), float(volccy))
        for ts, o, h, l, c, _vol, volccy, _volq, confirm in rows
        if confirm == "1" and start_ts <= int(ts) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


def to_mark_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    recs = [
        (int(ts), float(o), float(h), float(l), float(c))
        for ts, o, h, l, c, confirm in rows
        if confirm == "1" and start_ts <= int(ts) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


def to_funding_input_records(rows: list, start_ts: int, end_ts: int) -> list[tuple]:
    recs = [
        (int(r["fundingTime"]), float(r["fundingRate"]))
        for r in rows
        if start_ts <= int(r["fundingTime"]) < end_ts
    ]
    recs.sort(key=lambda r: r[0])
    return recs


# --- persist (deterministic given validated records) ---------------------------

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
        # .result() is called in the same order every time so a failure is
        # reported deterministically (fail-closed, RULES §1), not by whichever
        # thread happens to finish first.
        with ThreadPoolExecutor(max_workers=4) as pool:
            trade_future = pool.submit(fetch_trade_candles, inst_id, bar, start_ts, end_ts)
            mark_future = pool.submit(fetch_mark_candles, inst_id, bar, start_ts, end_ts)
            funding_future = pool.submit(fetch_funding_history, inst_id, start_ts, end_ts)
            instrument_future = pool.submit(fetch_instrument, inst_id)

            trade_rows = trade_future.result()
            mark_rows = mark_future.result()
            funding_rows = funding_future.result()
            instrument = instrument_future.result()

        trade_bars = data.to_bars(to_trade_records(trade_rows, start_ts, end_ts), interval_ms)
        mark_bars = data.to_mark_bars(to_mark_records(mark_rows, start_ts, end_ts), interval_ms)
        funding_records = data.to_funding_records(
            to_funding_input_records(funding_rows, start_ts, end_ts)
        )

        trade_gaps = data.detect_gaps(trade_bars, interval_ms)
        # detect_gaps only reads .open_ts, so MarkBar (structurally compatible)
        # works here without a Bar-shaped adapter.
        mark_gaps = data.detect_gaps(mark_bars, interval_ms)

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
                "dataset_hash": data.dataset_hash(trade_bars),
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
                "dataset_hash": data.mark_dataset_hash(mark_bars),
            },
            "funding": {
                "records_fetched": len(funding_rows),
                "records_valid": len(funding_records),
                "dataset_hash": data.funding_dataset_hash(funding_records),
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inst-id", default="BTC-USDT-SWAP")
    p.add_argument("--bar", default="5m", choices=list(INTERVAL_MS))
    p.add_argument("--days", type=int, default=90, help="window size when --start/--end omitted")
    p.add_argument("--start-ts", type=int, default=None, help="explicit window start, ms epoch")
    p.add_argument("--end-ts", type=int, default=None, help="explicit window end, ms epoch")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--allow-incomplete", action="store_true",
        help="persist even if trade/mark bar coverage is short of expected_bars",
    )
    args = p.parse_args()

    manifest = build(
        inst_id=args.inst_id, bar=args.bar,
        start_ts=args.start_ts, end_ts=args.end_ts, days=args.days,
        out_dir=args.out_dir, strict=not args.allow_incomplete,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
