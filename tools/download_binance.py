"""Unified Binance USD-M Futures data pipeline — single entry point.

Downloads monthly/daily ZIP archives from data.binance.vision S3 bucket
using a flat, highly concurrent ThreadPoolExecutor to maximize bandwidth,
fetches the July 2026 funding rate tail from the REST API, then compiles
validated Parquet tapes (trade, mark, index, premium, funding) per symbol.

Profiles: test (4 symbols, 1yr), early (10 symbols, 3yr), full (56 symbols, 5yr).

Usage:
  python tools/download_binance.py --profile full
  python tools/download_binance.py --profile full --skip-funding-tail --skip-download  # compile only
  python tools/download_binance.py --profile full --limit-symbols 5 --workers 10       # quick subset
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import tape  # noqa: E402

# Standard universes mapped to Binance symbols
# full profile: top 56 USD-M perpetuals by 24h quote volume (snapshot 2026-07-20)
FULL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BANKUSDT", "SOLUSDT", "ZECUSDT", "AKEUSDT",
    "XRPUSDT", "ACEUSDT", "HYPEUSDT", "ESPORTSUSDT", "DOGEUSDT", "PUMPUSDT",
    "TLMUSDT", "BNBUSDT", "1000PEPEUSDT", "BCHUSDT", "WLDUSDT", "BUSDT",
    "ADAUSDT", "NEARUSDT", "SUIUSDT", "KAITOUSDT", "LABUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "PROMUSDT", "AVAAIUSDT", "EVAAUSDT", "1000BONKUSDT",
    "ONDOUSDT", "ENAUSDT", "DOTUSDT", "FILUSDT", "XLMUSDT", "PAXGUSDT",
    "TRUMPUSDT", "TAOUSDT", "1000XECUSDT", "AAVEUSDT", "UNIUSDT", "HOMEUSDT",
    "DEXEUSDT", "PENGUUSDT", "VANRYUSDT", "HANAUSDT", "ALICEUSDT", "ZEREBROUSDT",
    "SYNUSDT", "JTOUSDT", "VELVETUSDT", "USUSDT", "ETCUSDT", "XAUTUSDT",
    "ALLOUSDT", "REUSDT",
]

UNIVERSES = {
    "test": {
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "days": 365,
    },
    "early": {
        "symbols": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "DOGEUSDT",
            "LINKUSDT",
            "DOTUSDT",
            "NEARUSDT",
        ],
        "days": 1095,  # 3 years
    },
    "full": {
        "symbols": FULL_SYMBOLS,
        "days": 1826,  # 5 years
    },
}

# Fixed end timestamp for reproducibility (2026-07-19T15:20:00Z)
END_TS = 1784474400000
DAY_MS = 86_400_000

# Base S3 URL
S3_BASE = "https://data.binance.vision/data/futures/um"

# Schemas matching tools/snapshot.py plus native flow proxy fields
_TRADE_SCHEMA = pa.schema([
    ("open_ts", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("quote_volume", pa.float64()),
    ("trade_count", pa.int64()),
    ("taker_buy_base_volume", pa.float64()),
    ("taker_buy_quote_volume", pa.float64()),
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


def fetch_funding_tail(symbols: list[str], out_file: Path) -> dict:
    """Fetch July 2026 funding rate tail from Binance REST API in parallel."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing tail if present, fetch only missing symbols
    tail_data: dict = {}
    if out_file.exists():
        with open(out_file) as f:
            tail_data = json.load(f)

    missing = [s for s in symbols if s not in tail_data]
    if not missing:
        print(f"  All {len(symbols)} symbols already in funding tail, skipping.")
        return tail_data

    START_TS = 1782825600000
    END_TS = 1784505600000

    def _fetch_one(sym: str) -> tuple[str, list | None, str | None]:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={START_TS}&endTime={END_TS}&limit=1000"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            records = [{"funding_time": int(x["fundingTime"]), "rate": float(x["fundingRate"])} for x in data]
            return sym, records, None
        except Exception as e:
            return sym, None, str(e)

    print(f"  Fetching funding tail for {len(missing)} symbols (parallel, 8 workers)...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in missing}
        for future in as_completed(futures):
            sym, records, err = future.result()
            if records is not None:
                tail_data[sym] = records
                print(f"    {sym}: {len(records)} funding events.")
            else:
                print(f"    {sym}: ERROR — {err}")

    with open(out_file, "w") as f:
        json.dump(tail_data, f, indent=2)
    print(f"  Funding tail saved to {out_file} ({len(tail_data)} symbols).")
    return tail_data


def get_months_and_days(start_ts: int, end_ts: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Return lists of (year, month) and (year, month, day) covering the range."""
    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)

    months = []
    curr_yr, curr_mon = start_dt.year, start_dt.month
    while (curr_yr < 2026) or (curr_yr == 2026 and curr_mon < 7):
        months.append((curr_yr, curr_mon))
        curr_mon += 1
        if curr_mon > 12:
            curr_mon = 1
            curr_yr += 1

    days = []
    if end_dt.year == 2026 and end_dt.month == 7:
        for d in range(1, end_dt.day + 1):
            days.append((2026, 7, d))

    return months, days


def download_task(url: str, cache_path: Path) -> bool:
    """Download a file from url and cache it. Return True if downloaded/loaded, False on 404."""
    if cache_path.exists():
        return True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                content = resp.read()
                temp_path.write_bytes(content)
                temp_path.rename(cache_path)
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == max_retries - 1:
                raise e
        except Exception as e:
            if attempt == max_retries - 1:
                if temp_path.exists():
                    temp_path.unlink()
                raise e
        time.sleep(1.0)  # short backoff before retry
    return False


def parse_csv_from_zip(zip_path: Path) -> list[list[str]]:
    """Extract and parse the CSV file from a cached ZIP archive."""
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            content = f.read().decode("utf-8")
            lines = content.splitlines()
            if not lines:
                return []
            reader = csv.reader(lines)
            rows = list(reader)
            if rows:
                try:
                    int(rows[0][0])
                except (ValueError, IndexError):
                    # Header row detected (cannot parse first field as int)
                    rows = rows[1:]
            return rows


def process_klines(rows: list[list[str]], start_ts: int, end_ts: int) -> list[tuple]:
    """Parse kline CSV rows and filter by time window."""
    records = []
    for r in rows:
        if len(r) < 11:
            continue
        ts = int(r[0])
        if start_ts <= ts < end_ts:
            records.append((
                ts,
                float(r[1]),   # open
                float(r[2]),   # high
                float(r[3]),   # low
                float(r[4]),   # close
                float(r[5]),   # volume
                float(r[6]),   # quote_volume
                int(r[8]),     # trade_count
                float(r[9]),   # taker_buy_base_volume
                float(r[10]),  # taker_buy_quote_volume
            ))
    return records


def process_mark_klines(rows: list[list[str]], start_ts: int, end_ts: int) -> list[tuple]:
    """Parse price kline CSV rows."""
    records = []
    for r in rows:
        if len(r) < 5:
            continue
        ts = int(r[0])
        if start_ts <= ts < end_ts:
            records.append((
                ts,
                float(r[1]),  # open
                float(r[2]),  # high
                float(r[3]),  # low
                float(r[4]),  # close
            ))
    return records


def process_funding(rows: list[list[str]], start_ts: int, end_ts: int) -> list[tuple]:
    """Parse funding rate CSV rows."""
    records = []
    for r in rows:
        if len(r) < 3:
            continue
        ts = int(r[0])
        if start_ts <= ts < end_ts:
            records.append((
                ts,
                float(r[2]),  # rate
            ))
    return records


def validate_premium_bars(records: list[tuple], interval_ms: int) -> list[tape.MarkBar]:
    """Validate premium index bars which can have negative prices since it is a spread."""
    bars: list[tape.MarkBar] = []
    prev_ts = None
    for idx, rec in enumerate(records):
        ts, o, h, l, c = rec
        if ts % interval_ms != 0:
            raise ValueError(f"premium[{idx}]: open_ts {ts} not aligned to interval {interval_ms}")
        if prev_ts is not None and ts <= prev_ts:
            raise ValueError(f"premium[{idx}]: open_ts {ts} not strictly increasing")
        if not (pa.lib.math.isfinite(o) and pa.lib.math.isfinite(h) and pa.lib.math.isfinite(l) and pa.lib.math.isfinite(c)):
            raise ValueError(f"premium[{idx}]: non-finite OHLC")
        if not (l <= min(o, c) and max(o, c) <= h):
            raise ValueError(f"premium[{idx}]: inconsistent OHLC (o={o}, h={h}, l={l}, c={c})")
        prev_ts = ts
        bars.append(tape.MarkBar(ts, o, h, l, c))
    return bars


def align_and_fill_trade_bars(records: list[tuple], master_grid: list[int]) -> list[tuple]:
    """Align trade records to master grid, forward-filling prices and setting volume/counts to 0."""
    rec_dict = {r[0]: r for r in records}
    aligned = []
    last_val = None
    
    # First, find the first available record to BFill if start is missing
    if records:
        first_val = records[0]
    else:
        first_val = (0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0, 0.0, 0.0)
        
    for ts in master_grid:
        if ts in rec_dict:
            r = rec_dict[ts]
            aligned.append(r)
            last_val = r
        else:
            fill_src = last_val if last_val is not None else first_val
            aligned.append((
                ts,
                fill_src[4],  # open = prev close
                fill_src[4],  # high = prev close
                fill_src[4],  # low = prev close
                fill_src[4],  # close = prev close
                0.0,          # volume = 0
                0.0,          # quote_volume = 0
                0,            # trade_count = 0
                0.0,          # taker_buy_base_volume = 0
                0.0,          # taker_buy_quote_volume = 0
            ))
    return aligned


def align_and_fill_price_bars(records: list[tuple], master_grid: list[int]) -> list[tuple]:
    """Align price records to master grid, forward-filling all price fields."""
    rec_dict = {r[0]: r for r in records}
    aligned = []
    last_val = None
    
    if records:
        first_val = records[0]
    else:
        first_val = (0, 1.0, 1.0, 1.0, 1.0)
        
    for ts in master_grid:
        if ts in rec_dict:
            r = rec_dict[ts]
            aligned.append(r)
            last_val = r
        else:
            fill_src = last_val if last_val is not None else first_val
            aligned.append((
                ts,
                fill_src[4],  # open = prev close
                fill_src[4],  # high = prev close
                fill_src[4],  # low = prev close
                fill_src[4],  # close = prev close
            ))
    return aligned


def compile_tapes(sym: str, start_ts: int, end_ts: int, months: list, days: list, cache_dir: Path, out_dir: Path) -> None:
    """Compile cached ZIP files into target Parquet tapes and manifest."""
    t0 = time.time()
    tapes = ["klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines", "fundingRate"]
    data_by_tape = {t: [] for t in tapes}

    for tape_name in tapes:
        # Load Monthly
        for y, m in months:
            cache_path = cache_dir / sym / tape_name / f"{y}-{m:02d}.zip"
            if cache_path.exists():
                rows = parse_csv_from_zip(cache_path)
                data_by_tape[tape_name].extend(rows)

        # Load Daily
        if tape_name != "fundingRate":
            for y, m, d in days:
                cache_path = cache_dir / sym / tape_name / f"{y}-{m:02d}-{d:02d}.zip"
                if cache_path.exists():
                    rows = parse_csv_from_zip(cache_path)
                    data_by_tape[tape_name].extend(rows)

    # Add July funding rate tail
    tail_file = Path("data/funding_tail.json")
    if tail_file.exists():
        with open(tail_file) as f:
            tail_json = json.load(f)
            if sym in tail_json:
                for ev in tail_json[sym]:
                    data_by_tape["fundingRate"].append([str(ev["funding_time"]), "8", str(ev["rate"])])

    # Parse and Filter
    trade_recs = process_klines(data_by_tape["klines"], start_ts, end_ts)
    mark_recs = process_mark_klines(data_by_tape["markPriceKlines"], start_ts, end_ts)
    index_recs = process_mark_klines(data_by_tape["indexPriceKlines"], start_ts, end_ts)
    premium_recs = process_mark_klines(data_by_tape["premiumIndexKlines"], start_ts, end_ts)
    funding_recs = process_funding(data_by_tape["fundingRate"], start_ts, end_ts)

    # Sort & Deduplicate
    trade_recs = sorted(list(set(trade_recs)), key=lambda x: x[0])
    mark_recs = sorted(list(set(mark_recs)), key=lambda x: x[0])
    index_recs = sorted(list(set(index_recs)), key=lambda x: x[0])
    premium_recs = sorted(list(set(premium_recs)), key=lambda x: x[0])
    funding_recs = sorted(list(set(funding_recs)), key=lambda x: x[0])

    # Construct math master grid of timestamps
    expected_bars = (end_ts - start_ts) // 300_000
    master_grid = [start_ts + i * 300_000 for i in range(expected_bars)]

    # Align and gap-fill all price/trade tapes to the master grid
    trade_recs = align_and_fill_trade_bars(trade_recs, master_grid)
    mark_recs = align_and_fill_price_bars(mark_recs, master_grid)
    index_recs = align_and_fill_price_bars(index_recs, master_grid)
    premium_recs = align_and_fill_price_bars(premium_recs, master_grid)

    # Core Validation via lab/tape
    trade_to_verify = [x[:6] for x in trade_recs]
    validated_trade_bars = tape.to_bars(trade_to_verify, 300_000)
    validated_mark_bars = tape.to_mark_bars(mark_recs, 300_000)
    validated_index_bars = tape.to_mark_bars(index_recs, 300_000)
    validated_premium_bars = validate_premium_bars(premium_recs, 300_000)
    validated_funding_recs = [tape.FundingRecord(funding_time=x[0], rate=x[1]) for x in funding_recs]

    # Deriving Dataset hashes
    trade_hash = tape.trade_tape_hash(validated_trade_bars)
    mark_hash = tape.mark_tape_hash(validated_mark_bars)
    index_hash = tape.mark_tape_hash(validated_index_bars)
    premium_hash = tape.mark_tape_hash(validated_premium_bars)
    funding_hash = tape.funding_tape_hash(validated_funding_recs)

    # Save to staging
    staging_dir = out_dir / f".build-{sym}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Write trade parquet
    trade_table = pa.table({
        "open_ts": [x[0] for x in trade_recs],
        "open": [x[1] for x in trade_recs],
        "high": [x[2] for x in trade_recs],
        "low": [x[3] for x in trade_recs],
        "close": [x[4] for x in trade_recs],
        "volume": [x[5] for x in trade_recs],
        "quote_volume": [x[6] for x in trade_recs],
        "trade_count": [x[7] for x in trade_recs],
        "taker_buy_base_volume": [x[8] for x in trade_recs],
        "taker_buy_quote_volume": [x[9] for x in trade_recs],
    }, schema=_TRADE_SCHEMA)
    pq.write_table(trade_table, staging_dir / "trade_bars_5m.parquet")

    # Write price/index/premium
    for name, recs, schema in [("mark", mark_recs, _MARK_SCHEMA),
                               ("index", index_recs, _MARK_SCHEMA),
                               ("premium", premium_recs, _MARK_SCHEMA)]:
        table = pa.table({
            "open_ts": [x[0] for x in recs],
            "open": [x[1] for x in recs],
            "high": [x[2] for x in recs],
            "low": [x[3] for x in recs],
            "close": [x[4] for x in recs],
        }, schema=schema)
        pq.write_table(table, staging_dir / f"{name}_bars_5m.parquet")

    # Write funding events
    funding_table = pa.table({
        "funding_time": [x[0] for x in funding_recs],
        "rate": [x[1] for x in funding_recs],
    }, schema=_FUNDING_SCHEMA)
    pq.write_table(funding_table, staging_dir / "funding_events.parquet")

    # Write manifest.json
    expected_bars = (end_ts - start_ts) // 300_000
    manifest = {
        "bar": "5m",
        "symbol": sym,
        "requested_start_ts": start_ts,
        "requested_end_ts": end_ts,
        "trade": {
            "dataset_hash": trade_hash,
            "n_records": len(trade_recs),
            "coverage_complete": len(trade_recs) == expected_bars,
        },
        "mark": {
            "dataset_hash": mark_hash,
            "n_records": len(mark_recs),
            "coverage_complete": len(mark_recs) == expected_bars,
        },
        "index": {
            "dataset_hash": index_hash,
            "n_records": len(index_recs),
            "coverage_complete": len(index_recs) == expected_bars,
        },
        "premium": {
            "dataset_hash": premium_hash,
            "n_records": len(premium_recs),
            "coverage_complete": len(premium_recs) == expected_bars,
        },
        "funding": {
            "dataset_hash": funding_hash,
            "n_records": len(funding_recs),
        },
    }
    with open(staging_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Rename
    target_dir = out_dir / f"binance-{sym.lower()}-5m-{start_ts}-{end_ts}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    staging_dir.rename(target_dir)

    dt = time.time() - t0
    print(
        f"  [SUCCESS] Compiled {sym} in {dt:.1f}s | trade_hash={trade_hash} | bars={len(trade_recs)}/{expected_bars}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile", choices=sorted(UNIVERSES), default="test",
        help="Standard profile to build",
    )
    p.add_argument(
        "--limit-symbols", type=int, default=None,
        help="Limit number of symbols to build",
    )
    p.add_argument(
        "--days", type=int, default=None,
        help="Override number of days to fetch",
    )
    p.add_argument(
        "--workers", type=int, default=100,
        help="Number of concurrent S3 downloader connections (default 100)",
    )
    p.add_argument(
        "--compile-workers", type=int, default=4,
        help="Number of symbols to compile concurrently (default 4, CPU-bound)",
    )
    p.add_argument(
        "--skip-funding-tail", action="store_true",
        help="Skip funding tail fetch (use existing data/funding_tail.json if present)",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip S3 download phase (compile from cached ZIPs only)",
    )
    args = p.parse_args()

    conf = UNIVERSES[args.profile]
    symbols = conf["symbols"]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    days = args.days if args.days is not None else conf["days"]
    start_ts = END_TS - days * DAY_MS

    print(
        f"Starting flat parallel Binance USD-M download for profile={args.profile!r} ({len(symbols)} symbols, {days} days)"
    )
    print(f"Time window: [{start_ts}, {END_TS}) ms epoch")

    # 0. Fetch funding rate tail for July 2026
    if not args.skip_funding_tail:
        print("\n--- Step 0: Funding tail fetch ---")
        tail_file = Path("data/funding_tail.json")
        fetch_funding_tail(symbols, tail_file)

    months, days_list = get_months_and_days(start_ts, END_TS)
    print(f"Plan per symbol: {len(months)} monthly archives + {len(days_list)} daily archives.")

    # 1. Generate all ZIP download tasks across all symbols, tapes and times
    download_tasks = []
    cache_dir = Path("data/cache")
    tapes = ["klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines", "fundingRate"]

    for sym in symbols:
        for tape_name in tapes:
            # Monthly Tasks
            for y, m in months:
                mon_str = f"{m:02d}"
                if tape_name == "fundingRate":
                    url = f"{S3_BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{y}-{mon_str}.zip"
                else:
                    url = f"{S3_BASE}/monthly/{tape_name}/{sym}/5m/{sym}-5m-{y}-{mon_str}.zip"
                cache_path = cache_dir / sym / tape_name / f"{y}-{mon_str}.zip"
                download_tasks.append((url, cache_path))

            # Daily Tasks (Only for prices/klines, not funding rate daily archives)
            if tape_name != "fundingRate":
                for y, m, d in days_list:
                    day_str = f"{m:02d}-{d:02d}"
                    url = f"{S3_BASE}/daily/{tape_name}/{sym}/5m/{sym}-5m-{y}-{day_str}.zip"
                    cache_path = cache_dir / sym / tape_name / f"{y}-{day_str}.zip"
                    download_tasks.append((url, cache_path))

    # Deduplicate download tasks
    download_tasks = sorted(list(set(download_tasks)), key=lambda x: x[0])
    total_files = len(download_tasks)
    print(f"Total files to fetch/verify: {total_files}")

    # 2. Run flat download in parallel using max connections
    t_start = time.time()
    if not args.skip_download:
        downloaded_count = 0
        skipped_count = 0
        failed_tasks = []

        print(f"Downloading using ThreadPoolExecutor(max_workers={args.workers})...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_task, url, cache_path): (url, cache_path)
                for url, cache_path in download_tasks
            }
            for future in as_completed(futures):
                url, cache_path = futures[future]
                try:
                    res = future.result()
                    if res:
                        downloaded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to download {url}: {e}")
                    failed_tasks.append(url)

        dt_download = time.time() - t_start
        print(
            f"\nDownload phase finished in {dt_download:.1f}s. "
            f"Verified: {downloaded_count} files | Ignored (404/Optional): {skipped_count} files."
        )
        if failed_tasks:
            print(f"  [CRITICAL] {len(failed_tasks)} downloads failed. Universe build aborted.")
            sys.exit(1)
    else:
        print("  [SKIP] --skip-download set, using cached ZIPs only.")

    # 3. Compile parquet snapshots in parallel (CPU-bound but independent per symbol)
    print(f"\nCompiling snapshots (compile_workers={args.compile_workers})...")
    out_dir = Path("data/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_compiles: list[str] = []
    with ThreadPoolExecutor(max_workers=args.compile_workers) as executor:
        futures = {
            executor.submit(compile_tapes, sym, start_ts, END_TS, months, days_list, cache_dir, out_dir): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [CRITICAL] Symbol {sym} compilation failed: {e}")
                import traceback
                traceback.print_exc()
                failed_compiles.append(sym)

    if failed_compiles:
        print(f"\nBinance universe compilation failed ({len(failed_compiles)} symbols).")
        sys.exit(1)

    dt_total = time.time() - t_start
    print(f"\nBinance universe build completed successfully in {dt_total:.1f}s.")


if __name__ == "__main__":
    main()
