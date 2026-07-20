"""Build multi-symbol snapshot universes in parallel (test, early, or custom)."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import data  # noqa: E402

# Standard universes defined in ARCHITECTURE §8.1.1
UNIVERSES = {
    "test": {
        "symbols": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"],
        "days": 365,
    },
    "early": {
        "symbols": [
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
            "XRP-USDT-SWAP",
            "ADA-USDT-SWAP",
            "AVAX-USDT-SWAP",
            "DOGE-USDT-SWAP",
            "LINK-USDT-SWAP",
            "DOT-USDT-SWAP",
            "NEAR-USDT-SWAP",
        ],
        "days": 1095,  # 3 years
    },
}

# Fixed end timestamp for reproducibility (2026-07-19T15:20:00Z)
END_TS = 1784474400000
DAY_MS = 86_400_000


def build_one(sym: str, start_ts: int, end_ts: int) -> None:
    """Build a single instrument snapshot, skipping if it already exists."""
    out_dir = Path(f"data/snapshots/okx-{sym.lower()}-5m-{start_ts}-{end_ts}")
    if out_dir.exists():
        print(f"  [SKIP] Already exists at {out_dir}")
        return

    print(f"  [START] Building snapshot for {sym}...")
    t0 = time.time()
    try:
        manifest = data.build(
            inst_id=sym,
            bar="5m",
            start_ts=start_ts,
            end_ts=end_ts,
            strict=True,
        )
        dt = time.time() - t0
        print(
            f"  [SUCCESS] {sym} finished in {dt:.1f}s | dataset_hash={manifest['trade']['dataset_hash']}"
        )
    except Exception as e:
        dt = time.time() - t0
        print(f"  [ERROR] Failed building {sym} after {dt:.1f}s: {e}")
        raise e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile", choices=sorted(UNIVERSES), default="test",
        help="Standard profile from ARCHITECTURE §8.1.1",
    )
    p.add_argument(
        "--limit-symbols", type=int, default=None,
        help="Limit number of symbols to build (for quicker test runs)",
    )
    p.add_argument(
        "--days", type=int, default=None,
        help="Override number of historical days to fetch",
    )
    p.add_argument(
        "--workers", type=int, default=3,
        help="Number of symbols to build concurrently (default 3)",
    )
    args = p.parse_args()

    conf = UNIVERSES[args.profile]
    symbols = conf["symbols"]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    days = args.days if args.days is not None else conf["days"]
    start_ts = END_TS - days * DAY_MS

    print(
        f"Starting parallel universe build for profile={args.profile!r} ({len(symbols)} symbols, {days} days)"
    )
    print(f"Time window: [{start_ts}, {END_TS}) ms epoch")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_one, sym, start_ts, END_TS): sym
            for sym in symbols
        }
        failed = False
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [CRITICAL] Symbol {sym} failed: {e}")
                failed = True

    if failed:
        print("\nUniverse build failed.")
        sys.exit(1)

    dt_total = time.time() - t_start
    print(f"\nUniverse build completed successfully in {dt_total:.1f}s.")


if __name__ == "__main__":
    main()
