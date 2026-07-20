"""Fetch July 2026 funding rate tail from Binance REST API.

Supports --profile for consistent symbol sets with download_binance.py.
Usage: python tools/fetch_funding_tail.py [--profile test|early|full]
"""

import argparse
import json
import urllib.request
import urllib.error
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.download_binance import UNIVERSES  # noqa: E402


# From 2026-07-01T00:00:00Z to 2026-07-20T00:00:00Z
START_TS = 1782825600000
END_TS = 1784505600000


def fetch(symbols: list[str], out_file: Path) -> dict:
    """Fetch funding tail for the given symbols."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tail_data: dict = {}
    if out_file.exists():
        with open(out_file) as f:
            tail_data = json.load(f)

    for sym in symbols:
        if sym in tail_data:
            continue
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={START_TS}&endTime={END_TS}&limit=1000"
        print(f"  Fetching {sym}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                tail_data[sym] = [
                    {"funding_time": int(x["fundingTime"]), "rate": float(x["fundingRate"])}
                    for x in data
                ]
            print(f"    Success: {len(data)} funding events.")
        except urllib.error.HTTPError as e:
            print(f"    HTTP Error for {sym}: {e.code} {e.reason}")
        except Exception as e:
            print(f"    Error for {sym}: {e}")
        time.sleep(0.3)

    with open(out_file, "w") as f:
        json.dump(tail_data, f, indent=2)
    print(f"Funding rate tail saved to {out_file} ({len(tail_data)} symbols).")
    return tail_data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile", choices=sorted(UNIVERSES), default="early",
        help="Standard profile (test, early, full)",
    )
    p.add_argument(
        "--limit-symbols", type=int, default=None,
        help="Limit number of symbols to fetch",
    )
    args = p.parse_args()

    symbols = UNIVERSES[args.profile]["symbols"]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    print(f"Fetching funding rate tail for profile={args.profile!r} ({len(symbols)} symbols)")
    out_file = Path("data/funding_tail.json")
    fetch(symbols, out_file)


if __name__ == "__main__":
    main()
