"""Thin CLI: load a verified snapshot, run Phase 3 outcome observation,
persist the report next to the manifest it was computed from.

All the actual work lives in tools/load_snapshot.py (I/O + re-verification)
and lab/observe.py (pure measurement) — this file only wires the two
together and prints/writes the result.

This CLI decides on every 5m bar (ARCHITECTURE §9.1's V0 event definition)
— a plumbing/sanity baseline proving the data and simulator are honest, not
the official Stage B interval-geometry result (§8.1's "primary decision
candidate: 1h", derived via §8.3's 5m->15m/1h/4h aggregation). The output
is labeled `observation_purpose: "plumbing_sanity_baseline"` /
`official_hunter_geometry: false` so it can't be mistaken for the latter.

Run:  python3 tools/run_observation.py --snapshot-dir data/snapshots/okx-btc-usdt-swap-5m-1776698400000-1784474400000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import observe as observe_module  # noqa: E402
from tools import load_snapshot  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-dir", type=Path, required=True)
    args = p.parse_args()

    loaded = load_snapshot.load(args.snapshot_dir)
    report = observe_module.observe(loaded.trade_bars, loaded.funding_events)

    output = {
        "phase": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # This CLI runs every 5m bar as a candidate decision (ARCHITECTURE
        # §9.1's V0 event definition), not the §8.1 "primary decision
        # candidate: 1h" Hunter geometry. It's a plumbing/sanity baseline —
        # proves the data + simulator are honest (near-zero zero-cost
        # expectancy, no lookahead, no double-counted costs) — not the
        # official Stage B interval-geometry result. Do not read this as an
        # edge finding at any interval.
        "decision_interval": "5m",
        "simulation_interval": "5m",
        "observation_purpose": "plumbing_sanity_baseline",
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

    out_path = args.snapshot_dir / "observations.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))

    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
