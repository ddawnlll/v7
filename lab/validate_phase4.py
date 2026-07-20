"""Phase 4 geometry gate validation.

Runs the wide_1h HunterSpec (read from specs/hunter_candidate_v0.json) against
all 10 early-profile OKX snapshots on the remote box and checks all 7 gates
from ROADMAP Phase 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import events, sim  # noqa: E402
from lab.events import Setup  # noqa: E402
from tools.data import load  # noqa: E402


def _load_hunter_spec() -> Setup:
    """Load the locked HunterSpec Candidate V0 from its versioned JSON spec."""
    spec_path = Path("specs/hunter_candidate_v0.json")
    spec = json.loads(spec_path.read_text())
    assert spec["spec_version"] == "hunter-candidate-v0", \
        f"unexpected spec version: {spec['spec_version']}"
    assert spec["simulation_interval"] == "5m", \
        f"unexpected simulation interval: {spec['simulation_interval']}"

    interval_map = {"1h": 12, "15m": 3, "4h": 48}
    factor = interval_map[spec["decision_interval"]]

    return Setup(
        label="wide_1h",
        k_stop=spec["k_stop"],
        reward_risk=spec["target_R"],
        max_holding_bars=spec["max_holding_minutes"] // 5,  # 5m bars
        decision_interval_factor=factor,
        decision_interval_label=spec["decision_interval"],
    )

# Symbols in the early profile (ARCHITECTURE §8.1.1 / build_universe.py)
EARLY_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
    "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "DOGE-USDT-SWAP", "LINK-USDT-SWAP",
    "DOT-USDT-SWAP", "NEAR-USDT-SWAP",
]

# Pilot universe window (ROADMAP Phase 4)
START_TS = 1689866400000  # 2023-07-20T15:20:00Z
END_TS = 1784474400000    # 2026-07-19T15:20:00Z

SNAPSHOT_BASE = Path("data/snapshots")

# Gates from ROADMAP Phase 4
GATES = {
    "aggregate_timeout": 0.50,
    "hard_timeout_per_segment": 0.70,
    "median_cost": 0.10,
    "p90_cost": 0.25,
    "ambiguous_bar_rate": 0.005,
    "valid_coverage": 0.995,
    "symbol_consistency": 8,  # >= 8/10 symbols must pass all gates
}


def cost_r(report: dict) -> float:
    """Round-trip cost as fraction of risk: fee_R + slippage_R."""
    return report["fee_r"]["median"] + report["slippage_r"]["median"]


def cost_r_p90(report: dict) -> float:
    return report["fee_r"]["p90"] + report["slippage_r"]["p90"]


def main() -> None:
    hunter = _load_hunter_spec()
    results: dict[str, dict] = {}

    for sym in EARLY_SYMBOLS:
        dirname = f"okx-{sym.lower()}-5m-{START_TS}-{END_TS}"
        snap_dir = SNAPSHOT_BASE / dirname
        if not snap_dir.exists():
            print(f"[SKIP] {sym}: snapshot not found at {snap_dir}")
            results[sym] = {"status": "missing"}
            continue

        loaded = load(snap_dir)
        report = events.observe(
            loaded.trade_bars, loaded.funding_events, setups=(hunter,)
        )
        r = report["wide_1h"]

        coverage = r["coverage"]
        timeout_rate = r["exit_reason_rate"]["time"]
        ambiguous_rate = r["ambiguous_bar_rate"]
        median_cost = cost_r(r)
        p90_cost = cost_r_p90(r)

        # Gate checks for this symbol
        gate_results = {
            "coverage_ok": coverage is not None and coverage >= GATES["valid_coverage"],
            "timeout_ok": timeout_rate is not None and timeout_rate < GATES["hard_timeout_per_segment"],
            "ambiguous_ok": ambiguous_rate is not None and ambiguous_rate < GATES["ambiguous_bar_rate"],
            "median_cost_ok": median_cost is not None and median_cost < GATES["median_cost"],
            "p90_cost_ok": p90_cost is not None and p90_cost < GATES["p90_cost"],
        }
        all_passed = all(gate_results.values())

        results[sym] = {
            "n_candidates": r["n_candidates"],
            "n_simulated": r["n_simulated"],
            "coverage": coverage,
            "timeout_rate": timeout_rate,
            "ambiguous_bar_rate": ambiguous_rate,
            "median_cost_R": median_cost,
            "p90_cost_R": p90_cost,
            "exit_reason_rate": r["exit_reason_rate"],
            "net_r_median": r["net_r_all_taker_conservative"]["median"],
            "gates": gate_results,
            "all_passed": all_passed,
        }

    # Aggregate
    n_simulated_total = sum(
        r["n_simulated"] for r in results.values()
        if isinstance(r.get("n_simulated"), int)
    )
    timeouts_total = sum(
        r["n_simulated"] * (r["timeout_rate"] or 0)
        for r in results.values()
        if isinstance(r.get("n_simulated"), int)
    )
    agg_timeout = timeouts_total / n_simulated_total if n_simulated_total else None

    symbols_passing = sum(
        1 for r in results.values() if r.get("all_passed", False)
    )
    n_symbols_with_data = sum(
        1 for r in results.values() if r.get("status") != "missing"
    )

    output = {
        "phase": 4,
        "setup": "wide_1h",
        "hunter_spec": "specs/hunter_candidate_v0.json",
        "symbols": list(results.keys()),
        "gates_required": GATES,
        "per_symbol": results,
        "aggregate": {
            "n_simulated_total": n_simulated_total,
            "aggregate_timeout_rate": agg_timeout,
            "aggregate_timeout_ok": agg_timeout is not None and agg_timeout < GATES["aggregate_timeout"],
            "symbols_passing_all_gates": symbols_passing,
            "symbols_with_data": n_symbols_with_data,
            "symbol_consistency_ok": symbols_passing >= GATES["symbol_consistency"],
        },
        "all_gates_passed": (
            agg_timeout is not None and agg_timeout < GATES["aggregate_timeout"]
            and symbols_passing >= GATES["symbol_consistency"]
        ),
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    # Write to disk
    out_path = Path("data/snapshots") / "phase4_geometry_gates.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}", file=sys.stderr)

    if not output["all_gates_passed"]:
        print("\n*** GATE CHECK FAILED ***", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n*** ALL GATES PASSED ***", file=sys.stderr)


if __name__ == "__main__":
    main()
