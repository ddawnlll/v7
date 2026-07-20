"""Evidence generator (ROADMAP Phase 6).

Loads Binance pilot universe snapshots, builds candidate events with
HunterSpec V0, extracts directional features, and evaluates the declared
market hypothesis against the baseline ladder on untouched OOS data.
Produces a PASS / FAIL / HOLD verdict."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.data import load
from lab.events import build_events, Setup, EventInput, CandidateEvent
from lab.features import build_event_features, FEATURE_NAMES, FEATURE_DIM
from lab.evaluate import (
    evaluate,
    reconcile,
    verify_splits,
    make_abstain_predictor,
    make_always_take_predictor,
    make_random_predictor,
    make_linear_predictor,
    make_tree_predictor,
    make_momentum_alignment_predictor,
    shuffled_control_check,
    PredictionContext,
    EvalMetrics,
    SplitEval,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis H1
# ═══════════════════════════════════════════════════════════════════════════════

HYPOTHESIS = (
    "H1: Directional momentum alignment — when both short-term (5m) and "
    "medium-term (1h) directional returns are positive (aligned with the "
    "event side in sign-flipped feature space), the event is more likely "
    "to achieve positive net_R than random selection. Predictor: TAKE "
    "when return_5m > 0 AND return_1h > 0, ABSTAIN otherwise."
)

# ═══════════════════════════════════════════════════════════════════════════════
# HunterSpec V0 (from specs/hunter_candidate_v0.json)
# ═══════════════════════════════════════════════════════════════════════════════

HUNTER_SPEC = Setup(
    "hunter_v0",
    k_stop=2.0,
    reward_risk=3.0,
    max_holding_bars=288,
    decision_interval_factor=12,
    decision_interval_label="1h",
)

# Split boundary: 2025-10-20T00:00:00Z (specs/split_candidate_v0.json)
SPLIT_TS = 1760918400000

PILOT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT",
    "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT",
]

BINANCE_DIR = Path("/teamspace/studios/this_studio/v7/data/snapshots")


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_metrics(m: EvalMetrics, label: str) -> str:
    if m.n_events == 0:
        return f"{label}: 0 events"
    if m.n_taken == 0:
        return (
            f"{label}: {m.n_events} events, 0 taken "
            f"(coverage={m.coverage:.1%})"
        )
    return (
        f"{label}: {m.n_events} events, {m.n_taken} taken "
        f"(coverage={m.coverage:.1%}), "
        f"mean={m.mean_net_r:+.4f}R, median={m.median_net_r:+.4f}R, "
        f"std={m.std_net_r:.4f}, sharpe={m.sharpe:+.2f}, "
        f"win_rate={m.win_rate:.1%}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def run_evidence(seed: int = 42) -> dict:
    t0 = time.time()

    # ---- 1. Load snapshots ----
    inputs: list[EventInput] = []
    for sym in PILOT_SYMBOLS:
        snap_dir = BINANCE_DIR / f"binance-{sym.lower()}-5m-1689866400000-1784474400000"
        if not snap_dir.exists():
            print(f"SKIP {sym}: snapshot not found", file=sys.stderr)
            continue
        snap = load(snap_dir, allow_mark_gaps=True)
        inputs.append(EventInput(
            symbol=sym, trade_bars=snap.trade_bars,
            funding_events=snap.funding_events,
        ))
        print(f"  loaded {sym}: {len(snap.trade_bars)} bars, {len(snap.funding_events)} funding events", file=sys.stderr)

    print(f"Total: {len(inputs)} symbols loaded in {time.time() - t0:.1f}s\n", file=sys.stderr)

    # ---- 2. Build events ----
    events = build_events(inputs, split_ts=SPLIT_TS, setup=HUNTER_SPEC)
    train_n = sum(1 for e in events if e.split == "train")
    test_n = sum(1 for e in events if e.split == "test")
    print(f"Events: {len(events)} total ({train_n} train, {test_n} test)\n", file=sys.stderr)

    # ---- 3. Verify splits ----
    if not verify_splits(events, SPLIT_TS):
        raise RuntimeError("split verification failed — purge or boundary violation")

    # ---- 4. Build features ----
    bars_by_sym = {inp.symbol: inp.trade_bars for inp in inputs}
    features = build_event_features(events, bars_by_sym)
    print(f"Features: {len(features)} event feature vectors (dim={FEATURE_DIM})\n", file=sys.stderr)

    # ---- 5. Collect train contexts/targets for ML baselines ----
    train_ctxs: list[PredictionContext] = []
    train_targets: list[float] = []
    for e in events:
        if e.split != "train":
            continue
        feats = features[e.event_id]
        train_ctxs.append(PredictionContext(
            event_id=e.event_id, symbol=e.symbol,
            side=e.side, decision_ts=e.decision_ts,
            features=feats.copy(),
        ))
        train_targets.append(e.locked_outcome.net_r)

    # ---- 6. Evaluate ----
    predictors: dict[str, callable] = {
        "abstain": make_abstain_predictor(),
        "always_take": make_always_take_predictor(),
        "random": make_random_predictor(),
        "linear_ols": make_linear_predictor(train_ctxs, train_targets, seed=seed),
        "tree_cart3": make_tree_predictor(train_ctxs, train_targets, seed=seed),
        "H1_momentum_align": make_momentum_alignment_predictor(),
    }

    results: dict[str, SplitEval] = {}
    for name, pred in predictors.items():
        result = evaluate(events, features, pred, seed=seed, fit_predictor=False)
        if not reconcile(result):
            raise RuntimeError(f"reconciliation failed for {name}")
        results[name] = result

    # ---- 7. Shuffled-label control for linear ----
    shuffled = shuffled_control_check(
        events, features, make_linear_predictor, seed=seed, n_shuffles=10,
    )
    shuffled_test_means = [s.mean_net_r for s in shuffled if s.mean_net_r is not None]

    # ---- 8. Report ----
    print("=" * 72)
    print("EVIDENCE — FIRST FALSIFIABLE HYPOTHESIS (ROADMAP Phase 6)")
    print("=" * 72)
    print(f"\n{HYPOTHESIS}\n")
    print(f"Data: Binance USD-M perpetuals, 5m bars")
    print(f"Window: 2023-07-20 – 2026-07-19 (3 years)")
    print(f"Symbols: {len(inputs)} ({', '.join(PILOT_SYMBOLS[:len(inputs)])})")
    print(f"Setup: k_stop={HUNTER_SPEC.k_stop}, reward_risk={HUNTER_SPEC.reward_risk}, max_hold={HUNTER_SPEC.max_holding_bars} bars (24h)")
    print(f"Split: chronological, boundary=2025-10-20T00:00:00Z, outcome-end purged")
    print(f"Features: {', '.join(FEATURE_NAMES)} (directional)")
    print(f"Seed: {seed}")
    print()

    for name, result in results.items():
        print(f"--- {name} ---")
        print(f"  {_fmt_metrics(result.train, 'TRAIN')}")
        print(f"  {_fmt_metrics(result.test, 'TEST')}")
        print()

    print("--- shuffled-label control (linear OLS, n=10) ---")
    if shuffled_test_means:
        print(f"  test mean_net_r: {np.mean(shuffled_test_means):+.4f} ± {np.std(shuffled_test_means):.4f}")
        print(f"  range: [{min(shuffled_test_means):+.4f}, {max(shuffled_test_means):+.4f}]")
    else:
        print("  (no results)")
    print()

    # ---- 9. Verdict ----
    h1_test = results["H1_momentum_align"].test
    base_test = results["abstain"].test
    always_test = results["always_take"].test

    if h1_test.mean_net_r is None:
        verdict = "HOLD (no trades)"
    elif always_test.mean_net_r is not None and h1_test.mean_net_r > always_test.mean_net_r:
        verdict = (
            f"PASS — H1 mean_net_r={h1_test.mean_net_r:+.4f}R > "
            f"always-take baseline ({always_test.mean_net_r:+.4f}R)"
        )
    elif h1_test.mean_net_r > 0:
        verdict = f"HOLD — H1 positive ({h1_test.mean_net_r:+.4f}R) but always-take is None"
    else:
        verdict = f"FAIL — H1 mean_net_r={h1_test.mean_net_r:+.4f}R <= 0"

    print(f"VERDICT: {verdict}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    return {
        "hypothesis": HYPOTHESIS,
        "verdict": verdict,
        "baselines": {
            name: {
                "train": {
                    "n_events": r.train.n_events,
                    "n_taken": r.train.n_taken,
                    "coverage": r.train.coverage,
                    "mean_net_r": r.train.mean_net_r,
                    "median_net_r": r.train.median_net_r,
                    "sharpe": r.train.sharpe,
                    "win_rate": r.train.win_rate,
                },
                "test": {
                    "n_events": r.test.n_events,
                    "n_taken": r.test.n_taken,
                    "coverage": r.test.coverage,
                    "mean_net_r": r.test.mean_net_r,
                    "median_net_r": r.test.median_net_r,
                    "sharpe": r.test.sharpe,
                    "win_rate": r.test.win_rate,
                },
            }
            for name, r in results.items()
        },
        "shuffled_control": {
            "n_shuffles": len(shuffled),
            "test_mean_mean_net_r": float(np.mean(shuffled_test_means)) if shuffled_test_means else None,
            "test_std_mean_net_r": float(np.std(shuffled_test_means)) if shuffled_test_means else None,
        },
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true", help="JSON output only")
    args = p.parse_args()

    report = run_evidence(seed=args.seed)
    if args.json:
        # Convert numpy types
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            return obj
        print(json.dumps(_clean(report), indent=2))
    else:
        pass  # already printed in run_evidence


if __name__ == "__main__":
    main()
