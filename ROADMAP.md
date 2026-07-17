# ROADMAP

Each phase has one exit criterion. A phase is done when its exit command passes on the remote box — not before.

## Phase 0 — Bootstrap (current)
Data integrity gate + salvage audit queue (`data_lake`, cost params first).
**Exit:** one verified dataset snapshot (monotonic, gap-flagged, deduped, hashed) built fail-closed from raw parquet.

## Phase 1 — Outcome engine
Minimal `outcome_engine.py` (~300–500 lines): barrier walk, fees/slippage/funding, `net_R`, MAE/MFE, exit reason.
**Exit:** parity vs frozen v7-engine simulation on N random trades within tolerance, plus 20 trades hand-verified against candles.

## Phase 2 — Evaluation harness + baselines
Event-end purged splits, frozen holdout, negative controls, rank IC / calibration / expectancy metrics.
Baseline ladder: random, always-NO_TRADE, linear, XGBoost.
**Exit:** shuffled-label run reports no edge; baselines reproduce from `dataset_hash + config_hash + seed`.

## Phase 3 — First hypotheses
Simple economic hypotheses (cross-sectional momentum/reversal, funding/carry, flow) as rules first, models second.
**Exit:** each hypothesis has an OOS verdict with costs — including "no edge", which counts as a result.

## Phase 4 — Neural sequence model
Small temporal encoder, multi-task outcome heads. Only if Phase 2/3 infrastructure is trusted.
**Exit:** beats the baseline ladder OOS on the same data, splits, and costs — or is rejected.

## Phase 5 — Policy / execution (locked)
Position sizing, portfolio replay, offline RL, live execution.
**Unlocks only** after a Phase 3/4 edge survives multiple untouched OOS periods and 2× cost stress.
