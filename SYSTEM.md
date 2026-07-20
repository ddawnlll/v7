# SYSTEM.md — v7 architecture and boundaries

## What this is

A measurement instrument, not a trading system. It exists to answer one question:
can a declared hypothesis improve outcome selection on untouched data?

Everything below is either an **authority** (one file owns one truth) or a
**tool** (I/O, CLI, disk, network). No file imports upward from lab/ to tools/.
Authorities share no money-computing logic — `sim.py` is the sole economic truth.

## Layered authorities

```
Phase 0: indicators.py   — Mathematical primitives (ATR, etc.)
Phase 1: sim.py          — Deterministic truth core (net_R, labels)
Phase 2: market.py       — Bar/tape reality (validation, hashing)
Phase 3–4: events.py     — Candidate-event authority (observe + build_events)
Phase 5: evaluate.py     — Evaluation authority (baselines, ledger)
Phase 6: features.py      — Declared feature surface (next phase)
```

### Phase 0–1: indicator and simulation (LOCKED)

`indicators.py` computes ATR and other research primitives. Pure functions,
causal (only bars ≤ t), finite-or-NaN contract. Never imported by sim.py.

`sim.py` is the sole money-computing authority. Given a `TradeSpec`
(stop/target/timeout on 5m bars), it walks bar-by-bar and produces a
`TradeOutcome` with `net_r`, `mae_r`, `mfe_r`, and cost breakdown. No other
file computes money or labels. Tag: `simulation-authority` at commit `8117950`.

### Phase 2: market reality

`market.py` defines `Bar`, `MarkBar`, `FundingRecord` and provides tape
validation (`to_bars`, `to_mark_bars`), gap detection, bar aggregation for
derived intervals (15m/1h/4h), and deterministic SHA-256 tape hashing.

### Phase 3–4: candidate events (LOCKED)

`events.py` is the single candidate-event authority:
- `observe(trade_bars, funding_events, setups)` — Phase 3 descriptive statistics
- `build_events(inputs, split_ts, setup)` — Phase 4 locked event rows
- Both share `_candidate_decisions()` — geometry defined once.
- `Setup` dataclass: `k_stop`, `reward_risk`, `max_holding_bars`, `decision_interval_factor`
- `EventInput`: per-symbol input (trade bars + funding), no I/O types.
- HunterSpec V0: `wide_1h` — k_stop=2.0, reward_risk=3.0, max_holding_bars=288 (24h),
  1h decision interval on 5m simulation path. Spec: `specs/hunter_candidate_v0.json`.

Tag: `outcome-contract-authority`.

### Phase 5: evaluation (LOCKED)

`evaluate.py` is the evaluation authority:
- **PredictionContext**: predictor receives only event_id, symbol, side,
  decision_ts, and precomputed causal features. No locked_outcome, future
  bars, or split access.
- **Action contract**: TAKE/ABSTAIN. Event side is fixed by Phase 4.
  No arithmetic outcome inversion.
- **Immutable ledger**: `SplitEval.ledger` is a tuple of `EventLedger` rows.
  All metrics derived exclusively from ledger. `reconcile()` re-derives
  independently and checks identity.
- **Baselines**: abstain (floor), always-take (raw expectancy), random,
  linear OLS with intercept, regression tree (CART, max_depth=3).
- **Shuffled-label control**: re-fits predictor from scratch on shuffled
  train labels only. Frozen test split untouched.
- **Fail-closed**: invalid split → ValueError, missing features → KeyError,
  `verify_splits()` checks chronological purge.

Tag: `evaluation-authority`.

## Split discipline

- Chronological splits only. Train events: `outcome_end_ts < frozen_test_start_ts`.
  Test events: `decision_ts >= frozen_test_start_ts`.
- Split boundary: `2025-10-20T00:00:00Z` (`1760918400000` ms epoch).
  Spec: `specs/split_candidate_v0.json`.
- Train events whose outcome spans into test are purged (leakage prevention).
- Frozen test split is physically separated — fit() never sees it.
- No fixed percentage. Test size determined by predeclared evidence requirement.

## Data pipeline

Two exchange sources, one target:

```
OKX API → tools/snapshot.py → Parquet tapes + manifest
Binance S3 → tools/download_binance.py → Parquet tapes + manifest
```

`tools/build_universe.py` orchestrates parallel builds across symbols.
Data lives on the execution box at `data/snapshots/` (gitignored).

**No synthetic data.** Missing bars are gaps, not zero-volume candles
(RULES §1). Snapshots with gaps have `coverage_complete: false`.

### Pilot universe (Phase 4)

10 OKX symbols × 3 years (`early` profile), `[1689866400000, 1784474400000)`:
BTC, ETH, SOL, XRP, ADA, AVAX, DOGE, LINK, DOT, NEAR (all USDT perpetual swaps).

## Cost model

Per-side taker fee (0.04%) + slippage (1 bps). Conservative default from
`sim.TradeSpec`. Not venue-calibrated — a limit-target exit would cost less.

## Rules (see RULES.md)

1. Fail closed — missing/invalid input raises. No fallbacks, no synthetic data.
2. Code is liability — core budget ~5k lines, hard cap 10k.
3. One concern per file — no abstractions with one caller.
4. `sim.py` is sole money authority — nothing else computes net_r.
5. Evidence is a command the operator ran — never prose, never a report.
6. Every trade must be hand-verifiable against raw candles.

## Roadmap

- **Phase 0–5**: LOCKED (see tags above)
- **NOW: Phase 6** — First falsifiable hypothesis. One hypothesis, small declared
  feature surface, no architecture search, OOS verdict.
- **Phase 7**: Minimal predictive challenger (single model family).
- **Phase 8**: Additional model capacity (only if simpler model shows signal).
- **Phase 9**: Execution policy research (position management).
- **Phase 10**: Paper and micro-live validation.
- **Conditional**: Execution research, hardware acceleration — gated behind
  measured limitations.
