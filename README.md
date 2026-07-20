# v7

Minimal, fail-closed alpha research lab. Successor to `v7-engine` (frozen, read-only).

**Goal:** build a lab that can say *"edge"* or *"no edge"* — and be trusted either way.

## Layout

```
lab/
  market.py       Bar/tape reality — validation, aggregation, hashing
  indicators.py   Pure, causal research primitives (ATR, etc.)
  sim.py          Deterministic truth core — net_R, labels, outcomes
  events.py       Candidate-event authority — observe() + build_events()
  evaluate.py     Evaluation authority — baselines, negative controls, ledger
tools/
  data.py         OKX data acquisition — build, load, verify, observe, universe CLI
  download_binance.py  Binance data acquisition + compilation
  build_universe.py    Multi-symbol parallel build
  export_llm.py        LLM context snapshot builder
specs/
  hunter_candidate_v0.json   Locked HunterSpec V0 geometry (wide_1h)
  split_candidate_v0.json    Chronological train/test split boundary
```

## Status

Phase 0–5 locked. Now in **Phase 6** (first falsifiable hypothesis).

- Phase 0: `indicator-authority` tag
- Phase 1: `simulation-authority` tag
- Phase 2: verified data snapshot (BTC-USDT-SWAP, 90-day)
- Phase 3: outcome observation (237 tests)
- Phase 4: `outcome-contract-authority` tag — HunterSpec V0, 7/7 geometry gates
- Phase 5: `evaluation-authority` tag — PredictionContext safety, TAKE/ABSTAIN, immutable ledger

**285/285 tests pass** from clean checkout on remote box (CPython 3.12.3).

See [ROADMAP.md](ROADMAP.md) for evidence. Rules: [RULES.md](RULES.md).
