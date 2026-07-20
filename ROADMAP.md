# ROADMAP

This roadmap is evidence-gated.

It does not promise a model architecture, trading strategy, accelerator, market,
timeframe, or profitable result in advance. Each phase answers one question.
Only the next phase may be implemented. Later phases remain locked until their
entry condition is satisfied.

A phase ends with one of three verdicts:

* **PASS** — the claim is supported; the next phase unlocks.
* **HOLD** — evidence or correctness is incomplete; remain in the phase.
* **FAIL** — the tested assumption is rejected; preserve the validated core and
  revise only the failed assumption.

A poor research result is never permission to rewrite a validated lower layer.

---

## Current status

**NOW:** Phase 3 — outcome observation on the verified snapshot.

**NEXT:** Phase 4 — outcome contract.

**LOCKED:** Research hypotheses, models, execution policies, RL, live trading,
and hardware acceleration.

Phase 0 (indicator) and Phase 1 (simulation) authorities are recorded as the
`indicator-authority` and `simulation-authority` git tags, both at the verified
commit whose full suite passes from a clean checkout on the execution box
(commit `8117950`, 227/227 tests, remote box, CPython 3.12.3).

Phase 2 exited — one immutable dataset snapshot recorded:

* Source: OKX, `BTC-USDT-SWAP`, `5m` bar
* Window: `[1776698400000, 1784474400000)` ms epoch
  (`2026-04-20T15:20:00Z` – `2026-07-19T15:20:00Z`, 90 days)
* Build command: `python3 tools/snapshot.py build --start-ts 1776698400000 --end-ts 1784474400000`
* `trade_dataset_hash`: `d08d100a35de8f8c65d6502d8e236aa7fb88626e4b2a398a20fe3a4a0a4c123a`
* `mark_dataset_hash`: `bb4d118460ecbb8580344bc9b9883558d1e4d4fa983b03c9c3d77f7482d059a9`
* `funding_dataset_hash`: `89dbf4481b57025a727b96cfdff412aed4fec0499f1c1eb8e3f12595bea3959c`
* `instrument_hash`: `23d6876bc2533ce4f04aafa23a469285d02493f01b28a0dce277923b1a381d23`
* `coverage_complete: true` (trade + mark, 25,920/25,920 bars each), `gap_count: 0` (both)
* Reproduced 3 times across independent live fetches on the execution box —
  identical hashes each time
* Snapshot files live at `data/snapshots/` on the execution box (gitignored,
  not committed — see RULES §15); the manifest above is the recorded,
  rerunnable evidence per this phase's exit requirement

---

## Phase 0 — Indicator authority

### Question

Do the research primitives implement their declared mathematical and causal
contracts?

### Required evidence

* Hand-computed golden values
* Future-append causality
* Dirty-data and recovery behavior
* Finite-or-NaN output contract
* Numerical overflow and extreme-input tests
* Structural input failures raise deterministically
* Full test suite passes from a clean checkout

### Exit

A verified commit SHA is recorded as the indicator authority.

After exit, this phase is reopened only for a reproducible correctness defect,
not because a later strategy performs poorly.

---

## Phase 1 — Simulation authority

### Question

Can one trade be converted into an economically correct, deterministic and
hand-verifiable outcome?

### Scope

* Entry-fill convention
* Stop, target and timeout semantics
* Open-gap behavior
* Same-bar ambiguity
* Directional slippage
* Entry and exit fees
* Funding events
* MAE and MFE semantics
* Gross return, execution return, net return and net_R
* Event-end causality
* Canonical outcome serialization

### Required evidence

* Hand-computed golden trades
* LONG/SHORT mirror tests
* Stop/target/gap/timeout cases
* Cost decomposition identities
* Post-exit bar mutation invariance
* Post-exit funding-value mutation invariance
* Malformed consumed inputs fail closed
* All economic outputs remain finite
* Frozen outcome hashes match on local and remote machines
* Full suite passes from the same clean commit

### Exit

A verified commit SHA is recorded as the simulation authority.

Nothing else in the repository may calculate money, labels or trade outcomes.

---

## Phase 2 — Verified data snapshot

### Question

Can one bounded market-data snapshot be reproduced and trusted as real input?

### Scope

The smallest snapshot sufficient to exercise the truth core.

The market, instrument, timeframe and date range are selected at the start of
this phase and recorded as data choices, not assumed permanently by the roadmap.

### Required evidence

* Source identity and retrieval parameters
* Completed-candle filtering
* Timestamp ordering
* Duplicate detection
* Gap detection and explicit gap representation
* OHLC consistency
* Instrument metadata and units
* Dataset hash
* Byte-identical or semantically identical rebuild according to the declared
  serialization contract

### Exit

One immutable dataset snapshot and its build command are recorded.

No multi-market expansion occurs in this phase.

---

## Phase 3 — Outcome observation

### Question

What outcome structures actually exist in the verified data before prediction
is attempted?

### Work

Use the locked simulation authority to measure observable outcome properties,
such as:

* Favorable and adverse excursion
* Time to outcome
* Target-before-stop base rates
* Cost sensitivity
* Ambiguous-bar frequency
* Coverage under candidate outcome definitions

No model is trained and no profitable geometry is presumed.

### Exit

A bounded set of empirical observations is produced from development data.

If no economically meaningful outcome definition is found, the phase returns
FAIL. The indicator, simulation and data authorities remain unchanged.

---

## Phase 4 — Outcome contract

### Question

Can one outcome definition be frozen without using validation or holdout
performance to choose it?

### Required artifact

One versioned outcome specification defining:

* Decision timestamp
* Entry convention
* Risk definition
* Exit rules
* Maximum horizon
* Outcome end timestamp
* Cost assumptions
* Invalid and ambiguous cases

### Exit

Golden labels generated by the outcome engine match hand-verified trades.

The outcome definition is frozen before predictive model selection begins.

---

## Phase 5 — Evaluation authority

### Question

Can the research pipeline distinguish signal from noise without leaking future
information?

### Required evidence

* Event-end purged splits
* Untouched frozen holdout
* Shuffled-label negative control
* Always-abstain baseline
* Random predictor
* Simple linear baseline
* Tree baseline
* Deterministic reproduction from dataset hash, config hash and seed
* Per-trade outcomes reconcile exactly with aggregate metrics

### Exit

Noise produces no edge and all baseline results are reproducible.

If shuffled labels produce edge, every result from the evaluator is void until
the defect is found.

---

## Phase 6 — First falsifiable hypothesis

### Question

Does one explicitly stated market hypothesis improve outcome selection outside
the data used to formulate it?

### Rules

* One hypothesis at a time
* A small, declared feature surface
* No architecture search
* No hidden outcome changes
* No cost changes after results are viewed
* Abstention and coverage are reported
* Both positive and negative verdicts are retained

### Exit

The hypothesis receives an OOS verdict:

* PASS — measurable improvement over the baseline ladder
* FAIL — no reproducible improvement
* HOLD — evidence is insufficient

A failed hypothesis does not reopen lower authorities.

---

## Phase 7 — Minimal predictive challenger

### Entry condition

Phase 6 shows that the tested outcome is at least partially distinguishable
from the available past information.

### Question

Can the smallest reasonable predictive model improve selection over simple
rules and baselines?

### Rules

Model family is selected only at the start of this phase.

The roadmap does not preselect neural networks, boosted trees, sequence models
or any other architecture.

### Required evidence

* Same data, splits, outcomes and costs as the baselines
* Calibration
* Performance by confidence and coverage
* Untouched OOS evaluation
* Stability under cost stress
* Negative controls remain clean

### Exit

The model either beats the baseline ladder under the fixed contract or is
rejected.

---

## Phase 8 — Additional model capacity

### Entry condition

A simpler model has demonstrated reproducible but capacity-limited signal.

### Question

Does additional temporal or representational capacity improve OOS economic
results under the unchanged contract?

The architecture is selected from evidence available at this point. No model
family is promised in advance.

### Exit

The challenger improves the locked evaluation criteria without increasing
leakage, fragility or unexplained complexity.

Otherwise it is rejected.

---

## Phase 9 — Execution policy research

### Entry condition

An entry decision process has survived multiple untouched OOS periods and
cost stress using a fixed, deterministic exit baseline.

### Question

Can position management improve realized net_R without weakening immutable
risk limits?

### Ladder

1. Fixed stop, target and timeout
2. Simple deterministic exit challengers
3. Supervised HOLD/CLOSE challenger
4. Sequential decision method only if simpler challengers leave measurable
   unresolved value

No reinforcement-learning implementation is presumed by the roadmap.

### Immutable limits

An execution policy may not:

* Widen the protective stop
* Increase initial account risk
* Add to a losing position
* Change leverage after entry
* Bypass portfolio or daily loss limits

### Exit

The challenger improves net expectancy or downside behavior on untouched data
without merely converting large winners into small wins.

---

## Phase 10 — Paper and micro-live validation

### Entry condition

A complete decision and execution path survives untouched OOS periods and
declared cost stress.

### Question

Does observed execution agree with the simulation contract closely enough for
the result to remain economically valid?

### Progression

* Shadow decisions
* Paper execution
* Minimum-size isolated live execution

### Required evidence

* Decision reproducibility
* Fill and slippage reconciliation
* Fee and funding reconciliation
* Backtest-to-paper divergence
* Paper-to-live divergence
* Operational failures and missed decisions
* Predeclared stop conditions

### Exit

The system receives a live verdict. A profitable backtest alone cannot satisfy
this phase.

---

## Conditional performance work

Hardware acceleration is not a roadmap phase.

After the scalar authority is locked, representative workloads are benchmarked.

* If runtime is acceptable, no fast path is created.
* If runtime blocks research, a compiled CPU challenger is considered.
* Parallel CPU execution is considered only after measurement.
* A GPU event scanner is considered only if measured workloads justify its
  additional audit surface.

Any accelerated path must reproduce the scalar event authority under parity
tests. It never becomes the sole economic truth implementation.

---

## Evaluation split discipline

### 75/25 is not a scientific law or financial standard

The commonly used 75/25 (or 80/20, 70/15/15) split ratios are practical
starting templates, not universally mandated rules.

**General ML resources** (Google for Developers) show 80/20 as an example and
70/15/15 as a better practice, but explicitly state these are not mandatory
percentages — the test set must simply be large enough, statistically meaningful,
and representative of real usage data.

**Time series resources** (scikit-learn `TimeSeriesSplit`) recommend no specific
percentage. The core rules are:

- Split chronologically
- Never train on future data to predict the past
- Train set grows from the past while a later time segment is tested

**Finance literature** notes that financial data is not independent — adjacent
rows may have overlapping feature and label windows. Standard K-fold or random
splits can be misleading. Purged cross-validation (removing overlapping labels
from train) and embargo (reducing near-term dependency) are recommended over
any fixed percentage.

### Correct split architecture

```
ALL TIME SERIES
│
├── DEVELOPMENT
│   ├── purged walk-forward fold 1
│   ├── purged walk-forward fold 2
│   ├── purged walk-forward fold 3
│   └── feature, label, threshold and model selection
│
└── FROZEN TEST
    └── only final OOS evaluation
```

### Correct contract

Replace a fixed percentage with:

```yaml
split:
  ordering: chronological
  development: all observations before frozen_test_start
  frozen_test: observations from frozen_test_start onward
  purge_by: outcome_end_ts
  test_size_rule: predeclared_before_modeling
```

### What frozen test size must satisfy

For Hunter, the test partition must satisfy:

1. **Sufficient independent trade events.** Tens of thousands of candles alone
   is insufficient; if Hunter selects only 15 trades the test is weak.
2. **Sufficiently long calendar span.** 100 trades from the same three-day
   volatility burst are not 100 independent pieces of evidence.
3. **Never used during model development.**
4. **Should ideally span multiple market regimes.** However, looking at the test
   period and adjusting its boundary constitutes test contamination — the date
   boundary must be pre-recorded.
5. **Outcome overlap is purged.** Any train row whose label uses prices after
   the test start must be removed from the training set.

### First-pass guideline

For the initial dataset pass:

1. Verify the total available date range
2. Set aside the latest portion as a frozen holdout
3. Record the split timestamp in the manifest
4. Use the remaining development period for geometry analysis
5. Use purged walk-forward validation inside development
6. Do not touch the frozen partition during feature, label, or model selection

A first-pass choice of roughly the final 20–25% is a reasonable engineering
default, described as:

> **Initial holdout preference — not a scientifically mandatory ratio.**

For example, with four years of data: first 3 years → development, last 1 year
→ frozen test (~75/25). But with only eight months of data, the last two months
may not provide sufficient evidence. With ten years of data, reserving the last
2.5 years may waste too much training data.

### What the roadmap must say

The roadmap must not lock a fixed percentage before Hunter's geometry (base
rate, annual candidate count, average holding horizon) is known. The correct
statement is:

> Frozen test size: TBD by a predeclared evidence requirement,
> before feature, label, threshold or model selection.

---

## Roadmap discipline

Only the **NOW** phase contains implementation detail.

The **NEXT** phase may contain its entry and exit conditions.

All later phases describe questions and gates only. Model architecture,
instrument universe, timeframe, risk geometry, execution method and accelerator
remain undecided until evidence makes the decision necessary.
