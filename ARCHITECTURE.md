ARCHITECTURE

1. Purpose

This repository is a minimal, fail-closed research system for answering one question honestly:

Can past market information identify rare, asymmetric trade opportunities that remain economically positive outside the data used to design them?

The system is intentionally small. It is not a general trading platform, a feature factory, an AutoML framework, or an autonomous research organization.

The architecture separates:





market truth,



economic truth,



causal observations,



predictive models,



deterministic decisions,



risk constraints.

A later layer may depend on an earlier authority. An earlier authority must never depend on a later research result.

2. Core principle

The repository grows only when a measured limitation requires a new component.

one bounded question
        ↓
smallest valid experiment
        ↓
reproducible evidence
        ↓
PASS / HOLD / FAIL
        ↓
only then unlock one additional component


Complexity is not evidence. More files, features, models, metrics, agents, and tests do not imply a better trading system.

3. Current research target

The first target is a Hunter entry system:

Select a small number of decision points with unusually favorable asymmetric outcomes, while abstaining from ordinary or uncertain opportunities.

The initial action space is:

TRADE
NO_TRADE


Direction may later be represented as separate LONG and SHORT candidate rows. Position management is initially fixed by the locked simulation contract.

The first system does not learn execution actions such as HOLD, CLOSE, PARTIAL_CLOSE, or MOVE_STOP. Those are later challengers and remain locked until entry edge exists under a fixed exit policy.

4. Non-goals

The initial architecture does not include:





reinforcement learning,



neural networks,



automatic feature mining,



automatic strategy generation,



multi-agent orchestration,



GPU or CUDA execution,



order-book reconstruction,



tick-level simulation,



portfolio optimization,



adaptive leverage,



live exchange execution,



dozens of independent data pipelines,



hundreds of active features,



large hyperparameter searches.

These are not permanently forbidden. They are unavailable until a simpler system demonstrates the specific limitation they are intended to solve.

5. System overview

┌──────────────────────────────────────────────────────────────┐
│                       RAW MARKET TRUTH                       │
│  5m last-price bars │ mark-price bars │ funding │ metadata  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     DATA INTEGRITY GATE                      │
│ ordering │ duplicates │ gaps │ OHLC │ completed bars │ hash │
└──────────────────────────────┬───────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│    CAUSAL OBSERVATIONS   │   │       ECONOMIC TRUTH        │
│  derived bars + features │   │ locked stop/target/timeout  │
│  known at decision time  │   │ fees/slippage/funding/net_R │
└──────────────┬───────────┘   └──────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       EVENT DATASET                          │
│ event_id │ times │ features │ raw outcomes │ split metadata │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      MODEL CHALLENGERS                       │
│ classifier │ regressor │ later: ranker / survival / sequence│
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  DETERMINISTIC DECISION POLICY               │
│ model outputs → threshold and veto rules → TRADE / NO_TRADE │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    IMMUTABLE RISK KERNEL                     │
│ fixed initial risk │ no stop widening │ exposure boundaries │
└──────────────────────────────────────────────────────────────┘


6. Authority boundaries

6.1 Indicator authority

lab/indicators.py is the only authority for causal mathematical primitives.

It contains pure functions only:





no network access,



no file access,



no global state,



no model code,



no trade settlement,



no hidden data cleaning.

Every feature primitive must declare and test:





required history,



mathematical definition,



causality,



invalid-input behavior,



gap behavior,



NaN behavior,



numerical bounds where applicable.

Appending future bars must never alter an already computed value.

6.2 Simulation authority

lab/sim.py is the only authority for trade economics.

It owns:





entry convention,



stop behavior,



target behavior,



timeout behavior,



gap fills,



same-bar precedence,



maker/taker fees,



directional slippage,



funding events,



MAE and MFE,



gross return,



execution return,



net return,



net_R,



canonical outcome hashing.

No dataset, model, evaluator, report, or runtime may independently recalculate trade PnL.

Execution valuation authority

`lab/sim.py` is also the sole authority for execution-state economic valuation.

It owns:

liquidation value at an explicitly defined executable future fill,
fees and directional slippage applied to an early close,
fixed-exit counterfactual replay,
same-entry execution-policy outcome comparison,
terminal outcome settlement after an execution action.

The V0 early-close contract is:

1. a completed 5m bar produces an execution state,
2. the execution policy chooses `HOLD` or `CLOSE`,
3. `CLOSE` is filled at the next valid 5m bar open,
4. taker fees and directional slippage are applied,
5. `HOLD` advances to the next simulation state unless the locked stop,
   target, or timeout has already terminated the trade.

An execution model, policy, dataset, evaluator, or reporting layer must never
independently calculate liquidation value, counterfactual PnL, or incremental
net_R.

6.3 Tape authority

lab/tape.py owns validation and deterministic bar aggregation.

It does not download data and does not generate features or labels.

It owns:





raw schemas,



timestamp normalization,



completed-bar checks,



OHLC validation,



duplicate detection,



gap detection,



deterministic 5m → 15m / 1h / 4h aggregation,



snapshot hashing.

Network and disk orchestration live outside the core.

6.4 Events authority

lab/events.py owns event construction and time alignment.

It defines:





what one candidate row represents,



which data is available at decision time,



planned entry time,



simulation start point,



outcome end time,



invalidation caused by gaps or insufficient history,



chronological development/test assignment,



purge rules.

It joins features and outcomes by event_id. It does not calculate indicators or economic outcomes itself.

Execution trajectory dataset

`lab/events.py` may construct execution-state trajectories only after a
Hunter entry has been generated by a frozen entry policy.

Each execution-state row contains:

`event_id`,
`trade_id`,
`state_ts`,
`entry_policy_version`,
`execution_policy_version`,
`risk_spec_version`,
information available at `state_ts`,
elapsed and remaining horizon,
current position-state variables,
terminal and validity metadata.

Execution trajectories follow the same outcome-end purge, gap invalidation,
and decision-time causality rules as entry events.

Fold assignment is per `event_id`: every execution-state row of one trade
belongs to the fold of that trade's `outcome_end_ts`. No trade's rows may
straddle a train/validation boundary — splitting chronologically by
`state_ts` would leak the same trade's future price path into training.

Within a fold, a trade's execution-state rows are not independent
observations. The effective sample size for any execution statistic is the
number of trades, not the number of rows. Training procedures must account
for this (e.g. row weight = 1 / trade_row_count). The clustered evaluation
in §6.7 handles this on the scoring side.

A ghost trajectory may continue recording the exogenous market path after an
execution policy closes a paper or simulated trade. Ghost rows are research
observations only and must never alter the realized live outcome.

No execution-state feature may observe bars completed after `state_ts`.

6.5 Feature authority

lab/features.py owns the small active feature set.

It may call functions from lab/indicators.py, but it must not contain alternate implementations of indicator formulas.

The active V0 feature set is limited to a small number of economically distinct families. Old AlphaForge features remain outside the active matrix until they pass salvage review.

6.6 Model authority

lab/models.py owns all initial predictive challengers behind one small interface.

The first active models are:





simple event classifier,



simple direct net_R regressor,



small tree classifier challenger,



small tree regressor challenger.

Models return predictions. They do not open trades, choose risk, calculate PnL, or modify the dataset.

Execution continuation models

The first learned execution challenger is not a generic reinforcement-learning
agent.

It is a finite-horizon continuation-value regressor for a locked `HOLD /
CLOSE` optimal-stopping problem.

The model estimates:

`continuation_value_R(state_t)`

It does not calculate close value, trade PnL, position size, risk, or
actions.

The initial execution model ladder is:

D0 — fixed stop / target / timeout champion,
D0.1 — deterministic `HOLD / CLOSE` challengers,
D1 — pooled shallow-tree continuation-value regressor,
D2 — richer or time-bucketed value models only if D1 reveals a measured
     limitation,
D3 — expanded action space such as `PARTIAL_CLOSE` or `MOVE_TO_BREAKEVEN`,
D4 — offline reinforcement learning only if the simulator-based control
     formulation becomes insufficient.

Neural execution models, actor-critic methods, CQL, IQL, and similar
offline-RL systems remain locked until a simpler execution challenger
demonstrates a concrete limitation that requires them.

6.7 Evaluation authority

lab/eval.py owns:

chronological folds,

outcome-end purging,

out-of-fold predictions,

optional pre-validation gap audits,

probability calibration when sample size permits,

threshold evaluation,

frozen-test evaluation,

compact reports.

The evaluator consumes locked outcomes. It never recreates labels from price data.

It scores a policy; it does not contain one. Threshold evaluation reports what each
candidate threshold would produce — selecting one and acting on it belongs to 6.8.

Execution-policy evaluation

Execution challengers are evaluated against the fixed-exit champion using the
same Hunter entries and the same market paths.

For each `event_id`:

`delta_net_R = execution_challenger_net_R - fixed_exit_net_R`

The primary statistical comparison is paired by `event_id`.

Evaluation must use:

chronological out-of-fold forward replay,
outcome-end purging,
same-entry paired deltas,
block or cluster bootstrap that respects overlapping trades,
execution-cost stress,
action-latency stress,
fold-level and regime-level incremental results.

In-sample backward-induction values, hindsight-optimal exits, and
training-set continuation estimates are diagnostics only. They must not be
reported as deployable execution performance.

The evaluator may report an oracle stopping upper bound, but that value is
never a model score or promotion result.

6.8 Policy authority

lab/policy.py owns the deterministic mapping from a prediction frame to an action.

It is a separate file because RULES §9 requires the prediction/decision boundary to be
structural, not conventional: models forecast, a small versioned policy decides, and
neither side crosses. A boundary that lives inside one file is a comment, not a boundary —
the first refactor that needs "just one probability here" erases it silently.

The policy:

is a pure function of the prediction frame and its own versioned parameters,

reads no price data, no dataset, no fold assignment,

computes no economic quantity — net_R comes from lab/sim.py alone,

carries an explicit version identifier that appears in every result it produces,

stays small enough (~50 lines) to be read in full during any result review.

The V0 action space is TRADE / NO_TRADE. Execution actions (HOLD, CLOSE, MOVE_STOP)
are later challengers and remain locked per §17.

Because the policy is versioned and tiny, changing a threshold is a visible diff against
a recorded identifier rather than an untracked constant inside the evaluator.

Execution stopping policy

The V0 execution policy is a small, versioned, deterministic function:

`CLOSE` when:

`close_value_R > predicted_continuation_value_R + action_margin`

otherwise:

`HOLD`.

`close_value_R` comes only from `lab/sim.py`.

`predicted_continuation_value_R` comes only from the execution model.

`action_margin` belongs to a frozen `ExecSpec`, not to the model.
It is selected per §14 threshold rules: development OOF only,
stable-region selection, frozen before forward replay. A single-point
maximum is treated as fragile — the selected margin must come from a stable
plateau, not a narrow spike in paired delta.

An execution-policy version is pinned when a trade opens and remains
unchanged until that trade terminates. A newly promoted policy may manage
only newly opened trades.

The policy may not:

widen the initial stop,
increase leverage,
increase initial risk,
add to a losing position,
modify the locked Hunter entry decision.

7. Minimal file layout

This section maps every file the architecture anticipates, and the phase that creates it.
It is a boundary map, not a schedule and not a status report — phase status lives in
ROADMAP's "Current status" section and is never duplicated here.

Marks: [*] exists   [ ] planned, phase known   [?] conditional, not promised

lab/                              authorities — imported, never executed
├── indicators.py         P0  [*] causal primitives. Never imports sim.py.
├── sim.py                P1  [*] economic truth: net_R, labels, outcomes. §14.
├── tape.py               P0  [*] bar contract, OHLC, gaps, 5m→15m/1h/4h, hashing.
├── observe.py            P3  [*] model-free geometry measurement. A consumer, not
│                                 an authority — it reads sim.py, it decides nothing.
├── events.py             P4  [*] event rows, decision-time availability, purge.
├── eval.py               P5  [ ] purged folds, OOF, negative controls, ladder.
├── policy.py             P5  [ ] prediction → action. ~50 lines, versioned. §6.8.
├── features.py           P6  [ ] the single feature engine. Calls indicators.py.
├── models.py             P7  [ ] challengers behind one interface. Not an authority.
└── tests/                        one test file per module, 1:1
    ├── test_indicators.py    [*]
    ├── test_sim.py           [*]
    ├── test_tape.py          [*]
    ├── test_observe.py       [*]
    ├── test_events.py        [*]
    ├── test_eval.py          [ ]
    ├── test_policy.py        [ ]
    ├── test_features.py      [ ]
    └── test_models.py        [ ]

tools/                            the edge — network, disk and wall-clock live only here
├── snapshot.py           P2  [*] dataset builder: build, load, observe.
├── experiment.py         P5  [ ] the one experiment CLI: dataset_hash + config_hash + seed.
├── shadow.py             P10 [?] paper/live reconciliation.
└── tests/
    ├── test_snapshot.py      [*]
    └── test_experiment.py    [ ]

salvage/                      [?] quarantine, §2. Absent from disk until something enters it.
└── feature_catalog.json   [?] the old 97-feature catalog, §12.3. Documentation, never
                               a runtime dependency — nothing imports from salvage/.

data/snapshots/               [*] gitignored. Snapshots are rebuilt from a recorded
                                  command, never committed. §15.

The target is nine core Python modules: authorities (indicators, sim,
tape, events, features, models, eval, policy) plus observe.py, which measures
rather than
decides.

7.1 Phases that add no file

Most later phases extend an existing authority rather than creating one:

P8, additional model capacity — a new challenger inside models.py.

P9, execution policy research — policy.py grows an action space. A separate execution
policy file appears only if the entry and execution decisions cannot share one versioned
identifier without confusion.

P10, paper and micro-live — tools/shadow.py, outside the core.

Conditional performance work — any faster path is a parity-gated second function inside
lab/sim.py, never a separate engine file. §14 and §18.

7.2 When a new core file is justified

A new core file is added only when placing the code in an existing authority would
violate a boundary. Files are not created merely to mirror abstract concepts.

The test is whether the file owns a decision that can be wrong. If a reviewer cannot
name the economic or causal claim that breaks when the file breaks, the code belongs
inside an existing authority.

Two consequences follow, and they point in opposite directions:

A file is not split for tidiness. A long file that a reviewer can read top to bottom
and fully judge is worth more than three short files that must be read together. §13.

A file is not merged to reduce the count. lab/policy.py is ~50 lines and could sit
inside eval.py without any line-count argument against it. It does not, because the
boundary it protects is the point. §6.8.

Empty modules, placeholder classes and stub signatures for later phases are not part of
this layout. A planned file is created by the phase that needs it, with its tests, in the
same commit — not reserved in advance. §3, and ROADMAP's rule that only the NOW phase
contains implementation detail.

8. Market data design

8.1 Initial source

The initial bounded snapshot uses one linear perpetual instrument from one exchange.

The V0 shape is:

canonical raw interval: 5m
primary decision candidate: 1h
interval challengers: 15m and 4h
simulation path: 5m
timezone: UTC


The first plumbing instrument may be a high-liquidity perpetual such as BTC-USDT-SWAP. This is a data-pipeline choice, not a claim that BTC contains the best alpha.

8.1.1 Universe rollout (forward specification, not current scope)

ROADMAP Phase 2 is explicitly single-instrument: "No multi-market expansion occurs in this phase." The tiers below record where the universe is headed once Phase 2 exits and Stage B/pilot work begins — they are not built now and do not unlock this phase early.

test profile: 4 symbols, 1 year — first multi-symbol pipeline exercise
early profile: 10 symbols, 3 years — pilot universe (interval geometry, HunterSpec selection)
full profile: 56 symbols, 5+ years if available — main evaluation universe

Each tier reuses the same single-instrument build command in a loop; no new orchestration code is written until a tier is actually reached (RULES §3). Symbol selection within a tier is by UniverseSpec (liquidity-ranked, no hand-picking), not by this document.

8.2 Raw tapes

The snapshot contains separate immutable tapes:

trade_bars_5m.parquet
mark_bars_5m.parquet
funding_events.parquet
instrument.json
manifest.json


Last-price bars

Used for:





local interval aggregation,



features,



entry fill convention,



stop and target path,



timeout exit,



MAE and MFE.

Mark-price bars

Used only where the simulation contract requires mark price, initially funding valuation.

Funding events

Funding remains an event sequence. It is not duplicated or forward-filled into every candle row.

Instrument metadata

Contains the contract and precision information required to interpret volume, price, and order units.

8.3 Derived intervals

All larger intervals are derived locally from the validated 5m authority:

5m
├── 15m
├── 1h
└── 4h


A derived bar is valid only when every required 5m constituent is present and valid.

The system does not maintain separate exchange-downloaded 15m, 1h, and 4h authorities.

8.4 Gap policy

Missing price bars are never silently filled.

A gap is:





recorded,



propagated into derived interval validity,



used to invalidate any feature or outcome path that requires the missing interval.

Failing closed is preferred to inventing a price path.

9. Event and time contract

9.1 V0 event definition

The initial event generator is deliberately simple:

Every completed decision bar with sufficient valid history is a candidate event.

There is no CUSUM trigger, volatility trigger, or learned event detector in V0.

Event filtering may become a challenger only after overlap and redundancy are measured.

9.2 Required timestamps

Every candidate row carries:

event_id
feature_cutoff_ts
decision_ts
planned_entry_ts
fill_ts
outcome_end_ts


Required ordering:

feature_cutoff_ts <= decision_ts < fill_ts <= outcome_end_ts


The exact equality rules are versioned in DatasetSpec.

9.3 Initial fill convention

A simple V0 convention may be:

decision information: completed decision bar
decision moment: immediately after its close
entry: next valid 5m bar open
outcome scan: begins after entry


This is an explicit research contract, not a claim of zero real-world latency.

No feature may observe the entry bar if that bar begins after the decision.

10. Snapshot and dataset stages

The system builds data in distinct stages.

Stage A — immutable market snapshot

Outputs:





validated raw tapes,



gap report,



metadata,



content hashes,



manifest.

No features, labels, models, or train/test split decisions are required to validate the raw snapshot.

Stage B — interval geometry dataset

Using the same 5m tape, generate candidate events for:





15m,



1h,



4h.

No model is trained.

Measure:





event count,



overlap,



effective independent coverage,



MFE and MAE,



time to favorable/adverse excursion,



candidate target-before-stop frequencies,



same-5m-bar ambiguity,



cost and funding impact.

Real-time horizons, not equal bar counts, are compared across intervals.

Stage C — frozen Hunter outcome contract

One HunterSpec is selected using development data only and then frozen before model research.

It declares:





decision interval,



side generation,



entry convention,



stop,



target,



maximum real-time horizon,



fee assumptions,



slippage assumptions,



funding treatment,



ambiguous-bar behavior.

The locked simulator then generates raw outcomes for every valid event.

Stage D — model dataset

Join by event_id:

event metadata
+ active causal features
+ raw locked outcomes
+ split assignment


The raw outcome columns are retained even when a classification or regression target is derived from them.

Stage E — execution research, conditionally unlocked

Stage E remains locked until one frozen Hunter entry model and fixed-exit
policy pass the frozen-test gate.

Stage E0 — model-free execution headroom

Using the same Hunter entries and locked market paths, calculate:

fixed-exit net_R,
valid early-close liquidation values,
hindsight best valid stopping value,
oracle incremental headroom,
headroom by exit reason,
headroom by elapsed-time bucket,
headroom by symbol and liquidity group,
headroom by volatility and market regime.

The hindsight stopping result is an unattainable upper bound. It is used
only to decide whether execution research has enough economic room to
justify additional complexity.

No execution model is trained in Stage E0.

Stage E1 — deterministic execution challengers

Evaluate small deterministic `HOLD / CLOSE` policies against the fixed-exit
champion on the same entries.

Examples may include:

elapsed-time close,
MFE giveback close,
adverse-momentum close,
volatility-shock close.

Only challengers with positive paired out-of-fold incremental value remain
active.

Stage E2 — continuation-value optimal stopping

Train a shallow continuation-value regressor by backward induction on
development folds.

At each valid execution state:

`CLOSE` has the liquidation value supplied by `lab/sim.py`,
`HOLD` has the estimated conditional continuation value,
the versioned execution policy selects one action.

Promotion results come only from chronological out-of-fold forward replay
and paired same-entry evaluation.

Stage E3 — expanded execution actions

Actions such as `PARTIAL_CLOSE` and `MOVE_TO_BREAKEVEN` remain locked until
Stage E2 demonstrates a measured limitation that cannot be solved by the
`HOLD / CLOSE` formulation.

Stage E4 — reinforcement-learning challengers

Generic offline or online reinforcement learning is unlocked only when:

entry edge remains positive under fixed exits,
deterministic and continuation-value challengers have been evaluated,
the expanded control problem cannot be represented reliably by the simpler
  simulator-based formulation,
the immutable risk kernel remains outside the learned policy.

11. Development and frozen test

The dataset is split chronologically:

older history                         newer history
┌──────────────────────────────┬────────────────────────┐
│         DEVELOPMENT          │      FROZEN TEST       │
└──────────────────────────────┴────────────────────────┘


The frozen-test boundary is recorded before feature, model, threshold, or policy selection.

No universal percentage is assumed. The boundary must provide sufficient calendar coverage and sufficient rare positive outcomes for a meaningful final verdict.

Development uses expanding walk-forward folds.

A training event is removed when its outcome_end_ts reaches into the validation interval.

Optional gaps between train and validation are sensitivity audits, not universal fixed percentages.

The frozen test is used only after one candidate model and policy have been predeclared.

12. Feature policy

12.1 Active V0 features

The first active matrix contains approximately 8–12 features across distinct families:





short return,



medium momentum,



normalized volatility,



ATR relative to price,



range or compression,



volume deviation,



dollar-volume change,



price position relative to a reference,



candle body or wick structure.

The exact list is frozen after the interval and HunterSpec are chosen.

Multiple near-identical lookbacks are not included in V0.

12.2 Feature admission rule

Every active feature must answer:





What historical information does it represent?



Why might that information relate to a Hunter outcome?



Is it causal at feature_cutoff_ts?



What happens when required bars are missing?



Does another active feature already represent nearly the same information?



Can it be removed independently?

12.3 Old 97-feature catalog

The old feature library is preserved as salvage, not imported as an active dataset.

Each old feature receives one status:

LOCKED      verified primitive and contract
CANDIDATE   plausible, not yet admitted
DUPLICATE   redundant with a smaller representation
BLOCKED     requires unavailable data
REJECT      leakage, unclear meaning, or invalid contract


A feature family returns only as a controlled ablation:

baseline feature set
vs.
baseline + one salvaged family


It remains only if it improves out-of-fold economic selection across multiple periods without creating fragility.

13. Model architecture

13.1 Models are challengers, not authorities

No model owns market truth, labels, PnL, risk, or trade execution.

Every model receives the same frozen event rows, split contract, features, and outcomes.

13.2 Initial ladder

M0  Always NO_TRADE
M1  Random score
M2  Linear event classifier
M3  Robust linear net_R regressor
M4  Small tree event classifier
M5  Small tree net_R regressor


The first classification target is derived from locked outcomes, for example:

TARGET
STOP
TIMEOUT


The first regression target is:

realized net_R


Classification and regression are evaluated independently before they are combined.

13.3 Multi-model policy

A multi-model system is unlocked only when both model families demonstrate independent out-of-fold value.

Possible prediction object:

p_target
p_stop
p_timeout
predicted_net_R


A small deterministic policy converts predictions into an action:

TRADE when:
    event expectancy exceeds a frozen threshold
    and direct predicted net_R passes a frozen veto
otherwise:
    NO_TRADE


The policy is a pure function. It contains no model fitting and no market-data access.

13.4 Probability calibration

Calibration is optional and gated by sample sufficiency.

If fold-level positive-event counts are too small, model scores are treated as ranking scores rather than trustworthy probabilities.

No forced probability interpretation is allowed.

13.5 Later challengers

The following remain locked:





candidate ranking,



competing-risk survival,



temporal sequence encoders,



shared multi-task neural models,



reinforcement-learning execution.

Each challenger must solve an observed limitation of the active simpler model.

14. Threshold selection

The classification default threshold of 0.5 has no privileged economic meaning.

Thresholds are selected only from development out-of-fold predictions.

For each candidate threshold, report:





selected-trade count,



coverage,



target/stop/timeout rates,



average net_R,



total net_R,



maximum drawdown in R,



maximum loss streak,



fold-level expectancy,



cost-stress result,



neighboring-threshold stability.

The policy selects a stable region, not a single isolated maximum.

A threshold that performs well only at one narrow value is treated as fragile.

The selected model, calibration method, policy, and threshold are frozen before the final test.

15. Evaluation outputs

The first report remains compact.

Required outputs:

candidate count
selected trade count
coverage
average net_R
total net_R
target / stop / timeout counts
fold-by-fold net_R
2x cost-stress net_R
maximum drawdown_R
maximum loss streak
frozen-test verdict


Prediction metrics such as log loss, Brier score, PR-AUC, or regression error are diagnostics. Economic outcomes remain separate and are always computed from the locked simulation results.

Verdicts are:

PASS  evidence supports continuing
HOLD  correctness or evidence is insufficient
FAIL  tested hypothesis is rejected


A failed model does not invalidate locked indicators, simulation, or market data.

16. Risk architecture

The risk kernel is deterministic and independent of model confidence.

It may enforce:





fixed initial account risk,



maximum concurrent exposure,



leverage bounds,



no stop widening,



no adding to losing positions,



daily or portfolio loss limits.

The model may abstain. It may not override risk limits.

Initially, the simulator uses fixed stop, target, and timeout behavior.

Adaptive exits remain locked.

17. Execution research and conditional reinforcement learning

The first learned execution problem is treated as optimal stopping, not
generic reinforcement learning. Under the V0 assumptions: the action space is
`HOLD / CLOSE`, the market path is exogenous, stop/target/timeout remain
fixed, ghost trajectories preserve the locked future market path, and
`lab/sim.py` can calculate the executable early-close value at every valid
state. Therefore the only learned quantity is the conditional continuation
value of holding.

Offline reinforcement learning is not introduced merely because decisions
occur sequentially. It requires a measured limitation of the
optimal-stopping or simulator-based control formulation.

18. Performance architecture

The scalar Python simulation remains the economic authority.

Performance work follows measurement:

scalar Python
    ↓ only if too slow
compiled CPU path
    ↓ only if still blocking
parallel CPU
    ↓ only if justified
GPU event scanner


Any fast path must reproduce scalar outcomes under parity tests.

A GPU path may accelerate event scanning. It must not become an independent PnL authority.

19. Reproducibility

A complete experiment is identified by:

market_snapshot_hash
dataset_spec_hash
hunter_spec_hash
feature_set_hash
split_spec_hash
model_config_hash
policy_config_hash
seed
code_commit_sha

For execution experiments, reproducibility additionally requires:

trajectory_snapshot_hash,
exec_spec_hash,
execution_state_schema_hash,
execution_model_config_hash,
execution_policy_config_hash,
entry_policy_version,
execution_policy_version,
risk_spec_version.

Every execution result must identify the fixed-exit comparator used for its
paired delta calculation.


The experiment must be rerunnable from these identifiers.

A small append-only trial record is sufficient initially. A database, experiment server, dashboard, or orchestration platform is not required.

20. Complexity gate

A proposed component is rejected unless all four questions have concrete answers:





Which measured limitation does it solve?



Which experiment is impossible or unreliable without it?



Is there a smaller solution?



Can it be removed independently when it fails?

“Recommended by the literature” is not sufficient by itself.

21. Build order

1. Lock indicators
2. Lock scalar simulation
3. Write DatasetSpec V0
4. Build one verified 5m market snapshot
5. Derive 15m / 1h / 4h bars locally
6. Run model-free interval and outcome geometry
7. Freeze one HunterSpec
8. Freeze chronological test boundary
9. Admit 8–12 V0 features
10. Run linear classifier and regression baselines
11. Run small tree challengers
12. Combine models only if both add OOF value
13. Evaluate one frozen policy on the frozen test
14. Salvage old feature families one at a time
15. Unlock later challengers only from measured need
16. Prove one frozen Hunter entry policy under fixed exits
17. Write and freeze ExecSpec V0 (lives in EXEC_SPEC_V0.md at repo root,
    referenced by hash; thresholds and margins are versioned there, not in
    ARCHITECTURE.md)
18. Enable append-only ghost-trajectory recording (prerequisite for paper
    trading — trajectory data is not retroactively recoverable from
    snapshots alone)
19. Measure model-free oracle execution headroom
20. Stop execution research if economically useful headroom is insufficient
21. Run deterministic HOLD / CLOSE challengers
22. Train the shallow continuation-value optimal-stopping challenger
23. Evaluate incremental value using paired chronological OOF replay
24. Promote one execution policy only after paired robustness gates pass
25. Unlock expanded actions or reinforcement learning only from a measured
    limitation


22. Definition of a healthy repository

The architecture is healthy when:





one person can explain the full path from raw bar to final net_R,



every economic number comes from one simulation authority,



future data cannot affect past features,



malformed data fails closed,



a model can be replaced without changing labels or PnL,



a failed hypothesis does not trigger a full rewrite,



inactive ideas remain documentation, not runtime dependencies,



the active core stays small enough to audit manually.

The goal is not to build the most sophisticated trading laboratory.

The goal is to build the smallest system capable of telling the truth.