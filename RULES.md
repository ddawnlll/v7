# RULES

These rules are fixed. Changing one requires a commit that states what failed without it.

## 1. Fail closed
Any stage that receives missing, malformed, or unexpected input **raises**. No fallback paths.
A pipeline that cannot prove its input is real data produces **no artifact**.

## 2. Salvage quarantine
- Nothing imports from `salvage/`. Nothing imports from `v7-engine`.
- Code moves `salvage/ → lab/` only after: line-by-line audit + its own tests + an audit note in the commit message.
- Deleting from `salvage/` is always allowed and always welcome.

## 3. Size budget
Core (`lab/`) target ~5,000 lines, hard cap 10,000. At the cap: stop adding, start pruning.
Files ≤ 400 lines. Functions ≤ 50 lines.

## 4. Single authorities
One dataset builder. One feature engine. One outcome engine. One evaluation harness.
If a second way to compute something appears, one of the two must die in the same PR.

## 5. Economic units
The only performance unit is `net_R = net_return / risk_fraction`.
Raw fractional returns are never reported as "R". Every column name carries its unit
(`_net_r`, `_net_return`, `_bps`). Unit mismatches are bugs, not conventions.

## 6. Leakage discipline
- Purge is event-end based: no training row whose `outcome_end_ts >= validation_start`.
- The frozen holdout is touched by **nothing** — no pruning, no threshold selection, no retraining.
- Features must pass the causality test: changing data after `t` must not change the feature at `t`.

## 7. Negative controls
Every experiment also runs with shuffled labels. If the pipeline finds "edge" in noise,
every result from that pipeline is void until the defect is found.

## 8. Determinism
Every result is reproducible from `dataset_hash + config_hash + seed`.
A claim without a rerunnable command is not a claim.

## 9. Prediction | decision boundary
Models output a prediction frame (forecast + uncertainty). A separate pure policy
(~50 lines, versioned) turns predictions into decisions. Neither side crosses.

## 10. Baseline ladder
No neural model runs before: random predictor, always-NO_TRADE, linear model, XGBoost —
same data, same splits, same costs. NN must beat the ladder or explain why not.

## 11. Completion protocol
Nothing is "done" until the operator has run the verify command (on the remote box)
and seen it pass. Claude's summary is not evidence; the command output is.

## 12. No process theater
No ACCP reports, no lock ledgers, no phase ceremonies. Docs are: README, RULES, ROADMAP,
and code comments that state constraints. Everything else is a test.

## 13. Minimal file surface
Prefer the fewest files that keep each file independently auditable. A file earns its
existence by being one coherent audit unit — something a reviewer can read top to bottom
in one sitting and fully judge on its own. Do not split a concern across files for
tidiness, and do not merge unrelated concerns to cut the count. Fewer files, each with
high standalone audit value: this is how hallucination is caught before it ships.

## 14. Deterministic truth core (`lab/sim/`)
The simulation is the single source of economic truth: it defines `net_R`, labels, and
outcomes. Nothing else in the repo computes money. Inside `lab/sim/`:
- No wall-clock, no global RNG, no network, no env reads. Every function is a pure
  function of its inputs; any randomness is an explicit seed argument.
- Iterate in sorted order — never rely on dict/set insertion order.
- Same inputs → byte-identical output hash, on any machine (enforced by a cross-machine
  determinism test: same result on the operator's box and the remote box).
- One reference engine (scalar, readable, hand-verifiable) defines truth. Any faster path
  (vectorized tape, CUDA) is parity-gated against the reference and is never the sole path.
