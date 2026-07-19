# RULES

These rules are fixed. Changing one requires a commit that states what failed without it.

## 1. Fail closed
Structural invalidity (wrong types, mismatched lengths, bad parameters) **raises** — these
are programming errors. Cell-level dirty market data (NaN, inf, impossible OHLC bars)
follows each module's contract: typically NaN-out or segment-reset rather than raise, so a
single bad observation does not crash the pipeline. No fallback paths. A pipeline that
cannot prove its input is real data produces **no artifact**.

## 2. Salvage quarantine
- Nothing imports from `salvage/`. Nothing imports from `v7-engine`.
- Code moves `salvage/ → lab/` only after: line-by-line audit + its own tests + an audit note in the commit message.
- Deleting from `salvage/` is always allowed and always welcome.

## 3. Size budget
Core (`lab/`) target ~5,000 lines, hard cap 10,000. At the cap: stop adding, start pruning.
One file per concern — a whole subsystem lives in one file read in a single pass
(`lab/sim.py`, `lab/indicators.py`), not scattered across a folder of fragments (see §13).
Per-file line count is bounded by the core budget above, not a fixed cap; functions stay
small (≤ ~50 lines) so the single file stays scannable top to bottom.

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
ARCHITECTURE, and code comments that state constraints. Everything else is a test.

ARCHITECTURE was added because the authority-boundary map (which file owns which
economic/causal decision, RULES §4) grew too large to keep solely in ROADMAP's
phase-gated language without either bloating ROADMAP or losing the map entirely.
It describes structure and boundaries, never phase status — status stays in
ROADMAP's "Current status" section, not duplicated here.

## 13. Minimal file surface
Prefer the fewest files that keep each file independently auditable. A file earns its
existence by being one coherent audit unit — something a reviewer can read top to bottom
in one sitting and fully judge on its own. Do not split a concern across files for
tidiness, and do not merge unrelated concerns to cut the count. Fewer files, each with
high standalone audit value: this is how hallucination is caught before it ships.

## 14. Deterministic truth core (`lab/sim.py`)
The simulation is the single source of economic truth: it defines `net_R`, labels, and
outcomes. Nothing else in the repo computes money. Inside `lab/sim.py`:
- No wall-clock, no global RNG, no network, no env reads. Every function is a pure
  function of its inputs; any randomness is an explicit seed argument.
- Iterate in sorted order — never rely on dict/set insertion order.
- Same inputs → byte-identical output hash, on any machine (enforced by a cross-machine
  determinism test: same result on the operator's box and the remote box).
- One reference engine (scalar, readable, hand-verifiable) defines truth. Any faster path
  (vectorized tape, CUDA) is parity-gated against the reference and is never the sole path.

## 15. Remote execution discipline
The remote box has exactly one working directory (`~/v7`). No second clone, bundle, or
extract is left there for a spot-check — if a throwaway copy is ever needed for a one-off
comparison, it's deleted before the session ends. Ad hoc clones accumulate silently and
create doubt about which commit was actually verified.

Local is the sole source of truth. A one-way Mutagen sync (`sync-mode: one-way-replica`,
alpha = local repo including `.git`, beta = remote `~/v7`) keeps them identical; a direct
edit made on the remote is reverted, not merged — the remote is a mirror, never an editor.
This is why `git tag`/`git log` on the remote always match local without a manual `git
pull`: the sync carries `.git` itself, not just the working tree.

This does not relax §11: the operator still runs the verify command on the remote box and
sees it pass themselves. The sync only guarantees that "the code I have locally" and "the
code the remote is testing" are never two different things.
