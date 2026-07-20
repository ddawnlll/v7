# EXEC_SPEC_V0

Versioned execution specification. Concrete thresholds, margins, and gates
live here, not in ARCHITECTURE.md. Architecture describes invariant
boundaries; this document records the current frozen choices.

## 1. Purpose

Define the contract for execution research: what actions exist, how
liquidation value is computed, how stopping decisions are made, and what
evidence gates must be passed before an execution policy is promoted.

## 2. Preconditions

- One Hunter entry model and policy are frozen and have passed the
  fixed-exit frozen test.
- Market, simulation, dataset, evaluation, and policy authorities are
  locked.
- `ExecSpec V0` has been reviewed and frozen before any execution model
  training begins.

## 3. Action space

V0: `HOLD` / `CLOSE`.

Locked until a measured limitation:
- `PARTIAL_CLOSE`
- `MOVE_TO_BREAKEVEN`
- `ADD_TO_POSITION`
- `WIDEN_STOP`

## 4. Execution-state schema

One row per completed 5m bar within an open trade's holding window.
Columns:

- `event_id` — links to the Hunter entry event
- `trade_id` — unique per opened trade
- `state_ts` — timestamp of the completed bar that produced this state
- `entry_policy_version`
- `execution_policy_version`
- `risk_spec_version`
- `elapsed_bars` — bars since entry
- `remaining_bars` — bars until timeout
- `current_mae_r` — worst adverse excursion so far (from `lab/sim.py`)
- `current_mfe_r` — best favorable excursion so far (from `lab/sim.py`)
- `close_value_r` — liquidation value if CLOSE is chosen now (from `lab/sim.py`)
- `is_terminal` — stop, target, or timeout already reached
- `is_valid` — bar is complete, within range, no gap invalidation

## 5. Decision and fill timing

- A completed 5m bar produces an execution state.
- The execution policy observes the state at bar close.
- `CLOSE`: filled at the next valid 5m bar open.
- `HOLD`: advance to the next bar's state.
- The locked stop, target, and timeout always take precedence over the
  execution policy — if the bar already terminated the trade, the state
  is terminal and the policy is not consulted.

## 6. Liquidation-value contract

`close_value_r` for a `CLOSE` decision is computed exclusively by
`lab/sim.py`:

1. exit fill at the next valid 5m bar open,
2. taker fee applied to the exit leg,
3. directional slippage applied,
4. any funding settled during the holding period up to exit,
5. normalized by the trade's risk fraction.

No other module may compute a liquidation value, early-close PnL, or
counterfactual return.

## 7. Ghost-trajectory contract

After a paper or simulated `CLOSE`, ghost rows continue recording the
exogenous market path to the locked horizon. Ghost rows:

- carry `is_ghost = true`,
- record the same market state columns as live rows,
- have null `close_value_r` (the position is closed),
- never affect the realized outcome of any trade,
- are research observations only.

## 8. Continuation-value target

The execution model estimates:

`continuation_value_R(state_t)`

Defined as the expected net_R from holding through the remaining horizon
under the fixed-exit policy, conditional on information available at
`state_t`.

Training targets are produced by backward induction on development folds
using the locked simulator — no future information leaks into the target.

## 9. Deterministic policy

Before any learned model, deterministic `HOLD / CLOSE` challengers are
evaluated:

- elapsed-time close at fixed bar count,
- MFE giveback close (e.g. `CLOSE` if `current_mfe_r - current_return_r >
  threshold`),
- adverse-momentum close,
- volatility-shock close.

Each challenger is a small pure function. Only those with positive paired
out-of-fold `delta_net_R` remain active.

## 10. Development and frozen split

Same chronological boundary as the Hunter entry split. Execution states
belong to the fold of their parent `event_id`.

Outcome-end purge: no training row whose trade's outcome window extends
into the validation or test period.

## 11. Paired evaluation

For each `event_id`:

`delta_net_R = execution_challenger_net_R - fixed_exit_net_R`

Primary comparison: paired by `event_id`, chronological out-of-fold forward
replay, block bootstrap respecting overlapping trade windows.

## 12. Oracle headroom gate

Before any execution model is trained:

- Compute hindsight best valid stopping value (unattainable upper bound).
- Report oracle incremental headroom vs. fixed exit.
- Issue PASS only if economically useful headroom exists.
- HALT execution research if headroom is insufficient.

The oracle value is a ceiling, never a model score.

## 13. OOF promotion gates

An execution challenger is eligible for promotion only when all hold:

- positive mean paired `delta_net_R` on every development fold,
- stability under 2x cost stress,
- stability under +1 bar action latency,
- no single-fold degradation exceeding the declared tolerance,
- block-bootstrap confidence interval excludes zero.

## 14. Cost and latency stress

Every evaluation repeats under:
- 2x taker fees,
- 2x slippage,
- +1 bar decision-to-fill latency.

A challenger that fails any stress condition is not promoted.

## 15. Version pinning

- `ExecSpec` version is frozen before training begins.
- Execution-policy version is pinned when a trade opens.
- A trade keeps its pinned policy version until termination.
- A newly promoted policy manages only newly opened trades.
- Every result record carries `entry_policy_version`,
  `execution_policy_version`, and `risk_spec_version`.

## 16. Risk-kernel invariants

The execution policy may never:
- widen the initial stop,
- increase leverage,
- increase initial risk,
- add to a losing position,
- modify the locked Hunter entry decision.

These are enforced structurally, not by convention.

## 17. Explicit non-goals

Not included in V0:
- online gradient updates,
- automatic hot deployment,
- replay-buffer automation,
- symbol-specific adapters,
- actor-critic methods,
- CQL, IQL, or other generic offline-RL frameworks,
- partial closing,
- stop movement,
- position adding,
- leverage adaptation.
