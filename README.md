# v7

Minimal, fail-closed alpha research lab. Successor to `v7-engine` (frozen, read-only).

**Goal:** build a lab that can say *"edge"* or *"no edge"* — and be trusted either way.

## Mottos

1. **Code is liability.** Core budget ~5k lines, hard cap 10k. Auditability is a constraint, not a feature.
2. **Fail closed.** Missing or invalid input raises. No fallbacks. No synthetic data, ever, unless explicitly requested and loudly labeled.
3. **Evidence is a command the operator ran** — never prose, never a report.
4. **Every trade must be hand-verifiable** against raw candles. If you can't verify one trade by hand, the result is not observable and not trusted.
5. **Prediction and decision never mix.** Models forecast; a separate, tiny, versioned policy decides.

## Layout

One subsystem, one file (RULES §3, §13):

```
lab/
  sim.py          Deterministic truth core — defines net_R, labels, outcomes.
                  Reference engine (scalar, hand-verifiable). Nothing else computes money.
  indicators.py   Pure, causal research primitives. Never imported by sim.py.
  tests/          Verification. Every claim is a test.
tools/
  snapshot.py     Only file touching network/disk/wall-clock: build, load, observe.
salvage/          Quarantine for code copied from v7-engine, UNAUDITED — see
                  RULES.md §2. Currently empty; nothing has entered quarantine yet.
```

## Status

Phase 0 (indicator authority) and Phase 1 (simulation authority) locked: 227/227 tests pass from a clean checkout on the remote box, `indicator-authority`/`simulation-authority` tags recorded at commit `8117950`. Phase 2 (verified data snapshot) exited: one immutable BTC-USDT-SWAP 5m snapshot built and hash-verified, reproduced across independent live fetches. Now in Phase 3 (outcome observation). See [ROADMAP.md](ROADMAP.md) for the evidence. Rules of the house: [RULES.md](RULES.md).
