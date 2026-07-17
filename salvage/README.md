# salvage/ — QUARANTINE

Copied from `v7-engine` @ f1753a8. **UNAUDITED unless listed below. Do not import.**

## Audit ledger (2026-07-17)

**Moved to `lab/` (audited + tested):**
- `indicators/` (10 files) — pure, causal; all pass the lookahead test in `lab/tests/test_indicators_causality.py`; `microstructure.dollar_volume` fixed to reject mismatched input lengths
- `costs/fees.py`, `costs/slippage.py` — the parameter values (fee rates, bps) survived into
  `lab/sim/costs.py`, but the notional-based API was **rewritten** as fractional costs so the
  truth core computes net_R with no unit juggling. Old files deleted from lab (superseded).

**Deleted (broken):**
- `costs/combined.py`, `costs/__init__.py` — imported `r_costs` (never salvaged; broken R-unit conversions)
- `indicators/__init__.py` — hardwired to old `lib.*` paths

**Rejected, still quarantined here:**

| Path | Why it stays |
|------|--------------|
| `lib/costs/funding_impact.py` | Unit bug: divides quote-currency funding cost by price-unit risk (`atr*stop_mult`) — missing `notional/entry_price` factor. Same disease as `r_costs`. Outcome engine (Phase 1) will do funding in quote currency instead. |
| `lib/data_lake/sync.py`, `funding.py` | Depend on `lib.market_data.*` (Binance client stack, never salvaged). Phase 0 decides: salvage that stack or rewrite thin. |
| `lib/data_lake/guard.py` | Sentinel-tag + `sys.exit` approach; Phase 0 integrity gate will be hash-based and exception-based. Kept as reference. |

Exit paths for every file here: audited + tested → moves to `lab/`, or deleted. Nothing lives here permanently.
