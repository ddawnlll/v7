# salvage/ — QUARANTINE

Copied from `v7-engine` @ f1753a8. **UNAUDITED. Do not import.**

| Path | Why it was salvaged | Known caveats |
|------|--------------------|---------------|
| `lib/data_lake/` | Binance download/archive plumbing — boring, low-risk, tedious to rewrite | audit for silent fallbacks before use |
| `lib/costs/` | Fee/slippage/funding **parameters and formulas** are market facts | `r_costs.py` deliberately NOT salvaged (broken R-unit conversions) |
| `lib/indicators/` | Small pure functions | each needs a causality test on import; most are not needed in early phases |

Exit paths for every file here: audited + tested → moves to `lab/`, or deleted. Nothing lives here permanently.
