"""Feature extraction authority (ROADMAP Phase 6).

Pure: no I/O, no network, no wall-clock. Computes causal feature vectors
from bar data at decision timestamps. Features are directional: momentum
returns are sign-flipped for SHORT events so the model can learn alignment
between event direction and recent trend.

Contract locked in specs/feature_candidate_v0.json — names, order, lookbacks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from lab.events import CandidateEvent
from lab.indicators import compute_atr
from lab.market import Bar

# ═══════════════════════════════════════════════════════════════════════════════
# feature contract
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = (
    "return_5m",
    "return_15m",
    "return_1h",
    "atr_pct",
)

FEATURE_DIM = len(FEATURE_NAMES)
ATR_PERIOD = 14


# ═══════════════════════════════════════════════════════════════════════════════
# raw feature precomputation — O(n) one-pass per symbol
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_features(
    bars: list[Bar],
    decision_tss: Sequence[int],
) -> dict[int, np.ndarray]:
    """Precompute raw (non-directional) feature vectors for decision timestamps.

    One O(n) pass over bars. Returns dict mapping decision_ts → feature vector.
    Features: [return_5m, return_15m, return_1h, atr_pct]

    Fail-closed: raises KeyError for any decision_ts not found in bars,
    raises ValueError for NaN/inf features, raises ValueError for
    insufficient history (less than ATR_PERIOD bars before decision).
    """
    if not bars:
        raise ValueError("bars must not be empty")

    n = len(bars)
    closes = np.array([b.close for b in bars], dtype=float)
    highs = np.array([b.high for b in bars], dtype=float)
    lows = np.array([b.low for b in bars], dtype=float)

    atr = compute_atr(highs.tolist(), lows.tolist(), closes.tolist(), period=ATR_PERIOD)
    atr_arr = np.array(atr, dtype=float)

    ts_to_idx: dict[int, int] = {b.open_ts: i for i, b in enumerate(bars)}

    result: dict[int, np.ndarray] = {}
    for ts in decision_tss:
        bar_before_ts = ts - 300_000
        idx = ts_to_idx.get(bar_before_ts)
        if idx is None:
            raise KeyError(
                f"decision_ts={ts}: bar at {bar_before_ts} not found in bars"
            )

        if idx < ATR_PERIOD:
            raise ValueError(
                f"decision_ts={ts}: insufficient history (idx={idx}, need >={ATR_PERIOD})"
            )

        def _ret(lookback: int) -> float:
            ref = closes[idx - lookback]
            if ref == 0.0:
                raise ValueError(
                    f"decision_ts={ts}: zero close at lookback={lookback}"
                )
            return float((closes[idx] - ref) / ref)

        return_5m = _ret(1)
        return_15m = _ret(3)
        return_1h = _ret(12)

        atr_val = float(atr_arr[idx])
        if not np.isfinite(atr_val) or atr_val <= 0:
            raise ValueError(f"decision_ts={ts}: non-finite or non-positive ATR={atr_val}")

        atr_pct = atr_val / float(closes[idx])
        if not np.isfinite(atr_pct):
            raise ValueError(f"decision_ts={ts}: non-finite atr_pct")

        feats = np.array([return_5m, return_15m, return_1h, atr_pct], dtype=float)
        if not np.all(np.isfinite(feats)):
            raise ValueError(f"decision_ts={ts}: non-finite feature values")

        result[ts] = feats

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# directional event features
# ═══════════════════════════════════════════════════════════════════════════════

def build_event_features(
    events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
) -> dict[str, np.ndarray]:
    """Build directional feature vectors for candidate events.

    Returns ``{event_id: feature_vector}``. Momentum returns are multiplied
    by ``side_sign`` (+1 for LONG, -1 for SHORT) so the model can learn
    directional alignment. ATR is unsigned (always positive).

    Same-timestamp LONG and SHORT events receive mirrored return features.
    """
    # Collect unique (symbol, decision_ts) pairs
    symbol_tss: dict[str, set[int]] = {}
    for e in events:
        symbol_tss.setdefault(e.symbol, set()).add(e.decision_ts)

    # Precompute raw features per symbol
    raw_features: dict[str, dict[int, np.ndarray]] = {}
    for sym, tss in symbol_tss.items():
        bars = bars_by_symbol.get(sym)
        if bars is None:
            raise KeyError(f"no bars for symbol {sym}")
        raw_features[sym] = precompute_features(bars, sorted(tss))

    # Apply directional mirror
    result: dict[str, np.ndarray] = {}
    for e in events:
        raw = raw_features[e.symbol][e.decision_ts].copy()
        side_sign = 1.0 if e.side == "LONG" else -1.0
        # Only momentum features are directional; ATR stays unsigned
        raw[0] *= side_sign  # return_5m
        raw[1] *= side_sign  # return_15m
        raw[2] *= side_sign  # return_1h
        # raw[3] is atr_pct — unsigned, stays as-is
        result[e.event_id] = raw

    return result
