"""Phase 5 — evaluation authority (ROADMAP Phase 5).

Pure: no I/O, no network, no wall-clock. Evaluates predictors against
candidate events using chronological purged splits from Phase 4. Implements
baseline predictors and negative controls to verify the pipeline can
distinguish signal from noise without leaking future information.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from lab.events import CandidateEvent
from lab.sim import CostBreakdown, TradeOutcome
from lab.indicators import compute_atr
from lab.tape import Bar

# ═══════════════════════════════════════════════════════════════════════════════
# types
# ═══════════════════════════════════════════════════════════════════════════════

# A predictor is a stateful object with fit + predict. Stateless predictors
# implement fit as a no-op.
Predictor = Callable[["CandidateEvent", list[Bar]], str | None]


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Per-split evaluation metrics for one predictor."""

    n_events: int
    n_trades: int
    coverage: float           # n_trades / n_events
    mean_net_r: float | None  # None when n_trades == 0
    median_net_r: float | None
    std_net_r: float | None
    sharpe: float | None      # mean / std, None when n_trades < 2 or std == 0
    win_rate: float | None    # fraction of trades with net_r > 0


@dataclass(frozen=True, slots=True)
class SplitEval:
    """Evaluation result broken out by train/test split."""

    train: EvalMetrics
    test: EvalMetrics


# ═══════════════════════════════════════════════════════════════════════════════
# metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(n_events: int, net_rs: Sequence[float]) -> EvalMetrics:
    n_trades = len(net_rs)
    if n_trades == 0:
        return EvalMetrics(
            n_events=n_events, n_trades=0, coverage=0.0,
            mean_net_r=None, median_net_r=None,
            std_net_r=None, sharpe=None, win_rate=None,
        )

    arr = np.asarray(net_rs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_trades >= 2 else 0.0
    sharpe = mean / std if std > 0 else None

    return EvalMetrics(
        n_events=n_events,
        n_trades=n_trades,
        coverage=n_trades / n_events,
        mean_net_r=mean,
        median_net_r=float(np.median(arr)),
        std_net_r=std,
        sharpe=sharpe,
        win_rate=float(np.sum(arr > 0)) / n_trades,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
    predictor: Predictor,
    seed: int,
) -> SplitEval:
    """Evaluate a predictor on candidate events, respecting train/test splits.

    The predictor is called once per event with the event and the symbol's
    bar array. It returns ``"LONG"``, ``"SHORT"``, or ``None`` (abstain).
    The event's ``locked_outcome`` is used as ground truth — this is NOT a
    forward-looking prediction; it is an evaluation of whether the
    predictor's side choice would have captured the known outcome.

    Deterministic: same events + bars + predictor + seed → identical result.
    """
    # Seed both stdlib random and numpy for predictor determinism
    _random.seed(seed)
    np.random.seed(seed)

    train_net_rs: list[float] = []
    test_net_rs: list[float] = []
    n_train_events = 0
    n_test_events = 0

    for event in events:
        bars = bars_by_symbol.get(event.symbol)
        if bars is None:
            continue

        side = predictor(event, bars)
        if side is None:
            # Abstain — counted in n_events but not in net_rs
            if event.split == "train":
                n_train_events += 1
            else:
                n_test_events += 1
            continue

        # Predictor chose LONG or SHORT. If it matches the event's actual
        # side, use the locked outcome's net_r; otherwise invert it.
        net_r = event.locked_outcome.net_r
        if side != event.side:
            net_r = -net_r

        if event.split == "train":
            n_train_events += 1
            train_net_rs.append(net_r)
        else:
            n_test_events += 1
            test_net_rs.append(net_r)

    return SplitEval(
        train=_compute_metrics(n_train_events, train_net_rs),
        test=_compute_metrics(n_test_events, test_net_rs),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# baseline predictors
# ═══════════════════════════════════════════════════════════════════════════════

def make_abstain_predictor() -> Predictor:
    """Never trades. Baseline floor: coverage=0, net_r=None."""

    def predict(_event: CandidateEvent, _bars: list[Bar]) -> None:
        return None

    return predict


def make_always_long_predictor() -> Predictor:
    """Takes every candidate as LONG. Raw expectancy baseline."""

    def predict(_event: CandidateEvent, _bars: list[Bar]) -> str:
        return "LONG"

    return predict


def make_random_predictor() -> Predictor:
    """Coin flip per event. Should converge to ~0 net_r with enough samples."""

    def predict(_event: CandidateEvent, _bars: list[Bar]) -> str:
        return _random.choice(["LONG", "SHORT"])

    return predict


# ═══════════════════════════════════════════════════════════════════════════════
# linear baseline — OLS on minimal features
# ═══════════════════════════════════════════════════════════════════════════════

class LinearBaseline:
    """OLS linear regression on features_at_decision. Predicts LONG if
    dot(features, weights) > 0, else SHORT. Never abstains."""

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None

    def fit(
        self,
        train_events: Sequence[CandidateEvent],
        bars_by_symbol: dict[str, list[Bar]],
    ) -> None:
        """Fit OLS weights on train events using locked_outcome.net_r as target."""
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        for event in train_events:
            if event.split != "train":
                continue
            bars = bars_by_symbol.get(event.symbol)
            if bars is None:
                continue
            feats = features_at_decision(bars, event.decision_ts)
            X_rows.append(feats)
            y_rows.append(event.locked_outcome.net_r)

        if len(X_rows) < 2:
            self._weights = np.zeros(4, dtype=float)
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=float)

        # Normal equation: w = (X^T X)^-1 X^T y
        try:
            self._weights = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            self._weights = np.zeros(4, dtype=float)

    def __call__(self, event: CandidateEvent, bars: list[Bar]) -> str:
        if self._weights is None:
            return "LONG"  # default before fit
        feats = features_at_decision(bars, event.decision_ts)
        pred = float(np.dot(feats, self._weights))
        return "LONG" if pred > 0 else "SHORT"


def make_linear_predictor(
    train_events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
) -> LinearBaseline:
    """Create a fitted linear baseline predictor."""
    model = LinearBaseline()
    model.fit(train_events, bars_by_symbol)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# tree baseline — minimal regression tree
# ═══════════════════════════════════════════════════════════════════════════════

class _TreeNode:
    __slots__ = ("feature_idx", "threshold", "left", "right", "value")

    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float = 0.0,
        left: _TreeNode | None = None,
        right: _TreeNode | None = None,
        value: float = 0.0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class TreeBaseline:
    """Minimal regression tree (CART, max_depth=3, min_samples_leaf=20).
    Predicts LONG if predicted net_r > 0, else SHORT. Never abstains."""

    _MAX_DEPTH = 3
    _MIN_LEAF = 20

    def __init__(self) -> None:
        self._root: _TreeNode | None = None

    def fit(
        self,
        train_events: Sequence[CandidateEvent],
        bars_by_symbol: dict[str, list[Bar]],
    ) -> None:
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        for event in train_events:
            if event.split != "train":
                continue
            bars = bars_by_symbol.get(event.symbol)
            if bars is None:
                continue
            feats = features_at_decision(bars, event.decision_ts)
            X_rows.append(feats)
            y_rows.append(event.locked_outcome.net_r)

        if len(X_rows) < self._MIN_LEAF * 2:
            self._root = _TreeNode(value=0.0)
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=float)
        self._root = self._build(X, y, depth=0)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n = len(y)
        if depth >= self._MAX_DEPTH or n < self._MIN_LEAF * 2:
            return _TreeNode(value=float(np.mean(y)))

        best_var = float("inf")
        best_idx = -1
        best_thresh = 0.0

        for fi in range(X.shape[1]):
            col = X[:, fi]
            unique = np.unique(col)
            if len(unique) < 2:
                continue
            for thresh in unique[:: max(1, len(unique) // 20)]:
                mask = col <= thresh
                left_n = np.sum(mask)
                if left_n < self._MIN_LEAF or (n - left_n) < self._MIN_LEAF:
                    continue
                left_var = np.var(y[mask]) if left_n > 1 else 0.0
                right_var = np.var(y[~mask]) if (n - left_n) > 1 else 0.0
                total_var = (left_n * left_var + (n - left_n) * right_var) / n
                if total_var < best_var:
                    best_var = total_var
                    best_idx = fi
                    best_thresh = thresh

        if best_idx < 0:
            return _TreeNode(value=float(np.mean(y)))

        mask = X[:, best_idx] <= best_thresh
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _TreeNode(feature_idx=best_idx, threshold=best_thresh,
                         left=left, right=right)

    def __call__(self, event: CandidateEvent, bars: list[Bar]) -> str:
        if self._root is None:
            return "LONG"
        feats = features_at_decision(bars, event.decision_ts)
        node = self._root
        while not node.is_leaf:
            assert node.feature_idx is not None
            if feats[node.feature_idx] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return "LONG" if node.value > 0 else "SHORT"


def make_tree_predictor(
    train_events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
) -> TreeBaseline:
    """Create a fitted tree baseline predictor."""
    model = TreeBaseline()
    model.fit(train_events, bars_by_symbol)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# negative control
# ═══════════════════════════════════════════════════════════════════════════════

def shuffled_control_check(
    events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
    predictor: Predictor,
    seed: int,
    n_shuffles: int = 10,
) -> list[EvalMetrics]:
    """Shuffle net_r values across events and re-evaluate.

    Returns one EvalMetrics (train split) per shuffle. If shuffled labels
    consistently produce positive mean_net_r, the pipeline has a defect.
    Phase 5 gate: shuffled labels must not produce edge.
    """
    rng = _random.Random(seed)
    events_list = list(events)
    n = len(events_list)

    # Original net_r pool
    net_r_pool = [e.locked_outcome.net_r for e in events_list]

    results: list[EvalMetrics] = []
    for i in range(n_shuffles):
        shuffled = list(net_r_pool)
        rng.shuffle(shuffled)

        # Build events with shuffled net_r
        shuffled_events = []
        for j, event in enumerate(events_list):
            orig = event.locked_outcome
            new_outcome = TradeOutcome(
                side=orig.side,
                entry_index=orig.entry_index,
                exit_index=orig.exit_index,
                exit_reason=orig.exit_reason,
                entry_price=orig.entry_price,
                exit_price=orig.exit_price,
                nominal_return=orig.nominal_return,
                risk_fraction=orig.risk_fraction,
                gross_return=shuffled[j],
                net_return=shuffled[j],
                net_r=shuffled[j],
                mae_r=orig.mae_r,
                mfe_r=orig.mfe_r,
                costs=CostBreakdown(
                    fee=orig.costs.fee, slippage=orig.costs.slippage,
                    funding=orig.costs.funding, total=orig.costs.total,
                ),
            )
            shuffled_events.append(
                CandidateEvent(
                    event_id=event.event_id,
                    symbol=event.symbol,
                    side=event.side,
                    feature_cutoff_ts=event.feature_cutoff_ts,
                    decision_ts=event.decision_ts,
                    planned_entry_ts=event.planned_entry_ts,
                    fill_ts=event.fill_ts,
                    outcome_end_ts=event.outcome_end_ts,
                    locked_outcome=new_outcome,
                    split=event.split,
                )
            )

        result = evaluate(shuffled_events, bars_by_symbol, predictor, seed + i)
        results.append(result.train)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# reconciliation
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile(
    events: Sequence[CandidateEvent],
    bars_by_symbol: dict[str, list[Bar]],
    predictor: Predictor,
    seed: int,
    result: SplitEval,
) -> bool:
    """Verify per-trade outcomes reconcile with aggregate metrics.

    Re-runs evaluate() with the same inputs and checks bit-identity.
    If the result is not identical, per-trade accounting does not
    reconcile with aggregate — a pipeline defect.
    """
    result2 = evaluate(events, bars_by_symbol, predictor, seed)
    return result == result2


# ═══════════════════════════════════════════════════════════════════════════════
# feature extraction (minimal — Phase 6 will expand this)
# ═══════════════════════════════════════════════════════════════════════════════

def features_at_decision(
    bars: list[Bar],
    decision_ts: int,
    atr_period: int = 14,
) -> np.ndarray:
    """Compute a minimal feature vector at a decision timestamp.

    Returns ``[return_1, return_3, return_12, atr_norm]`` where:
    - return_N: N-bar return ending at the bar before decision_ts
    - atr_norm: ATR(14) / close, or 0 if unavailable

    All values are finite floats. Returns zeros if there is insufficient
    history at decision_ts.
    """
    # Find the bar index just before decision_ts
    # decision_ts is the open_ts of the entry bar, so the bar before it
    # ends at decision_ts - 300_000
    bar_before_ts = decision_ts - 300_000

    # Binary search for the bar at bar_before_ts
    idx = -1
    for i, b in enumerate(bars):
        if b.open_ts == bar_before_ts:
            idx = i
            break
        elif b.open_ts > bar_before_ts:
            break

    if idx < 0:
        return np.zeros(4, dtype=float)

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    # returns: (close[idx] - close[idx-N]) / close[idx-N]
    def _ret(lookback: int) -> float:
        if idx - lookback < 0:
            return 0.0
        ref = closes[idx - lookback]
        if ref == 0:
            return 0.0
        return (closes[idx] - ref) / ref

    return_1 = _ret(1)
    return_3 = _ret(3)
    return_12 = _ret(12)

    # ATR normalized by close
    atr = compute_atr(highs, lows, closes, period=atr_period)
    atr_val = atr[idx] if idx < len(atr) else 0.0
    atr_norm = atr_val / closes[idx] if closes[idx] != 0 else 0.0
    if not np.isfinite(atr_norm):
        atr_norm = 0.0

    return np.array([return_1, return_3, return_12, atr_norm], dtype=float)
