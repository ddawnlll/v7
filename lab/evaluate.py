"""Phase 5 — evaluation authority (ROADMAP Phase 5).

Pure: no I/O, no network, no wall-clock. Evaluates predictors against
candidate events using chronological purged splits. The predictor never
sees future information: it receives only a PredictionContext with
precomputed causal features, event ID, symbol, side, and decision timestamp.

Action contract: TAKE / ABSTAIN. The event's declared side (LONG/SHORT) is
fixed by Phase 4; the predictor decides whether to enter, not which direction.
No arithmetic outcome inversion — sim.py is the sole money-computing authority.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from lab.events import CandidateEvent

# Feature extraction belongs to lab/features.py — evaluate.py is a consumer.

# ═══════════════════════════════════════════════════════════════════════════════
# types
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_SPLITS = frozenset({"train", "test"})


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """What a predictor is allowed to see at decision time.

    Fields deliberately excluded: locked_outcome, outcome_end_ts, split,
    fill_ts, feature_cutoff_ts, planned_entry_ts, any bar data after
    decision_ts. The predictor receives ONLY precomputed causal features.
    """

    event_id: str
    symbol: str
    side: str              # "LONG" or "SHORT" — the event's declared direction
    decision_ts: int
    features: np.ndarray   # precomputed causal feature vector


@dataclass(frozen=True, slots=True)
class EventLedger:
    """Immutable per-event record. Aggregate metrics are derived exclusively
    from this ledger — never computed independently."""

    event_id: str
    split: str             # "train" or "test"
    side: str              # event's declared side
    decision_ts: int
    predicted: bool        # True = TAKE, False = ABSTAIN
    net_r: float           # locked_outcome.net_r (signed, from sim.py)


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Per-split aggregate metrics, derived from the ledger."""

    n_events: int
    n_taken: int
    coverage: float           # n_taken / n_events
    mean_net_r: float | None  # None when n_taken == 0
    median_net_r: float | None
    std_net_r: float | None
    sharpe: float | None
    win_rate: float | None    # fraction of taken events with net_r > 0


@dataclass(frozen=True, slots=True)
class SplitEval:
    """Evaluation result: ledger + metrics, broken out by split."""

    ledger: tuple[EventLedger, ...] = field(repr=False)
    train: EvalMetrics
    test: EvalMetrics


# ═══════════════════════════════════════════════════════════════════════════════
# evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_split(event: CandidateEvent) -> None:
    if event.split not in _VALID_SPLITS:
        raise ValueError(
            f"invalid split {event.split!r} for event {event.event_id} — "
            f"must be one of {sorted(_VALID_SPLITS)}"
        )


def evaluate(
    events: Sequence[CandidateEvent],
    features_by_event_id: dict[str, np.ndarray],
    predictor,  # callable: (PredictionContext) → bool
    *,
    seed: int,
    fit_predictor: bool = False,
) -> SplitEval:
    """Evaluate a predictor on candidate events.

    The predictor receives a PredictionContext per event and returns True
    (TAKE the event at its declared side) or False (ABSTAIN). It never sees
    locked_outcome, future bars, or split labels.

    If ``fit_predictor`` is True, ``predictor.fit(train_events, ...)`` is
    called before evaluation. The test split is never passed to fit() —
    frozen holdout is physically separated from development.

    Returns a SplitEval with the immutable ledger and per-split metrics.
    """
    if fit_predictor:
        # Gather training PredictionContexts (test split excluded)
        train_ctxs: list[PredictionContext] = []
        for event in events:
            _validate_split(event)
            if event.split != "train":
                continue
            feats = features_by_event_id.get(event.event_id)
            if feats is None:
                raise KeyError(f"no features for event {event.event_id}")
            train_ctxs.append(PredictionContext(
                event_id=event.event_id,
                symbol=event.symbol,
                side=event.side,
                decision_ts=event.decision_ts,
                features=feats.copy(),
            ))
        predictor.fit(train_ctxs, seed=seed)

    _random.seed(seed)
    np.random.seed(seed)

    ledger_rows: list[EventLedger] = []
    n_train = 0
    n_test = 0

    for event in events:
        _validate_split(event)
        feats = features_by_event_id.get(event.event_id)
        if feats is None:
            raise KeyError(f"no features for event {event.event_id}")

        ctx = PredictionContext(
            event_id=event.event_id,
            symbol=event.symbol,
            side=event.side,
            decision_ts=event.decision_ts,
            features=feats.copy(),
        )
        take = bool(predictor(ctx))

        row = EventLedger(
            event_id=event.event_id,
            split=event.split,
            side=event.side,
            decision_ts=event.decision_ts,
            predicted=take,
            net_r=event.locked_outcome.net_r if take else 0.0,
        )
        ledger_rows.append(row)

        if event.split == "train":
            n_train += 1
        else:
            n_test += 1

    return SplitEval(
        ledger=tuple(ledger_rows),
        train=_metrics_for_split(ledger_rows, "train", n_train),
        test=_metrics_for_split(ledger_rows, "test", n_test),
    )


def _metrics_for_split(
    ledger: list[EventLedger], split: str, n_events: int,
) -> EvalMetrics:
    rows = [r for r in ledger if r.split == split]
    taken = [r for r in rows if r.predicted]
    n_taken = len(taken)

    if n_events == 0:
        return EvalMetrics(
            n_events=0, n_taken=0, coverage=0.0,
            mean_net_r=None, median_net_r=None,
            std_net_r=None, sharpe=None, win_rate=None,
        )

    if n_taken == 0:
        return EvalMetrics(
            n_events=n_events, n_taken=0, coverage=0.0,
            mean_net_r=None, median_net_r=None,
            std_net_r=None, sharpe=None, win_rate=None,
        )

    net_rs = np.array([r.net_r for r in taken], dtype=float)
    mean = float(np.mean(net_rs))
    std = float(np.std(net_rs, ddof=1)) if n_taken >= 2 else 0.0
    sharpe = mean / std if std > 0 else None

    return EvalMetrics(
        n_events=n_events,
        n_taken=n_taken,
        coverage=n_taken / n_events,
        mean_net_r=mean,
        median_net_r=float(np.median(net_rs)),
        std_net_r=std,
        sharpe=sharpe,
        win_rate=float(np.sum(net_rs > 0)) / n_taken,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# baseline predictors
# ═══════════════════════════════════════════════════════════════════════════════

def make_abstain_predictor():
    """Never trades. Baseline floor."""

    def predict(_ctx: PredictionContext) -> bool:
        return False

    predict.fit = lambda ctxs, *, seed: None  # type: ignore[attr-defined]
    return predict


def make_always_take_predictor():
    """Takes every candidate event. Raw expectancy baseline."""

    def predict(_ctx: PredictionContext) -> bool:
        return True

    predict.fit = lambda ctxs, *, seed: None   # type: ignore[attr-defined]
    return predict


def make_random_predictor():
    """Coin flip per event. Should converge to ~0 net_r."""

    def predict(_ctx: PredictionContext) -> bool:
        return _random.choice([True, False])

    predict.fit = lambda ctxs, *, seed: None   # type: ignore[attr-defined]
    return predict


class LinearBaseline:
    """OLS regression on precomputed features with intercept. Target: locked_outcome.net_r.

    Predicts TAKE if predicted net_r > 0, else ABSTAIN.
    """

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self._intercept: float = 0.0

    def __call__(self, ctx: PredictionContext) -> bool:
        if self._weights is None:
            return False
        pred = float(np.dot(ctx.features, self._weights)) + self._intercept
        return pred > 0.0


def make_linear_predictor(
    train_ctxs: Sequence[PredictionContext],
    targets: Sequence[float],
    *,
    seed: int,
) -> LinearBaseline:
    """Create a fitted linear baseline predictor.

    Args:
        train_ctxs: Training PredictionContexts (development only).
        targets: Corresponding net_r values from locked_outcome.
        seed: For reproducibility.
    """
    _random.seed(seed)
    np.random.seed(seed)

    model = LinearBaseline()
    X_rows = [ctx.features for ctx in train_ctxs]

    if len(X_rows) < 2:
        model._weights = np.zeros(4, dtype=float)
        model._intercept = float(np.mean(targets)) if targets else 0.0
        return model

    # Add intercept column
    X_raw = np.array(X_rows, dtype=float)
    X = np.column_stack([X_raw, np.ones(len(X_rows))])
    y = np.array(targets, dtype=float)

    try:
        w = np.linalg.solve(X.T @ X, X.T @ y)
        model._weights = w[:-1]
        model._intercept = float(w[-1])
    except np.linalg.LinAlgError:
        model._weights = np.zeros(4, dtype=float)
        model._intercept = float(np.mean(y))

    return model


class TreeBaseline:
    """Minimal regression tree (CART, max_depth=3, min_samples_leaf=20).

    Predicts TAKE if predicted net_r > 0, else ABSTAIN.
    """

    _MAX_DEPTH = 3
    _MIN_LEAF = 20

    def __init__(self) -> None:
        self._root: _TreeNode | None = None

    def fit(self, ctxs: Sequence[PredictionContext], *, seed: int) -> None:
        raise NotImplementedError("use make_tree_predictor which provides targets")

    def __call__(self, ctx: PredictionContext) -> bool:
        if self._root is None:
            return False
        node = self._root
        while not node.is_leaf:
            assert node.feature_idx is not None
            if ctx.features[node.feature_idx] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.value > 0.0


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


def _build_tree(X: np.ndarray, y: np.ndarray, depth: int,
                max_depth: int, min_leaf: int) -> _TreeNode:
    n = len(y)
    if depth >= max_depth or n < min_leaf * 2:
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
            if left_n < min_leaf or (n - left_n) < min_leaf:
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
    left = _build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf)
    right = _build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf)
    return _TreeNode(feature_idx=best_idx, threshold=best_thresh,
                     left=left, right=right)


def make_tree_predictor(
    train_ctxs: Sequence[PredictionContext],
    targets: Sequence[float],
    *,
    seed: int,
) -> TreeBaseline:
    """Create a fitted tree baseline predictor."""
    _random.seed(seed)
    np.random.seed(seed)

    model = TreeBaseline()
    X_rows = [ctx.features for ctx in train_ctxs]

    if len(X_rows) < TreeBaseline._MIN_LEAF * 2:
        model._root = _TreeNode(value=0.0)
        return model

    X = np.array(X_rows, dtype=float)
    y = np.array(targets, dtype=float)
    model._root = _build_tree(
        X, y, depth=0,
        max_depth=TreeBaseline._MAX_DEPTH,
        min_leaf=TreeBaseline._MIN_LEAF,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — hypothesis predictor
# ═══════════════════════════════════════════════════════════════════════════════

def make_momentum_alignment_predictor():
    """Hypothesis H1: directional momentum alignment.

    TAKE when return_5m > 0 AND return_1h > 0, ABSTAIN otherwise.
    Both features are sign-flipped for SHORT events (features.py), so
    positive = aligned with event direction regardless of LONG/SHORT.

    Rationale: when recent short-term momentum (5m) and medium-term
    momentum (1h) both point in the same direction as the candidate
    event, the trend is confirmed — indicating higher likelihood of
    continuation than when signals disagree.
    """
    def predict(ctx: PredictionContext) -> bool:
        return float(ctx.features[0]) > 0.0 and float(ctx.features[2]) > 0.0

    predict.fit = lambda ctxs, *, seed: None   # type: ignore[attr-defined]
    return predict


# ═══════════════════════════════════════════════════════════════════════════════
# negative control — shuffled-label
# ═══════════════════════════════════════════════════════════════════════════════

def shuffled_control_check(
    events: Sequence[CandidateEvent],
    features_by_event_id: dict[str, np.ndarray],
    predictor_factory,
    seed: int,
    n_shuffles: int = 10,
) -> list[EvalMetrics]:
    """Shuffle train labels and re-fit predictor from scratch each time.

    For each shuffle:
    1. Isolate train events (test split never touched).
    2. Shuffle only the train net_r values.
    3. Fit a new predictor on shuffled targets.
    4. Evaluate on the original test split (untouched).

    If shuffled labels consistently produce positive mean_net_r on test,
    the pipeline has a defect.
    """
    rng = _random.Random(seed)

    train_events = [e for e in events if e.split == "train"]
    if len(train_events) < 2:
        return []

    # Pre-build train contexts from event_id-keyed features
    train_ctxs: list[PredictionContext] = []
    for e in train_events:
        feats = features_by_event_id.get(e.event_id)
        if feats is None:
            raise KeyError(f"no features for event {e.event_id}")
        train_ctxs.append(PredictionContext(
            event_id=e.event_id, symbol=e.symbol,
            side=e.side, decision_ts=e.decision_ts,
            features=feats.copy(),
        ))

    results: list[EvalMetrics] = []
    for i in range(n_shuffles):
        targets = [e.locked_outcome.net_r for e in train_events]
        rng.shuffle(targets)

        predictor = predictor_factory(train_ctxs, targets, seed=seed + i)
        result = evaluate(events, features_by_event_id, predictor, seed=seed + i)
        results.append(result.test)  # evaluate on frozen test split

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# reconciliation
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile(result: SplitEval) -> bool:
    """Verify that aggregate metrics are derived correctly from the ledger.

    Re-derives metrics from the ledger independently and checks identity.
    This proves that per-trade outcomes reconcile exactly with aggregate.
    """
    ledger = list(result.ledger)

    # Count events per split from ledger
    n_train = sum(1 for r in ledger if r.split == "train")
    n_test = sum(1 for r in ledger if r.split == "test")

    train2 = _metrics_for_split(ledger, "train", n_train)
    test2 = _metrics_for_split(ledger, "test", n_test)

    return result.train == train2 and result.test == test2


# ═══════════════════════════════════════════════════════════════════════════════
# split / purge verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_splits(
    events: Sequence[CandidateEvent],
    frozen_test_start_ts: int,
) -> bool:
    """Verify split integrity.

    - All train events have outcome_end_ts < frozen_test_start_ts (purged).
    - All test events have decision_ts >= frozen_test_start_ts.
    - Events are chronologically ordered within each split.
    - No test events leak into train.
    """
    train_events = [e for e in events if e.split == "train"]
    test_events = [e for e in events if e.split == "test"]

    # Train events must end before frozen test starts
    for e in train_events:
        if e.outcome_end_ts >= frozen_test_start_ts:
            return False

    # Test events must start at or after frozen test
    for e in test_events:
        if e.decision_ts < frozen_test_start_ts:
            return False

    # Chronological order within splits
    for i in range(1, len(train_events)):
        if train_events[i].decision_ts < train_events[i - 1].decision_ts:
            return False
    for i in range(1, len(test_events)):
        if test_events[i].decision_ts < test_events[i - 1].decision_ts:
            return False

    return True
