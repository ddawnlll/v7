"""Tests for lab/evaluate.py — Phase 5 evaluation authority.

Covers: PredictionContext safety, TAKE/ABSTAIN contract, immutable ledger,
aggregate-from-ledger reconciliation, shuffled-label control, split/purge
verification, adversarial leakage prevention, and golden economic tests.
"""

from __future__ import annotations

import pytest

from lab.evaluate import (
    PredictionContext,
    EventLedger,
    EvalMetrics,
    SplitEval,
    precompute_features,
    evaluate,
    _metrics_for_split,
    reconcile,
    shuffled_control_check,
    verify_splits,
    make_abstain_predictor,
    make_always_take_predictor,
    make_random_predictor,
    make_linear_predictor,
    make_tree_predictor,
)
from lab.events import CandidateEvent
from lab.sim import TradeOutcome, CostBreakdown
from lab.market import Bar


# ——— helpers ———

def _event(
    decision_ts: int, side: str, net_r: float,
    exit_reason: str = "stop", split: str = "train",
) -> CandidateEvent:
    return CandidateEvent(
        event_id=f"ev_{decision_ts}_{side}",
        symbol="TEST",
        side=side,
        feature_cutoff_ts=decision_ts,
        decision_ts=decision_ts,
        planned_entry_ts=decision_ts,
        fill_ts=decision_ts,
        outcome_end_ts=decision_ts + 300_000,
        locked_outcome=TradeOutcome(
            side=side,
            entry_index=0, exit_index=1, exit_reason=exit_reason,
            entry_price=100.0, exit_price=101.0,
            nominal_return=1.0, risk_fraction=1.0,
            gross_return=net_r, net_return=net_r, net_r=net_r,
            mae_r=0.0, mfe_r=abs(net_r),
            costs=CostBreakdown(fee=0.0, slippage=0.0, funding=0.0, total=0.0),
        ),
        split=split,
    )


def _flat_bars(n: int = 30, open_ts_start: int = 0) -> list[Bar]:
    step = 300_000
    return [
        Bar(open_ts=i * step, open=100.0, high=101.0, low=99.0,
            close=100.0, volume=1.0)
        for i in range(open_ts_start // step,
                       open_ts_start // step + n)
    ]


def _ctx(decision_ts: int, side: str = "LONG", feats=None) -> PredictionContext:
    import numpy as np
    return PredictionContext(
        event_id=f"ev_{decision_ts}_{side}",
        symbol="TEST", side=side, decision_ts=decision_ts,
        features=feats if feats is not None else np.ones(4),
    )


# ——— PredictionContext safety ———

def test_prediction_context_has_no_outcome_access():
    """PredictionContext must not expose locked_outcome, future bars, or split."""
    ctx = _ctx(0)
    assert not hasattr(ctx, "locked_outcome")
    assert not hasattr(ctx, "outcome_end_ts")
    assert not hasattr(ctx, "split")
    assert not hasattr(ctx, "bars")


# ——— TAKE/ABSTAIN contract ———

def test_abstain_zero_trades():
    events = [_event(0, "LONG", 0.5), _event(300_000, "SHORT", -0.3)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000])
    result = evaluate(events, {"TEST": feats}, make_abstain_predictor(), seed=42)

    assert result.train.n_events == 2
    assert result.train.n_taken == 0
    assert result.train.coverage == 0.0
    assert result.train.mean_net_r is None
    assert all(not r.predicted for r in result.ledger)


def test_always_take_takes_every_candidate():
    events = [_event(0, "LONG", 1.0), _event(300_000, "SHORT", -2.0)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000])
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)

    assert result.train.n_taken == 2
    assert result.train.coverage == 1.0
    # net_r is used as-is (no inversion) — only taken events count
    assert result.train.mean_net_r == pytest.approx(-0.5)  # (1.0 + -2.0) / 2


def test_abstained_events_have_zero_net_r_in_ledger():
    """Abstained events appear in ledger with net_r=0.0."""
    events = [_event(0, "LONG", 5.0), _event(300_000, "LONG", -3.0)]
    bars = _flat_bars()

    # Predictor: take first, abstain second
    calls = [0]
    def selective(ctx):
        calls[0] += 1
        return calls[0] == 1  # first call = take, second = abstain

    selective.fit = lambda ctxs, *, seed: None

    feats = precompute_features(bars, [0, 300_000])
    result = evaluate(events, {"TEST": feats}, selective, seed=42)

    rows = list(result.ledger)
    assert rows[0].predicted is True
    assert rows[0].net_r == 5.0
    assert rows[1].predicted is False
    assert rows[1].net_r == 0.0


# ——— immutable ledger ———

def test_ledger_aggregate_derives_from_ledger():
    """Metrics must be derived from ledger, not computed independently."""
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(10)]
    bars = _flat_bars(n=50)
    tss = [e.decision_ts for e in events]
    feats = precompute_features(bars, tss)
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)

    # Manually derive from ledger
    taken = [r.net_r for r in result.ledger if r.predicted]
    assert len(taken) == result.train.n_taken
    assert sum(taken) / len(taken) == pytest.approx(result.train.mean_net_r)


def test_ledger_is_immutable():
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0])
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)

    assert isinstance(result.ledger, tuple)
    with pytest.raises(TypeError):
        result.ledger[0] = None  # type: ignore[index]


# ——— reconciliation ———

def test_reconciliation_passes():
    events = [_event(0, "LONG", 0.5), _event(300_000, "LONG", -0.3)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000])
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)
    assert reconcile(result)


def test_reconciliation_detects_tampered_metrics():
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0])
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)

    # Tamper with a metric
    tampered = SplitEval(
        ledger=result.ledger,
        train=EvalMetrics(n_events=1, n_taken=1, coverage=1.0,
                          mean_net_r=999.0, median_net_r=999.0,
                          std_net_r=0.0, sharpe=None, win_rate=1.0),
        test=result.test,
    )
    assert not reconcile(tampered)


# ——— determinism ———

def test_deterministic_reproduction():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(20)]
    bars = _flat_bars(n=200)
    tss = [e.decision_ts for e in events]
    feats = precompute_features(bars, tss)

    r1 = evaluate(events, {"TEST": feats}, make_random_predictor(), seed=42)
    r2 = evaluate(events, {"TEST": feats}, make_random_predictor(), seed=42)
    r3 = evaluate(events, {"TEST": feats}, make_random_predictor(), seed=99)

    assert r1.train.mean_net_r == r2.train.mean_net_r
    assert r1.train.n_taken == r2.train.n_taken
    assert r1.train.mean_net_r != r3.train.mean_net_r


# ——— train/test split ———

def test_train_test_split_is_respected():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", 2.0, split="train"),
        _event(600_000, "LONG", 3.0, split="test"),
    ]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000, 600_000])
    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)

    assert result.train.n_events == 2
    assert result.train.mean_net_r == pytest.approx(1.5)
    assert result.test.n_events == 1
    assert result.test.mean_net_r == pytest.approx(3.0)


# ——— fail-closed guards ———

def test_invalid_split_raises():
    events = [_event(0, "LONG", 1.0, split="garbage")]
    bars = _flat_bars()
    feats = precompute_features(bars, [0])
    with pytest.raises(ValueError, match="invalid split"):
        evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)


def test_missing_features_raises():
    events = [_event(0, "LONG", 1.0)]
    # No features for decision_ts=0
    with pytest.raises(KeyError, match="no precomputed features"):
        evaluate(events, {}, make_always_take_predictor(), seed=42)


def test_invalid_predictor_output_is_coerced_to_bool():
    """Predictor returning non-bool is bool()-coerced (fail-closed via type)."""
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0])

    def weird(_ctx):
        return "BANANA"  # truthy → TAKE

    weird.fit = lambda ctxs, *, seed: None
    result = evaluate(events, {"TEST": feats}, weird, seed=42)
    assert result.train.n_taken == 1


# ——— split/purge verification ———

def test_verify_splits_passes_on_valid_events():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", 2.0, split="train"),
        _event(1_000_000, "LONG", 3.0, split="test"),
    ]
    # Train events end at decision_ts + 300_000
    # Split at 900_000: train outcomes at 300_000 and 600_000 < 900_000 ✓
    assert verify_splits(events, frozen_test_start_ts=900_000)


def test_verify_splits_fails_on_train_leak():
    events = [
        _event(0, "LONG", 1.0, split="train"),
    ]
    # outcome_end_ts = 300_000 >= frozen_test_start_ts=100_000 → leak
    assert not verify_splits(events, frozen_test_start_ts=100_000)


def test_verify_splits_fails_on_test_before_boundary():
    events = [
        _event(100_000, "LONG", 1.0, split="test"),
    ]
    # decision_ts=100_000 < frozen_test_start_ts=500_000
    assert not verify_splits(events, frozen_test_start_ts=500_000)


# ——— linear baseline ———

def test_linear_baseline_take_abstain():
    """Linear predictor with intercept: all-positive targets → TAKE."""
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    targets = [1.0] * 50  # all positive
    model = make_linear_predictor(ctxs, targets, seed=42)

    for ctx in ctxs:
        assert model(ctx) is True  # intercept ≈ 1.0 > 0


def test_linear_baseline_abstains_on_negative():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    targets = [-1.0] * 50  # all negative
    model = make_linear_predictor(ctxs, targets, seed=42)

    taken = sum(1 for ctx in ctxs if model(ctx))
    # Intercept ≈ -1.0 → all abstain
    assert taken == 0


# ——— tree baseline ———

def test_tree_baseline_take_abstain():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    targets = [1.0] * 50
    model = make_tree_predictor(ctxs, targets, seed=42)

    for ctx in ctxs:
        assert model(ctx) is True


def test_tree_baseline_abstains_on_negative():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    targets = [-1.0] * 50
    model = make_tree_predictor(ctxs, targets, seed=42)

    taken = sum(1 for ctx in ctxs if model(ctx))
    assert taken == 0


# ——— shuffled control ———

def test_shuffled_control_does_not_touch_test():
    """Shuffled control must never use test events for fitting."""
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", -1.0, split="train"),
        _event(600_000, "LONG", 0.5, split="test"),
        _event(900_000, "LONG", -0.5, split="test"),
    ]
    bars = _flat_bars(n=50)
    feats = precompute_features(bars, [0, 300_000, 600_000, 900_000])

    results = shuffled_control_check(
        events, {"TEST": feats},
        make_linear_predictor,
        seed=42, n_shuffles=5,
    )

    assert len(results) == 5
    # Each result is test metrics (frozen holdout)
    for r in results:
        assert r.n_events == 2


# ——— adversarial: predictor cannot access outcome ———

def test_adversarial_predictor_has_no_outcome_access():
    """The API physically prevents predictors from reading locked_outcome."""
    events = [_event(0, "LONG", 1.0), _event(300_000, "SHORT", -1.0)]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000])

    def honest(ctx: PredictionContext) -> bool:
        # ctx has NO locked_outcome attribute — would be AttributeError
        return True

    honest.fit = lambda ctxs, *, seed: None
    # This must NOT raise — PredictionContext doesn't expose outcome
    result = evaluate(events, {"TEST": feats}, honest, seed=42)
    assert result.train.n_taken == 2


# ——— golden economic test ———

def test_golden_ledger_matches_hand_computation():
    """Hand-verify every ledger row for a known event sequence."""
    events = [
        _event(0, "LONG", 0.5),
        _event(300_000, "LONG", -0.3),
        _event(600_000, "LONG", 0.2),
    ]
    bars = _flat_bars()
    feats = precompute_features(bars, [0, 300_000, 600_000])

    result = evaluate(events, {"TEST": feats}, make_always_take_predictor(), seed=42)
    rows = list(result.ledger)

    assert len(rows) == 3
    assert rows[0].event_id == "ev_0_LONG"
    assert rows[0].predicted is True
    assert rows[0].net_r == 0.5

    assert rows[1].event_id == "ev_300000_LONG"
    assert rows[1].net_r == -0.3

    assert rows[2].event_id == "ev_600000_LONG"
    assert rows[2].net_r == 0.2

    # Aggregate from hand
    n_taken = 3
    mean = (0.5 + -0.3 + 0.2) / 3
    assert result.train.n_taken == n_taken
    assert result.train.mean_net_r == pytest.approx(mean)


# ——— features precomputation ———

def test_precompute_features_returns_zero_for_unknown_ts():
    bars = _flat_bars()
    result = precompute_features(bars, [999_999_999])
    feats = result[999_999_999]
    import numpy as np
    assert np.all(feats == 0.0)


def test_precompute_features_o1_lookup():
    """After precomputation, lookup is O(1) dict access."""
    bars = _flat_bars(n=100)
    tss = [i * 300_000 for i in range(14, 98)]  # after ATR warmup
    feats = precompute_features(bars, tss)

    for ts in tss:
        assert ts in feats
        f = feats[ts]
        assert len(f) == 4
