"""Tests for lab/evaluate.py — Phase 5 evaluation authority."""

from __future__ import annotations

import pytest

from lab.features import build_event_features
from lab.evaluate import (
    PredictionContext,
    EventLedger,
    EvalMetrics,
    SplitEval,
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


def _event(decision_ts: int, side: str, net_r: float,
           exit_reason: str = "stop", split: str = "train") -> CandidateEvent:
    return CandidateEvent(
        event_id=f"ev_{decision_ts}_{side}",
        symbol="TEST", side=side,
        feature_cutoff_ts=decision_ts, decision_ts=decision_ts,
        planned_entry_ts=decision_ts, fill_ts=decision_ts,
        outcome_end_ts=decision_ts + 300_000,
        locked_outcome=TradeOutcome(
            side=side, entry_index=0, exit_index=1, exit_reason=exit_reason,
            entry_price=100.0, exit_price=101.0,
            nominal_return=1.0, risk_fraction=1.0,
            gross_return=net_r, net_return=net_r, net_r=net_r,
            mae_r=0.0, mfe_r=abs(net_r),
            costs=CostBreakdown(fee=0.0, slippage=0.0, funding=0.0, total=0.0),
        ),
        split=split,
    )


def _flat_bars(n: int = 30) -> list[Bar]:
    return [Bar(open_ts=i * 300_000, open=100.0, high=101.0, low=99.0,
                close=100.0, volume=1.0) for i in range(n)]


def _ctx(decision_ts: int, side: str = "LONG", feats=None) -> PredictionContext:
    import numpy as np
    return PredictionContext(
        event_id=f"ev_{decision_ts}_{side}",
        symbol="TEST", side=side, decision_ts=decision_ts,
        features=feats if feats is not None else np.ones(4),
    )


# --- PredictionContext safety ---

def test_prediction_context_has_no_outcome_access():
    ctx = _ctx(0)
    assert not hasattr(ctx, "locked_outcome")
    assert not hasattr(ctx, "outcome_end_ts")
    assert not hasattr(ctx, "split")


# --- TAKE/ABSTAIN ---

def test_abstain_zero_trades():
    events = [_event(0, "LONG", 0.5), _event(300_000, "SHORT", -0.3)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_abstain_predictor(), seed=42)
    assert result.train.n_events == 2
    assert result.train.n_taken == 0
    assert result.train.coverage == 0.0
    assert result.train.mean_net_r is None


def test_always_take_takes_every_candidate():
    events = [_event(0, "LONG", 1.0), _event(300_000, "SHORT", -2.0)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    assert result.train.n_taken == 2
    assert result.train.mean_net_r == pytest.approx(-0.5)


def test_abstained_events_have_zero_net_r_in_ledger():
    events = [_event(0, "LONG", 5.0), _event(300_000, "LONG", -3.0)]
    bars = _flat_bars()
    calls = [0]
    def selective(ctx): calls[0] += 1; return calls[0] == 1
    selective.fit = lambda ctxs, *, seed: None
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, selective, seed=42)
    rows = list(result.ledger)
    assert rows[0].predicted is True and rows[0].net_r == 5.0
    assert rows[1].predicted is False and rows[1].net_r == 0.0


# --- immutable ledger ---

def test_ledger_aggregate_derives_from_ledger():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(10)]
    bars = _flat_bars(n=50)
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    taken = [r.net_r for r in result.ledger if r.predicted]
    assert len(taken) == result.train.n_taken
    assert sum(taken) / len(taken) == pytest.approx(result.train.mean_net_r)


def test_ledger_is_immutable():
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    assert isinstance(result.ledger, tuple)
    with pytest.raises(TypeError):
        result.ledger[0] = None  # type: ignore


# --- reconciliation ---

def test_reconciliation_passes():
    events = [_event(0, "LONG", 0.5), _event(300_000, "LONG", -0.3)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    assert reconcile(result)


def test_reconciliation_detects_tampered_metrics():
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    tampered = SplitEval(
        ledger=result.ledger,
        train=EvalMetrics(n_events=1, n_taken=1, coverage=1.0,
                          mean_net_r=999.0, median_net_r=999.0,
                          std_net_r=0.0, sharpe=None, win_rate=1.0),
        test=result.test,
    )
    assert not reconcile(tampered)


# --- determinism ---

def test_deterministic_reproduction():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(20)]
    bars = _flat_bars(n=200)
    feats = build_event_features(events, {"TEST": bars})
    r1 = evaluate(events, feats, make_random_predictor(), seed=42)
    r2 = evaluate(events, feats, make_random_predictor(), seed=42)
    r3 = evaluate(events, feats, make_random_predictor(), seed=99)
    assert r1.train.mean_net_r == r2.train.mean_net_r
    assert r1.train.n_taken == r2.train.n_taken
    assert r1.train.mean_net_r != r3.train.mean_net_r


# --- train/test split ---

def test_train_test_split_is_respected():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", 2.0, split="train"),
        _event(600_000, "LONG", 3.0, split="test"),
    ]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    assert result.train.n_events == 2
    assert result.train.mean_net_r == pytest.approx(1.5)
    assert result.test.n_events == 1
    assert result.test.mean_net_r == pytest.approx(3.0)


# --- fail-closed ---

def test_invalid_split_raises():
    events = [_event(0, "LONG", 1.0, split="garbage")]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    with pytest.raises(ValueError, match="invalid split"):
        evaluate(events, feats, make_always_take_predictor(), seed=42)


def test_missing_features_raises():
    events = [_event(0, "LONG", 1.0)]
    with pytest.raises(KeyError, match="no features"):
        evaluate(events, {}, make_always_take_predictor(), seed=42)


def test_invalid_predictor_output_is_coerced_to_bool():
    events = [_event(0, "LONG", 1.0)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    def weird(_ctx): return "BANANA"
    weird.fit = lambda ctxs, *, seed: None
    result = evaluate(events, feats, weird, seed=42)
    assert result.train.n_taken == 1


# --- split/purge verification ---

def test_verify_splits_passes_on_valid_events():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", 2.0, split="train"),
        _event(1_000_000, "LONG", 3.0, split="test"),
    ]
    assert verify_splits(events, frozen_test_start_ts=900_000)


def test_verify_splits_fails_on_train_leak():
    events = [_event(0, "LONG", 1.0, split="train")]
    assert not verify_splits(events, frozen_test_start_ts=100_000)


def test_verify_splits_fails_on_test_before_boundary():
    events = [_event(100_000, "LONG", 1.0, split="test")]
    assert not verify_splits(events, frozen_test_start_ts=500_000)


# --- linear baseline ---

def test_linear_baseline_take_abstain():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    model = make_linear_predictor(ctxs, [1.0] * 50, seed=42)
    for ctx in ctxs:
        assert model(ctx) is True


def test_linear_baseline_abstains_on_negative():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    model = make_linear_predictor(ctxs, [-1.0] * 50, seed=42)
    assert sum(1 for ctx in ctxs if model(ctx)) == 0


# --- tree baseline ---

def test_tree_baseline_take_abstain():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    model = make_tree_predictor(ctxs, [1.0] * 50, seed=42)
    for ctx in ctxs:
        assert model(ctx) is True


def test_tree_baseline_abstains_on_negative():
    ctxs = [_ctx(i * 300_000) for i in range(50)]
    model = make_tree_predictor(ctxs, [-1.0] * 50, seed=42)
    assert sum(1 for ctx in ctxs if model(ctx)) == 0


# --- shuffled control ---

def test_shuffled_control_does_not_touch_test():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", -1.0, split="train"),
        _event(600_000, "LONG", 0.5, split="test"),
        _event(900_000, "LONG", -0.5, split="test"),
    ]
    bars = _flat_bars(n=50)
    feats = build_event_features(events, {"TEST": bars})
    results = shuffled_control_check(events, feats, make_linear_predictor, seed=42, n_shuffles=5)
    assert len(results) == 5
    for r in results:
        assert r.n_events == 2


# --- adversarial ---

def test_adversarial_predictor_has_no_outcome_access():
    events = [_event(0, "LONG", 1.0), _event(300_000, "SHORT", -1.0)]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    def honest(ctx: PredictionContext) -> bool: return True
    honest.fit = lambda ctxs, *, seed: None
    result = evaluate(events, feats, honest, seed=42)
    assert result.train.n_taken == 2


# --- golden economic test ---

def test_golden_ledger_matches_hand_computation():
    events = [
        _event(0, "LONG", 0.5),
        _event(300_000, "LONG", -0.3),
        _event(600_000, "LONG", 0.2),
    ]
    bars = _flat_bars()
    feats = build_event_features(events, {"TEST": bars})
    result = evaluate(events, feats, make_always_take_predictor(), seed=42)
    rows = list(result.ledger)
    assert len(rows) == 3
    assert rows[0].net_r == 0.5
    assert rows[1].net_r == -0.3
    assert rows[2].net_r == 0.2
    assert result.train.mean_net_r == pytest.approx((0.5 - 0.3 + 0.2) / 3)
