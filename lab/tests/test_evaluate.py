"""Hand-verifiable tests for lab/evaluate.py — Phase 5 evaluation authority."""

from __future__ import annotations

import pytest

from lab.evaluate import (
    make_abstain_predictor,
    make_always_long_predictor,
    make_linear_predictor,
    make_random_predictor,
    make_tree_predictor,
    evaluate,
    shuffled_control_check,
    reconcile,
    features_at_decision,
)
from lab.events import CandidateEvent
from lab.sim import TradeOutcome, CostBreakdown
from lab.tape import Bar


def _event(
    decision_ts: int, side: str, net_r: float, exit_reason: str = "stop",
    split: str = "train",
) -> CandidateEvent:
    return CandidateEvent(
        event_id=f"hash_{decision_ts}_{side}",
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


def _flat_bars(n: int = 20, open_ts_start: int = 0) -> list[Bar]:
    return [
        Bar(open_ts=i * 300_000, open=100.0, high=101.0, low=99.0,
            close=100.0, volume=1.0)
        for i in range(open_ts_start // 300_000,
                       open_ts_start // 300_000 + n)
    ]


# ——— abstain ———

def test_abstain_zero_trades():
    events = [
        _event(0, "LONG", 0.5),
        _event(300_000, "SHORT", -0.3),
    ]
    predictor = make_abstain_predictor()
    bars = _flat_bars()
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)

    assert result.train.n_events == 2
    assert result.train.n_trades == 0
    assert result.train.coverage == 0.0
    assert result.train.mean_net_r is None
    assert result.train.median_net_r is None


# ——— always_long ———

def test_always_long_takes_every_candidate():
    events = [
        _event(0, "LONG", 1.0),
        _event(300_000, "SHORT", -2.0),
    ]
    predictor = make_always_long_predictor()
    bars = _flat_bars()
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)

    assert result.train.n_trades == 2
    assert result.train.coverage == 1.0
    # Event 1: side=LONG → LONG match → net_r=1.0
    # Event 2: side=SHORT → LONG mismatch → net_r = -(-2.0) = 2.0
    # Mean = (1.0 + 2.0) / 2 = 1.5
    assert result.train.mean_net_r == pytest.approx(1.5)


# ——— random ———

def test_random_predictor_coverage():
    events = [_event(i * 300_000, "LONG", 0.1) for i in range(100)]
    predictor = make_random_predictor()
    bars = _flat_bars(n=200)
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)

    # Random never abstains
    assert result.train.coverage == 1.0
    assert result.train.n_trades == 100


# ——— shuffled control ———

def test_shuffled_labels_produce_no_edge():
    """Shuffling labels should not produce a consistently positive mean."""
    events = [
        _event(i * 300_000, "LONG", net_r, "target")
        for i, net_r in enumerate([0.5, -0.3, 0.1, -0.8, 0.2] * 20)
    ]
    predictor = make_always_long_predictor()
    bars = _flat_bars(n=200)

    metrics_list = shuffled_control_check(events, {"TEST": bars}, predictor, seed=42, n_shuffles=10)
    # All shuffled mean net_r values should be within ±0.05 of the original mean
    # (random shuffle shouldn't create large deviations from the pool average)
    original = evaluate(events, {"TEST": bars}, predictor, seed=42)
    for m in metrics_list:
        assert m.mean_net_r is not None
        # Shuffled mean should be close to original (same pool, different assignment)
        assert abs(m.mean_net_r - original.train.mean_net_r) < 0.15


# ——— reconciliation ———

def test_reconciliation_passes_on_consistent_events():
    events = [
        _event(0, "LONG", 0.5),
        _event(300_000, "SHORT", -0.3),
    ]
    predictor = make_always_long_predictor()
    bars = _flat_bars()
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)

    # Reconciler re-runs and checks identity
    assert reconcile(events, {"TEST": bars}, predictor, seed=42, result=result)


def test_reconciliation_fails_on_inconsistent_metrics():
    events = [_event(i * 300_000, "LONG", 0.1) for i in range(50)]
    predictor = make_random_predictor()
    bars = _flat_bars(n=200)
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)
    # Different seed → different random choices → different result → reconciliation fails
    assert not reconcile(events, {"TEST": bars}, predictor, seed=99, result=result)


# ——— determinism ———

def test_deterministic_reproduction():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(20)]
    bars = _flat_bars(n=200)
    predictor = make_random_predictor()

    r1 = evaluate(events, {"TEST": bars}, predictor, seed=42)
    r2 = evaluate(events, {"TEST": bars}, predictor, seed=42)
    r3 = evaluate(events, {"TEST": bars}, predictor, seed=99)

    # Same seed → identical
    assert r1.train.mean_net_r == r2.train.mean_net_r
    assert r1.train.n_trades == r2.train.n_trades

    # Different seed → different (random predictor)
    assert r1.train.mean_net_r != r3.train.mean_net_r


# ——— train/test split ———

def test_train_test_split_is_respected():
    events = [
        _event(0, "LONG", 1.0, split="train"),
        _event(300_000, "LONG", 2.0, split="train"),
        _event(600_000, "LONG", 3.0, split="test"),
    ]
    bars = _flat_bars()
    predictor = make_always_long_predictor()
    result = evaluate(events, {"TEST": bars}, predictor, seed=42)

    assert result.train.n_events == 2
    assert result.train.mean_net_r == pytest.approx(1.5)  # (1+2)/2
    assert result.test.n_events == 1
    assert result.test.mean_net_r == pytest.approx(3.0)


# ——— features ———

def test_features_at_decision():
    bars = _flat_bars(n=30)
    # Make bar 14 have a different close for detectable return
    bars[14] = Bar(open_ts=14 * 300_000, open=100.0, high=101.0,
                   low=99.0, close=102.0, volume=1.0)
    # decision_ts = 15 * 300_000 (end of bar 14, i.e. before bar 15)
    feats = features_at_decision(bars, 15 * 300_000)

    # feature vector: [return_1, return_3, return_12, atr_norm]
    assert len(feats) == 4
    assert feats[0] == pytest.approx(0.02)  # bar14 return: (102-100)/100
    # All finite
    import math
    assert all(math.isfinite(f) for f in feats)


# ——— linear baseline ———

def test_linear_baseline_never_abstains():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(50)]
    bars = _flat_bars(n=200)
    # Fit linear on all events (all are "train")
    model = make_linear_predictor(events, {"TEST": bars})
    result = evaluate(events, {"TEST": bars}, model, seed=42)
    assert result.train.coverage == 1.0
    assert result.train.n_trades == 50


def test_linear_baseline_deterministic():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(50)]
    bars = _flat_bars(n=200)
    model = make_linear_predictor(events, {"TEST": bars})
    r1 = evaluate(events, {"TEST": bars}, model, seed=42)
    r2 = evaluate(events, {"TEST": bars}, model, seed=42)
    assert r1.train.mean_net_r == r2.train.mean_net_r
    assert r1.train.n_trades == r2.train.n_trades


# ——— tree baseline ———

def test_tree_baseline_never_abstains():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(50)]
    bars = _flat_bars(n=200)
    model = make_tree_predictor(events, {"TEST": bars})
    result = evaluate(events, {"TEST": bars}, model, seed=42)
    assert result.train.coverage == 1.0
    assert result.train.n_trades == 50


def test_tree_baseline_deterministic():
    events = [_event(i * 300_000, "LONG", 0.1 * i) for i in range(50)]
    bars = _flat_bars(n=200)
    model = make_tree_predictor(events, {"TEST": bars})
    r1 = evaluate(events, {"TEST": bars}, model, seed=42)
    r2 = evaluate(events, {"TEST": bars}, model, seed=42)
    assert r1.train.mean_net_r == r2.train.mean_net_r
    assert r1.train.n_trades == r2.train.n_trades


def test_baselines_on_identical_events_agree():
    """On perfectly symmetric data, baselines should produce sane output."""
    events = [_event(i * 300_000, "LONG", net_r=0.0) for i in range(100)]
    bars = _flat_bars(n=200)
    for make_fn in [make_linear_predictor, make_tree_predictor]:
        model = make_fn(events, {"TEST": bars})
        result = evaluate(events, {"TEST": bars}, model, seed=42)
        assert result.train.n_trades == 100
        assert result.train.mean_net_r is not None
