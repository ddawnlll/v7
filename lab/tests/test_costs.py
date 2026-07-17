"""Cost primitive checks (quote-currency units)."""

import pytest

from lab.costs import estimate_fee, estimate_maker_fee, estimate_taker_fee, get_slippage


def test_taker_fee():
    assert estimate_fee(10_000.0, "taker") == pytest.approx(4.0)  # 0.04%
    assert estimate_taker_fee(10_000.0) == pytest.approx(4.0)


def test_maker_fee():
    assert estimate_fee(10_000.0, "maker") == pytest.approx(1.0)  # 0.01%
    assert estimate_maker_fee(10_000.0) == pytest.approx(1.0)


def test_fee_custom_rate():
    assert estimate_fee(10_000.0, "taker", taker_rate=0.0005) == pytest.approx(5.0)


def test_slippage_explicit_pct():
    assert get_slippage(10_000.0, 0.0, slippage_pct=0.05) == pytest.approx(5.0)


def test_slippage_estimated_small_trade():
    # notional far below liquidity → floor at 0.01%
    assert get_slippage(100.0, 1_000_000.0) == pytest.approx(0.01)


def test_slippage_estimated_scales_with_size():
    # notional = 2x liquidity → 0.02% of notional
    assert get_slippage(200.0, 100.0) == pytest.approx(200.0 * 0.0002)


def test_slippage_fails_closed_without_liquidity():
    with pytest.raises(ValueError):
        get_slippage(10_000.0, 0.0)
    with pytest.raises(ValueError):
        get_slippage(10_000.0, -5.0)
