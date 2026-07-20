#!/usr/bin/env python3
"""
V7-Lite — deterministic, auditable trading engine in one file.

Design goals:
- Pure, causal indicators.
- LONG_NOW / SHORT_NOW / NO_TRADE decisions.
- One setup: trend-continuation pullback.
- Deterministic scoring and explicit reasons.
- Single-source execution costs.
- Bar-by-bar backtest with stop, target, time exit, fees, and slippage.
- No machine learning.
- No lookahead.

Expected CSV columns:
timestamp,open,high,low,close,volume

Example:
python v7-lite.py backtest --csv BTCUSDT_1h.csv
python v7-lite.py signal --csv BTCUSDT_1h.csv
python v7-lite.py selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


EPS = 1e-12


class Action(str, Enum):
    LONG_NOW = "LONG_NOW"
    SHORT_NOW = "SHORT_NOW"
    NO_TRADE = "NO_TRADE"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"Non-finite bar: {self}")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(f"Non-positive OHLC: {self}")
        if self.volume < 0:
            raise ValueError(f"Negative volume: {self}")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High inconsistent: {self}")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low inconsistent: {self}")


@dataclass(frozen=True)
class EngineConfig:
    # Indicators
    fast_ema: int = 20
    slow_ema: int = 50
    atr_period: int = 14
    volume_period: int = 20

    # Setup
    pullback_atr_distance: float = 0.45
    max_extension_atr: float = 1.10
    min_body_atr: float = 0.08
    max_body_atr: float = 1.20
    min_volume_ratio: float = 0.70
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.050

    # Decision — all seven signal components must reach trade_score.
    # Default 8 means: trend(2)+pullback(2)+trigger(2)+volatility(1)=7 is
    # NOT enough; you also need either volume(1) or structure(1).
    trade_score: int = 8
    trend_points: int = 2
    pullback_points: int = 2
    trigger_points: int = 2
    volatility_points: int = 1
    volume_points: int = 1
    structure_points: int = 1

    # Risk / exits
    stop_atr: float = 1.5
    target_r: float = 2.0
    max_hold_bars: int = 12
    risk_per_trade: float = 0.01
    max_leverage: float = 5.0

    # Costs
    fee_rate_per_side: float = 0.0004
    slippage_rate_per_side: float = 0.0002

    # Backtest behavior
    starting_equity: float = 1_000.0
    # allow_same_bar_exit: if True, checks stop/target on the entry bar's high/low.
    # WARNING: the bar's high/low formed BEFORE the close (entry price), so
    # target hits on the entry bar are optimistic lookahead.  Stops are
    # pessimistic (harmless direction).  Default False to be conservative.
    allow_same_bar_exit: bool = False
    stop_first_when_both_hit: bool = True

    def validate(self) -> None:
        if self.fast_ema < 2 or self.slow_ema <= self.fast_ema:
            raise ValueError("Require 2 <= fast_ema < slow_ema")
        if self.atr_period < 2 or self.volume_period < 2:
            raise ValueError("Periods must be >= 2")
        if self.stop_atr <= 0 or self.target_r <= 0:
            raise ValueError("stop_atr and target_r must be positive")
        if not 0 < self.risk_per_trade <= 0.10:
            raise ValueError("risk_per_trade must be in (0, 0.10]")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.trade_score <= 0:
            raise ValueError("trade_score must be positive")


@dataclass(frozen=True)
class IndicatorState:
    ema_fast: float
    ema_slow: float
    atr: float
    atr_pct: float
    volume_ratio: float
    body_atr: float


@dataclass(frozen=True)
class Decision:
    action: Action
    score: int
    timestamp: str
    price: float
    stop_price: float | None
    target_price: float | None
    reasons: tuple[str, ...]
    rejections: tuple[str, ...]
    indicators: IndicatorState | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class Position:
    side: Side
    entry_bar_index: int
    entry_timestamp: str
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    initial_risk_cash: float
    entry_fee: float
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Trade:
    side: Side
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    score: int


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    ending_equity: float
    net_profit: float
    return_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_pct: float
    average_win_r: float
    average_loss_r: float
    trades_detail: tuple[Trade, ...] = field(repr=False)

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("trades_detail", None)
        return d


def ema(values: Sequence[float], period: int) -> list[float]:
    """
    Causal EMA seeded with SMA(period). Values before seed are NaN.
    """
    if period < 2:
        raise ValueError("EMA period must be >= 2")
    n = len(values)
    out = [math.nan] * n
    if n < period:
        return out

    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)

    prev = seed
    for i in range(period, n):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def atr(bars: Sequence[Bar], period: int) -> list[float]:
    """
    Wilder ATR. First ATR is seeded from period true ranges.
    """
    if period < 2:
        raise ValueError("ATR period must be >= 2")
    n = len(bars)
    out = [math.nan] * n
    if n < period + 1:
        return out

    tr = [math.nan] * n
    for i in range(1, n):
        prev_close = bars[i - 1].close
        b = bars[i]
        tr[i] = max(
            b.high - b.low,
            abs(b.high - prev_close),
            abs(b.low - prev_close),
        )

    seed_index = period
    seed_values = tr[1 : period + 1]
    seed = sum(seed_values) / period
    out[seed_index] = seed

    prev = seed
    for i in range(seed_index + 1, n):
        prev = ((period - 1) * prev + tr[i]) / period
        out[i] = prev
    return out


def rolling_mean(values: Sequence[float], period: int) -> list[float]:
    if period < 1:
        raise ValueError("period must be >= 1")
    n = len(values)
    out = [math.nan] * n
    if n < period:
        return out

    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def load_csv(path: str | Path) -> list[Bar]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    bars: list[Bar] = []
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = required - {x.strip().lower() for x in reader.fieldnames}
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        for row_no, row in enumerate(reader, start=2):
            try:
                bar = Bar(
                    timestamp=str(row[normalized["timestamp"]]),
                    open=float(row[normalized["open"]]),
                    high=float(row[normalized["high"]]),
                    low=float(row[normalized["low"]]),
                    close=float(row[normalized["close"]]),
                    volume=float(row[normalized["volume"]]),
                )
                bar.validate()
                bars.append(bar)
            except Exception as exc:
                raise ValueError(f"Invalid CSV row {row_no}: {exc}") from exc

    if len(bars) < 100:
        raise ValueError("Need at least 100 bars")

    # Timestamp validation: must be monotonically increasing, no duplicates.
    prev_ts: int | None = None
    for bar in bars:
        try:
            ts = int(bar.timestamp)
        except ValueError:
            raise ValueError(f"Non-numeric timestamp: {bar.timestamp}")
        if prev_ts is not None:
            if ts <= prev_ts:
                raise ValueError(
                    f"Timestamps must be strictly increasing; "
                    f"found {ts} after {prev_ts}"
                )
        prev_ts = ts
    return bars


class DeterministicEngine:
    def __init__(self, config: EngineConfig):
        config.validate()
        self.config = config

    def precompute(self, bars: Sequence[Bar]) -> dict[str, list[float]]:
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        return {
            "ema_fast": ema(closes, self.config.fast_ema),
            "ema_slow": ema(closes, self.config.slow_ema),
            "atr": atr(bars, self.config.atr_period),
            "volume_mean": rolling_mean(volumes, self.config.volume_period),
        }

    def indicator_state(
        self,
        bars: Sequence[Bar],
        indicators: dict[str, list[float]],
        i: int,
    ) -> IndicatorState | None:
        if i <= 0:
            return None

        ef = indicators["ema_fast"][i]
        es = indicators["ema_slow"][i]
        a = indicators["atr"][i]
        vm = indicators["volume_mean"][i]

        if not all(math.isfinite(x) for x in (ef, es, a, vm)):
            return None
        if a <= EPS or vm <= EPS:
            return None

        b = bars[i]
        return IndicatorState(
            ema_fast=ef,
            ema_slow=es,
            atr=a,
            atr_pct=a / b.close,
            volume_ratio=b.volume / vm,
            body_atr=abs(b.close - b.open) / a,
        )

    def decide(
        self,
        bars: Sequence[Bar],
        indicators: dict[str, list[float]],
        i: int,
    ) -> Decision:
        b = bars[i]
        state = self.indicator_state(bars, indicators, i)

        if state is None:
            return Decision(
                action=Action.NO_TRADE,
                score=0,
                timestamp=b.timestamp,
                price=b.close,
                stop_price=None,
                target_price=None,
                reasons=(),
                rejections=("insufficient_indicator_history",),
                indicators=None,
            )

        prev = bars[i - 1]
        cfg = self.config

        long_score = 0
        short_score = 0
        long_reasons: list[str] = []
        short_reasons: list[str] = []
        common_rejections: list[str] = []

        volatility_ok = cfg.min_atr_pct <= state.atr_pct <= cfg.max_atr_pct
        body_ok = cfg.min_body_atr <= state.body_atr <= cfg.max_body_atr
        volume_ok = state.volume_ratio >= cfg.min_volume_ratio

        if volatility_ok:
            long_score += cfg.volatility_points
            short_score += cfg.volatility_points
            long_reasons.append("volatility_regime_ok")
            short_reasons.append("volatility_regime_ok")
        else:
            common_rejections.append("volatility_regime_rejected")

        if volume_ok:
            long_score += cfg.volume_points
            short_score += cfg.volume_points
            long_reasons.append("volume_ok")
            short_reasons.append("volume_ok")
        else:
            common_rejections.append("volume_too_low")

        if not body_ok:
            common_rejections.append("trigger_candle_body_rejected")

        # Trend
        bullish_trend = state.ema_fast > state.ema_slow and b.close > state.ema_slow
        bearish_trend = state.ema_fast < state.ema_slow and b.close < state.ema_slow

        if bullish_trend:
            long_score += cfg.trend_points
            long_reasons.append("bullish_ema_structure")
        if bearish_trend:
            short_score += cfg.trend_points
            short_reasons.append("bearish_ema_structure")

        # Pullback proximity to fast EMA, while avoiding overextension.
        long_distance = abs(b.low - state.ema_fast) / state.atr
        short_distance = abs(b.high - state.ema_fast) / state.atr
        long_extension = max(0.0, b.close - state.ema_fast) / state.atr
        short_extension = max(0.0, state.ema_fast - b.close) / state.atr

        long_pullback = (
            bullish_trend
            and long_distance <= cfg.pullback_atr_distance
            and long_extension <= cfg.max_extension_atr
        )
        short_pullback = (
            bearish_trend
            and short_distance <= cfg.pullback_atr_distance
            and short_extension <= cfg.max_extension_atr
        )

        if long_pullback:
            long_score += cfg.pullback_points
            long_reasons.append("pullback_to_fast_ema")
        if short_pullback:
            short_score += cfg.pullback_points
            short_reasons.append("pullback_to_fast_ema")

        # Trigger: directional recovery plus prior-bar confirmation.
        long_trigger = (
            body_ok
            and b.close > b.open
            and b.close > prev.close
            and b.close >= state.ema_fast
        )
        short_trigger = (
            body_ok
            and b.close < b.open
            and b.close < prev.close
            and b.close <= state.ema_fast
        )

        if long_trigger:
            long_score += cfg.trigger_points
            long_reasons.append("bullish_recovery_trigger")
        if short_trigger:
            short_score += cfg.trigger_points
            short_reasons.append("bearish_recovery_trigger")

        # Structure confirmation: higher low for long, lower high for short.
        if b.low >= prev.low:
            long_score += cfg.structure_points
            long_reasons.append("higher_low_structure")
        if b.high <= prev.high:
            short_score += cfg.structure_points
            short_reasons.append("lower_high_structure")

        long_valid = (
            long_score >= cfg.trade_score
            and bullish_trend
            and long_pullback
            and long_trigger
            and volatility_ok
        )
        short_valid = (
            short_score >= cfg.trade_score
            and bearish_trend
            and short_pullback
            and short_trigger
            and volatility_ok
        )

        # Deterministic conflict resolution: no trade on tie/conflict.
        if long_valid and short_valid:
            return Decision(
                action=Action.NO_TRADE,
                score=max(long_score, short_score),
                timestamp=b.timestamp,
                price=b.close,
                stop_price=None,
                target_price=None,
                reasons=(),
                rejections=("conflicting_long_short_signal",),
                indicators=state,
            )

        if long_valid:
            stop = b.close - cfg.stop_atr * state.atr
            risk = b.close - stop
            target = b.close + cfg.target_r * risk
            return Decision(
                action=Action.LONG_NOW,
                score=long_score,
                timestamp=b.timestamp,
                price=b.close,
                stop_price=stop,
                target_price=target,
                reasons=tuple(long_reasons),
                rejections=tuple(common_rejections),
                indicators=state,
            )

        if short_valid:
            stop = b.close + cfg.stop_atr * state.atr
            risk = stop - b.close
            target = b.close - cfg.target_r * risk
            return Decision(
                action=Action.SHORT_NOW,
                score=short_score,
                timestamp=b.timestamp,
                price=b.close,
                stop_price=stop,
                target_price=target,
                reasons=tuple(short_reasons),
                rejections=tuple(common_rejections),
                indicators=state,
            )

        dominant_score = max(long_score, short_score)
        rejections = list(common_rejections)
        if not bullish_trend and not bearish_trend:
            rejections.append("trend_not_clear")
        if dominant_score < cfg.trade_score:
            rejections.append("score_below_threshold")
        if bullish_trend and not long_pullback:
            rejections.append("no_valid_long_pullback")
        if bearish_trend and not short_pullback:
            rejections.append("no_valid_short_pullback")
        if bullish_trend and not long_trigger:
            rejections.append("no_long_trigger")
        if bearish_trend and not short_trigger:
            rejections.append("no_short_trigger")

        return Decision(
            action=Action.NO_TRADE,
            score=dominant_score,
            timestamp=b.timestamp,
            price=b.close,
            stop_price=None,
            target_price=None,
            reasons=(),
            rejections=tuple(dict.fromkeys(rejections)),
            indicators=state,
        )


def adverse_entry_price(side: Side, close: float, slippage_rate: float) -> float:
    if side is Side.LONG:
        return close * (1.0 + slippage_rate)
    return close * (1.0 - slippage_rate)


def adverse_exit_price(side: Side, raw_price: float, slippage_rate: float) -> float:
    if side is Side.LONG:
        return raw_price * (1.0 - slippage_rate)
    return raw_price * (1.0 + slippage_rate)


def size_position(
    equity: float,
    side: Side,
    raw_entry: float,
    stop_price: float,
    cfg: EngineConfig,
) -> tuple[float, float, float]:
    entry = adverse_entry_price(side, raw_entry, cfg.slippage_rate_per_side)
    stop_distance = abs(entry - stop_price)
    if stop_distance <= EPS:
        raise ValueError("Stop distance is zero")

    risk_cash = equity * cfg.risk_per_trade
    qty_by_risk = risk_cash / stop_distance
    qty_by_leverage = (equity * cfg.max_leverage) / entry
    qty = min(qty_by_risk, qty_by_leverage)

    if qty <= EPS or not math.isfinite(qty):
        raise ValueError("Invalid position quantity")

    actual_risk_cash = qty * stop_distance
    return entry, qty, actual_risk_cash


def close_trade(
    position: Position,
    exit_timestamp: str,
    raw_exit_price: float,
    reason: str,
    cfg: EngineConfig,
) -> Trade:
    exit_price = adverse_exit_price(
        position.side,
        raw_exit_price,
        cfg.slippage_rate_per_side,
    )

    if position.side is Side.LONG:
        gross = (exit_price - position.entry_price) * position.quantity
    else:
        gross = (position.entry_price - exit_price) * position.quantity

    exit_fee = exit_price * position.quantity * cfg.fee_rate_per_side
    fees = position.entry_fee + exit_fee
    net = gross - fees
    r_multiple = net / position.initial_risk_cash if position.initial_risk_cash > EPS else 0.0

    return Trade(
        side=position.side,
        entry_timestamp=position.entry_timestamp,
        exit_timestamp=exit_timestamp,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        r_multiple=r_multiple,
        exit_reason=reason,
        score=position.score,
    )


def evaluate_exit(
    position: Position,
    bar: Bar,
    bars_held: int,
    cfg: EngineConfig,
) -> tuple[float, str] | None:
    if position.side is Side.LONG:
        stop_hit = bar.low <= position.stop_price
        target_hit = bar.high >= position.target_price
    else:
        stop_hit = bar.high >= position.stop_price
        target_hit = bar.low <= position.target_price

    if stop_hit and target_hit:
        if cfg.stop_first_when_both_hit:
            return position.stop_price, "stop_and_target_same_bar_stop_first"
        return position.target_price, "stop_and_target_same_bar_target_first"

    if stop_hit:
        return position.stop_price, "stop"
    if target_hit:
        return position.target_price, "target"
    if bars_held >= cfg.max_hold_bars:
        return bar.close, "time_exit"
    return None


def backtest(bars: Sequence[Bar], cfg: EngineConfig) -> BacktestResult:
    engine = DeterministicEngine(cfg)
    indicators = engine.precompute(bars)

    equity = cfg.starting_equity
    peak_equity = equity
    max_drawdown = 0.0
    position: Position | None = None
    trades: list[Trade] = []

    warmup = max(cfg.slow_ema, cfg.atr_period + 1, cfg.volume_period)

    for i in range(warmup, len(bars)):
        bar = bars[i]

        if position is not None:
            held = i - position.entry_bar_index
            exit_eval = evaluate_exit(position, bar, held, cfg)
            if exit_eval is not None:
                raw_exit, reason = exit_eval
                trade = close_trade(position, bar.timestamp, raw_exit, reason, cfg)
                # entry_fee was already deducted at open; only apply gross minus exit_fee.
                exit_fee = trade.fees - position.entry_fee
                equity += trade.gross_pnl - exit_fee
                trades.append(trade)
                position = None

                peak_equity = max(peak_equity, equity)
                if peak_equity > EPS:
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)

        if position is None:
            decision = engine.decide(bars, indicators, i)
            if decision.action is not Action.NO_TRADE:
                side = Side.LONG if decision.action is Action.LONG_NOW else Side.SHORT
                assert decision.stop_price is not None
                assert decision.target_price is not None

                # Guard: skip trade when equity is too low to size a valid position.
                if equity <= cfg.starting_equity * 0.001:
                    continue
                entry, qty, risk_cash = size_position(
                    equity=equity,
                    side=side,
                    raw_entry=bar.close,
                    stop_price=decision.stop_price,
                    cfg=cfg,
                )
                entry_fee = entry * qty * cfg.fee_rate_per_side
                equity -= entry_fee

                position = Position(
                    side=side,
                    entry_bar_index=i,
                    entry_timestamp=bar.timestamp,
                    entry_price=entry,
                    stop_price=decision.stop_price,
                    target_price=decision.target_price,
                    quantity=qty,
                    initial_risk_cash=risk_cash,
                    entry_fee=entry_fee,
                    score=decision.score,
                    reasons=decision.reasons,
                )

                if cfg.allow_same_bar_exit:
                    exit_eval = evaluate_exit(position, bar, 0, cfg)
                    if exit_eval is not None:
                        raw_exit, reason = exit_eval
                        trade = close_trade(position, bar.timestamp, raw_exit, reason, cfg)
                        # entry fee was already deducted, so add gross minus exit fee only.
                        exit_fee = trade.fees - position.entry_fee
                        equity += trade.gross_pnl - exit_fee
                        trades.append(trade)
                        position = None

                        peak_equity = max(peak_equity, equity)
                        if peak_equity > EPS:
                            drawdown = (peak_equity - equity) / peak_equity
                            max_drawdown = max(max_drawdown, drawdown)

    if position is not None:
        last = bars[-1]
        trade = close_trade(position, last.timestamp, last.close, "end_of_data", cfg)
        exit_fee = trade.fees - position.entry_fee
        equity += trade.gross_pnl - exit_fee
        trades.append(trade)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    r_values = [t.r_multiple for t in trades]

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > EPS else (
        math.inf if gross_profit > 0 else 0.0
    )

    expectancy_r = statistics.fmean(r_values) if r_values else 0.0
    avg_win_r = statistics.fmean([t.r_multiple for t in wins]) if wins else 0.0
    avg_loss_r = statistics.fmean([t.r_multiple for t in losses]) if losses else 0.0

    return BacktestResult(
        starting_equity=cfg.starting_equity,
        ending_equity=equity,
        net_profit=equity - cfg.starting_equity,
        return_pct=((equity / cfg.starting_equity) - 1.0) * 100.0,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown * 100.0,
        average_win_r=avg_win_r,
        average_loss_r=avg_loss_r,
        trades_detail=tuple(trades),
    )


def latest_signal(bars: Sequence[Bar], cfg: EngineConfig) -> Decision:
    engine = DeterministicEngine(cfg)
    indicators = engine.precompute(bars)
    return engine.decide(bars, indicators, len(bars) - 1)


def write_trades_csv(path: str | Path, trades: Sequence[Trade]) -> None:
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "side",
                "entry_timestamp",
                "exit_timestamp",
                "entry_price",
                "exit_price",
                "quantity",
                "gross_pnl",
                "fees",
                "net_pnl",
                "r_multiple",
                "exit_reason",
                "score",
            ],
        )
        writer.writeheader()
        for trade in trades:
            row = asdict(trade)
            row["side"] = trade.side.value
            writer.writerow(row)


def synthetic_bars(count: int = 500) -> list[Bar]:
    """
    Deterministic synthetic series for smoke tests.
    Not intended to demonstrate profitability.
    """
    bars: list[Bar] = []
    price = 100.0
    for i in range(count):
        drift = 0.08 if (i // 80) % 2 == 0 else -0.06
        wave = math.sin(i / 7.0) * 0.35
        open_ = price
        close = max(1.0, open_ + drift + wave)
        high = max(open_, close) + 0.30 + abs(math.sin(i / 5.0)) * 0.15
        low = min(open_, close) - 0.30 - abs(math.cos(i / 6.0)) * 0.15
        volume = 1_000.0 + 150.0 * math.sin(i / 11.0) + (i % 17) * 8.0
        bars.append(
            Bar(
                timestamp=str(i),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=max(1.0, volume),
            )
        )
        price = close
    return bars


def run_selftest() -> None:
    cfg = EngineConfig()
    cfg.validate()

    bars = synthetic_bars()
    for b in bars:
        b.validate()

    closes = [b.close for b in bars]
    e = ema(closes, 20)
    a = atr(bars, 14)

    assert len(e) == len(bars)
    assert len(a) == len(bars)
    assert math.isnan(e[18])
    assert math.isfinite(e[19])
    assert math.isnan(a[13])
    assert math.isfinite(a[14])

    # Causality check: appending future bars must not alter old indicator values.
    extended = bars + synthetic_bars(20)
    e2 = ema([b.close for b in extended], 20)
    a2 = atr(extended, 14)
    for i in range(len(bars)):
        if math.isfinite(e[i]):
            assert abs(e[i] - e2[i]) < 1e-12
        if math.isfinite(a[i]):
            assert abs(a[i] - a2[i]) < 1e-12

    engine = DeterministicEngine(cfg)
    ind = engine.precompute(bars)
    d1 = engine.decide(bars, ind, len(bars) - 1)
    d2 = engine.decide(bars, ind, len(bars) - 1)
    assert d1 == d2

    result = backtest(bars, cfg)
    assert math.isfinite(result.ending_equity)
    assert result.trades >= 0

    print("SELFTEST PASS")
    print(json.dumps(result.summary_dict(), indent=2, default=str))


def build_config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        atr_period=args.atr_period,
        volume_period=args.volume_period,
        pullback_atr_distance=args.pullback_atr_distance,
        max_extension_atr=args.max_extension_atr,
        min_volume_ratio=args.min_volume_ratio,
        min_body_atr=args.min_body_atr,
        max_body_atr=args.max_body_atr,
        min_atr_pct=args.min_atr_pct,
        max_atr_pct=args.max_atr_pct,
        stop_atr=args.stop_atr,
        target_r=args.target_r,
        max_hold_bars=args.max_hold_bars,
        risk_per_trade=args.risk_per_trade,
        max_leverage=args.max_leverage,
        fee_rate_per_side=args.fee_rate,
        slippage_rate_per_side=args.slippage_rate,
        starting_equity=args.starting_equity,
        trade_score=args.trade_score,
        allow_same_bar_exit=args.allow_same_bar_exit,
    )


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fast-ema", type=int, default=20)
    parser.add_argument("--slow-ema", type=int, default=50)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--volume-period", type=int, default=20)
    parser.add_argument("--pullback-atr-distance", type=float, default=0.45)
    parser.add_argument("--max-extension-atr", type=float, default=1.10)
    parser.add_argument("--min-volume-ratio", type=float, default=0.70)
    parser.add_argument("--min-body-atr", type=float, default=0.08)
    parser.add_argument("--max-body-atr", type=float, default=1.20)
    parser.add_argument("--min-atr-pct", type=float, default=0.002)
    parser.add_argument("--max-atr-pct", type=float, default=0.050)
    parser.add_argument("--trade-score", type=int, default=8)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=12)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--starting-equity", type=float, default=1000.0)
    parser.add_argument("--allow-same-bar-exit", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7-Lite deterministic trade engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="Run bar-by-bar backtest")
    p_backtest.add_argument("--csv", required=True)
    p_backtest.add_argument("--trades-out")
    add_common_config_args(p_backtest)

    p_signal = sub.add_parser("signal", help="Print latest deterministic signal")
    p_signal.add_argument("--csv", required=True)
    add_common_config_args(p_signal)

    sub.add_parser("selftest", help="Run built-in deterministic smoke tests")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.command == "selftest":
            run_selftest()
            return 0

        cfg = build_config_from_args(args)
        bars = load_csv(args.csv)

        if args.command == "signal":
            print(latest_signal(bars, cfg).to_json())
            return 0

        if args.command == "backtest":
            result = backtest(bars, cfg)
            print(json.dumps(result.summary_dict(), indent=2, default=str))
            if args.trades_out:
                write_trades_csv(args.trades_out, result.trades_detail)
                print(f"Trades written to: {args.trades_out}", file=sys.stderr)
            return 0

        raise RuntimeError(f"Unknown command: {args.command}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
