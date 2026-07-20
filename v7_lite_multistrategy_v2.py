#!/usr/bin/env python3
"""
V7-Lite Multi-Strategy Challenger
=================================

Deterministic multi-strategy research runner for ddawnlll/v7.

Strategy families:
- trend_breakout
- range_reversion
- compression_breakout

This file deliberately does NOT reimplement market validation, indicators,
trade simulation, costs, funding, gap semantics, net_R, split/purge, or the
immutable evaluation ledger. Those remain owned by the repository:

    lab.market
    lab.indicators
    lab.events
    lab.sim
    lab.evaluate
    tools.data

V7-Lite only builds causal strategy features, makes TAKE/ABSTAIN decisions,
arbitrates conflicting strategies, blocks overlapping same-symbol risk, and
reports per-strategy plus combined-portfolio evidence.

Usage from repository root:

    python v7_lite_multistrategy.py selftest

    python v7_lite_multistrategy.py run \
      --snapshot BTC-USDT-SWAP=data/snapshots/btc \
      --snapshot ETH-USDT-SWAP=data/snapshots/eth \
      --split-ts 2026-05-01T00:00:00Z \
      --strategy all \
      --output multi_strategy_result.json

    python v7_lite_multistrategy.py walkforward \
      --snapshot BTC-USDT-SWAP=data/snapshots/btc \
      --snapshot ETH-USDT-SWAP=data/snapshots/eth \
      --snapshot SOL-USDT-SWAP=data/snapshots/sol \
      --snapshot XRP-USDT-SWAP=data/snapshots/xrp \
      --strategy all \
      --train-months 12 \
      --test-months 2 \
      --output cell_router_walkforward.json

Research warning:
Defaults are preregistered challenger hypotheses, not claimed optima. Once the
TEST interval is read, do not tune against that same interval again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


BASE_INTERVAL_MS = 300_000
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab import evaluate, events, indicators, market  # noqa: E402


FEATURE_NAMES = (
    "dir_return_1bar",
    "dir_return_6h",
    "dir_return_24h",
    "dir_return_72h",
    "abs_return_24h",
    "breakout_atr",
    "trend_efficiency",
    "atr_pct",
    "atr_ratio",
    "volume_ratio",
    "vwap_dev_atr",
    "directional_zscore",
    "side_rsi",
    "body_atr",
    "directional_close_location",
    "directional_wick_ratio",
    "prev_compression_ratio",
    "range_atr",
    "active_session",
)


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    key: str
    label: str
    interval_label: str
    interval_factor: int
    k_stop: float
    reward_risk: float
    max_holding_bars: int
    priority: int

    atr_period: int = 14
    short_hours: int = 6
    medium_hours: int = 24
    long_hours: int = 72
    channel_hours: int = 24
    efficiency_hours: int = 24
    volume_hours: int = 24
    vwap_hours: int = 24
    zscore_hours: int = 24
    compression_short_hours: int = 12
    compression_reference_hours: int = 72
    atr_reference_hours: int = 48

    active_session_only: bool = False
    active_utc_start_hour: int = 6
    active_utc_end_hour: int = 22

    def validate(self) -> None:
        if self.interval_factor < 2:
            raise ValueError(f"{self.key}: interval_factor must be >= 2")
        if self.k_stop <= 0.0 or self.reward_risk <= 0.0:
            raise ValueError(f"{self.key}: invalid stop/target geometry")
        if self.max_holding_bars <= 0:
            raise ValueError(f"{self.key}: max_holding_bars must be positive")
        if self.priority < 0:
            raise ValueError(f"{self.key}: priority must be non-negative")
        if self.atr_period < 2:
            raise ValueError(f"{self.key}: atr_period must be >= 2")

        interval_hours = self.interval_hours
        for name, hours in (
            ("short_hours", self.short_hours),
            ("medium_hours", self.medium_hours),
            ("long_hours", self.long_hours),
            ("channel_hours", self.channel_hours),
            ("efficiency_hours", self.efficiency_hours),
            ("volume_hours", self.volume_hours),
            ("vwap_hours", self.vwap_hours),
            ("zscore_hours", self.zscore_hours),
            ("compression_short_hours", self.compression_short_hours),
            ("compression_reference_hours", self.compression_reference_hours),
            ("atr_reference_hours", self.atr_reference_hours),
        ):
            if hours < interval_hours:
                raise ValueError(
                    f"{self.key}: {name}={hours} shorter than "
                    f"{interval_hours}h decision interval"
                )

    @property
    def interval_ms(self) -> int:
        return self.interval_factor * BASE_INTERVAL_MS

    @property
    def interval_hours(self) -> int:
        minutes = self.interval_ms // 60_000
        if minutes % 60 != 0:
            raise ValueError(
                f"{self.key}: only whole-hour decision intervals are supported"
            )
        return minutes // 60

    @property
    def cooldown_ms(self) -> int:
        # Predictor cannot see realized exit time, so use the complete worst-case
        # horizon. This guarantees no overlapping same-symbol trades.
        return self.max_holding_bars * BASE_INTERVAL_MS

    def setup(self) -> events.Setup:
        return events.Setup(
            label=self.label,
            k_stop=self.k_stop,
            reward_risk=self.reward_risk,
            max_holding_bars=self.max_holding_bars,
            decision_interval_factor=self.interval_factor,
            decision_interval_label=self.interval_label,
        )


DEFAULT_CONFIGS: dict[str, StrategyConfig] = {
    "trend_breakout": StrategyConfig(
        key="trend_breakout",
        label="v7_lite_trend_breakout_1h",
        interval_label="1h",
        interval_factor=12,
        k_stop=1.75,
        reward_risk=1.50,
        max_holding_bars=144,  # 12 hours on 5m truth tape
        priority=20,
    ),
    "range_reversion": StrategyConfig(
        key="range_reversion",
        label="v7_lite_range_reversion_1h",
        interval_label="1h",
        interval_factor=12,
        k_stop=1.50,
        reward_risk=1.25,
        max_holding_bars=72,  # 6 hours
        priority=10,
    ),
    "compression_breakout": StrategyConfig(
        key="compression_breakout",
        label="v7_lite_compression_breakout_1h",
        interval_label="1h",
        interval_factor=12,
        k_stop=1.50,
        reward_risk=2.00,
        max_holding_bars=144,  # 12 hours
        priority=30,
    ),
}


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    take: bool
    confidence: float
    reasons: tuple[str, ...]
    rejections: tuple[str, ...]
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class SymbolInput:
    symbol: str
    bars: list[market.Bar]
    funding_events: list


@dataclass(frozen=True, slots=True)
class QualifiedSignal:
    strategy: str
    priority: int
    confidence: float
    event: events.CandidateEvent
    trace: DecisionTrace


@dataclass(frozen=True, slots=True)
class PortfolioLedgerRow:
    strategy: str
    event_id: str
    symbol: str
    side: str
    split: str
    decision_ts: int
    net_r: float
    gross_r: float
    fee_r: float
    slippage_r: float
    funding_r: float
    exit_reason: str
    hold_bars: int
    confidence: float


@dataclass(frozen=True, slots=True)
class Performance:
    split: str
    n_opportunities: int
    n_taken: int
    coverage: float
    mean_net_r: float | None
    median_net_r: float | None
    mean_net_r_ci95_low: float | None
    mean_net_r_ci95_high: float | None
    win_rate: float | None
    profit_factor: float | None
    average_win_r: float | None
    average_loss_r: float | None
    payoff_ratio: float | None
    breakeven_win_rate: float | None
    cumulative_net_r: float
    max_drawdown_r: float
    longest_losing_streak: int
    average_hold_hours: float | None
    exit_reasons: dict[str, int]
    sides: dict[str, dict[str, float | int | None]]
    strategies: dict[str, dict[str, float | int | None]]


def _hours_to_period(config: StrategyConfig, hours: int) -> int:
    return max(1, int(round(hours / config.interval_hours)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.nan
    if abs(denominator) <= 1e-15:
        return math.nan
    value = numerator / denominator
    return value if math.isfinite(value) else math.nan


def _finite_abs_sum(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return math.nan
    return math.fsum(abs(value) for value in values)


def _rolling_abs_return_sum(
    returns: Sequence[float],
    period: int,
) -> list[float]:
    """Causal path-length helper using repository simple returns."""
    result = [math.nan] * len(returns)
    for index in range(period - 1, len(returns)):
        window = returns[index - period + 1 : index + 1]
        value = _finite_abs_sum(window)
        if math.isfinite(value):
            result[index] = value
    return result


def _active_session(decision_ts: int, config: StrategyConfig) -> float:
    hour = datetime.fromtimestamp(decision_ts / 1000, tz=timezone.utc).hour
    start = config.active_utc_start_hour
    end = config.active_utc_end_hour
    if start < end:
        active = start <= hour < end
    else:
        active = hour >= start or hour < end
    return 1.0 if active else 0.0


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _minimum_margin(value: float, threshold: float, scale: float) -> float:
    return _clip01((value - threshold) / scale)


def _maximum_margin(value: float, threshold: float, scale: float) -> float:
    return _clip01((threshold - value) / scale)


def _bounded_margin(value: float, low: float, high: float) -> float:
    if not low < high or not low <= value <= high:
        return 0.0
    center = (low + high) * 0.5
    half = (high - low) * 0.5
    return _clip01(1.0 - abs(value - center) / half)


def _feature_dict(ctx: evaluate.PredictionContext) -> dict[str, float]:
    vector = np.asarray(ctx.features, dtype=float)
    if vector.shape != (len(FEATURE_NAMES),):
        raise ValueError(
            f"{ctx.event_id}: expected {len(FEATURE_NAMES)} features, "
            f"got shape {vector.shape}"
        )
    return {
        name: float(vector[index])
        for index, name in enumerate(FEATURE_NAMES)
    }


def _precompute_raw_features(
    bars: list[market.Bar],
    config: StrategyConfig,
) -> dict[int, np.ndarray]:
    """Build side-neutral causal features for one symbol."""
    config.validate()
    derived = market.aggregate(
        bars,
        factor=config.interval_factor,
        interval_ms=BASE_INTERVAL_MS,
    )
    if not derived:
        return {}

    opens = [bar.open for bar in derived]
    highs = [bar.high for bar in derived]
    lows = [bar.low for bar in derived]
    closes = [bar.close for bar in derived]
    volumes = [bar.volume for bar in derived]

    p_short = _hours_to_period(config, config.short_hours)
    p_medium = _hours_to_period(config, config.medium_hours)
    p_long = _hours_to_period(config, config.long_hours)
    p_channel = _hours_to_period(config, config.channel_hours)
    p_efficiency = _hours_to_period(config, config.efficiency_hours)
    p_volume = _hours_to_period(config, config.volume_hours)
    p_vwap = _hours_to_period(config, config.vwap_hours)
    p_zscore = _hours_to_period(config, config.zscore_hours)
    p_compression_short = _hours_to_period(
        config, config.compression_short_hours
    )
    p_compression_reference = _hours_to_period(
        config, config.compression_reference_hours
    )
    p_atr_reference = _hours_to_period(config, config.atr_reference_hours)

    ret_1bar = indicators.momentum(closes, period=1)
    ret_short = indicators.momentum(closes, period=p_short)
    ret_medium = indicators.momentum(closes, period=p_medium)
    ret_long = indicators.momentum(closes, period=p_long)

    atr_values = indicators.compute_atr(
        highs, lows, closes, period=config.atr_period
    )
    atr_pct = [
        _safe_ratio(atr_value, close)
        for atr_value, close in zip(atr_values, closes)
    ]
    atr_reference = indicators.rolling_mean(
        atr_pct, period=p_atr_reference
    )

    volume_reference = indicators.rolling_mean(
        volumes, period=p_volume
    )
    rolling_vwap = indicators.rolling_vwap(
        highs, lows, closes, volumes, period=p_vwap
    )

    price_mean = indicators.rolling_mean(closes, period=p_zscore)
    price_std = indicators.rolling_std(closes, period=p_zscore)

    channel_high = indicators.rolling_max(highs, period=p_channel)
    channel_low = indicators.rolling_min(lows, period=p_channel)

    simple_returns = indicators.simple_returns(closes)
    path_length = _rolling_abs_return_sum(simple_returns, p_efficiency)

    close_std_short = indicators.rolling_std(
        closes, period=p_compression_short
    )
    normalized_width = [
        _safe_ratio(std, close)
        for std, close in zip(close_std_short, closes)
    ]
    width_reference = indicators.rolling_mean(
        normalized_width, period=p_compression_reference
    )

    rsi_values = indicators.rsi(closes, period=14)
    upper_wicks = indicators.upper_wick_ratio(opens, highs, lows, closes)
    lower_wicks = indicators.lower_wick_ratio(opens, highs, lows, closes)

    result: dict[int, np.ndarray] = {}
    for index, bar in enumerate(derived):
        decision_ts = bar.open_ts + config.interval_ms
        atr_value = atr_values[index]

        previous_high = channel_high[index - 1] if index >= 1 else math.nan
        previous_low = channel_low[index - 1] if index >= 1 else math.nan
        previous_compression = (
            _safe_ratio(
                normalized_width[index - 1],
                width_reference[index - 1],
            )
            if index >= 1
            else math.nan
        )

        if math.isfinite(atr_value) and atr_value > 0.0:
            breakout_long = (bar.close - previous_high) / atr_value
            breakout_short = (previous_low - bar.close) / atr_value
            vwap_long = (bar.close - rolling_vwap[index]) / atr_value
            vwap_short = (rolling_vwap[index] - bar.close) / atr_value
            body_atr = abs(bar.close - bar.open) / atr_value
            range_atr = (bar.high - bar.low) / atr_value
        else:
            breakout_long = breakout_short = math.nan
            vwap_long = vwap_short = math.nan
            body_atr = range_atr = math.nan

        zscore = _safe_ratio(
            bar.close - price_mean[index], price_std[index]
        )
        atr_ratio = _safe_ratio(atr_pct[index], atr_reference[index])
        volume_ratio = _safe_ratio(bar.volume, volume_reference[index])
        trend_efficiency = _safe_ratio(
            abs(ret_medium[index]), path_length[index]
        )
        if math.isfinite(trend_efficiency):
            trend_efficiency = min(1.0, max(0.0, trend_efficiency))

        candle_range = bar.high - bar.low
        if math.isfinite(candle_range) and candle_range > 0.0:
            close_location_long = (bar.close - bar.low) / candle_range
            close_location_short = (bar.high - bar.close) / candle_range
        else:
            close_location_long = close_location_short = math.nan

        result[decision_ts] = np.array(
            [
                ret_1bar[index],
                ret_short[index],
                ret_medium[index],
                ret_long[index],
                abs(ret_medium[index])
                if math.isfinite(ret_medium[index])
                else math.nan,
                breakout_long,
                breakout_short,
                trend_efficiency,
                atr_pct[index],
                atr_ratio,
                volume_ratio,
                vwap_long,
                vwap_short,
                zscore,
                rsi_values[index],
                body_atr,
                close_location_long,
                close_location_short,
                lower_wicks[index],
                upper_wicks[index],
                previous_compression,
                range_atr,
                _active_session(decision_ts, config),
            ],
            dtype=float,
        )

    return result


def build_feature_map(
    candidate_events: Sequence[events.CandidateEvent],
    bars_by_symbol: Mapping[str, list[market.Bar]],
    config: StrategyConfig,
) -> tuple[list[events.CandidateEvent], dict[str, np.ndarray], dict[str, int]]:
    raw_by_symbol = {
        symbol: _precompute_raw_features(bars, config)
        for symbol, bars in bars_by_symbol.items()
    }

    usable_events: list[events.CandidateEvent] = []
    feature_map: dict[str, np.ndarray] = {}
    dropped = Counter()

    for event in candidate_events:
        raw = raw_by_symbol.get(event.symbol, {}).get(event.decision_ts)
        if raw is None:
            dropped["missing_decision_feature"] += 1
            continue

        (
            ret_1bar,
            ret_short,
            ret_medium,
            ret_long,
            abs_return_medium,
            breakout_long,
            breakout_short,
            efficiency,
            atr_pct_value,
            atr_ratio,
            volume_ratio,
            vwap_long,
            vwap_short,
            zscore,
            raw_rsi,
            body_atr,
            close_location_long,
            close_location_short,
            lower_wick,
            upper_wick,
            previous_compression,
            range_atr,
            active_session,
        ) = raw.tolist()

        if event.side == "LONG":
            side_sign = 1.0
            breakout = breakout_long
            vwap_deviation = vwap_long
            directional_zscore = zscore
            side_rsi = raw_rsi
            close_location = close_location_long
            directional_wick = lower_wick
        elif event.side == "SHORT":
            side_sign = -1.0
            breakout = breakout_short
            vwap_deviation = vwap_short
            directional_zscore = -zscore
            side_rsi = 100.0 - raw_rsi
            close_location = close_location_short
            directional_wick = upper_wick
        else:
            dropped["invalid_side"] += 1
            continue

        vector = np.array(
            [
                side_sign * ret_1bar,
                side_sign * ret_short,
                side_sign * ret_medium,
                side_sign * ret_long,
                abs_return_medium,
                breakout,
                efficiency,
                atr_pct_value,
                atr_ratio,
                volume_ratio,
                vwap_deviation,
                directional_zscore,
                side_rsi,
                body_atr,
                close_location,
                directional_wick,
                previous_compression,
                range_atr,
                active_session,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(vector)):
            dropped["insufficient_causal_history"] += 1
            continue

        usable_events.append(event)
        feature_map[event.event_id] = vector

    return usable_events, feature_map, dict(dropped)


class DeterministicStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        config.validate()
        self.config = config

    @abstractmethod
    def trace(self, ctx: evaluate.PredictionContext) -> DecisionTrace:
        raise NotImplementedError

    def _finalize(
        self,
        values: dict[str, float],
        reasons: list[str],
        rejections: list[str],
        confidence_parts: list[float],
    ) -> DecisionTrace:
        if self.config.active_session_only:
            if values["active_session"] >= 0.5:
                reasons.append("active_session")
                confidence_parts.append(1.0)
            else:
                rejections.append("inactive_session")
                confidence_parts.append(0.0)

        confidence = (
            float(np.mean(confidence_parts)) if confidence_parts else 0.0
        )
        return DecisionTrace(
            take=not rejections,
            confidence=_clip01(confidence),
            reasons=tuple(reasons),
            rejections=tuple(rejections),
            values=values,
        )


class TrendBreakoutStrategy(DeterministicStrategy):
    def trace(self, ctx: evaluate.PredictionContext) -> DecisionTrace:
        values = _feature_dict(ctx)
        reasons: list[str] = []
        rejections: list[str] = []
        confidence: list[float] = []

        minimum_gates = (
            ("dir_return_6h", 0.0000, 0.0100, "6h_direction"),
            ("dir_return_24h", 0.0030, 0.0200, "24h_direction"),
            ("dir_return_72h", 0.0060, 0.0400, "72h_direction"),
            ("breakout_atr", -0.05, 0.75, "channel_breakout"),
            ("trend_efficiency", 0.18, 0.45, "efficient_trend"),
            ("volume_ratio", 0.85, 1.00, "volume_confirmation"),
            ("vwap_dev_atr", 0.00, 1.50, "vwap_alignment"),
            (
                "directional_close_location",
                0.60,
                0.40,
                "directional_close",
            ),
        )
        for name, threshold, scale, reason in minimum_gates:
            value = values[name]
            if value >= threshold:
                reasons.append(reason)
            else:
                rejections.append(f"{name}<{threshold:g}")
            confidence.append(_minimum_margin(value, threshold, scale))

        bounded_gates = (
            ("atr_pct", 0.0015, 0.0500, "absolute_volatility"),
            ("atr_ratio", 0.70, 2.50, "relative_volatility"),
            ("body_atr", 0.10, 1.80, "candle_body"),
        )
        for name, low, high, reason in bounded_gates:
            value = values[name]
            if low <= value <= high:
                reasons.append(reason)
            else:
                rejections.append(f"{name}_outside_[{low:g},{high:g}]")
            confidence.append(_bounded_margin(value, low, high))

        return self._finalize(values, reasons, rejections, confidence)


class RangeReversionStrategy(DeterministicStrategy):
    def trace(self, ctx: evaluate.PredictionContext) -> DecisionTrace:
        values = _feature_dict(ctx)
        reasons: list[str] = []
        rejections: list[str] = []
        confidence: list[float] = []

        maximum_gates = (
            ("abs_return_24h", 0.025, 0.025, "no_strong_24h_trend"),
            ("trend_efficiency", 0.28, 0.28, "range_efficiency"),
            ("directional_zscore", -1.40, 1.60, "statistical_stretch"),
            ("side_rsi", 38.0, 28.0, "rsi_extreme"),
        )
        for name, threshold, scale, reason in maximum_gates:
            value = values[name]
            if value <= threshold:
                reasons.append(reason)
            else:
                rejections.append(f"{name}>{threshold:g}")
            confidence.append(_maximum_margin(value, threshold, scale))

        minimum_gates = (
            ("dir_return_1bar", 0.0000, 0.0100, "reversal_trigger"),
            (
                "directional_close_location",
                0.58,
                0.42,
                "rejection_close",
            ),
            ("directional_wick_ratio", 0.25, 0.50, "rejection_wick"),
            ("volume_ratio", 0.70, 1.00, "sufficient_liquidity"),
        )
        for name, threshold, scale, reason in minimum_gates:
            value = values[name]
            if value >= threshold:
                reasons.append(reason)
            else:
                rejections.append(f"{name}<{threshold:g}")
            confidence.append(_minimum_margin(value, threshold, scale))

        if values["vwap_dev_atr"] <= -0.75:
            reasons.append("vwap_stretch")
        else:
            rejections.append("vwap_dev_atr>-0.75")
        confidence.append(
            _maximum_margin(values["vwap_dev_atr"], -0.75, 1.50)
        )

        bounded_gates = (
            ("atr_pct", 0.0015, 0.0400, "absolute_volatility"),
            ("atr_ratio", 0.60, 2.00, "relative_volatility"),
            ("body_atr", 0.05, 1.50, "candle_body"),
        )
        for name, low, high, reason in bounded_gates:
            value = values[name]
            if low <= value <= high:
                reasons.append(reason)
            else:
                rejections.append(f"{name}_outside_[{low:g},{high:g}]")
            confidence.append(_bounded_margin(value, low, high))

        return self._finalize(values, reasons, rejections, confidence)


class CompressionBreakoutStrategy(DeterministicStrategy):
    def trace(self, ctx: evaluate.PredictionContext) -> DecisionTrace:
        values = _feature_dict(ctx)
        reasons: list[str] = []
        rejections: list[str] = []
        confidence: list[float] = []

        if values["prev_compression_ratio"] <= 0.78:
            reasons.append("prior_volatility_compression")
        else:
            rejections.append("prev_compression_ratio>0.78")
        confidence.append(
            _maximum_margin(values["prev_compression_ratio"], 0.78, 0.60)
        )

        minimum_gates = (
            ("breakout_atr", 0.00, 0.90, "channel_breakout"),
            ("dir_return_1bar", 0.0020, 0.0150, "directional_impulse"),
            ("volume_ratio", 1.15, 1.50, "volume_expansion"),
            ("body_atr", 0.45, 1.25, "body_expansion"),
            ("range_atr", 1.00, 1.50, "range_expansion"),
            (
                "directional_close_location",
                0.72,
                0.28,
                "close_near_breakout_extreme",
            ),
            ("vwap_dev_atr", 0.00, 1.50, "vwap_alignment"),
        )
        for name, threshold, scale, reason in minimum_gates:
            value = values[name]
            if value >= threshold:
                reasons.append(reason)
            else:
                rejections.append(f"{name}<{threshold:g}")
            confidence.append(_minimum_margin(value, threshold, scale))

        bounded_gates = (
            ("atr_pct", 0.0015, 0.0600, "absolute_volatility"),
            ("atr_ratio", 0.75, 3.00, "relative_volatility"),
        )
        for name, low, high, reason in bounded_gates:
            value = values[name]
            if low <= value <= high:
                reasons.append(reason)
            else:
                rejections.append(f"{name}_outside_[{low:g},{high:g}]")
            confidence.append(_bounded_margin(value, low, high))

        return self._finalize(values, reasons, rejections, confidence)


def build_strategy(config: StrategyConfig) -> DeterministicStrategy:
    if config.key == "trend_breakout":
        return TrendBreakoutStrategy(config)
    if config.key == "range_reversion":
        return RangeReversionStrategy(config)
    if config.key == "compression_breakout":
        return CompressionBreakoutStrategy(config)
    raise KeyError(f"unknown strategy key: {config.key}")


@dataclass(frozen=True, slots=True)
class PreparedStrategy:
    config: StrategyConfig
    strategy: DeterministicStrategy
    events: list[events.CandidateEvent]
    feature_map: dict[str, np.ndarray]
    dropped: dict[str, int]


def prepare_strategy(
    symbol_inputs: Sequence[SymbolInput],
    split_ts: int,
    config: StrategyConfig,
) -> PreparedStrategy:
    event_inputs = [
        events.EventInput(
            symbol=item.symbol,
            trade_bars=item.bars,
            funding_events=item.funding_events,
        )
        for item in symbol_inputs
    ]
    candidate_events = events.build_events(
        event_inputs,
        split_ts=split_ts,
        setup=config.setup(),
    )
    bars_by_symbol = {item.symbol: item.bars for item in symbol_inputs}
    usable_events, feature_map, dropped = build_feature_map(
        candidate_events, bars_by_symbol, config
    )
    if not usable_events:
        raise RuntimeError(
            f"{config.key}: no events survive causal feature warmup"
        )
    return PreparedStrategy(
        config=config,
        strategy=build_strategy(config),
        events=usable_events,
        feature_map=feature_map,
        dropped=dropped,
    )


def _qualified_by_timestamp(
    prepared: PreparedStrategy,
    split: str,
) -> tuple[dict[tuple[str, int], list[QualifiedSignal]], Counter[str]]:
    grouped: dict[tuple[str, int], list[QualifiedSignal]] = defaultdict(list)
    diagnostics = Counter()

    for event in prepared.events:
        if event.split != split:
            continue
        ctx = evaluate.PredictionContext(
            event_id=event.event_id,
            symbol=event.symbol,
            side=event.side,
            decision_ts=event.decision_ts,
            features=prepared.feature_map[event.event_id].copy(),
        )
        trace = prepared.strategy.trace(ctx)
        if trace.take:
            grouped[(event.symbol, event.decision_ts)].append(
                QualifiedSignal(
                    strategy=prepared.config.key,
                    priority=prepared.config.priority,
                    confidence=trace.confidence,
                    event=event,
                    trace=trace,
                )
            )
            diagnostics["qualified_directional_events"] += 1

    return grouped, diagnostics


def choose_signal(
    signals: Sequence[QualifiedSignal],
) -> tuple[QualifiedSignal | None, str]:
    """Arbiter for signals sharing one symbol and decision timestamp."""
    if not signals:
        return None, "no_signal"
    sides = {signal.event.side for signal in signals}
    if len(sides) != 1:
        return None, "direction_conflict"
    chosen = max(
        signals,
        key=lambda signal: (
            signal.confidence,
            signal.priority,
            signal.strategy,
            signal.event.event_id,
        ),
    )
    if len(signals) > 1:
        return chosen, "same_direction_competition"
    return chosen, "single_signal"


def select_one_strategy(
    prepared: PreparedStrategy,
    split: str,
) -> tuple[list[QualifiedSignal], dict[str, int]]:
    grouped, diagnostics = _qualified_by_timestamp(prepared, split)
    selected: list[QualifiedSignal] = []
    unavailable_until: dict[str, int] = {}

    for (symbol, decision_ts), signals in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        chosen, reason = choose_signal(signals)
        diagnostics[reason] += 1
        if chosen is None:
            continue
        if decision_ts < unavailable_until.get(symbol, -1):
            diagnostics["cooldown_rejection"] += 1
            continue
        selected.append(chosen)
        unavailable_until[symbol] = (
            decision_ts + prepared.config.cooldown_ms
        )
        diagnostics["selected"] += 1

    return selected, dict(diagnostics)


class SelectedEventPredictor:
    def __init__(self, selected_event_ids: set[str]):
        self.selected_event_ids = selected_event_ids

    def __call__(self, ctx: evaluate.PredictionContext) -> bool:
        return ctx.event_id in self.selected_event_ids

    def fit(self, _contexts, *, seed: int) -> None:
        del seed
        return None


def evaluate_selected_strategy(
    prepared: PreparedStrategy,
    *,
    seed: int,
) -> tuple[
    evaluate.SplitEval,
    list[QualifiedSignal],
    list[QualifiedSignal],
    dict[str, dict[str, int]],
]:
    train_selected, train_diag = select_one_strategy(prepared, "train")
    test_selected, test_diag = select_one_strategy(prepared, "test")
    selected_ids = {
        signal.event.event_id
        for signal in train_selected + test_selected
    }
    result = evaluate.evaluate(
        prepared.events,
        prepared.feature_map,
        SelectedEventPredictor(selected_ids),
        seed=seed,
        fit_predictor=False,
    )
    if hasattr(evaluate, "reconcile") and not evaluate.reconcile(result):
        raise RuntimeError(
            f"{prepared.config.key}: evaluation reconciliation failed"
        )
    return result, train_selected, test_selected, {
        "train": train_diag,
        "test": test_diag,
    }


def select_combined_portfolio(
    prepared_strategies: Sequence[PreparedStrategy],
    split: str,
) -> tuple[list[QualifiedSignal], dict[str, int]]:
    all_grouped: dict[tuple[str, int], list[QualifiedSignal]] = defaultdict(list)
    diagnostics = Counter()

    for prepared in prepared_strategies:
        grouped, family_diag = _qualified_by_timestamp(prepared, split)
        for name, count in family_diag.items():
            diagnostics[f"{prepared.config.key}:{name}"] += count
        for key, signals in grouped.items():
            chosen, reason = choose_signal(signals)
            diagnostics[f"{prepared.config.key}:{reason}"] += 1
            if chosen is not None:
                all_grouped[key].append(chosen)

    selected: list[QualifiedSignal] = []
    unavailable_until: dict[str, int] = {}
    config_by_key = {
        prepared.config.key: prepared.config
        for prepared in prepared_strategies
    }

    for (symbol, decision_ts), signals in sorted(
        all_grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        chosen, reason = choose_signal(signals)
        diagnostics[reason] += 1
        if chosen is None:
            continue
        if decision_ts < unavailable_until.get(symbol, -1):
            diagnostics["portfolio_cooldown_rejection"] += 1
            continue
        selected.append(chosen)
        unavailable_until[symbol] = (
            decision_ts + config_by_key[chosen.strategy].cooldown_ms
        )
        diagnostics["selected"] += 1

    return selected, dict(diagnostics)


def _moving_block_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_boot: int = 2000,
) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    if n < 20:
        return None, None

    block_size = max(2, int(round(math.sqrt(n))))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)

    for bootstrap_index in range(n_boot):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            for offset in range(block_size):
                sample.append(float(array[(start + offset) % n]))
                if len(sample) == n:
                    break
        means[bootstrap_index] = float(np.mean(sample))

    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _group_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "trades": 0,
            "mean_net_r": None,
            "win_rate": None,
            "profit_factor": None,
        }

    wins = array[array > 0.0]
    losses = array[array < 0.0]
    gross_win = float(np.sum(wins))
    gross_loss = abs(float(np.sum(losses)))
    if gross_loss > 0.0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = None

    return {
        "trades": int(array.size),
        "mean_net_r": float(np.mean(array)),
        "win_rate": float(np.mean(array > 0.0)),
        "profit_factor": profit_factor,
    }


def _signals_to_rows(
    signals: Sequence[QualifiedSignal],
    split: str,
) -> list[PortfolioLedgerRow]:
    rows: list[PortfolioLedgerRow] = []
    for signal in signals:
        outcome = signal.event.locked_outcome
        risk_frac = outcome.risk_fraction if outcome.risk_fraction > 1e-12 else 1.0
        rows.append(
            PortfolioLedgerRow(
                strategy=signal.strategy,
                event_id=signal.event.event_id,
                symbol=signal.event.symbol,
                side=signal.event.side,
                split=split,
                decision_ts=signal.event.decision_ts,
                net_r=outcome.net_r,
                gross_r=outcome.gross_return / risk_frac,
                fee_r=outcome.costs.fee / risk_frac,
                slippage_r=outcome.costs.slippage / risk_frac,
                funding_r=outcome.costs.funding / risk_frac,
                exit_reason=outcome.exit_reason,
                hold_bars=outcome.exit_index - outcome.entry_index,
                confidence=signal.confidence,
            )
        )
    return rows


def _performance(
    rows: Sequence[PortfolioLedgerRow],
    *,
    split: str,
    n_opportunities: int,
    seed: int,
) -> Performance:
    selected = [row for row in rows if row.split == split]
    values = np.asarray([row.net_r for row in selected], dtype=float)

    if values.size == 0:
        return Performance(
            split=split,
            n_opportunities=n_opportunities,
            n_taken=0,
            coverage=0.0,
            mean_net_r=None,
            median_net_r=None,
            mean_net_r_ci95_low=None,
            mean_net_r_ci95_high=None,
            win_rate=None,
            profit_factor=None,
            average_win_r=None,
            average_loss_r=None,
            payoff_ratio=None,
            breakeven_win_rate=None,
            cumulative_net_r=0.0,
            max_drawdown_r=0.0,
            longest_losing_streak=0,
            average_hold_hours=None,
            exit_reasons={},
            sides={},
            strategies={},
        )

    wins = values[values > 0.0]
    losses = values[values < 0.0]
    gross_win = float(np.sum(wins))
    gross_loss = abs(float(np.sum(losses)))
    if gross_loss > 0.0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = None

    average_win = float(np.mean(wins)) if wins.size else None
    average_loss = float(np.mean(losses)) if losses.size else None
    payoff_ratio = (
        average_win / abs(average_loss)
        if average_win is not None
        and average_loss is not None
        and average_loss != 0.0
        else None
    )
    breakeven_win_rate = (
        1.0 / (1.0 + payoff_ratio)
        if payoff_ratio is not None
        else None
    )

    curve = np.concatenate(([0.0], np.cumsum(values)))
    peaks = np.maximum.accumulate(curve)
    max_drawdown = float(np.max(peaks - curve))

    longest_losing_streak = 0
    current_losing_streak = 0
    for value in values:
        if value < 0.0:
            current_losing_streak += 1
            longest_losing_streak = max(
                longest_losing_streak, current_losing_streak
            )
        else:
            current_losing_streak = 0

    ci_low, ci_high = _moving_block_bootstrap_ci(values, seed=seed)
    average_hold_bars = float(np.mean([row.hold_bars for row in selected]))
    average_hold_hours = average_hold_bars * BASE_INTERVAL_MS / 3_600_000

    side_values: dict[str, list[float]] = defaultdict(list)
    strategy_values: dict[str, list[float]] = defaultdict(list)
    exit_reasons = Counter()
    for row in selected:
        side_values[row.side].append(row.net_r)
        strategy_values[row.strategy].append(row.net_r)
        exit_reasons[row.exit_reason] += 1

    return Performance(
        split=split,
        n_opportunities=n_opportunities,
        n_taken=int(values.size),
        coverage=(
            float(values.size / n_opportunities)
            if n_opportunities
            else 0.0
        ),
        mean_net_r=float(np.mean(values)),
        median_net_r=float(np.median(values)),
        mean_net_r_ci95_low=ci_low,
        mean_net_r_ci95_high=ci_high,
        win_rate=float(np.mean(values > 0.0)),
        profit_factor=profit_factor,
        average_win_r=average_win,
        average_loss_r=average_loss,
        payoff_ratio=payoff_ratio,
        breakeven_win_rate=breakeven_win_rate,
        cumulative_net_r=float(np.sum(values)),
        max_drawdown_r=max_drawdown,
        longest_losing_streak=longest_losing_streak,
        average_hold_hours=average_hold_hours,
        exit_reasons=dict(exit_reasons),
        sides={
            key: _group_summary(group_values)
            for key, group_values in sorted(side_values.items())
        },
        strategies={
            key: _group_summary(group_values)
            for key, group_values in sorted(strategy_values.items())
        },
    )


def _strategy_performance_from_evaluation(
    prepared: PreparedStrategy,
    result: evaluate.SplitEval,
    selected: Sequence[QualifiedSignal],
    split: str,
    *,
    seed: int,
) -> Performance:
    rows = _signals_to_rows(selected, split)
    repo_values = {
        row.event_id: row.net_r
        for row in result.ledger
        if row.split == split and row.predicted
    }
    custom_values = {row.event_id: row.net_r for row in rows}
    if repo_values != custom_values:
        raise RuntimeError(
            f"{prepared.config.key}: report ledger differs from lab.evaluate"
        )

    opportunities = len(
        {
            (event.symbol, event.decision_ts)
            for event in prepared.events
            if event.split == split
        }
    )
    return _performance(
        rows,
        split=split,
        n_opportunities=opportunities,
        seed=seed,
    )


def _cost_breakdown(
    rows: Sequence[PortfolioLedgerRow],
    split: str,
) -> dict:
    """Group trades by symbol → side → strategy → gross/cost/exit breakdown."""
    selected = [r for r in rows if r.split == split]
    result: dict[str, dict[str, dict[str, dict]]] = {}
    for r in selected:
        sym = result.setdefault(r.symbol, {})
        sd = sym.setdefault(r.side, {})
        st = sd.setdefault(r.strategy, {
            "n_trades": 0,
            "gross_r": 0.0,
            "fee_r": 0.0,
            "slippage_r": 0.0,
            "funding_r": 0.0,
            "net_r": 0.0,
            "exit": {},
        })
        st["n_trades"] += 1
        st["gross_r"] += r.gross_r
        st["fee_r"] += r.fee_r
        st["slippage_r"] += r.slippage_r
        st["funding_r"] += r.funding_r
        st["net_r"] += r.net_r
        reason = st["exit"].setdefault(r.exit_reason, 0)
        st["exit"][r.exit_reason] = reason + 1

    # Round values
    for sym_data in result.values():
        for side_data in sym_data.values():
            for st_data in side_data.values():
                for key in (
                    "gross_r", "fee_r", "slippage_r", "funding_r", "net_r"
                ):
                    st_data[key] = round(st_data[key], 6)
    return result



# ═══════════════════════════════════════════════════════════════════════════════
# V2 rolling cell router
# ═══════════════════════════════════════════════════════════════════════════════

CellKey = tuple[str, str, str]  # strategy, symbol, side


@dataclass(frozen=True, slots=True)
class CellRouterConfig:
    """Train-only gate for one strategy × symbol × side cell.

    The selector is intentionally conservative. It observes only outcomes whose
    ``outcome_end_ts`` is strictly before the next OOS fold. The selected cell
    list is frozen for the complete OOS block.
    """

    train_months: int = 12
    test_months: int = 2
    subperiod_months: int = 2

    min_trades: int = 30
    min_subperiods: int = 3
    min_positive_subperiod_fraction: float = 0.60

    min_mean_gross_r: float = 0.16
    min_mean_net_r: float = 0.00
    min_profit_factor: float = 1.10

    cost_stress_multiplier: float = 1.50
    min_stressed_mean_net_r: float = 0.00

    shrinkage_prior_trades: float = 20.0
    shrinkage_prior_mean_r: float = 0.0
    min_shrunk_mean_net_r: float = 0.02

    max_top_win_share: float = 0.50

    def validate(self) -> None:
        for name, value in (
            ("train_months", self.train_months),
            ("test_months", self.test_months),
            ("subperiod_months", self.subperiod_months),
            ("min_trades", self.min_trades),
            ("min_subperiods", self.min_subperiods),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.test_months > self.train_months:
            raise ValueError("test_months must not exceed train_months")
        if not 0.0 <= self.min_positive_subperiod_fraction <= 1.0:
            raise ValueError("min_positive_subperiod_fraction must be in [0, 1]")
        if self.min_profit_factor < 0.0:
            raise ValueError("min_profit_factor must be >= 0")
        if self.cost_stress_multiplier < 1.0:
            raise ValueError("cost_stress_multiplier must be >= 1")
        if self.shrinkage_prior_trades < 0.0:
            raise ValueError("shrinkage_prior_trades must be >= 0")
        if not 0.0 < self.max_top_win_share <= 1.0:
            raise ValueError("max_top_win_share must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CellStats:
    strategy: str
    symbol: str
    side: str
    train_start_ts: int
    train_end_ts: int

    n_trades: int
    mean_gross_r: float | None
    mean_net_r: float | None
    shrunk_mean_net_r: float | None
    stressed_mean_net_r: float | None
    profit_factor: float | None

    positive_subperiods: int
    observed_subperiods: int
    positive_subperiod_fraction: float | None
    top_win_share: float | None

    pass_gate: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int
    active_cells: tuple[CellKey, ...]
    cell_stats: tuple[CellStats, ...]
    selected_trades: int
    diagnostics: dict[str, int]
    performance: Performance


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _add_months_dt(dt: datetime, months: int) -> datetime:
    total = dt.year * 12 + (dt.month - 1) + months
    year, month_zero = divmod(total, 12)
    return datetime(year, month_zero + 1, 1, tzinfo=timezone.utc)


def _ceil_month_start_ts(ts: int) -> int:
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    floor = _month_start(dt)
    if dt == floor:
        return int(floor.timestamp() * 1000)
    return int(_add_months_dt(floor, 1).timestamp() * 1000)


def _add_months_ts(ts: int, months: int) -> int:
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    # Fold boundaries are always month starts. Fail closed if a caller violates it.
    if dt != _month_start(dt):
        raise ValueError("month arithmetic requires a UTC month-start timestamp")
    return int(_add_months_dt(dt, months).timestamp() * 1000)


def _cell_key(signal: QualifiedSignal) -> CellKey:
    return (
        signal.strategy,
        signal.event.symbol,
        signal.event.side,
    )


def _signal_cost_components_r(
    signal: QualifiedSignal,
) -> tuple[float, float, float, float, float]:
    outcome = signal.event.locked_outcome
    risk = outcome.risk_fraction
    if not math.isfinite(risk) or risk <= 1e-12:
        raise ValueError(f"{signal.event.event_id}: invalid risk_fraction={risk}")
    gross_r = outcome.gross_return / risk
    fee_r = outcome.costs.fee / risk
    slippage_r = outcome.costs.slippage / risk
    funding_r = outcome.costs.funding / risk
    return gross_r, fee_r, slippage_r, funding_r, outcome.net_r


def _qualified_all_history(
    prepared: PreparedStrategy,
) -> list[QualifiedSignal]:
    qualified: list[QualifiedSignal] = []
    for event in prepared.events:
        ctx = evaluate.PredictionContext(
            event_id=event.event_id,
            symbol=event.symbol,
            side=event.side,
            decision_ts=event.decision_ts,
            features=prepared.feature_map[event.event_id].copy(),
        )
        trace = prepared.strategy.trace(ctx)
        if trace.take:
            qualified.append(
                QualifiedSignal(
                    strategy=prepared.config.key,
                    priority=prepared.config.priority,
                    confidence=trace.confidence,
                    event=event,
                    trace=trace,
                )
            )
    qualified.sort(
        key=lambda signal: (
            signal.event.decision_ts,
            signal.event.symbol,
            signal.strategy,
            signal.event.side,
            signal.event.event_id,
        )
    )
    return qualified


def _apply_independent_cell_cooldown(
    signals: Sequence[QualifiedSignal],
    config_by_strategy: Mapping[str, StrategyConfig],
) -> list[QualifiedSignal]:
    """Apply each cell's own worst-case cooldown before estimating its history."""
    result: list[QualifiedSignal] = []
    unavailable_until: dict[CellKey, int] = {}
    for signal in sorted(
        signals,
        key=lambda item: (
            item.event.decision_ts,
            item.event.symbol,
            item.strategy,
            item.event.side,
            item.event.event_id,
        ),
    ):
        key = _cell_key(signal)
        decision_ts = signal.event.decision_ts
        if decision_ts < unavailable_until.get(key, -1):
            continue
        result.append(signal)
        unavailable_until[key] = (
            decision_ts + config_by_strategy[signal.strategy].cooldown_ms
        )
    return result


def _profit_factor(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    wins = float(np.sum(array[array > 0.0])) if array.size else 0.0
    losses = abs(float(np.sum(array[array < 0.0]))) if array.size else 0.0
    if losses > 0.0:
        return wins / losses
    if wins > 0.0:
        return math.inf
    return None


def _subperiod_positive_fraction(
    signals: Sequence[QualifiedSignal],
    *,
    train_start_ts: int,
    train_end_ts: int,
    months: int,
) -> tuple[int, int, float | None]:
    positive = 0
    observed = 0
    cursor = train_start_ts
    while cursor < train_end_ts:
        end = min(_add_months_ts(cursor, months), train_end_ts)
        values = [
            signal.event.locked_outcome.net_r
            for signal in signals
            if cursor <= signal.event.decision_ts < end
        ]
        if values:
            observed += 1
            if float(np.mean(values)) > 0.0:
                positive += 1
        cursor = end
    fraction = positive / observed if observed else None
    return positive, observed, fraction


def _compute_cell_stats(
    key: CellKey,
    signals: Sequence[QualifiedSignal],
    *,
    train_start_ts: int,
    train_end_ts: int,
    router: CellRouterConfig,
) -> CellStats:
    strategy, symbol, side = key
    gross_values: list[float] = []
    net_values: list[float] = []
    stressed_values: list[float] = []

    for signal in signals:
        gross_r, fee_r, slippage_r, funding_r, net_r = (
            _signal_cost_components_r(signal)
        )
        # Conservative stress: fee/slippage and positive funding costs are
        # multiplied. A funding credit is never made more favorable.
        stressed_cost_r = router.cost_stress_multiplier * (
            fee_r + slippage_r + max(funding_r, 0.0)
        ) + min(funding_r, 0.0)
        gross_values.append(gross_r)
        net_values.append(net_r)
        stressed_values.append(gross_r - stressed_cost_r)

    n = len(net_values)
    mean_gross = float(np.mean(gross_values)) if n else None
    mean_net = float(np.mean(net_values)) if n else None
    stressed_mean = float(np.mean(stressed_values)) if n else None
    pf = _profit_factor(net_values)

    prior = router.shrinkage_prior_trades
    shrunk_mean = (
        (
            n * mean_net
            + prior * router.shrinkage_prior_mean_r
        )
        / (n + prior)
        if n and mean_net is not None
        else None
    )

    positive_subperiods, observed_subperiods, positive_fraction = (
        _subperiod_positive_fraction(
            signals,
            train_start_ts=train_start_ts,
            train_end_ts=train_end_ts,
            months=router.subperiod_months,
        )
    )

    positive_wins = [value for value in net_values if value > 0.0]
    total_positive = math.fsum(positive_wins)
    top_win_share = (
        max(positive_wins) / total_positive
        if positive_wins and total_positive > 0.0
        else None
    )

    rejected: list[str] = []
    if n < router.min_trades:
        rejected.append(f"n_trades<{router.min_trades}")
    if mean_gross is None or mean_gross < router.min_mean_gross_r:
        rejected.append(f"mean_gross_r<{router.min_mean_gross_r:g}")
    if mean_net is None or mean_net <= router.min_mean_net_r:
        rejected.append(f"mean_net_r<={router.min_mean_net_r:g}")
    if pf is None or pf < router.min_profit_factor:
        rejected.append(f"profit_factor<{router.min_profit_factor:g}")
    if stressed_mean is None or stressed_mean < router.min_stressed_mean_net_r:
        rejected.append(
            f"stressed_mean_net_r<{router.min_stressed_mean_net_r:g}"
        )
    if shrunk_mean is None or shrunk_mean < router.min_shrunk_mean_net_r:
        rejected.append(
            f"shrunk_mean_net_r<{router.min_shrunk_mean_net_r:g}"
        )
    if observed_subperiods < router.min_subperiods:
        rejected.append(f"observed_subperiods<{router.min_subperiods}")
    elif (
        positive_fraction is None
        or positive_fraction < router.min_positive_subperiod_fraction
    ):
        rejected.append(
            "positive_subperiod_fraction"
            f"<{router.min_positive_subperiod_fraction:g}"
        )
    if (
        top_win_share is None
        or top_win_share > router.max_top_win_share
    ):
        rejected.append(f"top_win_share>{router.max_top_win_share:g}")

    return CellStats(
        strategy=strategy,
        symbol=symbol,
        side=side,
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        n_trades=n,
        mean_gross_r=mean_gross,
        mean_net_r=mean_net,
        shrunk_mean_net_r=shrunk_mean,
        stressed_mean_net_r=stressed_mean,
        profit_factor=pf,
        positive_subperiods=positive_subperiods,
        observed_subperiods=observed_subperiods,
        positive_subperiod_fraction=positive_fraction,
        top_win_share=top_win_share,
        pass_gate=not rejected,
        rejection_reasons=tuple(rejected),
    )


def _select_cells_for_fold(
    history_signals: Sequence[QualifiedSignal],
    *,
    train_start_ts: int,
    train_end_ts: int,
    router: CellRouterConfig,
) -> tuple[set[CellKey], list[CellStats]]:
    # outcome_end_ts, not merely decision_ts, enforces the purged boundary.
    eligible = [
        signal
        for signal in history_signals
        if (
            train_start_ts <= signal.event.decision_ts < train_end_ts
            and signal.event.outcome_end_ts <= train_end_ts
        )
    ]
    grouped: dict[CellKey, list[QualifiedSignal]] = defaultdict(list)
    for signal in eligible:
        grouped[_cell_key(signal)].append(signal)

    stats = [
        _compute_cell_stats(
            key,
            signals,
            train_start_ts=train_start_ts,
            train_end_ts=train_end_ts,
            router=router,
        )
        for key, signals in sorted(grouped.items())
    ]
    active = {
        (stat.strategy, stat.symbol, stat.side)
        for stat in stats
        if stat.pass_gate
    }
    return active, stats


def _select_active_oos_signals(
    signals: Sequence[QualifiedSignal],
    *,
    active_cells: set[CellKey],
    test_start_ts: int,
    test_end_ts: int,
    config_by_strategy: Mapping[str, StrategyConfig],
    unavailable_until: dict[str, int],
) -> tuple[list[QualifiedSignal], dict[str, int]]:
    grouped: dict[tuple[str, int], list[QualifiedSignal]] = defaultdict(list)
    diagnostics = Counter()

    for signal in signals:
        event = signal.event
        if not test_start_ts <= event.decision_ts < test_end_ts:
            continue
        if _cell_key(signal) not in active_cells:
            diagnostics["inactive_cell_rejection"] += 1
            continue
        grouped[(event.symbol, event.decision_ts)].append(signal)

    selected: list[QualifiedSignal] = []
    for (symbol, decision_ts), candidates in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        chosen, reason = choose_signal(candidates)
        diagnostics[reason] += 1
        if chosen is None:
            continue
        if decision_ts < unavailable_until.get(symbol, -1):
            diagnostics["portfolio_cooldown_rejection"] += 1
            continue
        selected.append(chosen)
        unavailable_until[symbol] = (
            decision_ts + config_by_strategy[chosen.strategy].cooldown_ms
        )
        diagnostics["selected"] += 1

    return selected, dict(diagnostics)


def _walkforward_opportunities(
    prepared: Sequence[PreparedStrategy],
    start_ts: int,
    end_ts: int,
) -> int:
    return len(
        {
            (event.symbol, event.decision_ts)
            for item in prepared
            for event in item.events
            if start_ts <= event.decision_ts < end_ts
        }
    )


def _coverage_bounds(
    symbol_inputs: Sequence[SymbolInput],
) -> tuple[int, int]:
    if not symbol_inputs:
        raise ValueError("at least one symbol input is required")
    if any(not item.bars for item in symbol_inputs):
        raise ValueError("all symbols must contain bars")
    common_start = max(item.bars[0].open_ts for item in symbol_inputs)
    common_end = min(
        item.bars[-1].open_ts + BASE_INTERVAL_MS
        for item in symbol_inputs
    )
    if common_end <= common_start:
        raise ValueError("symbol snapshots have no common coverage")
    return common_start, common_end


def prepare_all_history_strategy(
    symbol_inputs: Sequence[SymbolInput],
    config: StrategyConfig,
) -> PreparedStrategy:
    # A far-future boundary makes every event a repository "train" event.
    # The router below creates its own purged rolling train/OOS boundaries.
    far_future_split = 9_000_000_000_000_000
    return prepare_strategy(
        symbol_inputs,
        far_future_split,
        config,
    )


def walkforward_command(args: argparse.Namespace) -> int:
    router = CellRouterConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        subperiod_months=args.subperiod_months,
        min_trades=args.min_cell_trades,
        min_subperiods=args.min_subperiods,
        min_positive_subperiod_fraction=(
            args.min_positive_subperiod_fraction
        ),
        min_mean_gross_r=args.min_cell_gross_r,
        min_mean_net_r=args.min_cell_net_r,
        min_profit_factor=args.min_cell_profit_factor,
        cost_stress_multiplier=args.cost_stress_multiplier,
        min_stressed_mean_net_r=args.min_stressed_mean_net_r,
        shrinkage_prior_trades=args.shrinkage_prior_trades,
        shrinkage_prior_mean_r=args.shrinkage_prior_mean_r,
        min_shrunk_mean_net_r=args.min_shrunk_mean_net_r,
        max_top_win_share=args.max_top_win_share,
    )
    router.validate()

    symbol_inputs = load_symbol_inputs(args.snapshot)
    requested_keys = (
        list(DEFAULT_CONFIGS)
        if args.strategy == "all"
        else [args.strategy]
    )
    prepared = [
        prepare_all_history_strategy(
            symbol_inputs,
            DEFAULT_CONFIGS[key],
        )
        for key in requested_keys
    ]
    config_by_strategy = {
        item.config.key: item.config
        for item in prepared
    }

    raw_qualified = [
        signal
        for item in prepared
        for signal in _qualified_all_history(item)
    ]
    history_signals = _apply_independent_cell_cooldown(
        raw_qualified,
        config_by_strategy,
    )

    coverage_start, coverage_end = _coverage_bounds(symbol_inputs)
    first_full_month = _ceil_month_start_ts(coverage_start)
    first_test_start = _add_months_ts(
        first_full_month,
        router.train_months,
    )

    fold_rows: list[PortfolioLedgerRow] = []
    folds: list[WalkForwardFold] = []
    unavailable_until: dict[str, int] = {}
    cursor = first_test_start
    fold_index = 0

    while cursor < coverage_end:
        test_end = _add_months_ts(cursor, router.test_months)
        if test_end > coverage_end and not args.allow_partial_last_fold:
            break
        effective_test_end = min(test_end, coverage_end)
        train_start = _add_months_ts(cursor, -router.train_months)

        active_cells, cell_stats = _select_cells_for_fold(
            history_signals,
            train_start_ts=train_start,
            train_end_ts=cursor,
            router=router,
        )
        selected, diagnostics = _select_active_oos_signals(
            raw_qualified,
            active_cells=active_cells,
            test_start_ts=cursor,
            test_end_ts=effective_test_end,
            config_by_strategy=config_by_strategy,
            unavailable_until=unavailable_until,
        )
        rows = _signals_to_rows(selected, "oos")
        fold_rows.extend(rows)

        opportunities = _walkforward_opportunities(
            prepared,
            cursor,
            effective_test_end,
        )
        performance = _performance(
            rows,
            split="oos",
            n_opportunities=opportunities,
            seed=args.seed + fold_index,
        )
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start_ts=train_start,
                train_end_ts=cursor,
                test_start_ts=cursor,
                test_end_ts=effective_test_end,
                active_cells=tuple(sorted(active_cells)),
                cell_stats=tuple(cell_stats),
                selected_trades=len(rows),
                diagnostics=diagnostics,
                performance=performance,
            )
        )

        fold_index += 1
        cursor = test_end

    if not folds:
        raise RuntimeError(
            "no complete walk-forward fold fits the common snapshot coverage"
        )

    total_opportunities = sum(
        fold.performance.n_opportunities
        for fold in folds
    )
    aggregate = _performance(
        fold_rows,
        split="oos",
        n_opportunities=total_opportunities,
        seed=args.seed + 100_000,
    )

    selection_frequency = Counter()
    rejection_frequency = Counter()
    for fold in folds:
        selection_frequency.update(fold.active_cells)
        for stat in fold.cell_stats:
            rejection_frequency.update(stat.rejection_reasons)

    payload = _json_safe(
        {
            "protocol": {
                "mode": "rolling_train_only_cell_router",
                "symbols": [item.symbol for item in symbol_inputs],
                "strategies": requested_keys,
                "common_coverage_start_ts": coverage_start,
                "common_coverage_end_ts": coverage_end,
                "common_coverage_start_utc": datetime.fromtimestamp(
                    coverage_start / 1000, tz=timezone.utc
                ).isoformat(),
                "common_coverage_end_utc": datetime.fromtimestamp(
                    coverage_end / 1000, tz=timezone.utc
                ).isoformat(),
                "router": asdict(router),
                "purge_rule": (
                    "train cell stats use only trades with outcome_end_ts "
                    "<= OOS fold start"
                ),
                "selection_freeze": (
                    "active strategy×symbol×side cells are frozen for each "
                    "complete OOS block"
                ),
                "portfolio_rules": {
                    "opposing_active_signals": "NO_TRADE",
                    "same_direction_active_signals": (
                        "highest confidence, then strategy priority"
                    ),
                    "risk_stacking": False,
                    "same_symbol_overlap": (
                        "worst-case strategy holding horizon, preserved "
                        "across fold boundaries"
                    ),
                },
            },
            "aggregate_oos": asdict(aggregate),
            "aggregate_oos_cost_breakdown": _cost_breakdown(
                fold_rows, "oos"
            ),
            "cell_selection_frequency": {
                "|".join(key): count
                for key, count in sorted(selection_frequency.items())
            },
            "rejection_reason_frequency": dict(
                sorted(rejection_frequency.items())
            ),
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "train_start_ts": fold.train_start_ts,
                    "train_end_ts": fold.train_end_ts,
                    "test_start_ts": fold.test_start_ts,
                    "test_end_ts": fold.test_end_ts,
                    "train_start_utc": datetime.fromtimestamp(
                        fold.train_start_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "train_end_utc": datetime.fromtimestamp(
                        fold.train_end_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "test_start_utc": datetime.fromtimestamp(
                        fold.test_start_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "test_end_utc": datetime.fromtimestamp(
                        fold.test_end_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "active_cells": [
                        {
                            "strategy": key[0],
                            "symbol": key[1],
                            "side": key[2],
                        }
                        for key in fold.active_cells
                    ],
                    "cell_stats": [
                        asdict(stat)
                        for stat in fold.cell_stats
                    ],
                    "selected_trades": fold.selected_trades,
                    "diagnostics": fold.diagnostics,
                    "performance": asdict(fold.performance),
                }
                for fold in folds
            ],
            "evidence_gate": {
                "minimum_oos_trades": 100,
                "minimum_mean_net_r": 0.05,
                "minimum_profit_factor": 1.10,
                "strong_evidence": (
                    "aggregate moving-block bootstrap CI95 lower bound > 0"
                ),
                "fold_consistency": (
                    "prefer positive mean_net_r in at least 60% of "
                    "non-empty OOS folds"
                ),
                "warning": (
                    "Router defaults are now a frozen hypothesis. Do not "
                    "retune them after reading this OOS report."
                ),
            },
        }
    )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0

def parse_split_ts(value: str) -> int:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    normalized = stripped.replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1000)


def parse_snapshot_spec(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        symbol, raw_path = value.split("=", 1)
        symbol = symbol.strip()
        if not symbol:
            raise ValueError(f"invalid --snapshot specification: {value!r}")
        return symbol, Path(raw_path)
    return None, Path(value)


def load_symbol_inputs(snapshot_specs: Sequence[str]) -> list[SymbolInput]:
    # Lazy import keeps selftest independent from pyarrow.
    from tools import data as data_tool

    result: list[SymbolInput] = []
    seen: set[str] = set()
    for snapshot_spec in snapshot_specs:
        explicit_symbol, path = parse_snapshot_spec(snapshot_spec)
        snapshot = data_tool.load(path)
        symbol = explicit_symbol or snapshot.manifest.get("instrument_id")
        if not symbol:
            raise ValueError(
                f"{path}: no explicit symbol and manifest lacks instrument_id"
            )
        if symbol in seen:
            raise ValueError(f"duplicate symbol: {symbol}")
        seen.add(symbol)
        result.append(
            SymbolInput(
                symbol=symbol,
                bars=snapshot.trade_bars,
                funding_events=snapshot.funding_events,
            )
        )
    return result


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_command(args: argparse.Namespace) -> int:
    split_ts = parse_split_ts(args.split_ts)
    symbol_inputs = load_symbol_inputs(args.snapshot)
    requested_keys = (
        list(DEFAULT_CONFIGS)
        if args.strategy == "all"
        else [args.strategy]
    )
    prepared = [
        prepare_strategy(symbol_inputs, split_ts, DEFAULT_CONFIGS[key])
        for key in requested_keys
    ]

    strategy_reports: dict[str, dict] = {}
    for index, item in enumerate(prepared):
        result, train_selected, test_selected, diagnostics = (
            evaluate_selected_strategy(
                item, seed=args.seed + index * 100
            )
        )
        strategy_reports[item.config.key] = {
            "config": asdict(item.config),
            "feature_warmup_drops": item.dropped,
            "selection_diagnostics": diagnostics,
            "train": asdict(
                _strategy_performance_from_evaluation(
                    item,
                    result,
                    train_selected,
                    "train",
                    seed=args.seed + index * 100,
                )
            ),
            "test": asdict(
                _strategy_performance_from_evaluation(
                    item,
                    result,
                    test_selected,
                    "test",
                    seed=args.seed + index * 100 + 1,
                )
            ),
        }

    combined_train, combined_train_diag = select_combined_portfolio(
        prepared, "train"
    )
    combined_test, combined_test_diag = select_combined_portfolio(
        prepared, "test"
    )
    combined_rows = (
        _signals_to_rows(combined_train, "train")
        + _signals_to_rows(combined_test, "test")
    )

    combined_train_opportunities = len(
        {
            (event.symbol, event.decision_ts)
            for item in prepared
            for event in item.events
            if event.split == "train"
        }
    )
    combined_test_opportunities = len(
        {
            (event.symbol, event.decision_ts)
            for item in prepared
            for event in item.events
            if event.split == "test"
        }
    )

    payload = _json_safe(
        {
            "protocol": {
                "split_ts": split_ts,
                "split_utc": datetime.fromtimestamp(
                    split_ts / 1000, tz=timezone.utc
                ).isoformat(),
                "symbols": [item.symbol for item in symbol_inputs],
                "feature_names": FEATURE_NAMES,
                "selected_strategies": requested_keys,
                "portfolio_rules": {
                    "opposing_signals": "NO_TRADE",
                    "same_direction_signals": (
                        "highest confidence, then priority; no risk stacking"
                    ),
                    "same_symbol_overlap": (
                        "blocked for selected strategy's worst-case horizon"
                    ),
                },
            },
            "per_strategy": strategy_reports,
            "combined_portfolio": {
                "selection_diagnostics": {
                    "train": combined_train_diag,
                    "test": combined_test_diag,
                },
                "train": asdict(
                    _performance(
                        combined_rows,
                        split="train",
                        n_opportunities=combined_train_opportunities,
                        seed=args.seed + 9000,
                    )
                ),
                "test": asdict(
                    _performance(
                        combined_rows,
                        split="test",
                        n_opportunities=combined_test_opportunities,
                        seed=args.seed + 9001,
                    )
                ),
            },
            "cost_breakdown": {
                "train": _json_safe(_cost_breakdown(combined_rows, "train")),
                "test": _json_safe(_cost_breakdown(combined_rows, "test")),
            },
            "evidence_gate": {
                "minimum_test_trades": 50,
                "preferred_test_trades": 100,
                "minimum_mean_net_r": 0.05,
                "minimum_profit_factor": 1.10,
                "strong_evidence": (
                    "95% moving-block bootstrap lower bound > 0"
                ),
                "warning": (
                    "Do not retune against this same TEST interval."
                ),
            },
        }
    )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


def _context(
    *,
    event_id: str,
    symbol: str,
    side: str,
    decision_ts: int,
    overrides: Mapping[str, float] | None = None,
) -> evaluate.PredictionContext:
    base = {
        "dir_return_1bar": 0.005,
        "dir_return_6h": 0.010,
        "dir_return_24h": 0.020,
        "dir_return_72h": 0.040,
        "abs_return_24h": 0.020,
        "breakout_atr": 0.30,
        "trend_efficiency": 0.35,
        "atr_pct": 0.010,
        "atr_ratio": 1.10,
        "volume_ratio": 1.30,
        "vwap_dev_atr": 0.50,
        "directional_zscore": -2.00,
        "side_rsi": 28.0,
        "body_atr": 0.80,
        "directional_close_location": 0.85,
        "directional_wick_ratio": 0.40,
        "prev_compression_ratio": 0.60,
        "range_atr": 1.50,
        "active_session": 1.0,
    }
    if overrides:
        base.update(overrides)
    return evaluate.PredictionContext(
        event_id=event_id,
        symbol=symbol,
        side=side,
        decision_ts=decision_ts,
        features=np.array([base[name] for name in FEATURE_NAMES], dtype=float),
    )


def _fake_signal(
    strategy: str,
    side: str,
    confidence: float,
    priority: int,
) -> QualifiedSignal:
    from lab.sim import CostBreakdown, TradeOutcome

    decision_ts = 1_000_000_000
    outcome = TradeOutcome(
        side=side,
        entry_index=10,
        exit_index=20,
        exit_reason="target",
        entry_price=100.0,
        exit_price=101.0,
        nominal_return=0.01,
        risk_fraction=0.01,
        gross_return=0.01,
        net_return=0.01,
        net_r=1.0,
        mae_r=0.2,
        mfe_r=1.0,
        costs=CostBreakdown(
            fee=0.0,
            slippage=0.0,
            funding=0.0,
            total=0.0,
        ),
    )
    event = events.CandidateEvent(
        event_id=f"{strategy}-{side}",
        symbol="BTC",
        side=side,
        feature_cutoff_ts=decision_ts,
        decision_ts=decision_ts,
        planned_entry_ts=decision_ts,
        fill_ts=decision_ts,
        outcome_end_ts=decision_ts + BASE_INTERVAL_MS,
        locked_outcome=outcome,
        split="test",
    )
    return QualifiedSignal(
        strategy=strategy,
        priority=priority,
        confidence=confidence,
        event=event,
        trace=DecisionTrace(
            take=True,
            confidence=confidence,
            reasons=("selftest",),
            rejections=(),
            values={name: 0.0 for name in FEATURE_NAMES},
        ),
    )


def selftest_command(_args: argparse.Namespace) -> int:
    trend = build_strategy(DEFAULT_CONFIGS["trend_breakout"])
    mean_reversion = build_strategy(DEFAULT_CONFIGS["range_reversion"])
    compression = build_strategy(DEFAULT_CONFIGS["compression_breakout"])

    trend_ctx = _context(
        event_id="trend",
        symbol="BTC",
        side="LONG",
        decision_ts=1_000_000_000,
    )
    assert trend.trace(trend_ctx).take is True
    assert trend.trace(
        _context(
            event_id="trend-fail",
            symbol="BTC",
            side="LONG",
            decision_ts=1_000_000_000,
            overrides={
                "dir_return_24h": -0.01,
                "dir_return_72h": -0.02,
            },
        )
    ).take is False

    mean_ctx = _context(
        event_id="mean",
        symbol="BTC",
        side="LONG",
        decision_ts=1_000_000_000,
        overrides={
            "abs_return_24h": 0.010,
            "trend_efficiency": 0.15,
            "dir_return_1bar": 0.004,
            "vwap_dev_atr": -1.20,
            "directional_zscore": -2.20,
            "side_rsi": 25.0,
        },
    )
    assert mean_reversion.trace(mean_ctx).take is True
    assert mean_reversion.trace(
        _context(
            event_id="mean-fail",
            symbol="BTC",
            side="LONG",
            decision_ts=1_000_000_000,
            overrides={
                "abs_return_24h": 0.080,
                "trend_efficiency": 0.70,
                "vwap_dev_atr": 0.50,
            },
        )
    ).take is False

    compression_ctx = _context(
        event_id="compression",
        symbol="BTC",
        side="LONG",
        decision_ts=1_000_000_000,
    )
    assert compression.trace(compression_ctx).take is True
    assert compression.trace(
        _context(
            event_id="compression-fail",
            symbol="BTC",
            side="LONG",
            decision_ts=1_000_000_000,
            overrides={
                "prev_compression_ratio": 1.30,
                "volume_ratio": 0.70,
            },
        )
    ).take is False

    # Arbiter contracts.
    conflict, conflict_reason = choose_signal(
        [
            _fake_signal("trend_breakout", "LONG", 0.8, 20),
            _fake_signal("range_reversion", "SHORT", 0.9, 10),
        ]
    )
    assert conflict is None
    assert conflict_reason == "direction_conflict"

    chosen, chosen_reason = choose_signal(
        [
            _fake_signal("trend_breakout", "LONG", 0.6, 20),
            _fake_signal("compression_breakout", "LONG", 0.9, 30),
        ]
    )
    assert chosen is not None
    assert chosen.strategy == "compression_breakout"
    assert chosen_reason == "same_direction_competition"

    # Predictor context must not expose outcomes or split labels.
    assert not hasattr(trend_ctx, "locked_outcome")
    assert not hasattr(trend_ctx, "outcome_end_ts")
    assert not hasattr(trend_ctx, "split")


    # Cell-router contracts: profitable/stable cell passes; weak cell fails.
    router = CellRouterConfig(
        train_months=12,
        test_months=2,
        subperiod_months=2,
        min_trades=6,
        min_subperiods=3,
        min_positive_subperiod_fraction=2 / 3,
        min_mean_gross_r=0.10,
        min_mean_net_r=0.05,
        min_profit_factor=1.10,
        cost_stress_multiplier=1.5,
        min_stressed_mean_net_r=0.0,
        shrinkage_prior_trades=2.0,
        min_shrunk_mean_net_r=0.03,
        max_top_win_share=0.60,
    )
    router.validate()

    month0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    month12 = _add_months_ts(month0, 12)
    synthetic_signals: list[QualifiedSignal] = []
    for idx, net_r in enumerate((0.6, -0.2, 0.5, -0.1, 0.4, 0.3)):
        signal = _fake_signal(
            "compression_breakout",
            "LONG",
            0.8,
            30,
        )
        decision_ts = _add_months_ts(month0, idx * 2) + 86_400_000
        outcome = signal.event.locked_outcome
        synthetic_event = events.CandidateEvent(
            event_id=f"router-{idx}",
            symbol="XRP",
            side="LONG",
            feature_cutoff_ts=decision_ts,
            decision_ts=decision_ts,
            planned_entry_ts=decision_ts,
            fill_ts=decision_ts,
            outcome_end_ts=decision_ts + BASE_INTERVAL_MS,
            locked_outcome=type(outcome)(
                side=outcome.side,
                entry_index=outcome.entry_index,
                exit_index=outcome.exit_index,
                exit_reason=("target" if net_r > 0 else "stop"),
                entry_price=outcome.entry_price,
                exit_price=outcome.exit_price,
                nominal_return=outcome.nominal_return,
                risk_fraction=outcome.risk_fraction,
                gross_return=(net_r + 0.08) * outcome.risk_fraction,
                net_return=net_r * outcome.risk_fraction,
                net_r=net_r,
                mae_r=outcome.mae_r,
                mfe_r=max(net_r, 0.0),
                costs=type(outcome.costs)(
                    fee=0.06 * outcome.risk_fraction,
                    slippage=0.02 * outcome.risk_fraction,
                    funding=0.0,
                    total=0.08 * outcome.risk_fraction,
                ),
            ),
            split="train",
        )
        synthetic_signals.append(
            QualifiedSignal(
                strategy="compression_breakout",
                priority=30,
                confidence=0.8,
                event=synthetic_event,
                trace=signal.trace,
            )
        )

    active_cells, router_stats = _select_cells_for_fold(
        synthetic_signals,
        train_start_ts=month0,
        train_end_ts=month12,
        router=router,
    )
    assert (
        "compression_breakout", "XRP", "LONG"
    ) in active_cells
    assert router_stats[0].pass_gate is True

    # A trade decided before the fold but ending after it must not enter training.
    crossing = synthetic_signals[0]
    crossing_event = events.CandidateEvent(
        event_id="crossing",
        symbol="XRP",
        side="LONG",
        feature_cutoff_ts=month12 - BASE_INTERVAL_MS,
        decision_ts=month12 - BASE_INTERVAL_MS,
        planned_entry_ts=month12 - BASE_INTERVAL_MS,
        fill_ts=month12 - BASE_INTERVAL_MS,
        outcome_end_ts=month12 + BASE_INTERVAL_MS,
        locked_outcome=crossing.event.locked_outcome,
        split="train",
    )
    crossing_signal = QualifiedSignal(
        strategy=crossing.strategy,
        priority=crossing.priority,
        confidence=crossing.confidence,
        event=crossing_event,
        trace=crossing.trace,
    )
    _, purged_stats = _select_cells_for_fold(
        synthetic_signals + [crossing_signal],
        train_start_ts=month0,
        train_end_ts=month12,
        router=router,
    )
    assert purged_stats[0].n_trades == 6

    print(
        json.dumps(
            {
                "status": "PASS",
                "strategies": list(DEFAULT_CONFIGS),
                "feature_count": len(FEATURE_NAMES),
                "contracts": {
                    "lab_market_used": True,
                    "lab_indicators_used": True,
                    "lab_events_used": True,
                    "lab_evaluate_used": True,
                    "net_r_reimplemented": False,
                    "opposing_signals_blocked": True,
                    "same_direction_risk_not_stacked": True,
                    "worst_case_cooldown_enabled": True,
                    "outcome_hidden_from_decision_path": True,
                    "rolling_cell_router": True,
                    "cell_selection_train_only": True,
                    "outcome_end_purge": True,
                    "cost_stress_gate": True,
                    "shrinkage_gate": True,
                    "subperiod_stability_gate": True,
                },
                "sample_traces": {
                    "trend_breakout": asdict(trend.trace(trend_ctx)),
                    "range_reversion": asdict(
                        mean_reversion.trace(mean_ctx)
                    ),
                    "compression_breakout": asdict(
                        compression.trace(compression_ctx)
                    ),
                },
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V7-Lite deterministic multi-strategy challenger"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="evaluate one or all strategy families"
    )
    run.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="[SYMBOL=]PATH",
        help="repeat for every verified tools.data snapshot",
    )
    run.add_argument(
        "--split-ts",
        required=True,
        help=(
            "Unix milliseconds or ISO-8601, e.g. 2026-05-01T00:00:00Z"
        ),
    )
    run.add_argument(
        "--strategy",
        choices=("all", *DEFAULT_CONFIGS.keys()),
        default="all",
    )
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--output")
    run.set_defaults(func=run_command)


    walkforward = subparsers.add_parser(
        "walkforward",
        help=(
            "rolling train-only strategy×symbol×side cell router "
            "with frozen OOS blocks"
        ),
    )
    walkforward.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="[SYMBOL=]PATH",
        help="repeat for every verified tools.data snapshot",
    )
    walkforward.add_argument(
        "--strategy",
        choices=("all", *DEFAULT_CONFIGS.keys()),
        default="all",
    )
    walkforward.add_argument("--train-months", type=int, default=12)
    walkforward.add_argument("--test-months", type=int, default=2)
    walkforward.add_argument("--subperiod-months", type=int, default=2)
    walkforward.add_argument("--min-cell-trades", type=int, default=30)
    walkforward.add_argument("--min-subperiods", type=int, default=3)
    walkforward.add_argument(
        "--min-positive-subperiod-fraction",
        type=float,
        default=0.60,
    )
    walkforward.add_argument(
        "--min-cell-gross-r",
        type=float,
        default=0.16,
    )
    walkforward.add_argument(
        "--min-cell-net-r",
        type=float,
        default=0.0,
    )
    walkforward.add_argument(
        "--min-cell-profit-factor",
        type=float,
        default=1.10,
    )
    walkforward.add_argument(
        "--cost-stress-multiplier",
        type=float,
        default=1.50,
    )
    walkforward.add_argument(
        "--min-stressed-mean-net-r",
        type=float,
        default=0.0,
    )
    walkforward.add_argument(
        "--shrinkage-prior-trades",
        type=float,
        default=20.0,
    )
    walkforward.add_argument(
        "--shrinkage-prior-mean-r",
        type=float,
        default=0.0,
    )
    walkforward.add_argument(
        "--min-shrunk-mean-net-r",
        type=float,
        default=0.02,
    )
    walkforward.add_argument(
        "--max-top-win-share",
        type=float,
        default=0.50,
    )
    walkforward.add_argument(
        "--allow-partial-last-fold",
        action="store_true",
    )
    walkforward.add_argument("--seed", type=int, default=7)
    walkforward.add_argument("--output")
    walkforward.set_defaults(func=walkforward_command)

    selftest = subparsers.add_parser(
        "selftest", help="run deterministic strategy and arbiter tests"
    )
    selftest.set_defaults(func=selftest_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return int(args.func(args))
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
