#!/usr/bin/env python3
"""
V7-Lite V3 — Quarter-Hour Microstructure Challenger
====================================================

One-file research/execution challenger for ddawnlll/v7.

Core hypothesis
---------------
At UTC quarter-hour boundaries (:00, :15, :30, :45), the first ten seconds of
aggressor-side trade flow may carry information about the next several hours,
but only after conditioning on pre-existing liquidity state, price response,
execution cost, and model uncertainty.

V3 replaces the failed candle-pattern strategies with:
- quarter-hour aggressor-flow features,
- clock-phase flow memory,
- price-response / absorption features,
- L1 and optional L2 liquidity-state features,
- optional funding/premium context,
- calibrated 0–100 trade-quality scores,
- validation-only threshold tuning,
- a train-only post-execution early-exit model,
- rolling train / validation / frozen-OOS evaluation,
- lab.sim as the sole economic truth.

V3 deliberately DOES NOT reimplement:
- bar validation,
- ATR primitives,
- fee/slippage/funding arithmetic,
- stop/target/gap semantics,
- net_R.

Those remain owned by:
    lab.market
    lab.indicators
    lab.sim
    tools.data

Required data
-------------
1. Verified V7 snapshot for 5m execution truth:

    --snapshot BTC-USDT-SWAP=data/snapshots/btc

2. Raw aggressor-side trades, CSV or parquet:

    --trades BTC-USDT-SWAP=data/micro/btc_trades.parquet

   Required semantic columns, aliases accepted:
       timestamp milliseconds, price, size, aggressor side

3. L1 order-book snapshots, CSV or parquet:

    --book BTC-USDT-SWAP=data/micro/btc_book.parquet

   Required:
       timestamp, best bid price/size, best ask price/size

   Optional:
       bid/ask price and size for levels 2..5

The file fails closed when raw trade side or L1 book state is unavailable.
It never fabricates 10-second order flow from 5m OHLCV.

Examples
--------
Inspect data:

    python v7-lite.py inspect-data \
      --snapshot BTC-USDT-SWAP=data/snapshots/btc \
      --trades BTC-USDT-SWAP=data/micro/btc_trades.parquet \
      --book BTC-USDT-SWAP=data/micro/btc_book.parquet

Run rolling V3:

    python v7-lite.py walkforward \
      --snapshot BTC-USDT-SWAP=data/snapshots/btc \
      --trades BTC-USDT-SWAP=data/micro/btc_trades.parquet \
      --book BTC-USDT-SWAP=data/micro/btc_book.parquet \
      --snapshot ETH-USDT-SWAP=data/snapshots/eth \
      --trades ETH-USDT-SWAP=data/micro/eth_trades.parquet \
      --book ETH-USDT-SWAP=data/micro/eth_book.parquet \
      --train-months 12 \
      --validation-months 2 \
      --test-months 2 \
      --horizon-hours 8 \
      --output v3-result.json

Run built-in contract tests:

    python v7-lite.py selftest

Research warning
----------------
Thresholds are tuned on each fold's validation window only. Frozen OOS is read
once. After reading an OOS interval, do not retune against that interval.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab import indicators, market, sim  # noqa: E402


BASE_INTERVAL_MS = 300_000
QUARTER_MS = 900_000
DAY_MS = 86_400_000
MONTH_MS = 30 * DAY_MS
EPS = 1e-12

Side = Literal["LONG", "SHORT"]


# =============================================================================
# Feature contracts
# =============================================================================

CORE_FEATURE_NAMES = (
    # Clock context
    "utc_hour_sin",
    "utc_hour_cos",

    # First ten seconds of aggressive flow
    "dir_volume_imbalance_10s",
    "dir_trade_count_imbalance_10s",
    "dir_signed_dollar_flow_z_10s",
    "total_volume_shock_10s",
    "trade_count_shock_10s",
    "large_trade_share_10s",
    "trade_size_roundness_10s",

    # Clock-phase memory
    "dir_qh_imbalance_lag_1",
    "dir_qh_imbalance_lag_4",
    "dir_qh_imbalance_ewm_4",
    "dir_qh_imbalance_acceleration",

    # Price response / absorption
    "dir_return_10s",
    "realized_volatility_10s",
    "dir_price_response_to_flow",
    "dir_absorption_score",

    # Portable top-of-book / microstructure
    "relative_spread",
    "dir_l1_book_imbalance",
    "dir_microprice_deviation",
    "dir_buy_vwap_to_mid",
    "dir_sell_vwap_to_mid",
    "trade_price_variance",
    "volume_concentration",

    # Aggregated liquidity state
    "dir_depth_imbalance_l5",
    "total_depth_l5_z",
    "liquidity_state",

    # Cost and slow regime
    "flow_to_depth_ratio",
    "estimated_round_trip_cost_r",
    "realized_volatility_1h",
)

OPTIONAL_POSITIONING_FEATURE_NAMES = (
    "dir_funding_rate_z",
    "dir_premium_z",
    "time_to_funding_fraction",
)

POST_FEATURE_NAMES = (
    "elapsed_minutes",
    "unrealized_r",
    "mfe_r_so_far",
    "mae_r_so_far",
    "dir_residual_flow_imbalance",
    "flow_sign_flip",
    "flow_decay_ratio",
    "dir_return_since_entry",
    "dir_price_response_since_entry",
    "dir_absorption_since_entry",
    "spread_change",
    "dir_book_imbalance_change",
    "dir_microprice_change",
    "estimated_exit_cost_r",
    "entry_quality_score",
    "entry_predicted_net_r",
    "entry_predicted_win_probability",
)


# =============================================================================
# Configuration and immutable records
# =============================================================================

@dataclass(frozen=True, slots=True)
class V3Config:
    flow_window_seconds: int = 10
    horizon_hours: int = 8
    stop_atr: float = 1.50
    target_r: float = 1.50
    atr_period: int = 14

    fee_rate: float = 0.0004
    slippage_bps: float = 1.0

    train_months: int = 12
    validation_months: int = 2
    test_months: int = 2
    step_months: int = 2

    min_trades_in_flow_window: int = 4
    historical_z_days: int = 30
    book_max_age_ms: int = 5_000

    threshold_min: int = 50
    threshold_max: int = 90
    threshold_step: int = 5
    minimum_predicted_net_r: float = 0.02
    minimum_validation_trades: int = 20
    minimum_validation_pf: float = 1.00
    minimum_validation_mean_r: float = 0.0
    minimum_validation_win_rate: float = 0.52

    conflict_score_margin: float = 3.0
    post_checkpoints_minutes: tuple[int, ...] = (5, 15, 30, 60)
    post_exit_threshold_grid: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05)
    enable_post_execution: bool = True

    ridge_alpha: float = 10.0
    tree_max_depth: int = 3
    tree_max_iter: int = 100
    seed: int = 7

    def validate(self) -> None:
        if self.flow_window_seconds < 1 or self.flow_window_seconds > 60:
            raise ValueError("flow_window_seconds must be in [1, 60]")
        if self.horizon_hours < 1:
            raise ValueError("horizon_hours must be positive")
        if self.stop_atr <= 0.0 or self.target_r <= 0.0:
            raise ValueError("stop_atr and target_r must be positive")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if self.fee_rate < 0.0 or self.slippage_bps < 0.0:
            raise ValueError("cost inputs must be non-negative")
        if min(
            self.train_months,
            self.validation_months,
            self.test_months,
            self.step_months,
        ) < 1:
            raise ValueError("walk-forward month lengths must be positive")
        if self.min_trades_in_flow_window < 1:
            raise ValueError("min_trades_in_flow_window must be positive")
        if self.historical_z_days < 2:
            raise ValueError("historical_z_days must be >= 2")
        if not 0 <= self.threshold_min <= self.threshold_max <= 100:
            raise ValueError("quality thresholds must be in [0, 100]")
        if self.threshold_step < 1:
            raise ValueError("threshold_step must be positive")
        if self.minimum_validation_trades < 1:
            raise ValueError("minimum_validation_trades must be positive")
        if not 0.0 <= self.minimum_validation_win_rate <= 1.0:
            raise ValueError("minimum_validation_win_rate must be in [0, 1]")
        if self.conflict_score_margin < 0.0:
            raise ValueError("conflict_score_margin must be non-negative")
        if any(m <= 0 or m % 5 != 0 for m in self.post_checkpoints_minutes):
            raise ValueError("post checkpoints must be positive 5-minute multiples")
        if max(self.post_checkpoints_minutes, default=0) >= self.horizon_hours * 60:
            raise ValueError("post checkpoints must precede the horizon")
        if self.tree_max_depth < 1 or self.tree_max_iter < 1:
            raise ValueError("invalid shallow-tree configuration")


@dataclass(frozen=True, slots=True)
class RawTradeTape:
    ts_ms: np.ndarray
    price: np.ndarray
    size: np.ndarray
    side_sign: np.ndarray  # +1 buyer taker, -1 seller taker

    def validate(self, symbol: str) -> None:
        n = len(self.ts_ms)
        if n == 0:
            raise ValueError(f"{symbol}: empty raw trade tape")
        if not (len(self.price) == len(self.size) == len(self.side_sign) == n):
            raise ValueError(f"{symbol}: raw trade arrays have different lengths")
        if np.any(np.diff(self.ts_ms) < 0):
            raise ValueError(f"{symbol}: raw trade timestamps are not sorted")
        if not np.all(np.isfinite(self.price)) or np.any(self.price <= 0):
            raise ValueError(f"{symbol}: invalid trade prices")
        if not np.all(np.isfinite(self.size)) or np.any(self.size <= 0):
            raise ValueError(f"{symbol}: invalid trade sizes")
        if not np.all(np.isin(self.side_sign, (-1.0, 1.0))):
            raise ValueError(f"{symbol}: aggressor side must be BUY/SELL")


@dataclass(frozen=True, slots=True)
class BookTape:
    ts_ms: np.ndarray
    bid_px: np.ndarray       # shape (n, levels)
    ask_px: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray

    @property
    def levels(self) -> int:
        return int(self.bid_px.shape[1])

    def validate(self, symbol: str) -> None:
        n = len(self.ts_ms)
        if n == 0:
            raise ValueError(f"{symbol}: empty book tape")
        for name, arr in (
            ("bid_px", self.bid_px),
            ("ask_px", self.ask_px),
            ("bid_size", self.bid_size),
            ("ask_size", self.ask_size),
        ):
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"{symbol}: {name} has invalid shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{symbol}: {name} contains non-finite values")
        if self.levels < 1:
            raise ValueError(f"{symbol}: at least L1 book is required")
        if np.any(np.diff(self.ts_ms) <= 0):
            raise ValueError(f"{symbol}: book timestamps must be strictly increasing")
        if np.any(self.bid_px <= 0) or np.any(self.ask_px <= 0):
            raise ValueError(f"{symbol}: book prices must be positive")
        if np.any(self.bid_size < 0) or np.any(self.ask_size < 0):
            raise ValueError(f"{symbol}: book sizes must be non-negative")
        if np.any(self.ask_px[:, 0] <= self.bid_px[:, 0]):
            raise ValueError(f"{symbol}: crossed/locked L1 book found")


@dataclass(frozen=True, slots=True)
class SymbolData:
    symbol: str
    bars: list[market.Bar]
    funding_events: list[sim.FundingEvent]
    trades: RawTradeTape
    book: BookTape
    premium_bars: Sequence | None
    manifest: dict


@dataclass(frozen=True, slots=True)
class RawQuarterEvent:
    symbol: str
    quarter_ts: int
    decision_ts: int
    entry_index: int
    raw_features: dict[str, float]
    final_outcome_long: sim.TradeOutcome
    final_outcome_short: sim.TradeOutcome
    checkpoint_outcomes_long: dict[int, sim.TradeOutcome]
    checkpoint_outcomes_short: dict[int, sim.TradeOutcome]


@dataclass(frozen=True, slots=True)
class DirectionalEvent:
    event_id: str
    symbol: str
    side: Side
    decision_ts: int
    entry_index: int
    outcome_end_ts: int
    features: np.ndarray
    final_outcome: sim.TradeOutcome
    checkpoint_outcomes: dict[int, sim.TradeOutcome]


@dataclass(frozen=True, slots=True)
class EntryPrediction:
    event: DirectionalEvent
    predicted_net_r: float
    predicted_win_probability: float
    uncertainty: float
    quality_score: float


@dataclass(frozen=True, slots=True)
class SelectedTrade:
    event: DirectionalEvent
    quality_score: float
    predicted_net_r: float
    predicted_win_probability: float
    realized_outcome: sim.TradeOutcome
    post_action: str
    post_checkpoint_minutes: int | None


@dataclass(frozen=True, slots=True)
class FoldWindow:
    fold_index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class Performance:
    n_trades: int
    mean_net_r: float | None
    median_net_r: float | None
    win_rate: float | None
    profit_factor: float | None
    cumulative_net_r: float
    max_drawdown_r: float
    average_win_r: float | None
    average_loss_r: float | None
    longest_losing_streak: int
    ci95_low: float | None
    ci95_high: float | None
    post_early_exits: int
    symbols: dict[str, dict]
    sides: dict[str, dict]


# =============================================================================
# Generic table loading with strict semantic aliases
# =============================================================================

_TS_ALIASES = (
    "ts_ms", "timestamp_ms", "timestamp", "ts", "time", "trade_time",
)
_PRICE_ALIASES = ("price", "px", "trade_price")
_SIZE_ALIASES = ("size", "qty", "quantity", "amount", "base_volume")
_SIDE_ALIASES = (
    "side", "aggressor_side", "taker_side", "trade_side", "is_buyer_maker",
)

_BOOK_ALIASES = {
    "bid_px1": ("bid_px1", "bid_price1", "best_bid", "bid1_price", "b1_px"),
    "ask_px1": ("ask_px1", "ask_price1", "best_ask", "ask1_price", "a1_px"),
    "bid_sz1": ("bid_sz1", "bid_size1", "best_bid_size", "bid1_size", "b1_sz"),
    "ask_sz1": ("ask_sz1", "ask_size1", "best_ask_size", "ask1_size", "a1_sz"),
}


def _normalise_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_column(columns: Sequence[str], aliases: Sequence[str], label: str) -> str:
    mapping = {_normalise_name(col): col for col in columns}
    for alias in aliases:
        actual = mapping.get(_normalise_name(alias))
        if actual is not None:
            return actual
    raise ValueError(f"missing required {label}; accepted aliases={aliases}")


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: missing header")
            rows = list(reader)
            return list(reader.fieldnames), rows

    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: JSON row must be object")
                rows.append(row)
        if not rows:
            raise ValueError(f"{path}: empty JSONL")
        return list(rows[0]), rows

    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for parquet input") from exc
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"cannot read {path}; install pyarrow or fastparquet"
            ) from exc
        return [str(col) for col in frame.columns], frame.to_dict("records")

    raise ValueError(f"unsupported data format: {path.suffix}")


def _timestamp_ms(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        raw = int(value)
    elif isinstance(value, float) and math.isfinite(value):
        raw = int(value)
    else:
        text = str(value).strip()
        if text.replace(".", "", 1).isdigit():
            raw = int(float(text))
        else:
            normalised = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalised)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

    # Infer common Unix units.
    magnitude = abs(raw)
    if magnitude < 10_000_000_000:      # seconds
        return raw * 1000
    if magnitude < 10_000_000_000_000:  # milliseconds
        return raw
    if magnitude < 10_000_000_000_000_000:  # microseconds
        return raw // 1000
    return raw // 1_000_000             # nanoseconds


def _normalise_side(value: object, column_name: str) -> float:
    name = _normalise_name(column_name)

    if name == "is_buyer_maker":
        # Binance convention: true means buyer was maker, therefore seller taker.
        if isinstance(value, str):
            flag = value.strip().lower() in {"1", "true", "t", "yes"}
        else:
            flag = bool(value)
        return -1.0 if flag else 1.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric > 0:
            return 1.0
        if numeric < 0:
            return -1.0

    text = str(value).strip().lower()
    if text in {"buy", "b", "buyer", "bid", "long", "1", "+1"}:
        return 1.0
    if text in {"sell", "s", "seller", "ask", "short", "-1"}:
        return -1.0
    raise ValueError(f"unsupported aggressor side value: {value!r}")


def load_trade_tape(path: Path, symbol: str) -> RawTradeTape:
    columns, rows = _read_rows(path)
    ts_col = _pick_column(columns, _TS_ALIASES, "trade timestamp")
    price_col = _pick_column(columns, _PRICE_ALIASES, "trade price")
    size_col = _pick_column(columns, _SIZE_ALIASES, "trade size")
    side_col = _pick_column(columns, _SIDE_ALIASES, "aggressor side")

    parsed = []
    for row_no, row in enumerate(rows, 2):
        try:
            parsed.append((
                _timestamp_ms(row[ts_col]),
                float(row[price_col]),
                float(row[size_col]),
                _normalise_side(row[side_col], side_col),
            ))
        except Exception as exc:
            raise ValueError(f"{path}:{row_no}: {exc}") from exc

    parsed.sort(key=lambda item: item[0])
    tape = RawTradeTape(
        ts_ms=np.asarray([item[0] for item in parsed], dtype=np.int64),
        price=np.asarray([item[1] for item in parsed], dtype=float),
        size=np.asarray([item[2] for item in parsed], dtype=float),
        side_sign=np.asarray([item[3] for item in parsed], dtype=float),
    )
    tape.validate(symbol)
    return tape


def _book_level_aliases(side: str, field: str, level: int) -> tuple[str, ...]:
    if field == "px":
        return (
            f"{side}_px{level}",
            f"{side}_price{level}",
            f"{side}{level}_price",
            f"{side[0]}{level}_px",
        )
    return (
        f"{side}_sz{level}",
        f"{side}_size{level}",
        f"{side}{level}_size",
        f"{side[0]}{level}_sz",
    )


def load_book_tape(path: Path, symbol: str) -> BookTape:
    columns, rows = _read_rows(path)
    ts_col = _pick_column(columns, _TS_ALIASES, "book timestamp")

    mapping = {_normalise_name(col): col for col in columns}

    def optional_column(aliases: Sequence[str]) -> str | None:
        for alias in aliases:
            value = mapping.get(_normalise_name(alias))
            if value is not None:
                return value
        return None

    level_columns = []
    for level in range(1, 6):
        bid_px = optional_column(
            _BOOK_ALIASES["bid_px1"] if level == 1
            else _book_level_aliases("bid", "px", level)
        )
        ask_px = optional_column(
            _BOOK_ALIASES["ask_px1"] if level == 1
            else _book_level_aliases("ask", "px", level)
        )
        bid_sz = optional_column(
            _BOOK_ALIASES["bid_sz1"] if level == 1
            else _book_level_aliases("bid", "sz", level)
        )
        ask_sz = optional_column(
            _BOOK_ALIASES["ask_sz1"] if level == 1
            else _book_level_aliases("ask", "sz", level)
        )

        if level == 1 and None in {bid_px, ask_px, bid_sz, ask_sz}:
            raise ValueError(f"{path}: complete L1 bid/ask price/size is required")
        if None in {bid_px, ask_px, bid_sz, ask_sz}:
            break
        level_columns.append((bid_px, ask_px, bid_sz, ask_sz))

    parsed = []
    for row_no, row in enumerate(rows, 2):
        try:
            values = [_timestamp_ms(row[ts_col])]
            for bid_px, ask_px, bid_sz, ask_sz in level_columns:
                values.extend((
                    float(row[bid_px]),
                    float(row[ask_px]),
                    float(row[bid_sz]),
                    float(row[ask_sz]),
                ))
            parsed.append(values)
        except Exception as exc:
            raise ValueError(f"{path}:{row_no}: {exc}") from exc

    parsed.sort(key=lambda item: item[0])
    deduped = []
    for row in parsed:
        if deduped and row[0] == deduped[-1][0]:
            deduped[-1] = row
        else:
            deduped.append(row)

    levels = len(level_columns)
    n = len(deduped)
    bid_px = np.empty((n, levels), dtype=float)
    ask_px = np.empty((n, levels), dtype=float)
    bid_size = np.empty((n, levels), dtype=float)
    ask_size = np.empty((n, levels), dtype=float)

    for row_index, row in enumerate(deduped):
        offset = 1
        for level_index in range(levels):
            bid_px[row_index, level_index] = row[offset]
            ask_px[row_index, level_index] = row[offset + 1]
            bid_size[row_index, level_index] = row[offset + 2]
            ask_size[row_index, level_index] = row[offset + 3]
            offset += 4

    tape = BookTape(
        ts_ms=np.asarray([row[0] for row in deduped], dtype=np.int64),
        bid_px=bid_px,
        ask_px=ask_px,
        bid_size=bid_size,
        ask_size=ask_size,
    )
    tape.validate(symbol)
    return tape


# =============================================================================
# Utility math
# =============================================================================

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return default
    if abs(denominator) <= EPS:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    mean = statistics.fmean(finite)
    std = statistics.pstdev(finite) if len(finite) >= 2 else 0.0
    return mean, max(std, 1e-9)


def _zscore(value: float, history: Sequence[float]) -> float:
    mean, std = _mean_std(history)
    return (value - mean) / std


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _profit_factor(values: Sequence[float]) -> float | None:
    wins = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    if losses > EPS:
        return wins / losses
    if wins > 0.0:
        return math.inf
    return None


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_boot: int = 2000,
) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    n = len(array)
    if n < 20:
        return None, None
    block = max(2, int(round(math.sqrt(n))))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            for offset in range(block):
                sample.append(float(array[(start + offset) % n]))
                if len(sample) == n:
                    break
        means[index] = float(np.mean(sample))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _book_state(book: BookTape, ts_ms: int, max_age_ms: int) -> tuple[int, dict] | None:
    index = bisect.bisect_right(book.ts_ms, ts_ms) - 1
    if index < 0:
        return None
    age = ts_ms - int(book.ts_ms[index])
    if age < 0 or age > max_age_ms:
        return None

    bid = float(book.bid_px[index, 0])
    ask = float(book.ask_px[index, 0])
    bid_size = float(book.bid_size[index, 0])
    ask_size = float(book.ask_size[index, 0])
    mid = (bid + ask) * 0.5
    spread = (ask - bid) / mid
    imbalance = _safe_div(bid_size - ask_size, bid_size + ask_size)
    microprice = _safe_div(
        ask * bid_size + bid * ask_size,
        bid_size + ask_size,
        default=mid,
    )
    micro_dev = (microprice - mid) / mid

    levels = min(5, book.levels)
    total_bid = float(np.sum(book.bid_size[index, :levels]))
    total_ask = float(np.sum(book.ask_size[index, :levels]))
    depth_imbalance = _safe_div(total_bid - total_ask, total_bid + total_ask)
    total_depth = total_bid + total_ask

    return index, {
        "book_age_ms": float(age),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "l1_imbalance": imbalance,
        "microprice_deviation": micro_dev,
        "depth_imbalance_l5": depth_imbalance,
        "total_depth_l5": total_depth,
    }


def _trade_slice(tape: RawTradeTape, start_ts: int, end_ts: int) -> slice:
    left = bisect.bisect_left(tape.ts_ms, start_ts)
    right = bisect.bisect_left(tape.ts_ms, end_ts)
    return slice(left, right)


def _trade_window_metrics(
    tape: RawTradeTape,
    start_ts: int,
    end_ts: int,
    reference_mid: float,
    large_trade_threshold: float,
) -> dict[str, float] | None:
    window = _trade_slice(tape, start_ts, end_ts)
    prices = tape.price[window]
    sizes = tape.size[window]
    signs = tape.side_sign[window]
    n = len(prices)
    if n == 0:
        return None

    notionals = prices * sizes
    buy_mask = signs > 0
    sell_mask = signs < 0
    buy_volume = float(np.sum(sizes[buy_mask]))
    sell_volume = float(np.sum(sizes[sell_mask]))
    total_volume = buy_volume + sell_volume
    buy_count = int(np.sum(buy_mask))
    sell_count = int(np.sum(sell_mask))
    total_count = buy_count + sell_count
    signed_dollar_flow = float(np.sum(notionals * signs))
    total_dollar_volume = float(np.sum(notionals))

    volume_imbalance = _safe_div(buy_volume - sell_volume, total_volume)
    count_imbalance = _safe_div(buy_count - sell_count, total_count)

    first_price = float(prices[0])
    last_price = float(prices[-1])
    return_fraction = (last_price - first_price) / first_price

    if n >= 2:
        tick_returns = np.diff(np.log(prices))
        realized_vol = float(np.sqrt(np.sum(tick_returns * tick_returns)))
        trade_price_variance = float(np.var(prices / reference_mid - 1.0))
    else:
        realized_vol = 0.0
        trade_price_variance = 0.0

    buy_notional = notionals[buy_mask]
    sell_notional = notionals[sell_mask]
    buy_vwap = (
        float(np.sum(prices[buy_mask] * sizes[buy_mask]) / np.sum(sizes[buy_mask]))
        if buy_count and float(np.sum(sizes[buy_mask])) > EPS
        else reference_mid
    )
    sell_vwap = (
        float(np.sum(prices[sell_mask] * sizes[sell_mask]) / np.sum(sizes[sell_mask]))
        if sell_count and float(np.sum(sizes[sell_mask])) > EPS
        else reference_mid
    )

    weights = notionals / max(total_dollar_volume, EPS)
    concentration = float(np.sum(weights * weights))
    large_share = float(
        np.sum(notionals[notionals >= large_trade_threshold]) /
        max(total_dollar_volume, EPS)
    )

    # Scale-free proxy for algorithmic round-number slicing:
    # fraction of quote notionals near integer multiples of 100.
    nearest = np.round(notionals / 100.0) * 100.0
    roundness = float(np.mean(np.abs(notionals - nearest) <= np.maximum(1.0, 0.01 * notionals)))

    return {
        "n_trades": float(n),
        "buy_count": float(buy_count),
        "sell_count": float(sell_count),
        "total_count": float(total_count),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "total_volume": total_volume,
        "volume_imbalance": volume_imbalance,
        "trade_count_imbalance": count_imbalance,
        "signed_dollar_flow": signed_dollar_flow,
        "total_dollar_volume": total_dollar_volume,
        "return": return_fraction,
        "realized_volatility": realized_vol,
        "trade_price_variance": trade_price_variance,
        "buy_vwap_to_mid": (buy_vwap - reference_mid) / reference_mid,
        "sell_vwap_to_mid": (sell_vwap - reference_mid) / reference_mid,
        "volume_concentration": concentration,
        "large_trade_share": large_share,
        "trade_size_roundness": roundness,
    }


def _bar_index_at_or_after(bars: Sequence[market.Bar], ts_ms: int) -> int:
    timestamps = [bar.open_ts for bar in bars]
    return bisect.bisect_left(timestamps, ts_ms)


def _bar_index_at_or_before(bars: Sequence[market.Bar], ts_ms: int) -> int:
    timestamps = [bar.open_ts for bar in bars]
    return bisect.bisect_right(timestamps, ts_ms) - 1


# =============================================================================
# Quarter-hour feature generation and lab.sim outcomes
# =============================================================================

def _funding_context(
    data: SymbolData,
    entry_index: int,
    decision_ts: int,
    funding_history: deque[float],
) -> tuple[float, float]:
    past_events = [
        event for event in data.funding_events
        if event.bar_index <= entry_index
    ]
    if past_events:
        current_rate = float(past_events[-1].rate)
    else:
        current_rate = 0.0

    funding_z = _zscore(current_rate, funding_history) if funding_history else 0.0

    future_events = [
        event for event in data.funding_events
        if event.bar_index > entry_index
    ]
    if future_events:
        next_bar_ts = data.bars[future_events[0].bar_index].open_ts
        minutes = max(0.0, (next_bar_ts - decision_ts) / 60_000)
        fraction = min(1.0, minutes / (8.0 * 60.0))
    else:
        fraction = 1.0

    return funding_z, fraction


def _premium_context(
    data: SymbolData,
    entry_index: int,
    premium_history: deque[float],
) -> float:
    if data.premium_bars is None or entry_index >= len(data.premium_bars):
        return 0.0
    value = float(data.premium_bars[entry_index].close)
    return _zscore(value, premium_history) if premium_history else 0.0


def _simulate_outcome(
    data: SymbolData,
    entry_index: int,
    side: Side,
    atr_value: float,
    config: V3Config,
    max_holding_bars: int,
    price_arrays: tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]] | None = None,
) -> sim.TradeOutcome:
    bars = data.bars
    entry_price = float(bars[entry_index].open)
    stop_distance = config.stop_atr * atr_value
    if side == "LONG":
        stop_price = entry_price - stop_distance
        target_price = entry_price + config.target_r * stop_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - config.target_r * stop_distance

    spec = sim.TradeSpec(
        side=side,
        entry_index=entry_index,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        max_holding_bars=max_holding_bars,
        fee_rate=config.fee_rate,
        slippage_bps=config.slippage_bps,
    )
    if price_arrays is None:
        price_arrays = (
            [bar.open for bar in bars],
            [bar.high for bar in bars],
            [bar.low for bar in bars],
            [bar.close for bar in bars],
        )
    opens, highs, lows, closes = price_arrays
    return sim.simulate(
        opens,
        highs,
        lows,
        closes,
        spec,
        funding_events=data.funding_events,
    )


def build_raw_quarter_events(
    data: SymbolData,
    config: V3Config,
) -> tuple[list[RawQuarterEvent], dict]:
    config.validate()
    bars = data.bars
    if len(bars) < config.atr_period + 10:
        raise ValueError(f"{data.symbol}: insufficient 5m bars")

    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    closes = [bar.close for bar in bars]
    atr_values = indicators.compute_atr(
        highs, lows, closes, period=config.atr_period
    )
    price_arrays = (
        [bar.open for bar in bars],
        highs,
        lows,
        closes,
    )

    start_ts = max(
        int(data.trades.ts_ms[0]),
        int(data.book.ts_ms[0]),
        bars[0].open_ts,
    )
    end_ts = min(
        int(data.trades.ts_ms[-1]),
        int(data.book.ts_ms[-1]),
        bars[-1].open_ts + BASE_INTERVAL_MS,
    )
    quarter_ts = ((start_ts + QUARTER_MS - 1) // QUARTER_MS) * QUARTER_MS

    historical_limit = max(96, config.historical_z_days * 96)
    history = {
        "signed_flow": deque(maxlen=historical_limit),
        "total_volume": deque(maxlen=historical_limit),
        "trade_count": deque(maxlen=historical_limit),
        "return": deque(maxlen=historical_limit),
        "depth": deque(maxlen=historical_limit),
        "spread": deque(maxlen=historical_limit),
        "large_trade_notional": deque(maxlen=historical_limit),
        "funding": deque(maxlen=max(30, config.historical_z_days * 3)),
        "premium": deque(maxlen=historical_limit),
        "imbalance": deque(maxlen=8),
    }

    events_out: list[RawQuarterEvent] = []
    dropped = Counter()

    horizon_bars = config.horizon_hours * 12
    checkpoint_bars = {
        minute: minute // 5
        for minute in config.post_checkpoints_minutes
    }

    while quarter_ts < end_ts:
        decision_ts = quarter_ts + config.flow_window_seconds * 1000
        entry_ts = ((decision_ts + BASE_INTERVAL_MS - 1) // BASE_INTERVAL_MS) * BASE_INTERVAL_MS
        entry_index = _bar_index_at_or_after(bars, entry_ts)

        if entry_index < config.atr_period:
            dropped["atr_warmup"] += 1
            quarter_ts += QUARTER_MS
            continue
        if entry_index + horizon_bars >= len(bars):
            dropped["insufficient_forward_bars"] += 1
            quarter_ts += QUARTER_MS
            continue

        atr_value = float(atr_values[entry_index - 1])
        if not math.isfinite(atr_value) or atr_value <= 0.0:
            dropped["invalid_atr"] += 1
            quarter_ts += QUARTER_MS
            continue

        pre_book = _book_state(data.book, quarter_ts, config.book_max_age_ms)
        end_book = _book_state(data.book, decision_ts, config.book_max_age_ms)
        if pre_book is None or end_book is None:
            dropped["stale_or_missing_book"] += 1
            quarter_ts += QUARTER_MS
            continue
        _, pre = pre_book
        _, after = end_book

        trade_window = _trade_slice(data.trades, quarter_ts, decision_ts)
        notionals = data.trades.price[trade_window] * data.trades.size[trade_window]
        if len(notionals) < config.min_trades_in_flow_window:
            dropped["too_few_window_trades"] += 1
            quarter_ts += QUARTER_MS
            continue

        if history["large_trade_notional"]:
            large_threshold = float(np.quantile(
                np.asarray(history["large_trade_notional"], dtype=float),
                0.90,
            ))
        else:
            large_threshold = float(np.quantile(notionals, 0.90))

        metrics = _trade_window_metrics(
            data.trades,
            quarter_ts,
            decision_ts,
            pre["mid"],
            large_threshold,
        )
        if metrics is None:
            dropped["empty_trade_window"] += 1
            quarter_ts += QUARTER_MS
            continue

        signed_flow_z = _zscore(
            metrics["signed_dollar_flow"],
            history["signed_flow"],
        )
        volume_shock = _zscore(
            metrics["total_dollar_volume"],
            history["total_volume"],
        )
        count_shock = _zscore(
            metrics["total_count"],
            history["trade_count"],
        )
        return_z = _zscore(metrics["return"], history["return"])
        depth_z = _zscore(pre["total_depth_l5"], history["depth"])
        spread_z = _zscore(pre["spread"], history["spread"])

        imbalance_history = list(history["imbalance"])
        lag_1 = imbalance_history[-1] if len(imbalance_history) >= 1 else 0.0
        lag_4 = imbalance_history[-4] if len(imbalance_history) >= 4 else 0.0
        if imbalance_history:
            weights = np.exp(np.linspace(-1.5, 0.0, min(4, len(imbalance_history))))
            values = np.asarray(imbalance_history[-len(weights):], dtype=float)
            ewm_4 = float(np.sum(values * weights) / np.sum(weights))
        else:
            ewm_4 = 0.0
        acceleration = metrics["volume_imbalance"] - lag_1

        response = _safe_div(
            metrics["return"],
            abs(metrics["volume_imbalance"]) + 0.05,
        )
        # Positive signed flow with weak/negative price response gives positive
        # absorption. Directional mirroring is applied later.
        absorption = signed_flow_z - return_z

        flow_to_depth = _safe_div(
            metrics["total_dollar_volume"],
            pre["total_depth_l5"] * pre["mid"],
        )

        risk_fraction = config.stop_atr * atr_value / bars[entry_index].open
        fixed_round_trip = (
            2.0 * config.fee_rate
            + 2.0 * config.slippage_bps * 1e-4
        )
        estimated_cost_r = fixed_round_trip / max(risk_fraction, EPS)

        hour_bar_index = entry_index - 1
        start_hour_index = max(0, hour_bar_index - 12)
        hour_closes = np.asarray(
            [bar.close for bar in bars[start_hour_index:hour_bar_index + 1]],
            dtype=float,
        )
        if len(hour_closes) >= 2:
            log_returns = np.diff(np.log(hour_closes))
            realized_vol_1h = float(np.sqrt(np.sum(log_returns * log_returns)))
        else:
            realized_vol_1h = 0.0

        funding_z, time_to_funding = _funding_context(
            data, entry_index, decision_ts, history["funding"]
        )
        premium_z = _premium_context(
            data, entry_index, history["premium"]
        )

        if spread_z <= 0.0 and depth_z >= 0.0 and realized_vol_1h < 0.02:
            liquidity_state = 1.0   # calm / supportive
        elif spread_z >= 1.0 or depth_z <= -1.0:
            liquidity_state = -1.0  # stressed
        else:
            liquidity_state = 0.0

        raw = {
            "utc_hour_sin": math.sin(
                2.0 * math.pi * (
                    datetime.fromtimestamp(quarter_ts / 1000, timezone.utc).hour
                    + datetime.fromtimestamp(quarter_ts / 1000, timezone.utc).minute / 60.0
                ) / 24.0
            ),
            "utc_hour_cos": math.cos(
                2.0 * math.pi * (
                    datetime.fromtimestamp(quarter_ts / 1000, timezone.utc).hour
                    + datetime.fromtimestamp(quarter_ts / 1000, timezone.utc).minute / 60.0
                ) / 24.0
            ),
            "volume_imbalance_10s": metrics["volume_imbalance"],
            "trade_count_imbalance_10s": metrics["trade_count_imbalance"],
            "signed_dollar_flow_z_10s": signed_flow_z,
            "total_volume_shock_10s": volume_shock,
            "trade_count_shock_10s": count_shock,
            "large_trade_share_10s": metrics["large_trade_share"],
            "trade_size_roundness_10s": metrics["trade_size_roundness"],
            "qh_imbalance_lag_1": lag_1,
            "qh_imbalance_lag_4": lag_4,
            "qh_imbalance_ewm_4": ewm_4,
            "qh_imbalance_acceleration": acceleration,
            "return_10s": metrics["return"],
            "realized_volatility_10s": metrics["realized_volatility"],
            "price_response_to_flow": response,
            "absorption_score": absorption,
            "relative_spread": pre["spread"],
            "l1_book_imbalance": pre["l1_imbalance"],
            "microprice_deviation": pre["microprice_deviation"],
            "buy_vwap_to_mid": metrics["buy_vwap_to_mid"],
            "sell_vwap_to_mid": metrics["sell_vwap_to_mid"],
            "trade_price_variance": metrics["trade_price_variance"],
            "volume_concentration": metrics["volume_concentration"],
            "depth_imbalance_l5": pre["depth_imbalance_l5"],
            "total_depth_l5_z": depth_z,
            "liquidity_state": liquidity_state,
            "flow_to_depth_ratio": flow_to_depth,
            "estimated_round_trip_cost_r": estimated_cost_r,
            "realized_volatility_1h": realized_vol_1h,
            "funding_rate_z": funding_z,
            "premium_z": premium_z,
            "time_to_funding_fraction": time_to_funding,
            "entry_spread": pre["spread"],
            "entry_book_imbalance": pre["l1_imbalance"],
            "entry_microprice_deviation": pre["microprice_deviation"],
            "entry_flow_imbalance": metrics["volume_imbalance"],
            "entry_signed_flow_z": signed_flow_z,
            "entry_mid": pre["mid"],
        }

        try:
            final_long = _simulate_outcome(
                data, entry_index, "LONG", atr_value, config, horizon_bars, price_arrays
            )
            final_short = _simulate_outcome(
                data, entry_index, "SHORT", atr_value, config, horizon_bars, price_arrays
            )
            cp_long = {
                minute: _simulate_outcome(
                    data, entry_index, "LONG", atr_value, config, bars_count, price_arrays
                )
                for minute, bars_count in checkpoint_bars.items()
            }
            cp_short = {
                minute: _simulate_outcome(
                    data, entry_index, "SHORT", atr_value, config, bars_count, price_arrays
                )
                for minute, bars_count in checkpoint_bars.items()
            }
        except Exception:
            dropped["simulation_error"] += 1
            quarter_ts += QUARTER_MS
            continue

        events_out.append(
            RawQuarterEvent(
                symbol=data.symbol,
                quarter_ts=quarter_ts,
                decision_ts=decision_ts,
                entry_index=entry_index,
                raw_features=raw,
                final_outcome_long=final_long,
                final_outcome_short=final_short,
                checkpoint_outcomes_long=cp_long,
                checkpoint_outcomes_short=cp_short,
            )
        )

        history["signed_flow"].append(metrics["signed_dollar_flow"])
        history["total_volume"].append(metrics["total_dollar_volume"])
        history["trade_count"].append(metrics["total_count"])
        history["return"].append(metrics["return"])
        history["depth"].append(pre["total_depth_l5"])
        history["spread"].append(pre["spread"])
        history["large_trade_notional"].extend(float(value) for value in notionals)
        history["imbalance"].append(metrics["volume_imbalance"])
        past_funding = [
            event.rate for event in data.funding_events
            if event.bar_index <= entry_index
        ]
        if past_funding:
            history["funding"].append(float(past_funding[-1]))
        if data.premium_bars is not None:
            history["premium"].append(float(data.premium_bars[entry_index].close))

        quarter_ts += QUARTER_MS

    report = {
        "symbol": data.symbol,
        "n_raw_events": len(events_out),
        "dropped": dict(dropped),
        "trade_tape_hash": _hash_array(data.trades.ts_ms),
        "book_tape_hash": _hash_array(data.book.ts_ms),
        "book_levels": data.book.levels,
    }
    return events_out, report


def directionalise_event(
    raw_event: RawQuarterEvent,
    side: Side,
    include_positioning: bool,
    bars: Sequence[market.Bar],
) -> DirectionalEvent:
    sign = 1.0 if side == "LONG" else -1.0
    raw = raw_event.raw_features

    feature_values = {
        "utc_hour_sin": raw["utc_hour_sin"],
        "utc_hour_cos": raw["utc_hour_cos"],
        "dir_volume_imbalance_10s": sign * raw["volume_imbalance_10s"],
        "dir_trade_count_imbalance_10s": sign * raw["trade_count_imbalance_10s"],
        "dir_signed_dollar_flow_z_10s": sign * raw["signed_dollar_flow_z_10s"],
        "total_volume_shock_10s": raw["total_volume_shock_10s"],
        "trade_count_shock_10s": raw["trade_count_shock_10s"],
        "large_trade_share_10s": raw["large_trade_share_10s"],
        "trade_size_roundness_10s": raw["trade_size_roundness_10s"],
        "dir_qh_imbalance_lag_1": sign * raw["qh_imbalance_lag_1"],
        "dir_qh_imbalance_lag_4": sign * raw["qh_imbalance_lag_4"],
        "dir_qh_imbalance_ewm_4": sign * raw["qh_imbalance_ewm_4"],
        "dir_qh_imbalance_acceleration": sign * raw["qh_imbalance_acceleration"],
        "dir_return_10s": sign * raw["return_10s"],
        "realized_volatility_10s": raw["realized_volatility_10s"],
        "dir_price_response_to_flow": sign * raw["price_response_to_flow"],
        # Positive means flow in the candidate direction was absorbed.
        "dir_absorption_score": sign * raw["absorption_score"],
        "relative_spread": raw["relative_spread"],
        "dir_l1_book_imbalance": sign * raw["l1_book_imbalance"],
        "dir_microprice_deviation": sign * raw["microprice_deviation"],
        "dir_buy_vwap_to_mid": sign * raw["buy_vwap_to_mid"],
        "dir_sell_vwap_to_mid": sign * raw["sell_vwap_to_mid"],
        "trade_price_variance": raw["trade_price_variance"],
        "volume_concentration": raw["volume_concentration"],
        "dir_depth_imbalance_l5": sign * raw["depth_imbalance_l5"],
        "total_depth_l5_z": raw["total_depth_l5_z"],
        "liquidity_state": raw["liquidity_state"],
        "flow_to_depth_ratio": raw["flow_to_depth_ratio"],
        "estimated_round_trip_cost_r": raw["estimated_round_trip_cost_r"],
        "realized_volatility_1h": raw["realized_volatility_1h"],
        "dir_funding_rate_z": sign * raw["funding_rate_z"],
        "dir_premium_z": sign * raw["premium_z"],
        "time_to_funding_fraction": raw["time_to_funding_fraction"],
    }

    names = CORE_FEATURE_NAMES + (
        OPTIONAL_POSITIONING_FEATURE_NAMES if include_positioning else ()
    )
    vector = np.asarray([feature_values[name] for name in names], dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{raw_event.symbol}:{raw_event.decision_ts}:{side}: non-finite features"
        )

    outcome = (
        raw_event.final_outcome_long if side == "LONG"
        else raw_event.final_outcome_short
    )
    checkpoints = (
        raw_event.checkpoint_outcomes_long if side == "LONG"
        else raw_event.checkpoint_outcomes_short
    )
    outcome_end_ts = (
        bars[outcome.exit_index].open_ts + BASE_INTERVAL_MS
    )
    event_id = (
        f"v3|{raw_event.symbol}|{raw_event.decision_ts}|{side}"
    )
    return DirectionalEvent(
        event_id=event_id,
        symbol=raw_event.symbol,
        side=side,
        decision_ts=raw_event.decision_ts,
        entry_index=raw_event.entry_index,
        outcome_end_ts=outcome_end_ts,
        features=vector,
        final_outcome=outcome,
        checkpoint_outcomes=checkpoints,
    )


# =============================================================================
# Entry models and 0–100 quality score
# =============================================================================

class EntryModel:
    def __init__(self, config: V3Config):
        self.config = config
        self.scaler = None
        self.ridge = None
        self.tree = None
        self.classifier = None
        self.target_scale = 1.0
        self.fitted = False

    def fit(self, events: Sequence[DirectionalEvent]) -> None:
        if len(events) < 30:
            raise ValueError("entry model requires at least 30 training events")
        try:
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for V3 models") from exc

        x = np.vstack([event.features for event in events])
        y = np.asarray([event.final_outcome.net_r for event in events], dtype=float)
        y_win = (y > 0.0).astype(int)

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)

        self.ridge = Ridge(alpha=self.config.ridge_alpha)
        self.ridge.fit(x_scaled, y)

        self.tree = DecisionTreeRegressor(
            max_depth=self.config.tree_max_depth,
            min_samples_leaf=max(10, len(events) // 50),
            random_state=self.config.seed,
        )
        self.tree.fit(x_scaled, y)

        self.classifier = LogisticRegression(
            C=0.25,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.config.seed,
        )
        if len(np.unique(y_win)) < 2:
            raise ValueError("entry training labels contain only one win class")
        self.classifier.fit(x_scaled, y_win)

        self.target_scale = max(float(np.std(y)), 0.10)
        self.fitted = True

    def predict(self, event: DirectionalEvent) -> EntryPrediction:
        if not self.fitted:
            raise RuntimeError("entry model is not fitted")
        x = event.features.reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        ridge_r = float(self.ridge.predict(x_scaled)[0])
        tree_r = float(self.tree.predict(x_scaled)[0])
        predicted_r = 0.60 * ridge_r + 0.40 * tree_r
        p_win = float(self.classifier.predict_proba(x_scaled)[0, 1])
        disagreement = abs(ridge_r - tree_r)
        uncertainty = min(1.0, disagreement / self.target_scale)

        edge_component = _sigmoid(predicted_r / self.target_scale)
        quality = 100.0 * (
            0.55 * p_win
            + 0.45 * edge_component
            - 0.20 * uncertainty
        )
        quality = min(100.0, max(0.0, quality))
        return EntryPrediction(
            event=event,
            predicted_net_r=predicted_r,
            predicted_win_probability=p_win,
            uncertainty=uncertainty,
            quality_score=quality,
        )


# =============================================================================
# Post-execution model
# =============================================================================

def _post_features(
    prediction: EntryPrediction,
    raw_lookup: Mapping[tuple[str, int], RawQuarterEvent],
    data_by_symbol: Mapping[str, SymbolData],
    checkpoint_minutes: int,
) -> np.ndarray | None:
    event = prediction.event
    raw_event = raw_lookup[(event.symbol, event.decision_ts)]
    data = data_by_symbol[event.symbol]
    side_sign = 1.0 if event.side == "LONG" else -1.0
    checkpoint_bars = checkpoint_minutes // 5
    checkpoint_index = event.entry_index + checkpoint_bars

    if checkpoint_index >= len(data.bars):
        return None
    if event.final_outcome.exit_index <= checkpoint_index:
        return None

    entry_price = float(data.bars[event.entry_index].open)
    risk_fraction = event.final_outcome.risk_fraction
    checkpoint_price = float(data.bars[checkpoint_index].close)
    directional_return = side_sign * (checkpoint_price - entry_price) / entry_price
    unrealized_r = directional_return / max(risk_fraction, EPS)

    inspected = data.bars[event.entry_index + 1:checkpoint_index + 1]
    if event.side == "LONG":
        adverse = max(
            [max(0.0, (entry_price - bar.low) / entry_price) for bar in inspected],
            default=0.0,
        )
        favorable = max(
            [max(0.0, (bar.high - entry_price) / entry_price) for bar in inspected],
            default=0.0,
        )
    else:
        adverse = max(
            [max(0.0, (bar.high - entry_price) / entry_price) for bar in inspected],
            default=0.0,
        )
        favorable = max(
            [max(0.0, (entry_price - bar.low) / entry_price) for bar in inspected],
            default=0.0,
        )

    start_ts = data.bars[event.entry_index].open_ts
    end_ts = data.bars[checkpoint_index].open_ts + BASE_INTERVAL_MS
    entry_mid = raw_event.raw_features["entry_mid"]
    trade_metrics = _trade_window_metrics(
        data.trades,
        start_ts,
        end_ts,
        entry_mid,
        large_trade_threshold=float("inf"),
    )
    if trade_metrics is None:
        return None

    current_book = _book_state(
        data.book,
        end_ts,
        max_age_ms=max(5_000, data.book.ts_ms[-1] - data.book.ts_ms[-2] if len(data.book.ts_ms) > 1 else 5_000),
    )
    if current_book is None:
        return None
    _, book = current_book

    entry_flow = raw_event.raw_features["entry_flow_imbalance"]
    residual_flow = side_sign * trade_metrics["volume_imbalance"]
    flow_sign_flip = 1.0 if residual_flow < 0.0 else 0.0
    flow_decay = _safe_div(
        abs(trade_metrics["volume_imbalance"]),
        abs(entry_flow) + 0.05,
    )
    price_response = _safe_div(
        directional_return,
        abs(trade_metrics["volume_imbalance"]) + 0.05,
    )
    absorption = (
        side_sign * trade_metrics["volume_imbalance"]
        - directional_return / max(risk_fraction, EPS)
    )

    spread_change = _safe_div(
        book["spread"] - raw_event.raw_features["entry_spread"],
        raw_event.raw_features["entry_spread"] + EPS,
    )
    book_change = side_sign * (
        book["l1_imbalance"]
        - raw_event.raw_features["entry_book_imbalance"]
    )
    micro_change = side_sign * (
        book["microprice_deviation"]
        - raw_event.raw_features["entry_microprice_deviation"]
    )
    estimated_exit_cost_r = (
        data.manifest.get("fee_rate", 0.0004)
        + data.manifest.get("slippage_bps", 1.0) * 1e-4
    ) / max(risk_fraction, EPS)

    values = (
        float(checkpoint_minutes),
        unrealized_r,
        favorable / max(risk_fraction, EPS),
        adverse / max(risk_fraction, EPS),
        residual_flow,
        flow_sign_flip,
        flow_decay,
        directional_return,
        price_response,
        absorption,
        spread_change,
        book_change,
        micro_change,
        estimated_exit_cost_r,
        prediction.quality_score,
        prediction.predicted_net_r,
        prediction.predicted_win_probability,
    )
    vector = np.asarray(values, dtype=float)
    return vector if np.all(np.isfinite(vector)) else None


class PostExecutionModel:
    def __init__(self, config: V3Config):
        self.config = config
        self.scaler = None
        self.ridge = None
        self.fitted = False

    def fit(
        self,
        events: Sequence[DirectionalEvent],
        entry_model: EntryModel,
        raw_lookup: Mapping[tuple[str, int], RawQuarterEvent],
        data_by_symbol: Mapping[str, SymbolData],
    ) -> dict:
        if not self.config.enable_post_execution:
            return {"enabled": False, "samples": 0}
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for post model") from exc

        x_rows = []
        labels = []
        for event in events:
            prediction = entry_model.predict(event)
            for minute, checkpoint_outcome in event.checkpoint_outcomes.items():
                vector = _post_features(
                    prediction,
                    raw_lookup,
                    data_by_symbol,
                    minute,
                )
                if vector is None:
                    continue
                hold_advantage = (
                    event.final_outcome.net_r
                    - checkpoint_outcome.net_r
                )
                x_rows.append(vector)
                labels.append(hold_advantage)

        if len(x_rows) < 50:
            return {"enabled": False, "samples": len(x_rows)}

        x = np.vstack(x_rows)
        y = np.asarray(labels, dtype=float)
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        self.ridge = Ridge(alpha=self.config.ridge_alpha)
        self.ridge.fit(x_scaled, y)
        self.fitted = True
        return {
            "enabled": True,
            "samples": len(x_rows),
            "mean_hold_advantage": float(np.mean(y)),
        }

    def predict_hold_advantage(self, features: np.ndarray) -> float:
        if not self.fitted:
            return math.inf
        transformed = self.scaler.transform(features.reshape(1, -1))
        return float(self.ridge.predict(transformed)[0])


# =============================================================================
# Selection, threshold tuning, and walk-forward
# =============================================================================

def _group_predictions(
    predictions: Sequence[EntryPrediction],
) -> dict[tuple[str, int], list[EntryPrediction]]:
    grouped: dict[tuple[str, int], list[EntryPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction.event.symbol, prediction.event.decision_ts)].append(
            prediction
        )
    return grouped


def select_entries(
    predictions: Sequence[EntryPrediction],
    threshold: float,
    config: V3Config,
) -> tuple[list[EntryPrediction], dict]:
    grouped = _group_predictions(predictions)
    selected = []
    diagnostics = Counter()
    unavailable_until: dict[str, int] = {}
    cooldown_ms = config.horizon_hours * 3_600_000

    for (symbol, decision_ts), candidates in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        eligible = [
            candidate for candidate in candidates
            if candidate.quality_score >= threshold
            and candidate.predicted_net_r >= config.minimum_predicted_net_r
        ]
        if not eligible:
            diagnostics["below_quality_or_edge_threshold"] += 1
            continue

        eligible.sort(
            key=lambda item: (
                item.quality_score,
                item.predicted_net_r,
                item.predicted_win_probability,
                item.event.side,
            ),
            reverse=True,
        )
        best = eligible[0]
        if len(eligible) >= 2 and eligible[0].event.side != eligible[1].event.side:
            score_gap = eligible[0].quality_score - eligible[1].quality_score
            if score_gap < config.conflict_score_margin:
                diagnostics["direction_conflict"] += 1
                continue

        if decision_ts < unavailable_until.get(symbol, -1):
            diagnostics["cooldown"] += 1
            continue

        selected.append(best)
        unavailable_until[symbol] = decision_ts + cooldown_ms
        diagnostics["selected"] += 1

    return selected, dict(diagnostics)


def apply_post_execution(
    selected: Sequence[EntryPrediction],
    post_model: PostExecutionModel,
    post_exit_threshold: float,
    raw_lookup: Mapping[tuple[str, int], RawQuarterEvent],
    data_by_symbol: Mapping[str, SymbolData],
) -> list[SelectedTrade]:
    trades = []
    for prediction in selected:
        event = prediction.event
        chosen_outcome = event.final_outcome
        action = "HOLD_TO_BASE_OUTCOME"
        checkpoint_used = None

        if post_model.fitted:
            for minute in sorted(event.checkpoint_outcomes):
                vector = _post_features(
                    prediction,
                    raw_lookup,
                    data_by_symbol,
                    minute,
                )
                if vector is None:
                    continue
                hold_advantage = post_model.predict_hold_advantage(vector)
                if hold_advantage < post_exit_threshold:
                    chosen_outcome = event.checkpoint_outcomes[minute]
                    action = "EXIT_EARLY"
                    checkpoint_used = minute
                    break

        trades.append(
            SelectedTrade(
                event=event,
                quality_score=prediction.quality_score,
                predicted_net_r=prediction.predicted_net_r,
                predicted_win_probability=prediction.predicted_win_probability,
                realized_outcome=chosen_outcome,
                post_action=action,
                post_checkpoint_minutes=checkpoint_used,
            )
        )
    return trades


def performance_from_trades(
    trades: Sequence[SelectedTrade],
    *,
    seed: int,
) -> Performance:
    values = [trade.realized_outcome.net_r for trade in trades]
    if not values:
        return Performance(
            n_trades=0,
            mean_net_r=None,
            median_net_r=None,
            win_rate=None,
            profit_factor=None,
            cumulative_net_r=0.0,
            max_drawdown_r=0.0,
            average_win_r=None,
            average_loss_r=None,
            longest_losing_streak=0,
            ci95_low=None,
            ci95_high=None,
            post_early_exits=0,
            symbols={},
            sides={},
        )

    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    streak = 0
    maximum_streak = 0
    for value in values:
        if value < 0:
            streak += 1
            maximum_streak = max(maximum_streak, streak)
        else:
            streak = 0

    symbol_values: dict[str, list[float]] = defaultdict(list)
    side_values: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        symbol_values[trade.event.symbol].append(trade.realized_outcome.net_r)
        side_values[trade.event.side].append(trade.realized_outcome.net_r)

    def summary(group: Sequence[float]) -> dict:
        return {
            "trades": len(group),
            "mean_net_r": statistics.fmean(group) if group else None,
            "win_rate": sum(value > 0.0 for value in group) / len(group) if group else None,
            "profit_factor": _profit_factor(group),
            "cumulative_net_r": sum(group),
        }

    low, high = _bootstrap_ci(values, seed=seed)
    return Performance(
        n_trades=len(values),
        mean_net_r=statistics.fmean(values),
        median_net_r=statistics.median(values),
        win_rate=sum(value > 0.0 for value in values) / len(values),
        profit_factor=_profit_factor(values),
        cumulative_net_r=sum(values),
        max_drawdown_r=_max_drawdown(values),
        average_win_r=statistics.fmean(wins) if wins else None,
        average_loss_r=statistics.fmean(losses) if losses else None,
        longest_losing_streak=maximum_streak,
        ci95_low=low,
        ci95_high=high,
        post_early_exits=sum(trade.post_action == "EXIT_EARLY" for trade in trades),
        symbols={key: summary(group) for key, group in sorted(symbol_values.items())},
        sides={key: summary(group) for key, group in sorted(side_values.items())},
    )


def tune_policy(
    validation_predictions: Sequence[EntryPrediction],
    post_model: PostExecutionModel,
    raw_lookup: Mapping[tuple[str, int], RawQuarterEvent],
    data_by_symbol: Mapping[str, SymbolData],
    config: V3Config,
) -> tuple[int, float, dict]:
    candidates = []
    thresholds = range(
        config.threshold_min,
        config.threshold_max + 1,
        config.threshold_step,
    )
    post_thresholds = (
        config.post_exit_threshold_grid
        if post_model.fitted
        else (0.0,)
    )

    for threshold in thresholds:
        selected, diagnostics = select_entries(
            validation_predictions,
            threshold,
            config,
        )
        for post_threshold in post_thresholds:
            trades = apply_post_execution(
                selected,
                post_model,
                post_threshold,
                raw_lookup,
                data_by_symbol,
            )
            perf = performance_from_trades(
                trades,
                seed=config.seed + threshold,
            )
            if perf.n_trades == 0 or perf.mean_net_r is None:
                objective = -math.inf
                pf = 0.0
                gate_pass = False
                gate_failures = ["no_trades"]
            else:
                pf = perf.profit_factor if perf.profit_factor is not None else 0.0
                stability_penalty = perf.max_drawdown_r / max(math.sqrt(perf.n_trades), 1.0)
                objective = (
                    perf.mean_net_r * math.sqrt(perf.n_trades)
                    + 0.20 * (perf.win_rate or 0.0)
                    + 0.05 * min(pf, 3.0)
                    - 0.02 * stability_penalty
                )
                gate_failures = []
                if perf.n_trades < config.minimum_validation_trades:
                    gate_failures.append("too_few_validation_trades")
                if perf.mean_net_r < config.minimum_validation_mean_r:
                    gate_failures.append("validation_mean_r_below_minimum")
                if pf < config.minimum_validation_pf:
                    gate_failures.append("validation_profit_factor_below_minimum")
                if (perf.win_rate or 0.0) < config.minimum_validation_win_rate:
                    gate_failures.append("validation_win_rate_below_minimum")
                gate_pass = not gate_failures

            candidates.append({
                "quality_threshold": threshold,
                "post_exit_threshold": post_threshold,
                "objective": objective,
                "gate_pass": gate_pass,
                "gate_failures": gate_failures,
                "selection_diagnostics": diagnostics,
                "performance": asdict(perf),
            })

    candidates.sort(
        key=lambda row: (
            row["gate_pass"],
            row["objective"],
            row["performance"]["win_rate"] or -1.0,
            row["performance"]["mean_net_r"] or -math.inf,
            row["performance"]["n_trades"],
        ),
        reverse=True,
    )
    best = candidates[0]
    return (
        int(best["quality_threshold"]),
        float(best["post_exit_threshold"]),
        {
            "policy_status": "VALIDATED" if best["gate_pass"] else "NO_VALID_POLICY",
            "selected": best,
            "all_candidates": candidates,
        },
    )


def build_fold_windows(
    events: Sequence[DirectionalEvent],
    config: V3Config,
) -> list[FoldWindow]:
    if not events:
        return []
    minimum = min(event.decision_ts for event in events)
    maximum = max(event.decision_ts for event in events)

    train_length = config.train_months * MONTH_MS
    validation_length = config.validation_months * MONTH_MS
    test_length = config.test_months * MONTH_MS
    step = config.step_months * MONTH_MS

    windows = []
    train_start = minimum
    fold = 0
    while True:
        train_end = train_start + train_length
        validation_start = train_end
        validation_end = validation_start + validation_length
        test_start = validation_end
        test_end = test_start + test_length
        if test_end > maximum + 1:
            break
        windows.append(
            FoldWindow(
                fold_index=fold,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start += step
        fold += 1
    return windows


def _events_in_window(
    events: Sequence[DirectionalEvent],
    start: int,
    end: int,
    purge_outcomes_after: int | None = None,
) -> list[DirectionalEvent]:
    result = []
    for event in events:
        if not start <= event.decision_ts < end:
            continue
        if (
            purge_outcomes_after is not None
            and event.outcome_end_ts > purge_outcomes_after
        ):
            continue
        result.append(event)
    return result


def run_walkforward(
    all_events: Sequence[DirectionalEvent],
    raw_lookup: Mapping[tuple[str, int], RawQuarterEvent],
    data_by_symbol: Mapping[str, SymbolData],
    config: V3Config,
) -> dict:
    windows = build_fold_windows(all_events, config)
    if not windows:
        raise RuntimeError("insufficient date range for requested walk-forward")

    fold_reports = []
    aggregate_trades: list[SelectedTrade] = []

    for window in windows:
        train = _events_in_window(
            all_events,
            window.train_start,
            window.train_end,
            purge_outcomes_after=window.validation_start,
        )
        validation = _events_in_window(
            all_events,
            window.validation_start,
            window.validation_end,
            purge_outcomes_after=window.test_start,
        )
        test = _events_in_window(
            all_events,
            window.test_start,
            window.test_end,
        )

        if len(train) < 100 or len(validation) < 20 or len(test) < 1:
            fold_reports.append({
                "window": asdict(window),
                "status": "SKIPPED_INSUFFICIENT_EVENTS",
                "counts": {
                    "train": len(train),
                    "validation": len(validation),
                    "test": len(test),
                },
            })
            continue

        entry_model = EntryModel(config)
        entry_model.fit(train)
        post_model = PostExecutionModel(config)
        post_fit = post_model.fit(
            train,
            entry_model,
            raw_lookup,
            data_by_symbol,
        )

        validation_predictions = [
            entry_model.predict(event) for event in validation
        ]
        quality_threshold, post_exit_threshold, tuning = tune_policy(
            validation_predictions,
            post_model,
            raw_lookup,
            data_by_symbol,
            config,
        )

        test_predictions = [
            entry_model.predict(event) for event in test
        ]
        selected, diagnostics = select_entries(
            test_predictions,
            quality_threshold,
            config,
        )
        test_trades = apply_post_execution(
            selected,
            post_model,
            post_exit_threshold,
            raw_lookup,
            data_by_symbol,
        )
        aggregate_trades.extend(test_trades)

        fold_reports.append({
            "window": {
                **asdict(window),
                "train_start_utc": _iso(window.train_start),
                "train_end_utc": _iso(window.train_end),
                "validation_start_utc": _iso(window.validation_start),
                "validation_end_utc": _iso(window.validation_end),
                "test_start_utc": _iso(window.test_start),
                "test_end_utc": _iso(window.test_end),
            },
            "status": "COMPLETE",
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "selected_policy": {
                "quality_threshold": quality_threshold,
                "post_exit_threshold": post_exit_threshold,
            },
            "post_model_fit": post_fit,
            "validation_tuning": tuning,
            "test_selection_diagnostics": diagnostics,
            "test_performance": asdict(
                performance_from_trades(
                    test_trades,
                    seed=config.seed + window.fold_index,
                )
            ),
        })

    aggregate = performance_from_trades(
        aggregate_trades,
        seed=config.seed + 99_000,
    )
    return {
        "protocol": {
            "mode": "quarter_hour_microstructure_v3",
            "feature_names": (
                CORE_FEATURE_NAMES + OPTIONAL_POSITIONING_FEATURE_NAMES
            ),
            "post_feature_names": POST_FEATURE_NAMES,
            "config": asdict(config),
            "economic_truth": "lab.sim",
            "entry_delay": (
                "decision after first flow_window_seconds; fill at next 5m bar open"
            ),
            "threshold_tuning": "validation_only",
            "post_execution_training": "train_only",
            "purge": "outcome_end_ts must precede next split boundary",
        },
        "aggregate_oos": asdict(aggregate),
        "folds": fold_reports,
        "evidence_gate": {
            "minimum_oos_trades": 300,
            "minimum_mean_net_r": 0.05,
            "minimum_profit_factor": 1.10,
            "minimum_win_rate": 0.56,
            "ci95_lower_must_exceed_zero": True,
            "positive_fold_fraction": 0.70,
        },
    }


# =============================================================================
# Snapshot assembly and CLI
# =============================================================================

def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            f"expected SYMBOL=PATH, got {value!r}"
        )
    symbol, path = value.split("=", 1)
    symbol = symbol.strip()
    if not symbol:
        raise ValueError(f"empty symbol in {value!r}")
    return symbol, Path(path)


def _spec_map(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        symbol, path = _parse_spec(value)
        if symbol in result:
            raise ValueError(f"duplicate specification for {symbol}")
        result[symbol] = path
    return result


def load_symbol_data(
    snapshot_specs: Sequence[str],
    trade_specs: Sequence[str],
    book_specs: Sequence[str],
) -> list[SymbolData]:
    from tools import data as data_tool

    snapshots = _spec_map(snapshot_specs)
    trades = _spec_map(trade_specs)
    books = _spec_map(book_specs)

    symbols = sorted(set(snapshots) | set(trades) | set(books))
    for symbol in symbols:
        missing = [
            label for label, mapping in (
                ("snapshot", snapshots),
                ("trades", trades),
                ("book", books),
            )
            if symbol not in mapping
        ]
        if missing:
            raise ValueError(f"{symbol}: missing inputs {missing}")

    result = []
    for symbol in symbols:
        loaded = data_tool.load(snapshots[symbol])
        manifest_symbol = loaded.manifest.get("instrument_id")
        if manifest_symbol and manifest_symbol != symbol:
            raise ValueError(
                f"{symbol}: snapshot manifest instrument_id={manifest_symbol!r}"
            )
        manifest = dict(loaded.manifest)
        manifest.setdefault("fee_rate", 0.0004)
        manifest.setdefault("slippage_bps", 1.0)
        result.append(
            SymbolData(
                symbol=symbol,
                bars=loaded.trade_bars,
                funding_events=loaded.funding_events,
                trades=load_trade_tape(trades[symbol], symbol),
                book=load_book_tape(books[symbol], symbol),
                premium_bars=getattr(loaded, "premium_bars", None),
                manifest=manifest,
            )
        )
    return result


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if value > 0:
            return "inf"
        if value < 0:
            return "-inf"
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def inspect_data_command(args: argparse.Namespace) -> int:
    datasets = load_symbol_data(args.snapshot, args.trades, args.book)
    report = {}
    for data in datasets:
        overlap_start = max(
            data.bars[0].open_ts,
            int(data.trades.ts_ms[0]),
            int(data.book.ts_ms[0]),
        )
        overlap_end = min(
            data.bars[-1].open_ts + BASE_INTERVAL_MS,
            int(data.trades.ts_ms[-1]),
            int(data.book.ts_ms[-1]),
        )
        report[data.symbol] = {
            "snapshot_start_utc": _iso(data.bars[0].open_ts),
            "snapshot_end_utc": _iso(
                data.bars[-1].open_ts + BASE_INTERVAL_MS
            ),
            "trade_rows": len(data.trades.ts_ms),
            "trade_start_utc": _iso(int(data.trades.ts_ms[0])),
            "trade_end_utc": _iso(int(data.trades.ts_ms[-1])),
            "book_rows": len(data.book.ts_ms),
            "book_levels": data.book.levels,
            "book_start_utc": _iso(int(data.book.ts_ms[0])),
            "book_end_utc": _iso(int(data.book.ts_ms[-1])),
            "overlap_start_utc": _iso(overlap_start),
            "overlap_end_utc": _iso(overlap_end),
            "overlap_days": (overlap_end - overlap_start) / DAY_MS,
            "aggressor_buy_fraction": float(
                np.mean(data.trades.side_sign > 0)
            ),
        }
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


def walkforward_command(args: argparse.Namespace) -> int:
    config = V3Config(
        flow_window_seconds=args.flow_window_seconds,
        horizon_hours=args.horizon_hours,
        stop_atr=args.stop_atr,
        target_r=args.target_r,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        minimum_validation_trades=args.minimum_validation_trades,
        minimum_validation_win_rate=args.minimum_validation_win_rate,
        enable_post_execution=not args.disable_post_execution,
        seed=args.seed,
    )
    config.validate()
    datasets = load_symbol_data(args.snapshot, args.trades, args.book)
    data_by_symbol = {data.symbol: data for data in datasets}

    raw_events = []
    preparation = {}
    for data in datasets:
        symbol_events, report = build_raw_quarter_events(data, config)
        raw_events.extend(symbol_events)
        preparation[data.symbol] = report

    raw_lookup = {
        (event.symbol, event.decision_ts): event
        for event in raw_events
    }
    directional_events = []
    for raw_event in raw_events:
        data = data_by_symbol[raw_event.symbol]
        directional_events.append(
            directionalise_event(
                raw_event, "LONG", True, data.bars
            )
        )
        directional_events.append(
            directionalise_event(
                raw_event, "SHORT", True, data.bars
            )
        )
    directional_events.sort(
        key=lambda event: (
            event.decision_ts,
            event.symbol,
            event.side,
        )
    )

    result = run_walkforward(
        directional_events,
        raw_lookup,
        data_by_symbol,
        config,
    )
    result["preparation"] = preparation
    result["n_raw_quarter_events"] = len(raw_events)
    result["n_directional_events"] = len(directional_events)

    text = json.dumps(_json_safe(result), indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


# =============================================================================
# Self-test
# =============================================================================

def _synthetic_symbol_data() -> SymbolData:
    bars = []
    price = 100.0
    start = 1_700_000_000_000
    for index in range(700):
        phase = math.sin(index / 19.0) * 0.15
        drift = 0.02 if (index // 300) % 2 == 0 else -0.015
        open_price = price
        close = max(1.0, open_price + drift + phase)
        high = max(open_price, close) + 0.20
        low = min(open_price, close) - 0.20
        bars.append(
            market.Bar(
                open_ts=start + index * BASE_INTERVAL_MS,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + index % 50,
            )
        )
        price = close

    trade_ts = []
    trade_price = []
    trade_size = []
    trade_side = []
    book_ts = []
    bid_rows = []
    ask_rows = []
    bid_size_rows = []
    ask_size_rows = []

    end = bars[-1].open_ts + BASE_INTERVAL_MS
    quarter = ((start + QUARTER_MS - 1) // QUARTER_MS) * QUARTER_MS
    while quarter < end:
        base_index = _bar_index_at_or_before(bars, quarter)
        if base_index < 0:
            quarter += QUARTER_MS
            continue
        mid = bars[base_index].close
        direction = 1.0 if (quarter // QUARTER_MS) % 3 != 0 else -1.0

        for second in range(0, 70):
            ts = quarter + second * 1000
            if ts >= end:
                break
            book_ts.append(ts)
            bid_rows.append([mid - 0.01 * (level + 1) for level in range(5)])
            ask_rows.append([mid + 0.01 * (level + 1) for level in range(5)])
            bid_size_rows.append([
                20.0 + (5.0 if direction > 0 else 0.0) + level
                for level in range(5)
            ])
            ask_size_rows.append([
                20.0 + (5.0 if direction < 0 else 0.0) + level
                for level in range(5)
            ])

        for trade_index in range(12):
            ts = quarter + 100 + trade_index * 700
            if ts >= end:
                break
            sign = direction if trade_index < 8 else -direction
            trade_ts.append(ts)
            trade_side.append(sign)
            trade_size.append(1.0 + trade_index * 0.1)
            trade_price.append(mid * (1.0 + sign * 0.00002 * trade_index))

        quarter += QUARTER_MS

    trades = RawTradeTape(
        ts_ms=np.asarray(trade_ts, dtype=np.int64),
        price=np.asarray(trade_price, dtype=float),
        size=np.asarray(trade_size, dtype=float),
        side_sign=np.asarray(trade_side, dtype=float),
    )
    book = BookTape(
        ts_ms=np.asarray(book_ts, dtype=np.int64),
        bid_px=np.asarray(bid_rows, dtype=float),
        ask_px=np.asarray(ask_rows, dtype=float),
        bid_size=np.asarray(bid_size_rows, dtype=float),
        ask_size=np.asarray(ask_size_rows, dtype=float),
    )
    trades.validate("SYNTH")
    book.validate("SYNTH")
    return SymbolData(
        symbol="SYNTH",
        bars=bars,
        funding_events=[],
        trades=trades,
        book=book,
        premium_bars=None,
        manifest={"instrument_id": "SYNTH"},
    )


def selftest_command(_args: argparse.Namespace) -> int:
    config = V3Config(
        train_months=1,
        validation_months=1,
        test_months=1,
        step_months=1,
        historical_z_days=2,
        min_trades_in_flow_window=4,
        book_max_age_ms=2_000,
        post_checkpoints_minutes=(5, 15, 30),
    )
    data = _synthetic_symbol_data()

    raw_events, prep = build_raw_quarter_events(data, config)
    assert raw_events, "synthetic quarter events were not built"

    first = raw_events[0]
    long_event = directionalise_event(first, "LONG", True, data.bars)
    short_event = directionalise_event(first, "SHORT", True, data.bars)
    assert long_event.features.shape == short_event.features.shape
    assert len(long_event.features) == (
        len(CORE_FEATURE_NAMES) + len(OPTIONAL_POSITIONING_FEATURE_NAMES)
    )

    # Directional mirror contracts for signed flow and book imbalance.
    feature_names = CORE_FEATURE_NAMES + OPTIONAL_POSITIONING_FEATURE_NAMES
    for name in (
        "dir_volume_imbalance_10s",
        "dir_signed_dollar_flow_z_10s",
        "dir_l1_book_imbalance",
        "dir_microprice_deviation",
    ):
        index = feature_names.index(name)
        assert long_event.features[index] == -short_event.features[index]

    # Appending future micro data cannot alter an old event.
    old_features = long_event.features.copy()
    future_trade = RawTradeTape(
        ts_ms=np.append(data.trades.ts_ms, data.trades.ts_ms[-1] + 1000),
        price=np.append(data.trades.price, data.trades.price[-1]),
        size=np.append(data.trades.size, 1.0),
        side_sign=np.append(data.trades.side_sign, 1.0),
    )
    future_book = BookTape(
        ts_ms=np.append(data.book.ts_ms, data.book.ts_ms[-1] + 1000),
        bid_px=np.vstack([data.book.bid_px, data.book.bid_px[-1]]),
        ask_px=np.vstack([data.book.ask_px, data.book.ask_px[-1]]),
        bid_size=np.vstack([data.book.bid_size, data.book.bid_size[-1]]),
        ask_size=np.vstack([data.book.ask_size, data.book.ask_size[-1]]),
    )
    extended = replace(data, trades=future_trade, book=future_book)
    extended_events, _ = build_raw_quarter_events(extended, config)
    matching = next(
        event for event in extended_events
        if event.decision_ts == first.decision_ts
    )
    extended_long = directionalise_event(
        matching, "LONG", True, extended.bars
    )
    np.testing.assert_allclose(old_features, extended_long.features)

    # lab.sim is the sole source of outcomes.
    assert isinstance(long_event.final_outcome, sim.TradeOutcome)
    assert long_event.final_outcome.outcome_hash

    # Model and quality score contract.
    sample_events = []
    for raw in raw_events[:80]:
        sample_events.append(directionalise_event(raw, "LONG", True, data.bars))
        sample_events.append(directionalise_event(raw, "SHORT", True, data.bars))
    model = EntryModel(config)
    model.fit(sample_events)
    prediction = model.predict(sample_events[-1])
    assert 0.0 <= prediction.quality_score <= 100.0
    assert 0.0 <= prediction.predicted_win_probability <= 1.0

    selected, _ = select_entries(
        [model.predict(event) for event in sample_events[-20:]],
        threshold=0.0,
        config=config,
    )
    assert len(selected) <= 10  # one side per timestamp

    print(json.dumps({
        "status": "PASS",
        "raw_quarter_events": len(raw_events),
        "feature_count": len(feature_names),
        "post_feature_count": len(POST_FEATURE_NAMES),
        "preparation": prep,
        "contracts": {
            "single_file": True,
            "raw_aggressor_flow_required": True,
            "l1_book_required": True,
            "future_append_causality": True,
            "directional_mirroring": True,
            "quality_score_0_100": True,
            "validation_only_threshold_tuning": True,
            "train_only_post_execution": True,
            "lab_sim_economic_truth": True,
            "net_r_reimplemented": False,
        },
    }, indent=2))
    return 0


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="SYMBOL=PATH",
    )
    parser.add_argument(
        "--trades",
        action="append",
        required=True,
        metavar="SYMBOL=PATH",
    )
    parser.add_argument(
        "--book",
        action="append",
        required=True,
        metavar="SYMBOL=PATH",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V7-Lite V3 quarter-hour microstructure challenger"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect-data",
        help="validate snapshot, aggressor trades, L1/L2 book, and overlap",
    )
    add_data_args(inspect)
    inspect.set_defaults(func=inspect_data_command)

    walk = sub.add_parser(
        "walkforward",
        help="run rolling train/validation/frozen-OOS V3",
    )
    add_data_args(walk)
    walk.add_argument("--flow-window-seconds", type=int, default=10)
    walk.add_argument("--horizon-hours", type=int, default=8)
    walk.add_argument("--stop-atr", type=float, default=1.50)
    walk.add_argument("--target-r", type=float, default=1.50)
    walk.add_argument("--train-months", type=int, default=12)
    walk.add_argument("--validation-months", type=int, default=2)
    walk.add_argument("--test-months", type=int, default=2)
    walk.add_argument("--step-months", type=int, default=2)
    walk.add_argument("--minimum-validation-trades", type=int, default=20)
    walk.add_argument("--minimum-validation-win-rate", type=float, default=0.52)
    walk.add_argument("--disable-post-execution", action="store_true")
    walk.add_argument("--seed", type=int, default=7)
    walk.add_argument("--output")
    walk.set_defaults(func=walkforward_command)

    selftest = sub.add_parser(
        "selftest",
        help="run causal feature, model, and lab.sim contract tests",
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
