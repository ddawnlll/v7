"""Phase 4 — Outcome contract (event dataset generation).

Pure: no I/O, no network, no wall-clock. Given LoadedSnapshots from tools/snapshot,
builds candidate event rows, assigns splits (train/test), and purges training
events whose holding period overlaps with the test period boundary to prevent
leakage (RULES §6).
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from lab import indicators, sim, tape
from lab.observe import Setup
from tools.snapshot import LoadedSnapshot



@dataclass(frozen=True, slots=True)
class CandidateEvent:
    """One candidate trade event, directional (LONG/SHORT), with simulated outcome."""

    event_id: str             # Hash of (symbol, decision_ts, side)
    symbol: str               # e.g., "BTC-USDT-SWAP"
    side: str                 # "LONG" or "SHORT"
    feature_cutoff_ts: int    # Timestamp when features are computed (before entry)
    decision_ts: int          # Close timestamp of decision bar
    planned_entry_ts: int     # Intended entry (open of next bar)
    fill_ts: int              # Actual fill timestamp
    outcome_end_ts: int       # Resolution timestamp of simulated trade (close of exit bar)
    locked_outcome: sim.TradeOutcome
    split: str                # "train" (development) or "test" (frozen holdout)


def build_events(
    snapshots: Sequence[LoadedSnapshot],
    split_ts: int,
    setup: Setup,
) -> list[CandidateEvent]:
    """Build candidate events from loaded snapshots, assigning splits and purging.

    Applies chronological splitting based on ``split_ts``. Any development ("train")
    trade whose outcome end timestamp overflows into the validation/test window
    is purged (RULES §6) to prevent look-ahead bias.
    """
    events: list[CandidateEvent] = []

    for snap in snapshots:
        symbol = snap.manifest["inst_id"]
        trade_bars = snap.trade_bars
        funding_events = snap.funding_events

        n = len(trade_bars)
        if n == 0:
            continue

        highs = [b.high for b in trade_bars]
        lows = [b.low for b in trade_bars]
        opens = [b.open for b in trade_bars]
        closes = [b.close for b in trade_bars]
        bar_open_ts = [b.open_ts for b in trade_bars]

        factor = setup.decision_interval_factor
        if factor == 1:
            atr = indicators.compute_atr(highs, lows, closes, period=14)
            last_entry_i = n - setup.max_holding_bars - 2
            candidates = []
            for i in range(14, max(14, last_entry_i + 1)):
                candidates.append((i + 1, atr[i]))
        else:
            derived = tape.aggregate(trade_bars, factor, 300_000)
            if len(derived) <= 14:
                continue
            d_atr = indicators.compute_atr(
                [b.high for b in derived], [b.low for b in derived],
                [b.close for b in derived], period=14
            )
            candidates = []
            for di in range(14, len(derived)):
                decision_close_ts = derived[di].open_ts + factor * 300_000
                idx = bisect.bisect_left(bar_open_ts, decision_close_ts)
                if (
                    idx >= n
                    or bar_open_ts[idx] != decision_close_ts
                    or idx + setup.max_holding_bars >= n
                ):
                    continue
                candidates.append((idx, d_atr[di]))

        for entry_index, a in candidates:
            if not math.isfinite(a):
                continue

            entry_price = opens[entry_index]
            stop_dist = setup.k_stop * a
            target_dist = setup.reward_risk * stop_dist

            # decision_ts is close of decision bar, which is the open of the entry bar
            decision_ts = bar_open_ts[entry_index]
            feature_cutoff_ts = decision_ts
            planned_entry_ts = decision_ts
            fill_ts = decision_ts

            for side in ("LONG", "SHORT"):
                is_long = side == "LONG"
                if is_long:
                    stop_price = entry_price - stop_dist
                    target_price = entry_price + target_dist
                else:
                    stop_price = entry_price + stop_dist
                    target_price = entry_price - target_dist

                spec = sim.TradeSpec(
                    side=side,
                    entry_index=entry_index,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    max_holding_bars=setup.max_holding_bars,
                )

                outcome = sim.simulate(opens, highs, lows, closes, spec, funding_events)
                outcome_end_ts = bar_open_ts[outcome.exit_index] + 300_000

                if decision_ts < split_ts:
                    # Train split. Purge validation leak:
                    if outcome_end_ts >= split_ts:
                        continue
                    split = "train"
                else:
                    split = "test"

                # Deterministic event ID
                event_id = hashlib.sha256(
                    f"{symbol}_{decision_ts}_{side}".encode("utf-8")
                ).hexdigest()

                events.append(
                    CandidateEvent(
                        event_id=event_id,
                        symbol=symbol,
                        side=side,
                        feature_cutoff_ts=feature_cutoff_ts,
                        decision_ts=decision_ts,
                        planned_entry_ts=planned_entry_ts,
                        fill_ts=fill_ts,
                        outcome_end_ts=outcome_end_ts,
                        locked_outcome=outcome,
                        split=split,
                    )
                )

    # Deterministic sorting
    events.sort(key=lambda e: (e.decision_ts, e.symbol, e.side))
    return events


def canonical_bytes(events: Sequence[CandidateEvent]) -> bytes:
    """Deterministic, round-trippable serialization of the candidate events."""
    lines = ["candidate-event-v0"]
    for e in events:
        lines.append(
            f"{e.event_id} {e.symbol} {e.side} {e.feature_cutoff_ts} {e.decision_ts} "
            f"{e.planned_entry_ts} {e.fill_ts} {e.outcome_end_ts} {e.split} "
            f"{e.locked_outcome.exit_index} {e.locked_outcome.exit_reason} "
            f"{e.locked_outcome.net_r!r}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def events_hash(events: Sequence[CandidateEvent]) -> str:
    """SHA-256 hex digest of the canonical events serialization."""
    return hashlib.sha256(canonical_bytes(events)).hexdigest()
