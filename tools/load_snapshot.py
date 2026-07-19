"""Snapshot loader — the read-side counterpart of build_snapshot.py.

Like build_snapshot.py, this touches pyarrow and disk, not the deterministic
core: lab/data.py never imports pyarrow (see requirements.txt), so this is
the only other place besides the acquisition tool that does.

A snapshot on disk is untrusted bytes until re-verified: this module
re-derives the trade/mark/funding dataset hashes from the loaded rows via
lab.data's own hashers and compares them against the manifest's recorded
hashes, and refuses a snapshot whose manifest says its tape coverage was
incomplete. Nothing downstream (lab/observe.py) ever sees unverified bars.
"""

from __future__ import annotations

import bisect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import data, sim  # noqa: E402  (path set above)


@dataclass(frozen=True, slots=True)
class LoadedSnapshot:
    """A snapshot's contents, already hash-verified against its manifest and
    ready to hand to lab/observe.py (or any other pure consumer)."""

    trade_bars: list[data.Bar]
    mark_bars: list[data.MarkBar]
    funding_events: list[sim.FundingEvent]
    manifest: dict


def _read_trade_bars(path: Path) -> list[data.Bar]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    return [
        data.Bar(open_ts=o, open=op, high=h, low=lo, close=c, volume=v)
        for o, op, h, lo, c, v in zip(
            cols["open_ts"], cols["open"], cols["high"], cols["low"],
            cols["close"], cols["volume"],
        )
    ]


def _read_mark_bars(path: Path) -> list[data.MarkBar]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    return [
        data.MarkBar(open_ts=o, open=op, high=h, low=lo, close=c)
        for o, op, h, lo, c in zip(
            cols["open_ts"], cols["open"], cols["high"], cols["low"], cols["close"],
        )
    ]


def _read_funding_records(path: Path) -> list[data.FundingRecord]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    return [
        data.FundingRecord(funding_time=t, rate=r)
        for t, r in zip(cols["funding_time"], cols["rate"])
    ]


def _funding_events(
    funding_records: Sequence[data.FundingRecord],
    trade_bars: Sequence[data.Bar],
    mark_bars: Sequence[data.MarkBar],
) -> list[sim.FundingEvent]:
    """Map each raw funding record onto a trade-bar index and a settlement
    mark price (ARCHITECTURE §8.2: mark bars are used only where the
    simulation contract requires mark price, initially funding valuation).

    A funding record at ``funding_time`` maps to the last trade bar whose
    ``open_ts <= funding_time`` (bisect on the sorted open_ts grid) — the
    bar during which the funding event settles. A record that falls before
    the first bar or after the last is out of the snapshot's covered range
    and is dropped, not guessed at.
    """
    open_ts = [b.open_ts for b in trade_bars]
    events: list[sim.FundingEvent] = []
    for rec in funding_records:
        idx = bisect.bisect_right(open_ts, rec.funding_time) - 1
        if idx < 0 or idx >= len(trade_bars):
            continue
        events.append(
            sim.FundingEvent(
                bar_index=idx, rate=rec.rate, mark_price=mark_bars[idx].close
            )
        )
    # sim.py requires strictly increasing bar_index; multiple funding
    # records can map to the same bar only if funding settles more often
    # than the bar grid, which the 8h OKX cadence never does at 5m — but
    # fail closed rather than silently dedupe if it ever happens.
    for prev, cur in zip(events, events[1:]):
        if cur.bar_index <= prev.bar_index:
            raise ValueError(
                f"funding events did not map to strictly increasing bar "
                f"indices ({prev.bar_index} -> {cur.bar_index}) — funding "
                "cadence finer than the bar grid, or a duplicate record"
            )
    return events


def load(snapshot_dir: Path) -> LoadedSnapshot:
    """Load, re-verify and return one snapshot's contents. Fails closed:
    raises if any tape's re-derived hash disagrees with the manifest, or if
    the manifest records incomplete coverage for trade or mark."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())

    if not manifest["trade"]["coverage_complete"]:
        raise ValueError(
            f"{snapshot_dir}: trade tape coverage_complete=false — refusing "
            "to observe outcomes on an incomplete snapshot"
        )
    if not manifest["mark"]["coverage_complete"]:
        raise ValueError(
            f"{snapshot_dir}: mark tape coverage_complete=false — refusing "
            "to observe outcomes on an incomplete snapshot"
        )

    trade_bars = _read_trade_bars(snapshot_dir / "trade_bars_5m.parquet")
    mark_bars = _read_mark_bars(snapshot_dir / "mark_bars_5m.parquet")
    funding_records = _read_funding_records(snapshot_dir / "funding_events.parquet")

    checks = [
        ("trade", data.dataset_hash(trade_bars), manifest["trade"]["dataset_hash"]),
        ("mark", data.mark_dataset_hash(mark_bars), manifest["mark"]["dataset_hash"]),
        ("funding", data.funding_dataset_hash(funding_records),
         manifest["funding"]["dataset_hash"]),
    ]
    for tape, recomputed, recorded in checks:
        if recomputed != recorded:
            raise ValueError(
                f"{snapshot_dir}: {tape} tape hash mismatch — recomputed "
                f"{recomputed}, manifest says {recorded}. The on-disk data "
                "does not match what was recorded; refusing to use it."
            )

    funding_events = _funding_events(funding_records, trade_bars, mark_bars)

    return LoadedSnapshot(
        trade_bars=trade_bars,
        mark_bars=mark_bars,
        funding_events=funding_events,
        manifest=manifest,
    )
