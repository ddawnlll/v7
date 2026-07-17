"""Funding data persistence — Parquet read/write for funding records.

Follows the medallion data-lake convention:
  <base_dir>/funding/<symbol>/<year>/<month>.parquet

Each Parquet file contains one partition of funding records sorted by
timestamp ascending.  Duplicate (symbol, timestamp) rows are resolved
deterministically — last-write-wins within the same write call,
read always returns stable ascending order.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import pandas as pd
    import pyarrow.parquet as pq
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from lib.market_data.binance.funding_service import FundingRecord

# Default base directory — override with base_dir parameter.
_DEFAULT_BASE = os.environ.get("DATA_LAKE_ROOT", "/data/lake")


def _funding_path(
    symbol: str,
    year: int,
    month: int,
    base_dir: str,
) -> str:
    """Build deterministic Parquet path for a funding partition."""
    return os.path.join(
        base_dir, "funding", symbol, str(year), f"{month:02d}.parquet",
    )


def _partition_range(
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, int, int]]:
    """Generate (year, month, start_ms, end_ms) tuples covering [start, end).

    Yields one tuple per month boundary crossed.
    """
    from datetime import datetime, timezone
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    partitions = []
    cursor = start_dt
    while cursor < end_dt:
        # Advance to next month
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1)
        part_end = min(next_month, end_dt)
        partitions.append((
            cursor.year, cursor.month,
            int(cursor.timestamp() * 1000),
            int(part_end.timestamp() * 1000),
        ))
        cursor = next_month
    return partitions


def write_funding_records(
    records: List[FundingRecord],
    symbol: str,
    start_time: int,
    end_time: int,
    base_dir: Optional[str] = None,
) -> str:
    """Write funding records to Parquet, returning the last path written.

    Args:
        records: Funding records to persist.
        symbol: Trading symbol for path isolation.
        start_time: Query start in ms (used for partition routing).
        end_time: Query end in ms.
        base_dir: Data-lake root directory.

    Returns:
        Path of the last Parquet partition written.
    """
    base = base_dir or _DEFAULT_BASE
    if not _HAS_PANDAS:
        raise ImportError("pandas/pyarrow required for funding persistence")

    # Validate
    for r in records:
        if r.timestamp <= 0:
            raise ValueError(f"Invalid timestamp {r.timestamp} in {r}")
        if not _isfinite(r.funding_rate):
            raise ValueError(f"Invalid funding_rate {r.funding_rate} in {r}")

    # Sort ascending by timestamp
    records = sorted(records, key=lambda r: r.timestamp)

    # Group by month partition
    partitions = _partition_range(start_time, end_time)
    # Route each record to its partition
    from collections import defaultdict
    by_partition = defaultdict(list)
    for r in records:
        dt = __import__("datetime").datetime.fromtimestamp(
            r.timestamp / 1000, tz=__import__("datetime").timezone.utc,
        )
        key = (dt.year, dt.month)
        by_partition[key].append(r)

    last_path = ""
    for year, month in sorted(by_partition.keys()):
        part_records = by_partition[(year, month)]
        # Remove duplicates: last timestamp wins
        seen = {}
        for r in part_records:
            seen[r.timestamp] = r
        deduped = sorted(seen.values(), key=lambda x: x.timestamp)

        df = pd.DataFrame([
            {"symbol": r.symbol, "timestamp": r.timestamp,
             "funding_rate": r.funding_rate, "source": r.source}
            for r in deduped
        ])
        path = _funding_path(symbol, year, month, base)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        last_path = path

    return last_path


def read_funding_records(
    symbol: str,
    start_time: int,
    end_time: int,
    base_dir: Optional[str] = None,
) -> List[FundingRecord]:
    """Read funding records from Parquet partitions.

    Args:
        symbol: Trading symbol.
        start_time: Query start in ms.
        end_time: Query end in ms.
        base_dir: Data-lake root directory.

    Returns:
        Chronologically sorted list of FundingRecord.
    """
    base = base_dir or _DEFAULT_BASE
    if not _HAS_PANDAS:
        raise ImportError("pandas/pyarrow required for funding persistence")

    records: list[FundingRecord] = []
    partitions = _partition_range(start_time, end_time)
    for year, month, ps, pe in partitions:
        path = _funding_path(symbol, year, month, base)
        if not os.path.exists(path):
            continue
        try:
            table = pq.read_table(path)
        except Exception:
            continue
        df = table.to_pandas()
        # Filter to query range
        in_range = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        df = df[in_range]
        for _, row in df.iterrows():
            records.append(FundingRecord(
                symbol=str(row["symbol"]),
                timestamp=int(row["timestamp"]),
                funding_rate=float(row["funding_rate"]),
                source=str(row.get("source", "binance")),
            ))

    return sorted(records, key=lambda r: r.timestamp)


def read_funding_events(
    symbol: str,
    start_time: int,
    end_time: int,
    base_dir: Optional[str] = None,
) -> list:
    """Read funding events (FundingEvent objects) from Parquet.

    Same as read_funding_records but returns simulation-contract
    FundingEvent instances.  The import is deferred to keep the
    lib domain from directly importing simulation at module level.

    Args:
        symbol: Trading symbol.
        start_time: Query start in ms.
        end_time: Query end in ms.
        base_dir: Data-lake root directory.

    Returns:
        Chronologically sorted list of FundingEvent.
    """
    from simulation.contracts.models import FundingEvent
    records = read_funding_records(symbol, start_time, end_time, base_dir=base_dir)
    return [
        FundingEvent(timestamp=r.timestamp, rate=r.funding_rate)
        for r in records
    ]


def _isfinite(value: float) -> bool:
    """Check value is a finite real number."""
    import math
    return not (math.isnan(value) or math.isinf(value))
