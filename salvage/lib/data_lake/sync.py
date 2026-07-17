"""
Data sync orchestrator — deterministic, idempotent OHLCV sync from Binance.

Reuses the existing market_data authority (BackfillOrchestrator, KlinesService,
StorageWriter, DataCatalog, BackfillCheckpoint) rather than introducing a new
downloader architecture.

Key design:
  - 12-symbol bootstrap set defined as BOOTSTRAP_SYMBOLS_12
  - Symbol/timeframe partitioning via --symbols and --intervals
  - Idempotent resume via BackfillCheckpoint (skips completed ranges)
  - SHA-256 checksum validation after download
  - Duplicate + missing candle detection built into result
  - Deterministic (same args → same files, no random scheduling)

Usage:
    from lib.data_lake.sync import DataSyncOrchestrator, BOOTSTRAP_SYMBOLS_12

    sync = DataSyncOrchestrator(data_dir="data_lake")
    result = sync.run(symbols=BOOTSTRAP_SYMBOLS_12, intervals=["1h"])
    print(f"Records: {result.total_records}, gaps={len(result.gaps)}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib.market_data.binance.backfill import BackfillOrchestrator
from lib.market_data.binance.checkpoint import BackfillCheckpoint
from lib.market_data.binance.client import BinanceClient
from lib.market_data.binance.funding_service import FundingService
from lib.market_data.binance.klines_service import KlinesService
from lib.market_data.binance.rate_limiter import BinanceRateLimiter
from lib.market_data.catalog import DataCatalog
from lib.market_data.contracts import KlineRecord
from lib.market_data.binance.klines_service import interval_to_minutes
from lib.market_data.quality import detect_duplicates, detect_gaps
from lib.market_data.storage import StorageWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 12-symbol bootstrap set
# ---------------------------------------------------------------------------
# Core liquid pairs covering BTC/ETH alts, major L1s, and oracle/defi.
# Sorted by convention: BTC-ETH anchor, then large-cap alts, then L1s.
BOOTSTRAP_SYMBOLS_12: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "ATOMUSDT",
)

# Default time range (last 3 full months from current UTC month)
_DEFAULT_MONTHS_BACK = 3


def _default_start_ms() -> int:
    """Start timestamp: N months before the current UTC month start."""
    now = datetime.now(timezone.utc)
    # Go back N months, then snap to month start
    target_month = now.month - _DEFAULT_MONTHS_BACK
    target_year = now.year
    while target_month < 1:
        target_month += 12
        target_year -= 1
    start = datetime(target_year, target_month, 1, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000)


def _now_ms() -> int:
    """Current UTC timestamp in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncResult:
    """Immutable result of a sync run."""
    symbols_requested: int
    intervals_requested: int
    total_records: int
    total_files: int
    gaps: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checksum_failures: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    checkpoint_path: str = ""

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.checksum_failures) == 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DataSyncOrchestrator:
    """Deterministic, idempotent OHLCV sync from Binance.

    Wraps the existing BackfillOrchestrator from lib/market_data/,
    adding duplicate/missing-candle detection and checksum validation
    on top of the checkpoint resume behaviour.
    """

    def __init__(
        self,
        data_dir: str = "data_lake",
        checkpoint_path: Optional[str] = None,
        verify_checksums: bool = True,
    ) -> None:
        self._data_dir = data_dir
        self._verify_checksums = verify_checksums

        # Shared infrastructure — same wiring as the existing backfill tests
        client = BinanceClient()
        klines = KlinesService(client)
        funding = FundingService(client)
        writer = StorageWriter(base_dir=data_dir)
        catalog_path = os.path.join(data_dir, "catalog.json")
        catalog = DataCatalog(catalog_path=catalog_path)
        limiter = BinanceRateLimiter()
        cp_path = checkpoint_path or os.path.join(data_dir, "checkpoint.json")
        checkpoint = BackfillCheckpoint(file_path=cp_path)

        self._checkpoint_path = cp_path
        self._backfill = BackfillOrchestrator(
            klines_service=klines,
            funding_service=funding,
            storage_writer=writer,
            catalog=catalog,
            rate_limiter=limiter,
            checkpoint=checkpoint,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: Optional[list[str]] = None,
        intervals: Optional[list[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        skip_checksum_verify: bool = False,
    ) -> SyncResult:
        """Execute a deterministic sync run.

        Args:
            symbols: List of symbols (default: BOOTSTRAP_SYMBOLS_12).
            intervals: List of intervals (default: ["1h"]).
            start_time: Start ms timestamp (default: 3 months ago).
            end_time: End ms timestamp (default: now).
            skip_checksum_verify: Skip post-download checksum verification.

        Returns:
            SyncResult with counts, gaps, duplicates, errors.
        """
        symbols = list(symbols or BOOTSTRAP_SYMBOLS_12)
        intervals = list(intervals or ["1h"])
        start_time = start_time or _default_start_ms()
        end_time = end_time or _now_ms()

        t0 = time.monotonic()

        logger.info(
            "DataSyncOrchestrator starting: %d symbols x %d intervals [%d, %d)",
            len(symbols), len(intervals), start_time, end_time,
        )

        # --- Phase 1: Backfill via existing authority ---
        stats = self._backfill.backfill(
            symbols=symbols,
            intervals=intervals,
            start_time=start_time,
            end_time=end_time,
        )

        # --- Phase 2: Post-sync quality checks ---
        gaps: list[dict] = []
        duplicates: list[dict] = []
        checksum_failures: list[str] = []
        total_files = 0

        for symbol in symbols:
            for interval in intervals:
                interval_min = interval_to_minutes(interval)
                try:
                    _gaps, _dups, _files = self._run_quality_checks(
                        symbol, interval, start_time, end_time, interval_min,
                    )
                    gaps.extend(_gaps)
                    duplicates.extend(_dups)
                    total_files += _files
                except Exception as exc:
                    logger.warning("Quality check failed for %s %s: %s", symbol, interval, exc)

        # --- Phase 3: Checksum verification ---
        if self._verify_checksums and not skip_checksum_verify:
            checksum_failures = self._verify_all_checksums(symbols, intervals, start_time, end_time)

        elapsed = time.monotonic() - t0

        return SyncResult(
            symbols_requested=len(symbols),
            intervals_requested=len(intervals),
            total_records=stats.get("total_records", 0),
            total_files=total_files,
            gaps=gaps,
            duplicates=duplicates,
            errors=stats.get("errors", []),
            checksum_failures=checksum_failures,
            elapsed_seconds=elapsed,
            checkpoint_path=self._checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _run_quality_checks(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        interval_min: int,
    ) -> tuple[list[dict], list[dict], int]:
        """Detect gaps and duplicates in downloaded data.

        Reads Parquet files from disk (flat naming convention as written
        by StorageWriter), reconstructs sorted record list,
        then runs gap and duplicate detection from lib.market_data.quality.

        Returns:
            (gaps, duplicates, file_count)
        """
        import pyarrow.parquet as pq
        import pandas as pd

        base = Path(self._data_dir)
        records: list[KlineRecord] = []
        file_count = 0

        # Walk parquet files for this symbol — StorageWriter writes:
        #   {base_dir}/raw/{symbol}/{symbol}_{interval}_{start}_{end}.parquet
        raw_dir = base / "raw" / symbol.upper()
        if not raw_dir.exists():
            return [], [], 0

        for pf in sorted(raw_dir.iterdir()):
            if pf.suffix != ".parquet":
                continue
            # Check if this file matches the requested interval
            # Filename pattern: SYMBOL_INTERVAL_START_END.parquet
            name = pf.stem  # without .parquet
            parts = name.split("_")
            if len(parts) >= 2 and parts[1] != interval:
                continue
            try:
                table = pq.read_table(str(pf))
                df = table.to_pandas()
                for _, row in df.iterrows():
                    records.append(KlineRecord(
                        symbol=str(row.get("symbol", symbol)),
                        timestamp=int(row["timestamp"]),
                        open=float(row.get("open", 0.0)),
                        high=float(row.get("high", 0.0)),
                        low=float(row.get("low", 0.0)),
                        close=float(row.get("close", 0.0)),
                        volume=float(row.get("volume", 0.0)),
                        quote_volume=float(row.get("quote_volume", 0.0)),
                        trade_count=int(row.get("trade_count", 0)),
                        taker_buy_volume=float(row.get("taker_buy_volume", 0.0)),
                        taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0.0)),
                        interval=str(row.get("interval", interval)),
                        source=str(row.get("source", "binance")),
                        is_closed=bool(row.get("is_closed", True)),
                    ))
                file_count += 1
            except Exception as exc:
                logger.debug("Skipping unreadable file %s: %s", pf, exc)

        if not records:
            return [], [], file_count

        # Sort by timestamp (defensive — files should already be sorted)
        records.sort(key=lambda r: r.timestamp)

        # Gap detection
        raw_gaps = detect_gaps(records, interval_min)
        gap_list = [
            {
                "symbol": symbol,
                "interval": interval,
                "gap_start_ms": gs,
                "gap_end_ms": ge,
            }
            for gs, ge in raw_gaps
        ]

        # Duplicate detection
        dup_indices = detect_duplicates(records)
        dup_list = [
            {
                "symbol": symbol,
                "interval": interval,
                "timestamp": records[i].timestamp,
                "index": i,
            }
            for i in dup_indices
        ]

        return gap_list, dup_list, file_count

    # ------------------------------------------------------------------
    # Checksum verification
    # ------------------------------------------------------------------

    def _verify_all_checksums(
        self,
        symbols: list[str],
        intervals: list[str],
        start_time: int,
        end_time: int,
    ) -> list[str]:
        """Verify every .parquet file against its .sha256 sidecar.

        Returns list of file paths that failed checksum verification.
        """
        writer = StorageWriter(base_dir=self._data_dir)
        failures: list[str] = []

        for symbol in symbols:
            raw_dir = Path(self._data_dir) / "raw" / symbol.upper()
            if not raw_dir.exists():
                continue
            for pf in sorted(raw_dir.iterdir()):
                if pf.suffix != ".parquet":
                    continue
                if not writer.verify_checksum(str(pf)):
                    failures.append(str(pf))

        return failures


# ---------------------------------------------------------------------------
# CLI-like summariser
# ---------------------------------------------------------------------------


def print_sync_result(result: SyncResult, verbose: bool = False) -> None:
    """Print a human-readable summary of a sync result."""
    status = "OK" if result.success else "FAIL"
    print(f"\n  Data sync [{status}]")
    print(f"  Symbols:        {result.symbols_requested}")
    print(f"  Intervals:      {result.intervals_requested}")
    print(f"  Total records:  {result.total_records:,}")
    print(f"  Total files:    {result.total_files}")
    print(f"  Gaps:           {len(result.gaps)}")
    print(f"  Duplicates:     {len(result.duplicates)}")
    print(f"  Checksum fails: {len(result.checksum_failures)}")
    print(f"  Errors:         {len(result.errors)}")
    print(f"  Elapsed:        {result.elapsed_seconds:.1f}s")
    print(f"  Checkpoint:     {result.checkpoint_path}")

    if verbose and result.gaps:
        print(f"\n  Gaps ({len(result.gaps)}):")
        for g in result.gaps[:10]:
            print(f"    {g['symbol']} {g['interval']}: {g['gap_start_ms']} → {g['gap_end_ms']}")
        if len(result.gaps) > 10:
            print(f"    ... and {len(result.gaps) - 10} more")

    if verbose and result.duplicates:
        print(f"\n  Duplicates ({len(result.duplicates)}):")
        for d in result.duplicates[:10]:
            print(f"    {d['symbol']} {d['interval']}: timestamp={d['timestamp']}")
        if len(result.duplicates) > 10:
            print(f"    ... and {len(result.duplicates) - 10} more")

    if result.errors:
        print(f"\n  Errors:")
        for e in result.errors[:5]:
            print(f"    - {e}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more")

    if result.checksum_failures:
        print(f"\n  Checksum failures:")
        for f in result.checksum_failures[:5]:
            print(f"    - {f}")
        if len(result.checksum_failures) > 5:
            print(f"    ... and {len(result.checksum_failures) - 5} more")
