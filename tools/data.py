"""Build and verify immutable Binance USD-M canonical market datasets.

This module replaces the old all-in-one ``tools/data.py``.  It has one job:
turn official Binance public archives into validated, reproducible Parquet
*tapes*.  It deliberately does NOT contain:

- feature engineering,
- labels or outcome observation,
- simulation,
- train/validation splitting,
- scikit-learn preprocessing,
- OKX/Binance L2 replay.

Those concerns must remain in separate modules.  In particular, scikit-learn
belongs in the train-only model pipeline, not in dataset construction.

Data flow
---------

    data.binance.vision ZIP + CHECKSUM
        -> SHA-256 verification
        -> Polars normalization
        -> Pandera schema validation
        -> Polars economic/temporal invariant gates
        -> PyArrow Parquet with canonical metadata
        -> content hash + file hash + immutable manifest
        -> DuckDB audit report

Public API
----------

``build(config)``
    Download, verify, normalize, validate, and atomically publish a dataset.

``verify(dataset_dir)``
    Re-open all published files, re-run validation, and compare hashes.

``load(dataset_dir)``
    Verify first, then return a lazy scanning handle.

``audit(dataset_dir)``
    Produce a DuckDB-backed summary without modifying the dataset.

Example
-------

    python dataset.py build \
        --symbols BTCUSDT,ETHUSDT \
        --start 2023-01-01 \
        --end 2026-07-01 \
        --out data/canonical/binance-usdm-pilot

    python dataset.py verify --dataset-dir data/canonical/binance-usdm-pilot
    python dataset.py audit  --dataset-dir data/canonical/binance-usdm-pilot

The requested interval is UTC and half-open: ``[start, end)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

try:
    import duckdb
    import pandera.polars as pa
    import polars as pl
    import pyarrow as arrow
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - dependency bootstrap path
    raise SystemExit(
        "Missing dataset dependencies. Install with:\n"
        "  pip install 'polars>=1.30' 'pyarrow>=18' "
        "'pandera[polars]>=0.24' 'duckdb>=1.3'\n"
        f"Original import error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Constants and contracts
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "alphaforge-binance-usdm-canonical-v1"
BUILDER_VERSION = "1.0.0"
EXCHANGE = "binance"
MARKET = "usdm"
BASE_URL = "https://data.binance.vision/data"
FAPI_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
DAY_MS = 86_400_000

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": DAY_MS,
}

KLINE_TYPES = (
    "klines",
    "markPriceKlines",
    "indexPriceKlines",
    "premiumIndexKlines",
)
ALL_DATA_TYPES = (*KLINE_TYPES, "fundingRate")

TAPE_BY_DATA_TYPE: dict[str, str] = {
    "klines": "bars",
    "markPriceKlines": "mark",
    "indexPriceKlines": "index",
    "premiumIndexKlines": "premium",
    "fundingRate": "funding",
}

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

# Binance public funding archives have changed headers over time.  We route
# aliases explicitly and fail if the timestamp/rate columns cannot be resolved.
FUNDING_TIMESTAMP_ALIASES = (
    "calc_time",
    "funding_time",
    "fundingtime",
    "time",
    "timestamp",
)
FUNDING_RATE_ALIASES = (
    "last_funding_rate",
    "funding_rate",
    "fundingrate",
    "rate",
)
FUNDING_INTERVAL_ALIASES = (
    "funding_interval_hours",
    "funding_interval",
    "interval_hours",
)

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 8
PARQUET_ROW_GROUP_SIZE = 256_000


# Pandera is used for table-boundary contracts.  Cross-column/time-series gates
# stay in native Polars expressions because they are clearer and faster there.
BAR_SCHEMA = pa.DataFrameSchema(
    {
        "exchange": pa.Column(str, nullable=False),
        "market": pa.Column(str, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "interval": pa.Column(str, nullable=False),
        "open_ts": pa.Column(int, nullable=False),
        "close_ts": pa.Column(int, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "base_volume": pa.Column(float, nullable=False),
        "quote_volume": pa.Column(float, nullable=False),
        "trade_count": pa.Column(int, nullable=False),
        "taker_buy_base_volume": pa.Column(float, nullable=False),
        "taker_buy_quote_volume": pa.Column(float, nullable=False),
        "is_synthetic": pa.Column(bool, nullable=False),
        "source_file": pa.Column(str, nullable=False),
        "source_sha256": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=False,
)

PRICE_SCHEMA = pa.DataFrameSchema(
    {
        "exchange": pa.Column(str, nullable=False),
        "market": pa.Column(str, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "interval": pa.Column(str, nullable=False),
        "open_ts": pa.Column(int, nullable=False),
        "close_ts": pa.Column(int, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "is_synthetic": pa.Column(bool, nullable=False),
        "source_file": pa.Column(str, nullable=False),
        "source_sha256": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=False,
)

FUNDING_SCHEMA = pa.DataFrameSchema(
    {
        "exchange": pa.Column(str, nullable=False),
        "market": pa.Column(str, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "funding_ts": pa.Column(int, nullable=False),
        "funding_rate": pa.Column(float, nullable=False),
        "funding_interval_hours": pa.Column(float, nullable=True),
        "source_file": pa.Column(str, nullable=False),
        "source_sha256": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=False,
)

SCHEMA_BY_TAPE = {
    "bars": BAR_SCHEMA,
    "mark": PRICE_SCHEMA,
    "index": PRICE_SCHEMA,
    "premium": PRICE_SCHEMA,
    "funding": FUNDING_SCHEMA,
}

SORT_COLUMNS_BY_TAPE = {
    "bars": ["symbol", "open_ts"],
    "mark": ["symbol", "open_ts"],
    "index": ["symbol", "open_ts"],
    "premium": ["symbol", "open_ts"],
    "funding": ["symbol", "funding_ts"],
}

PRIMARY_KEY_BY_TAPE = {
    "bars": ["symbol", "open_ts"],
    "mark": ["symbol", "open_ts"],
    "index": ["symbol", "open_ts"],
    "premium": ["symbol", "open_ts"],
    "funding": ["symbol", "funding_ts"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Immutable build configuration.

    ``end`` is exclusive.  Dates are interpreted as midnight UTC.
    """

    symbols: tuple[str, ...]
    start: date
    end: date
    out_dir: Path
    raw_cache_dir: Path = Path("data/raw-cache/binance")
    interval: str = "1m"
    data_types: tuple[str, ...] = ALL_DATA_TYPES
    workers: int = 6
    base_url: str = BASE_URL
    keep_raw: bool = False
    fetch_exchange_info: bool = True
    strict: bool = True
    allow_leading_missing_archives: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(s.strip().upper() for s in self.symbols if s.strip()))
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "out_dir", Path(self.out_dir))
        object.__setattr__(self, "raw_cache_dir", Path(self.raw_cache_dir))

        if not normalized:
            raise ValueError("symbols cannot be empty")
        if self.start >= self.end:
            raise ValueError(f"start must be before end: {self.start} >= {self.end}")
        if self.interval not in INTERVAL_MS:
            raise ValueError(
                f"unsupported interval {self.interval!r}; supported={sorted(INTERVAL_MS)}"
            )
        unknown = sorted(set(self.data_types) - set(ALL_DATA_TYPES))
        if unknown:
            raise ValueError(f"unsupported data types: {unknown}")
        if "klines" not in self.data_types:
            raise ValueError("klines is mandatory because it is the primary coverage tape")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.out_dir.exists():
            raise FileExistsError(
                f"{self.out_dir} already exists; canonical datasets are immutable"
            )

    @property
    def start_ts(self) -> int:
        return _date_to_ms(self.start)

    @property
    def end_ts(self) -> int:
        return _date_to_ms(self.end)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "interval": self.interval,
            "data_types": list(self.data_types),
            "workers": self.workers,
            "base_url": self.base_url,
            "strict": self.strict,
            "allow_leading_missing_archives": self.allow_leading_missing_archives,
        }


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    symbol: str
    data_type: str
    interval: str | None
    cadence: Literal["monthly", "daily"]
    period: str
    relative_path: str
    url: str
    checksum_url: str


@dataclass(frozen=True, slots=True)
class DownloadedArchive:
    spec: ArchiveSpec
    zip_path: Path
    checksum_path: Path
    sha256: str
    size_bytes: int
    downloaded: bool


@dataclass(frozen=True, slots=True)
class TapeArtifact:
    tape: str
    symbol: str
    relative_path: str
    rows: int
    first_ts: int | None
    last_ts: int | None
    content_sha256: str
    file_sha256: str
    size_bytes: int
    source_files: tuple[str, ...]
    source_sha256: tuple[str, ...]
    validation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    root: Path
    manifest: Mapping[str, Any]

    def scan(
        self,
        tape: Literal["bars", "mark", "index", "premium", "funding"],
        symbols: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        if tape not in SCHEMA_BY_TAPE:
            raise ValueError(f"unknown tape {tape!r}")
        paths = [
            self.root / artifact["relative_path"]
            for artifact in self.manifest["artifacts"]
            if artifact["tape"] == tape
            and (symbols is None or artifact["symbol"] in set(symbols))
        ]
        if not paths:
            raise FileNotFoundError(f"no files found for tape={tape!r}, symbols={symbols}")
        return pl.scan_parquet([str(path) for path in paths])


# ---------------------------------------------------------------------------
# Time and archive planning
# ---------------------------------------------------------------------------

def _date_to_ms(value: date) -> int:
    dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _iter_months(start: date, end: date) -> Iterator[date]:
    current = _month_start(start)
    while current < end:
        yield current
        current = _next_month(current)


def _iter_days(start: date, end: date) -> Iterator[date]:
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)


def _is_full_month(month: date, start: date, end: date) -> bool:
    return start <= month and _next_month(month) <= end


def _archive_relative_path(
    *,
    symbol: str,
    data_type: str,
    interval: str | None,
    cadence: Literal["monthly", "daily"],
    period: str,
) -> str:
    prefix = f"futures/um/{cadence}/{data_type}/{symbol}"
    if data_type in KLINE_TYPES:
        assert interval is not None
        return f"{prefix}/{interval}/{symbol}-{interval}-{period}.zip"
    if data_type == "fundingRate":
        return f"{prefix}/{symbol}-fundingRate-{period}.zip"
    raise ValueError(f"unsupported data type {data_type!r}")


def plan_archives(config: DatasetConfig) -> list[ArchiveSpec]:
    """Plan deterministic official archive URLs for the requested window.

    Kline-like tapes use monthly files for whole interior months and daily files
    for partial boundary months.  Funding archives are monthly-only and are
    filtered to the exact requested window after parsing.
    """

    specs: list[ArchiveSpec] = []
    seen: set[str] = set()

    for symbol in config.symbols:
        for data_type in config.data_types:
            if data_type == "fundingRate":
                for month in _iter_months(config.start, config.end):
                    period = month.strftime("%Y-%m")
                    relative = _archive_relative_path(
                        symbol=symbol,
                        data_type=data_type,
                        interval=None,
                        cadence="monthly",
                        period=period,
                    )
                    if relative in seen:
                        continue
                    seen.add(relative)
                    specs.append(
                        ArchiveSpec(
                            symbol=symbol,
                            data_type=data_type,
                            interval=None,
                            cadence="monthly",
                            period=period,
                            relative_path=relative,
                            url=f"{config.base_url.rstrip('/')}/{relative}",
                            checksum_url=f"{config.base_url.rstrip('/')}/{relative}.CHECKSUM",
                        )
                    )
                continue

            for month in _iter_months(config.start, config.end):
                month_end = _next_month(month)
                slice_start = max(config.start, month)
                slice_end = min(config.end, month_end)

                if _is_full_month(month, config.start, config.end):
                    periods: Iterable[tuple[Literal["monthly", "daily"], str]] = [
                        ("monthly", month.strftime("%Y-%m"))
                    ]
                else:
                    periods = [
                        ("daily", day.strftime("%Y-%m-%d"))
                        for day in _iter_days(slice_start, slice_end)
                    ]

                for cadence, period in periods:
                    relative = _archive_relative_path(
                        symbol=symbol,
                        data_type=data_type,
                        interval=config.interval,
                        cadence=cadence,
                        period=period,
                    )
                    if relative in seen:
                        continue
                    seen.add(relative)
                    specs.append(
                        ArchiveSpec(
                            symbol=symbol,
                            data_type=data_type,
                            interval=config.interval,
                            cadence=cadence,
                            period=period,
                            relative_path=relative,
                            url=f"{config.base_url.rstrip('/')}/{relative}",
                            checksum_url=f"{config.base_url.rstrip('/')}/{relative}.CHECKSUM",
                        )
                    )

    return sorted(
        specs,
        key=lambda x: (x.symbol, x.data_type, x.period, x.cadence),
    )


# ---------------------------------------------------------------------------
# Network and checksum handling
# ---------------------------------------------------------------------------

def _http_download(
    url: str,
    destination: Path,
    *,
    attempts: int = 4,
    timeout_s: int = 60,
) -> bool:
    """Download atomically. Return True when bytes were downloaded now."""

    if destination.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "alphaforge-dataset/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(partial, destination)
            return True
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(f"archive not found: {url}") from exc
            if 400 <= exc.code < 500:
                raise RuntimeError(f"non-retryable HTTP {exc.code}: {url}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            partial.unlink(missing_ok=True)
            last_error = exc

        if attempt < attempts - 1:
            time.sleep((2**attempt) + random.random() * 0.25)

    assert last_error is not None
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksum_file(path: Path, expected_filename: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty checksum file: {path}")
    parts = text.replace("*", " ").split()
    if not parts:
        raise ValueError(f"invalid checksum file: {path}")
    digest = parts[0].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"invalid SHA-256 in {path}: {digest!r}")
    if len(parts) > 1 and Path(parts[-1]).name != expected_filename:
        raise ValueError(
            f"checksum filename mismatch in {path}: expected {expected_filename!r}, "
            f"got {Path(parts[-1]).name!r}"
        )
    return digest


def _download_one(spec: ArchiveSpec, cache_root: Path) -> DownloadedArchive:
    zip_path = cache_root / spec.relative_path
    checksum_path = cache_root / f"{spec.relative_path}.CHECKSUM"

    downloaded_checksum = _http_download(spec.checksum_url, checksum_path)
    downloaded_zip = _http_download(spec.url, zip_path)
    expected = _parse_checksum_file(checksum_path, zip_path.name)
    actual = _sha256_file(zip_path)
    if actual != expected:
        # A stale/corrupt cache must never be silently reused.
        zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {spec.relative_path}: expected {expected}, got {actual}"
        )

    return DownloadedArchive(
        spec=spec,
        zip_path=zip_path,
        checksum_path=checksum_path,
        sha256=actual,
        size_bytes=zip_path.stat().st_size,
        downloaded=downloaded_zip or downloaded_checksum,
    )


def download_archives(
    specs: Sequence[ArchiveSpec],
    cache_root: Path,
    workers: int,
    *,
    allow_leading_missing_archives: bool,
) -> tuple[list[DownloadedArchive], list[dict[str, Any]]]:
    """Download and checksum-verify archives concurrently.

    Missing archives are normally fatal.  ``allow_leading_missing_archives`` is
    intended only for mixed-listing-date universes: it permits missing files
    before the first successful archive for a symbol/data_type, but never a
    hole after data has started.
    """

    results: dict[str, DownloadedArchive] = {}
    missing: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, spec, cache_root): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                archive = future.result()
                results[spec.relative_path] = archive
                print(
                    f"[OK] {spec.symbol:<12} {spec.data_type:<20} {spec.period}",
                    flush=True,
                )
            except FileNotFoundError as exc:
                missing.append({"spec": asdict(spec), "error": str(exc)})
                print(
                    f"[MISS] {spec.symbol:<12} {spec.data_type:<20} {spec.period}",
                    flush=True,
                )
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise

    if missing:
        if not allow_leading_missing_archives:
            sample = "\n".join(item["spec"]["relative_path"] for item in missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} required archives are missing; first entries:\n{sample}"
            )

        # Check that missing files occur only before the first successful period
        # for each symbol/data_type.  A missing archive inside or after coverage
        # is a real gap and remains fatal.
        successful_periods: dict[tuple[str, str], list[str]] = {}
        for archive in results.values():
            key = (archive.spec.symbol, archive.spec.data_type)
            successful_periods.setdefault(key, []).append(archive.spec.period)

        illegal: list[dict[str, Any]] = []
        for item in missing:
            spec_dict = item["spec"]
            key = (spec_dict["symbol"], spec_dict["data_type"])
            periods = successful_periods.get(key, [])
            if not periods or spec_dict["period"] >= min(periods):
                illegal.append(item)
        if illegal:
            sample = "\n".join(i["spec"]["relative_path"] for i in illegal[:10])
            raise FileNotFoundError(
                "missing archives occur inside/after observed coverage; refusing to build:\n"
                f"{sample}"
            )

    ordered = [results[key] for key in sorted(results)]
    return ordered, missing


# ---------------------------------------------------------------------------
# ZIP/CSV parsing with Polars
# ---------------------------------------------------------------------------

def _extract_single_csv(zip_path: Path, temp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise ValueError(
                f"{zip_path}: expected exactly one CSV member, found "
                f"{[m.filename for m in members]}"
            )
        member = members[0]
        output = temp_dir / Path(member.filename).name
        with archive.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        return output


def _csv_has_header(path: Path) -> bool:
    with path.open("rb") as handle:
        first_line = handle.readline(16_384).decode("utf-8-sig", errors="strict")
    if not first_line.strip():
        raise ValueError(f"empty CSV: {path}")
    first_cell = first_line.split(",", 1)[0].strip().strip('"')
    try:
        int(first_cell)
        return False
    except ValueError:
        return True


def _normalize_column_name(name: str) -> str:
    return name.strip().strip('"').lower().replace(" ", "_").replace("-", "_")


def _timestamp_ms_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Int64, strict=False)
    # USD-M archives are ms today.  Dividing values above 1e14 also keeps the
    # parser safe if an archive later follows the spot microsecond convention.
    return (
        pl.when(value >= 100_000_000_000_000)
        .then(value // 1_000)
        .otherwise(value)
    )


def _scan_archive_csv(archive: DownloadedArchive, temp_dir: Path) -> pl.LazyFrame:
    csv_path = _extract_single_csv(archive.zip_path, temp_dir)
    has_header = _csv_has_header(csv_path)

    if archive.spec.data_type in KLINE_TYPES:
        return pl.scan_csv(
            csv_path,
            has_header=has_header,
            new_columns=None if has_header else list(KLINE_COLUMNS),
            infer_schema=False,
            with_column_names=(
                (lambda cols: [_normalize_column_name(col) for col in cols])
                if has_header
                else None
            ),
        )

    if archive.spec.data_type == "fundingRate":
        # Headerless funding files are assumed to use Binance's documented
        # three-column archive order.  Headered files are alias-routed below.
        return pl.scan_csv(
            csv_path,
            has_header=has_header,
            new_columns=(
                None
                if has_header
                else ["calc_time", "funding_interval_hours", "last_funding_rate"]
            ),
            infer_schema=False,
            with_column_names=(
                (lambda cols: [_normalize_column_name(col) for col in cols])
                if has_header
                else None
            ),
        )

    raise ValueError(f"unsupported data type {archive.spec.data_type}")


def _resolve_alias(columns: Sequence[str], aliases: Sequence[str], label: str) -> str:
    normalized = {_normalize_column_name(column): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(f"cannot resolve {label}; columns={list(columns)}, aliases={list(aliases)}")


def _normalize_kline_archive(
    archive: DownloadedArchive,
    *,
    start_ts: int,
    end_ts: int,
    temp_dir: Path,
) -> pl.DataFrame:
    lf = _scan_archive_csv(archive, temp_dir)
    columns = lf.collect_schema().names()

    missing = [name for name in KLINE_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"{archive.spec.relative_path}: missing kline columns {missing}; got {columns}"
        )

    common = [
        pl.lit(EXCHANGE).alias("exchange"),
        pl.lit(MARKET).alias("market"),
        pl.lit(archive.spec.symbol).alias("symbol"),
        pl.lit(archive.spec.interval).alias("interval"),
        _timestamp_ms_expr("open_time").alias("open_ts"),
        _timestamp_ms_expr("close_time").alias("close_ts"),
        pl.col("open").cast(pl.Float64, strict=False).alias("open"),
        pl.col("high").cast(pl.Float64, strict=False).alias("high"),
        pl.col("low").cast(pl.Float64, strict=False).alias("low"),
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
    ]

    if archive.spec.data_type == "klines":
        expressions = [
            *common,
            pl.col("volume").cast(pl.Float64, strict=False).alias("base_volume"),
            pl.col("quote_volume").cast(pl.Float64, strict=False).alias("quote_volume"),
            pl.col("count").cast(pl.Int64, strict=False).alias("trade_count"),
            pl.col("taker_buy_volume")
            .cast(pl.Float64, strict=False)
            .alias("taker_buy_base_volume"),
            pl.col("taker_buy_quote_volume")
            .cast(pl.Float64, strict=False)
            .alias("taker_buy_quote_volume"),
            pl.lit(False).alias("is_synthetic"),
            pl.lit(archive.spec.relative_path).alias("source_file"),
            pl.lit(archive.sha256).alias("source_sha256"),
        ]
    else:
        expressions = [
            *common,
            pl.lit(False).alias("is_synthetic"),
            pl.lit(archive.spec.relative_path).alias("source_file"),
            pl.lit(archive.sha256).alias("source_sha256"),
        ]

    return (
        lf.select(expressions)
        .filter((pl.col("open_ts") >= start_ts) & (pl.col("open_ts") < end_ts))
        .sort("open_ts")
        .collect(engine="streaming")
    )


def _normalize_funding_archive(
    archive: DownloadedArchive,
    *,
    start_ts: int,
    end_ts: int,
    temp_dir: Path,
) -> pl.DataFrame:
    lf = _scan_archive_csv(archive, temp_dir)
    columns = lf.collect_schema().names()
    timestamp_col = _resolve_alias(columns, FUNDING_TIMESTAMP_ALIASES, "funding timestamp")
    rate_col = _resolve_alias(columns, FUNDING_RATE_ALIASES, "funding rate")

    normalized_names = {_normalize_column_name(column): column for column in columns}
    interval_col = next(
        (normalized_names[alias] for alias in FUNDING_INTERVAL_ALIASES if alias in normalized_names),
        None,
    )

    interval_expr = (
        pl.col(interval_col).cast(pl.Float64, strict=False)
        if interval_col is not None
        else pl.lit(None, dtype=pl.Float64)
    )

    return (
        lf.select(
            [
                pl.lit(EXCHANGE).alias("exchange"),
                pl.lit(MARKET).alias("market"),
                pl.lit(archive.spec.symbol).alias("symbol"),
                _timestamp_ms_expr(timestamp_col).alias("funding_ts"),
                pl.col(rate_col).cast(pl.Float64, strict=False).alias("funding_rate"),
                interval_expr.alias("funding_interval_hours"),
                pl.lit(archive.spec.relative_path).alias("source_file"),
                pl.lit(archive.sha256).alias("source_sha256"),
            ]
        )
        .filter((pl.col("funding_ts") >= start_ts) & (pl.col("funding_ts") < end_ts))
        .sort("funding_ts")
        .collect(engine="streaming")
    )


def normalize_archive(
    archive: DownloadedArchive,
    *,
    start_ts: int,
    end_ts: int,
    temp_dir: Path,
) -> pl.DataFrame:
    if archive.spec.data_type in KLINE_TYPES:
        return _normalize_kline_archive(
            archive,
            start_ts=start_ts,
            end_ts=end_ts,
            temp_dir=temp_dir,
        )
    if archive.spec.data_type == "fundingRate":
        return _normalize_funding_archive(
            archive,
            start_ts=start_ts,
            end_ts=end_ts,
            temp_dir=temp_dir,
        )
    raise ValueError(f"unsupported data type {archive.spec.data_type}")


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _count_true(df: pl.DataFrame, expression: pl.Expr) -> int:
    value = df.select(expression.sum().cast(pl.Int64).alias("count")).item()
    return int(value or 0)


def _null_counts(df: pl.DataFrame) -> dict[str, int]:
    row = df.null_count().row(0, named=True)
    return {str(key): int(value) for key, value in row.items()}


def _duplicate_count(df: pl.DataFrame, keys: Sequence[str]) -> int:
    return int(
        df.group_by(list(keys))
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().fill_null(0))
        .item()
    )


def _timestamp_column(tape: str) -> str:
    return "funding_ts" if tape == "funding" else "open_ts"


def _validate_schema(tape: str, df: pl.DataFrame) -> None:
    try:
        SCHEMA_BY_TAPE[tape].validate(df, lazy=True)
    except Exception as exc:
        raise ValueError(f"Pandera validation failed for tape={tape}: {exc}") from exc


def _validate_common(tape: str, df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        raise ValueError(f"tape={tape} is empty")

    _validate_schema(tape, df)
    nulls = _null_counts(df)
    illegal_nulls = {
        key: value
        for key, value in nulls.items()
        if value > 0 and not (tape == "funding" and key == "funding_interval_hours")
    }
    if illegal_nulls:
        raise ValueError(f"tape={tape} contains nulls: {illegal_nulls}")

    primary_key = PRIMARY_KEY_BY_TAPE[tape]
    duplicates = _duplicate_count(df, primary_key)
    if duplicates:
        raise ValueError(f"tape={tape} has {duplicates} duplicate primary-key rows")

    sort_columns = SORT_COLUMNS_BY_TAPE[tape]
    if not df.select(sort_columns).equals(df.sort(sort_columns).select(sort_columns)):
        raise ValueError(f"tape={tape} is not sorted by {sort_columns}")

    ts_column = _timestamp_column(tape)
    first_ts, last_ts = df.select(
        pl.col(ts_column).min().alias("first"),
        pl.col(ts_column).max().alias("last"),
    ).row(0)

    return {
        "rows": df.height,
        "null_counts": nulls,
        "duplicate_primary_keys": duplicates,
        "first_ts": int(first_ts),
        "last_ts": int(last_ts),
        "sorted_by": sort_columns,
    }


def _validate_price_rows(tape: str, df: pl.DataFrame) -> dict[str, Any]:
    report = _validate_common(tape, df)

    non_positive = _count_true(
        df,
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0),
    )
    bad_ohlc = _count_true(
        df,
        (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high")),
    )
    invalid_time = _count_true(df, pl.col("close_ts") <= pl.col("open_ts"))
    synthetic = _count_true(df, pl.col("is_synthetic"))

    if (tape != "premium" and non_positive) or bad_ohlc or invalid_time or synthetic:
        raise ValueError(
            f"tape={tape} failed price invariants: non_positive={non_positive}, "
            f"bad_ohlc={bad_ohlc}, invalid_time={invalid_time}, synthetic={synthetic}"
        )

    report.update(
        {
            "non_positive_price_rows": non_positive,
            "invalid_ohlc_rows": bad_ohlc,
            "invalid_close_time_rows": invalid_time,
            "synthetic_rows": synthetic,
        }
    )
    return report


def _validate_bars(df: pl.DataFrame) -> dict[str, Any]:
    report = _validate_price_rows("bars", df)
    negative_volume = _count_true(
        df,
        (pl.col("base_volume") < 0)
        | (pl.col("quote_volume") < 0)
        | (pl.col("trade_count") < 0)
        | (pl.col("taker_buy_base_volume") < 0)
        | (pl.col("taker_buy_quote_volume") < 0),
    )
    taker_exceeds_total = _count_true(
        df,
        (pl.col("taker_buy_base_volume") > pl.col("base_volume") + 1e-12)
        | (pl.col("taker_buy_quote_volume") > pl.col("quote_volume") + 1e-8),
    )
    if negative_volume or taker_exceeds_total:
        raise ValueError(
            "bars failed volume invariants: "
            f"negative={negative_volume}, taker_exceeds_total={taker_exceeds_total}"
        )
    report.update(
        {
            "negative_volume_rows": negative_volume,
            "taker_volume_exceeds_total_rows": taker_exceeds_total,
        }
    )
    return report


def _validate_funding(df: pl.DataFrame) -> dict[str, Any]:
    report = _validate_common("funding", df)
    non_finite = _count_true(df, pl.col("funding_rate").is_nan() | pl.col("funding_rate").is_infinite())
    implausible = _count_true(df, pl.col("funding_rate").abs() > 0.10)
    if non_finite or implausible:
        raise ValueError(
            f"funding failed invariants: non_finite={non_finite}, abs(rate)>10%={implausible}"
        )
    report.update(
        {
            "non_finite_rate_rows": non_finite,
            "implausible_rate_rows": implausible,
        }
    )
    return report


def _validate_exact_coverage(
    df: pl.DataFrame,
    *,
    interval_ms: int,
    requested_start_ts: int,
    requested_end_ts: int,
    allow_leading_gap: bool,
) -> dict[str, Any]:
    observed = df.get_column("open_ts")
    first_ts = int(observed[0])
    effective_start = first_ts if allow_leading_gap else requested_start_ts

    if first_ts != effective_start:
        raise ValueError(
            f"coverage starts at {first_ts}, expected {effective_start}; "
            "use allow_leading_missing_archives only for pre-listing periods"
        )

    expected_rows = (requested_end_ts - effective_start) // interval_ms
    if (requested_end_ts - effective_start) % interval_ms:
        raise ValueError("requested window is not aligned to the interval grid")

    if df.height != expected_rows:
        # Produce a compact sample of missing timestamps without allocating the
        # full expected grid for very large datasets.
        gap_rows = df.with_columns(
            pl.col("open_ts").diff().alias("delta")
        ).filter(pl.col("delta") != interval_ms)
        sample = gap_rows.select("open_ts", "delta").head(20).to_dicts()
        raise ValueError(
            f"coverage mismatch: rows={df.height}, expected={expected_rows}, "
            f"first={first_ts}, requested_end={requested_end_ts}, gaps={sample}"
        )

    bad_spacing = int((observed.diff().drop_nulls() != interval_ms).sum())
    if bad_spacing:
        raise ValueError(f"found {bad_spacing} non-{interval_ms}ms timestamp steps")

    expected_last = requested_end_ts - interval_ms
    actual_last = int(observed[-1])
    if actual_last != expected_last:
        raise ValueError(f"last open_ts={actual_last}, expected={expected_last}")

    return {
        "coverage_complete": True,
        "effective_start_ts": effective_start,
        "requested_end_ts": requested_end_ts,
        "expected_rows": expected_rows,
        "actual_rows": df.height,
        "gap_count": 0,
        "first_ts": first_ts,
        "last_ts": actual_last,
    }


def _validate_alignment(primary: pl.DataFrame, other: pl.DataFrame, tape: str) -> dict[str, Any]:
    """Verify that *other* timestamps align with *primary* via inner join.

    Row-identical alignment is NOT required — mark/index/premium feeds may
    start later or have sparse coverage relative to the traded-bar tape.
    """
    joined = primary.select("open_ts").join(
        other.select("open_ts"), on="open_ts", how="inner"
    )
    if joined.is_empty():
        raise ValueError(f"{tape}: no overlapping open_ts with bars")
    coverage = joined.height / primary.height
    return {"aligned_to_bars": True, "rows": other.height, "coverage_ratio": round(coverage, 6)}


def validate_tape(tape: str, df: pl.DataFrame) -> dict[str, Any]:
    if tape == "bars":
        return _validate_bars(df)
    if tape in {"mark", "index", "premium"}:
        return _validate_price_rows(tape, df)
    if tape == "funding":
        return _validate_funding(df)
    raise ValueError(f"unknown tape {tape}")


# ---------------------------------------------------------------------------
# Canonical hashing and Parquet persistence
# ---------------------------------------------------------------------------

def _hash_arrow_array(digest: "hashlib._Hash", array: arrow.Array) -> None:
    digest.update(str(array.type).encode("utf-8"))
    digest.update(struct.pack(">q", len(array)))
    for buffer in array.buffers():
        if buffer is None:
            digest.update(b"\x00")
        else:
            data = memoryview(buffer)
            digest.update(b"\x01")
            digest.update(struct.pack(">q", len(data)))
            digest.update(data)


def content_hash(df: pl.DataFrame) -> str:
    """Hash logical Arrow content, independent of Parquet compression/layout."""

    table = df.to_arrow().combine_chunks()
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    digest.update(struct.pack(">q", table.num_rows))
    digest.update(struct.pack(">q", table.num_columns))
    for field, chunked in zip(table.schema, table.columns):
        digest.update(field.name.encode("utf-8"))
        digest.update(b"\x00")
        array = chunked.chunk(0) if chunked.num_chunks else arrow.array([], type=field.type)
        _hash_arrow_array(digest, array)
    return digest.hexdigest()


def _write_parquet(
    df: pl.DataFrame,
    destination: Path,
    *,
    tape: str,
    symbol: str,
    logical_hash: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"alphaforge.schema_version": SCHEMA_VERSION.encode("utf-8"),
        b"alphaforge.builder_version": BUILDER_VERSION.encode("utf-8"),
        b"alphaforge.exchange": EXCHANGE.encode("utf-8"),
        b"alphaforge.market": MARKET.encode("utf-8"),
        b"alphaforge.tape": tape.encode("utf-8"),
        b"alphaforge.symbol": symbol.encode("utf-8"),
        b"alphaforge.content_sha256": logical_hash.encode("utf-8"),
    }
    table = df.to_arrow().replace_schema_metadata(metadata)
    pq.write_table(
        table,
        destination,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        write_statistics=True,
        store_schema=True,
        version="2.6",
    )


def _artifact_path(tape: str, symbol: str, interval: str) -> Path:
    suffix = "events" if tape == "funding" else interval
    return Path(tape) / f"symbol={symbol}" / f"{tape}_{suffix}.parquet"


# ---------------------------------------------------------------------------
# Metadata and dataset build
# ---------------------------------------------------------------------------

def _fetch_exchange_info(timeout_s: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        FAPI_EXCHANGE_INFO_URL,
        headers={"User-Agent": "alphaforge-dataset/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or "symbols" not in payload:
        raise ValueError("unexpected Binance exchangeInfo response")
    return payload


def _select_symbol_metadata(exchange_info: Mapping[str, Any], symbols: Sequence[str]) -> dict[str, Any]:
    by_symbol = {
        item.get("symbol"): item
        for item in exchange_info.get("symbols", [])
        if isinstance(item, dict)
    }
    missing = [symbol for symbol in symbols if symbol not in by_symbol]
    if missing:
        raise ValueError(f"symbols missing from current Binance exchangeInfo: {missing}")
    return {symbol: by_symbol[symbol] for symbol in symbols}


def _build_symbol_tapes(
    *,
    symbol: str,
    archives: Sequence[DownloadedArchive],
    config: DatasetConfig,
    staging_dir: Path,
) -> list[TapeArtifact]:
    artifacts: list[TapeArtifact] = []
    by_type: dict[str, list[DownloadedArchive]] = {}
    for archive in archives:
        if archive.spec.symbol == symbol:
            by_type.setdefault(archive.spec.data_type, []).append(archive)

    dataframes: dict[str, pl.DataFrame] = {}
    source_by_tape: dict[str, list[DownloadedArchive]] = {}

    with tempfile.TemporaryDirectory(prefix=f"normalize-{symbol}-") as temp:
        temp_dir = Path(temp)
        for data_type in config.data_types:
            tape = TAPE_BY_DATA_TYPE[data_type]
            inputs = sorted(
                by_type.get(data_type, []),
                key=lambda item: (item.spec.period, item.spec.cadence),
            )
            if not inputs:
                raise ValueError(f"no downloaded archives for {symbol}/{data_type}")

            frames = [
                normalize_archive(
                    archive,
                    start_ts=config.start_ts,
                    end_ts=config.end_ts,
                    temp_dir=temp_dir,
                )
                for archive in inputs
            ]
            non_empty = [frame for frame in frames if not frame.is_empty()]
            if not non_empty:
                raise ValueError(f"all archives normalized to zero rows for {symbol}/{data_type}")

            df = pl.concat(non_empty, how="vertical_relaxed").sort(SORT_COLUMNS_BY_TAPE[tape])
            dataframes[tape] = df
            source_by_tape[tape] = inputs

    # Per-tape schema and economic gates.
    validation: dict[str, dict[str, Any]] = {
        tape: validate_tape(tape, df) for tape, df in dataframes.items()
    }

    # The primary traded-bar tape owns exact coverage.  Price tapes must be
    # index-identical; no implicit forward fill or padding is permitted.
    coverage = _validate_exact_coverage(
        dataframes["bars"],
        interval_ms=INTERVAL_MS[config.interval],
        requested_start_ts=config.start_ts,
        requested_end_ts=config.end_ts,
        allow_leading_gap=config.allow_leading_missing_archives,
    )
    validation["bars"]["coverage"] = coverage

    for tape in ("mark", "index", "premium"):
        if tape in dataframes:
            validation[tape]["alignment"] = _validate_alignment(
                dataframes["bars"], dataframes[tape], tape
            )

    for tape, df in dataframes.items():
        logical_hash = content_hash(df)
        relative = _artifact_path(tape, symbol, config.interval)
        absolute = staging_dir / relative
        _write_parquet(
            df,
            absolute,
            tape=tape,
            symbol=symbol,
            logical_hash=logical_hash,
        )
        file_hash = _sha256_file(absolute)
        timestamp_col = _timestamp_column(tape)
        first_ts, last_ts = df.select(
            pl.col(timestamp_col).min().alias("first_ts"),
            pl.col(timestamp_col).max().alias("last_ts"),
        ).row(0)
        inputs = source_by_tape[tape]
        artifacts.append(
            TapeArtifact(
                tape=tape,
                symbol=symbol,
                relative_path=relative.as_posix(),
                rows=df.height,
                first_ts=int(first_ts) if first_ts is not None else None,
                last_ts=int(last_ts) if last_ts is not None else None,
                content_sha256=logical_hash,
                file_sha256=file_hash,
                size_bytes=absolute.stat().st_size,
                source_files=tuple(item.spec.relative_path for item in inputs),
                source_sha256=tuple(item.sha256 for item in inputs),
                validation=validation[tape],
            )
        )

    return sorted(artifacts, key=lambda item: item.tape)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build(config: DatasetConfig) -> dict[str, Any]:
    """Build one immutable canonical dataset and return its manifest."""

    config.out_dir.parent.mkdir(parents=True, exist_ok=True)
    config.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.out_dir.name}.build-", dir=config.out_dir.parent)
    )

    try:
        specs = plan_archives(config)
        _write_json(
            staging_dir / "archive_plan.json",
            {
                "schema_version": SCHEMA_VERSION,
                "archives": [asdict(spec) for spec in specs],
            },
        )

        archives, missing = download_archives(
            specs,
            config.raw_cache_dir,
            config.workers,
            allow_leading_missing_archives=config.allow_leading_missing_archives,
        )

        metadata: dict[str, Any] = {}
        if config.fetch_exchange_info:
            exchange_info = _fetch_exchange_info()
            metadata = _select_symbol_metadata(exchange_info, config.symbols)
            _write_json(
                staging_dir / "metadata" / "exchange_info_snapshot.json",
                {
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "warning": (
                        "This is a current metadata snapshot, not point-in-time history. "
                        "Historical parameter changes require a separate metadata tape."
                    ),
                    "symbols": metadata,
                },
            )

        artifacts: list[TapeArtifact] = []
        # Symbol builds are intentionally sequential: archive downloads already
        # saturate network concurrency, and per-symbol Polars parsing uses all CPU
        # threads internally.  This avoids memory spikes on 20-symbol builds.
        for symbol in config.symbols:
            print(f"[BUILD] {symbol}", flush=True)
            artifacts.extend(
                _build_symbol_tapes(
                    symbol=symbol,
                    archives=archives,
                    config=config,
                    staging_dir=staging_dir,
                )
            )

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "exchange": EXCHANGE,
            "market": MARKET,
            "status": "VERIFIED",
            "config": config.manifest_dict(),
            "source_policy": {
                "canonical_authority": "Binance Data Collection / data.binance.vision",
                "archive_checksum_required": True,
                "third_party_mirror_allowed": False,
                "synthetic_rows_allowed": False,
                "forward_fill_allowed": False,
            },
            "missing_leading_archives": missing,
            "metadata_snapshot_present": bool(metadata),
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "totals": {
                "archives": len(archives),
                "archive_bytes": sum(item.size_bytes for item in archives),
                "parquet_files": len(artifacts),
                "parquet_bytes": sum(item.size_bytes for item in artifacts),
                "rows": sum(item.rows for item in artifacts),
            },
        }

        # DuckDB audit is part of publication, not an optional afterthought.
        audit_report = _audit_staging(staging_dir, manifest)
        manifest["audit"] = audit_report
        _write_json(staging_dir / "audit.json", audit_report)
        _write_json(staging_dir / "manifest.json", manifest)

        os.rename(staging_dir, config.out_dir)

        if not config.keep_raw:
            for archive in archives:
                archive.zip_path.unlink(missing_ok=True)
                archive.checksum_path.unlink(missing_ok=True)

        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# DuckDB audit
# ---------------------------------------------------------------------------

def _quote_sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _duckdb_file_summary(path: Path, tape: str) -> dict[str, Any]:
    timestamp_col = _timestamp_column(tape)
    key_cols = PRIMARY_KEY_BY_TAPE[tape]
    key_sql = ", ".join(f'"{column}"' for column in key_cols)
    quoted_path = _quote_sql_path(path)

    connection = duckdb.connect(database=":memory:")
    try:
        row = connection.execute(
            f"""
            SELECT
                count(*) AS rows,
                min("{timestamp_col}") AS first_ts,
                max("{timestamp_col}") AS last_ts
            FROM read_parquet('{quoted_path}')
            """
        ).fetchone()
        duplicate_rows = connection.execute(
            f"""
            SELECT coalesce(sum(n - 1), 0)
            FROM (
                SELECT count(*) AS n
                FROM read_parquet('{quoted_path}')
                GROUP BY {key_sql}
                HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        parquet_meta = connection.execute(
            f"""
            SELECT
                count(DISTINCT row_group_id) AS row_groups,
                sum(row_group_num_rows) AS metadata_rows
            FROM parquet_metadata('{quoted_path}')
            """
        ).fetchone()
    finally:
        connection.close()

    return {
        "rows": int(row[0]),
        "first_ts": int(row[1]) if row[1] is not None else None,
        "last_ts": int(row[2]) if row[2] is not None else None,
        "duplicate_primary_keys": int(duplicate_rows),
        "row_groups": int(parquet_meta[0]),
        "metadata_rows": int(parquet_meta[1]),
    }


def _audit_staging(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for artifact in manifest["artifacts"]:
        path = root / artifact["relative_path"]
        summary = _duckdb_file_summary(path, artifact["tape"])
        if summary["rows"] != artifact["rows"]:
            raise ValueError(
                f"DuckDB row count mismatch for {artifact['relative_path']}: "
                f"duckdb={summary['rows']}, manifest={artifact['rows']}"
            )
        if summary["duplicate_primary_keys"]:
            raise ValueError(
                f"DuckDB found duplicate primary keys in {artifact['relative_path']}"
            )
        files.append({"relative_path": artifact["relative_path"], **summary})

    return {
        "engine": f"duckdb-{duckdb.__version__}",
        "verdict": "PASS",
        "files": files,
    }


def audit(dataset_dir: Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest = _read_manifest(root)
    return _audit_staging(root, manifest)


# ---------------------------------------------------------------------------
# Load and re-verification
# ---------------------------------------------------------------------------

def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "builder_version",
        "exchange",
        "market",
        "status",
        "config",
        "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest missing fields: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {manifest['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if manifest["exchange"] != EXCHANGE or manifest["market"] != MARKET:
        raise ValueError("manifest is not a Binance USD-M canonical dataset")
    if manifest["status"] != "VERIFIED":
        raise ValueError(f"dataset status is not VERIFIED: {manifest['status']!r}")
    return manifest


def _read_artifact(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def verify(dataset_dir: Path) -> dict[str, Any]:
    """Re-verify every artifact and fail on any mismatch."""

    root = Path(dataset_dir)
    manifest = _read_manifest(root)
    verified: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pl.DataFrame] = {}

    for artifact in manifest["artifacts"]:
        path = root / artifact["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"artifact missing: {path}")

        file_hash = _sha256_file(path)
        if file_hash != artifact["file_sha256"]:
            raise ValueError(
                f"file SHA mismatch for {artifact['relative_path']}: "
                f"expected={artifact['file_sha256']}, actual={file_hash}"
            )

        df = _read_artifact(path)
        validation = validate_tape(artifact["tape"], df)
        logical_hash = content_hash(df)
        if logical_hash != artifact["content_sha256"]:
            raise ValueError(
                f"content SHA mismatch for {artifact['relative_path']}: "
                f"expected={artifact['content_sha256']}, actual={logical_hash}"
            )
        if df.height != artifact["rows"]:
            raise ValueError(
                f"row count mismatch for {artifact['relative_path']}: "
                f"expected={artifact['rows']}, actual={df.height}"
            )

        frames[(artifact["symbol"], artifact["tape"])] = df
        verified.append(
            {
                "relative_path": artifact["relative_path"],
                "rows": df.height,
                "file_sha256": file_hash,
                "content_sha256": logical_hash,
                "validation": validation,
            }
        )

    interval = manifest["config"]["interval"]
    interval_ms = INTERVAL_MS[interval]
    requested_start_ts = int(manifest["config"]["start_ts"])
    requested_end_ts = int(manifest["config"]["end_ts"])
    allow_leading = bool(manifest["config"].get("allow_leading_missing_archives", False))

    for symbol in manifest["config"]["symbols"]:
        bars = frames[(symbol, "bars")]
        _validate_exact_coverage(
            bars,
            interval_ms=interval_ms,
            requested_start_ts=requested_start_ts,
            requested_end_ts=requested_end_ts,
            allow_leading_gap=allow_leading,
        )
        for tape in ("mark", "index", "premium"):
            frame = frames.get((symbol, tape))
            if frame is not None:
                _validate_alignment(bars, frame, tape)

    duckdb_audit = _audit_staging(root, manifest)
    result = {
        "verdict": "PASS",
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(root),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": verified,
        "duckdb_audit": duckdb_audit,
    }
    return result


def load(dataset_dir: Path) -> LoadedDataset:
    verify(dataset_dir)
    return LoadedDataset(root=Path(dataset_dir), manifest=_read_manifest(Path(dataset_dir)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not symbols:
        raise argparse.ArgumentTypeError("at least one symbol is required")
    return symbols


def _parse_data_types(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values) - set(ALL_DATA_TYPES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown data types {unknown}; allowed={list(ALL_DATA_TYPES)}"
        )
    return values


def _cmd_build(args: argparse.Namespace) -> None:
    config = DatasetConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        out_dir=args.out,
        raw_cache_dir=args.raw_cache,
        interval=args.interval,
        data_types=args.data_types,
        workers=args.workers,
        keep_raw=args.keep_raw,
        fetch_exchange_info=not args.no_exchange_info,
        strict=True,
        allow_leading_missing_archives=args.allow_leading_missing_archives,
    )
    manifest = build(config)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


def _cmd_verify(args: argparse.Namespace) -> None:
    result = verify(args.dataset_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


def _cmd_audit(args: argparse.Namespace) -> None:
    result = audit(args.dataset_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


def _cmd_plan(args: argparse.Namespace) -> None:
    # A temporary non-existing output path is sufficient because planning does
    # not touch it; DatasetConfig still enforces the same input contracts.
    config = DatasetConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        out_dir=args.out,
        raw_cache_dir=args.raw_cache,
        interval=args.interval,
        data_types=args.data_types,
        workers=args.workers,
        fetch_exchange_info=False,
        allow_leading_missing_archives=args.allow_leading_missing_archives,
    )
    print(
        json.dumps(
            {"config": config.manifest_dict(), "archives": [asdict(x) for x in plan_archives(config)]},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", type=_parse_symbols, required=True)
    parser.add_argument("--start", type=_parse_date, required=True, help="inclusive UTC date")
    parser.add_argument("--end", type=_parse_date, required=True, help="exclusive UTC date")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, default=Path("data/raw-cache/binance"))
    parser.add_argument("--interval", choices=sorted(INTERVAL_MS), default="1m")
    parser.add_argument(
        "--data-types",
        type=_parse_data_types,
        default=ALL_DATA_TYPES,
        help="comma-separated; default is all canonical macro tapes",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-leading-missing-archives", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="build and atomically publish a canonical dataset")
    _add_build_arguments(build_parser)
    build_parser.add_argument("--keep-raw", action="store_true")
    build_parser.add_argument("--no-exchange-info", action="store_true")
    build_parser.set_defaults(func=_cmd_build)

    plan_parser = sub.add_parser("plan", help="print archive URLs without downloading")
    _add_build_arguments(plan_parser)
    plan_parser.set_defaults(func=_cmd_plan)

    verify_parser = sub.add_parser("verify", help="re-verify an existing canonical dataset")
    verify_parser.add_argument("--dataset-dir", type=Path, required=True)
    verify_parser.set_defaults(func=_cmd_verify)

    audit_parser = sub.add_parser("audit", help="run DuckDB file/row-group audit")
    audit_parser.add_argument("--dataset-dir", type=Path, required=True)
    audit_parser.set_defaults(func=_cmd_audit)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
