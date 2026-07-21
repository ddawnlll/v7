from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import dataset


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source" / "data"
        self.cache = self.root / "cache"
        self.output = self.root / "canonical"
        self.symbol = "BTCUSDT"
        self.day = date(2024, 1, 15)
        self.start_ts = int(
            datetime(2024, 1, 15, tzinfo=timezone.utc).timestamp() * 1000
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_zip(self, relative_path: str, rows: list[list[object]]) -> Path:
        zip_path = self.source / relative_path
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = zip_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname=csv_path.name)
        csv_path.unlink()

        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        Path(f"{zip_path}.CHECKSUM").write_text(
            f"{digest}  {zip_path.name}\n", encoding="utf-8"
        )
        return zip_path

    def _kline_rows(self, *, omit_index: int | None = None) -> list[list[object]]:
        rows: list[list[object]] = []
        for index in range(1_440):
            if index == omit_index:
                continue
            open_ts = self.start_ts + index * 60_000
            close_ts = open_ts + 59_999
            price = 100 + index * 0.001
            rows.append(
                [
                    open_ts,
                    price,
                    price + 1,
                    price - 1,
                    price + 0.5,
                    10,
                    close_ts,
                    1_000,
                    5,
                    4,
                    400,
                    0,
                ]
            )
        return rows

    def _prepare_source(self, *, omit_bar: int | None = None) -> None:
        period = self.day.isoformat()
        rows = self._kline_rows(omit_index=omit_bar)
        for data_type in dataset.KLINE_TYPES:
            relative = dataset._archive_relative_path(
                symbol=self.symbol,
                data_type=data_type,
                interval="1m",
                cadence="daily",
                period=period,
            )
            self._write_zip(relative, rows)

        funding_relative = dataset._archive_relative_path(
            symbol=self.symbol,
            data_type="fundingRate",
            interval=None,
            cadence="monthly",
            period="2024-01",
        )
        self._write_zip(
            funding_relative,
            [
                ["calc_time", "funding_interval_hours", "last_funding_rate"],
                [self.start_ts + 8 * 3_600_000, 8, "0.0001"],
            ],
        )

    def _config(self) -> dataset.DatasetConfig:
        return dataset.DatasetConfig(
            symbols=(self.symbol,),
            start=self.day,
            end=date(2024, 1, 16),
            out_dir=self.output,
            raw_cache_dir=self.cache,
            base_url=self.source.as_uri(),
            workers=2,
            keep_raw=True,
            fetch_exchange_info=False,
        )

    def test_build_verify_audit_and_lazy_load(self) -> None:
        self._prepare_source()
        manifest = dataset.build(self._config())

        self.assertEqual(manifest["status"], "VERIFIED")
        self.assertEqual(len(manifest["artifacts"]), 5)
        self.assertEqual(dataset.verify(self.output)["verdict"], "PASS")
        self.assertEqual(dataset.audit(self.output)["verdict"], "PASS")

        loaded = dataset.load(self.output)
        self.assertEqual(loaded.scan("bars").select("open_ts").collect().height, 1_440)

    def test_missing_primary_bar_fails_closed(self) -> None:
        self._prepare_source(omit_bar=700)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            dataset.build(self._config())
        self.assertFalse(self.output.exists())

    def test_published_file_tampering_is_detected(self) -> None:
        self._prepare_source()
        dataset.build(self._config())
        bars = self.output / "bars" / f"symbol={self.symbol}" / "bars_1m.parquet"
        with bars.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "file SHA mismatch"):
            dataset.verify(self.output)


if __name__ == "__main__":
    unittest.main()
