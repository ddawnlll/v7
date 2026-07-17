"""
Real data guard for V7 Engine.

Every research/training/candidate script MUST call assert_real_data()
after loading data. Synthetic data is FORBIDDEN in any code path that
produces research results, handoff packages, or candidate evaluations.

Synthetic data is ONLY permitted in:
  - Unit tests
  - CI pipeline integrity checks
  - Feature computation unit tests
"""

from __future__ import annotations

import sys
from typing import Any

# Sentinel object that synthetic loader attaches to mark data
_SENTINEL_KEY = "_v7_data_source"
_SENTINEL_SYNTHETIC = "SYNTHETIC"
_SENTINEL_REAL = "REAL"


def tag_as_synthetic(ohlcv: dict) -> dict:
    """Mark a data dict as synthetic (for testing only)."""
    ohlcv[_SENTINEL_KEY] = _SENTINEL_SYNTHETIC
    return ohlcv


def tag_as_real(ohlcv: dict) -> dict:
    """Mark a data dict as real Binance data."""
    ohlcv[_SENTINEL_KEY] = _SENTINEL_REAL
    return ohlcv


def is_real_data(ohlcv: dict) -> bool:
    """Check if the data dict is tagged as real."""
    if _SENTINEL_KEY not in ohlcv:
        return False
    return ohlcv[_SENTINEL_KEY] == _SENTINEL_REAL


def assert_real_data(ohlcv: dict) -> None:
    """Assert that the data dict contains real Binance data.

    Raises SystemExit(1) with a clear error message if the data
    is synthetic or untagged. This is a HARD STOP — not a warning.
    """
    if is_real_data(ohlcv):
        return

    if _SENTINEL_KEY in ohlcv and ohlcv[_SENTINEL_KEY] == _SENTINEL_SYNTHETIC:
        msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  SENTETİK VERİ TESPİT EDİLDİ                             ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Bu script production/research sonucu üretemez.           ║\n"
            "║  generate_synthetic_ohlcv() sadece unit testte            ║\n"
            "║  kullanılabilir.                                          ║\n"
            "║                                                          ║\n"
            "║  Çözüm: load_cached_data() ile gerçek Binance verisi     ║\n"
            "║  yükleyin. data/raw/ altında 4 sembol mevcut.            ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )
    else:
        msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  VERİ KAYNAĞI BELİRSİZ                                    ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  OHLCV dict'inde '_v7_data_source' etiketi yok.           ║\n"
            "║  load_cached_data() ile gerçek veri yükleyin veya         ║\n"
            "║  generate_synthetic_ohlcv() sonrası tag_as_synthetic()    ║\n"
            "║  ile etiketleyin (sadece test için).                     ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )

    print(msg, file=sys.stderr)
    sys.exit(1)
