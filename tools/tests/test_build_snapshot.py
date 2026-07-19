"""Offline tests for tools/build_snapshot.py's pure-logic guards.

Deliberately does NOT test the network fetch functions themselves — that
would mean either hitting live OKX from CI (flaky, slow, rate-limited) or
mocking urllib deeply enough that the test proves nothing about the real
endpoint. What's covered here is the part that's actually deterministic and
was actually wrong: window validation in build(), and the pagination
cursor/max_pages guards in _paginate_bounded(), both fixed after an external
review found build() would silently discard a half-given window and
_paginate_bounded() would silently loop on a stuck cursor or truncate a
too-large window with no error.
"""
from unittest import mock

import pytest

from tools import build_snapshot as bs


# --- build() window validation ------------------------------------------------

def test_partial_bound_start_only_rejected(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        bs.build(start_ts=1776698400000, end_ts=None, out_dir=tmp_path / "snap")


def test_partial_bound_end_only_rejected(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        bs.build(start_ts=None, end_ts=1784474400000, out_dir=tmp_path / "snap")


def test_unsupported_bar_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported bar"):
        bs.build(bar="1m", out_dir=tmp_path / "snap")


def test_inverted_window_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be before"):
        bs.build(start_ts=1784474400000, end_ts=1776698400000, out_dir=tmp_path / "snap")


def test_misaligned_window_rejected(tmp_path):
    with pytest.raises(ValueError, match="align to the"):
        bs.build(start_ts=1776698400001, end_ts=1784474400000, out_dir=tmp_path / "snap")


def test_nonpositive_days_rejected(tmp_path):
    with pytest.raises(ValueError, match="days must be positive"):
        bs.build(days=0, out_dir=tmp_path / "snap")


def test_existing_out_dir_rejected(tmp_path):
    existing = tmp_path / "snap"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        bs.build(start_ts=1776698400000, end_ts=1776698700000, out_dir=existing)


# --- _paginate_bounded() guards -----------------------------------------------

def test_pagination_stuck_cursor_raises():
    with mock.patch.object(bs, "_get_retry") as get_retry:
        get_retry.return_value = [[str(1000), "1", "1", "1", "1", "1"]]  # same page forever
        with pytest.raises(RuntimeError, match="did not advance"):
            bs._paginate_bounded(
                "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=3
            )


def test_pagination_max_pages_exhausted_raises():
    counter = {"n": 2000}

    def descending_page(_path, _params):
        counter["n"] -= 1
        return [[str(counter["n"]), "1", "1", "1", "1", "1"]]

    with mock.patch.object(bs, "_get_retry", side_effect=descending_page):
        with pytest.raises(RuntimeError, match="exhausted max_pages"):
            bs._paginate_bounded(
                "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=3
            )


def test_pagination_normal_termination_still_works():
    """Regression guard: the two new checks above must not break the
    ordinary case of a cursor that strictly advances to stop_ts."""
    pages = [
        [[str(500), "1", "1", "1", "1", "1"]],
        [[str(0), "1", "1", "1", "1", "1"]],
    ]
    with mock.patch.object(bs, "_get_retry", side_effect=lambda _path, _params: pages.pop(0)):
        rows = bs._paginate_bounded(
            "/fake", {}, "9999", lambda row: int(row[0]), 0, limit=1, max_pages=5
        )
    assert len(rows) == 2
