"""Tests for the Timings accumulator + report formatter.

Pure Python — no Isaac / CUDA / SAM3 dependencies. The accumulator is the
backbone of the end-of-run performance report we print so demo-day
runs leave behind a sense of where the wall-clock went.
"""
from __future__ import annotations

import time

import pytest

from autonomy.timing import Timings, format_timing_report


# ── Accumulation ─────────────────────────────────────────────────────────────
def test_single_call_records_time_and_count():
    t = Timings()
    with t.time("bucket"):
        time.sleep(0.005)
    snap = t.snapshot()
    assert snap["bucket"]["count"] == 1
    assert snap["bucket"]["total_s"] >= 0.004    # sleep is at-least, not exact


def test_multiple_calls_accumulate():
    t = Timings()
    for _ in range(3):
        with t.time("bucket"):
            time.sleep(0.001)
    snap = t.snapshot()
    assert snap["bucket"]["count"] == 3
    assert snap["bucket"]["total_s"] >= 0.0025


def test_separate_buckets_are_independent():
    t = Timings()
    with t.time("a"):
        time.sleep(0.001)
    with t.time("b"):
        time.sleep(0.003)
    with t.time("a"):
        time.sleep(0.001)
    snap = t.snapshot()
    assert snap["a"]["count"] == 2
    assert snap["b"]["count"] == 1
    assert snap["b"]["total_s"] > snap["a"]["total_s"] / 2.0


def test_nested_timers_record_independently():
    """Nested ``with t.time(...)`` blocks record both buckets correctly.
    Wall-clock for outer should be >= inner.
    """
    t = Timings()
    with t.time("outer"):
        with t.time("inner"):
            time.sleep(0.005)
        time.sleep(0.005)
    snap = t.snapshot()
    assert snap["outer"]["total_s"] >= snap["inner"]["total_s"]
    assert snap["outer"]["count"] == 1
    assert snap["inner"]["count"] == 1


def test_exception_in_body_still_records():
    """If the body raises, the timer still updates totals/count and the
    exception propagates."""
    t = Timings()
    with pytest.raises(RuntimeError):
        with t.time("x"):
            time.sleep(0.001)
            raise RuntimeError("boom")
    snap = t.snapshot()
    assert snap["x"]["count"] == 1
    assert snap["x"]["total_s"] >= 0.0005


def test_record_manually_adds_to_bucket():
    """Some external timers (e.g. CUDA events) report their own duration.
    Allow direct record(bucket, seconds, count=1)."""
    t = Timings()
    t.record("ext", 0.250, count=4)
    t.record("ext", 0.050, count=1)
    snap = t.snapshot()
    assert snap["ext"]["count"] == 5
    assert snap["ext"]["total_s"] == pytest.approx(0.300, abs=1e-9)


def test_empty_timings_snapshot_is_empty_dict():
    assert Timings().snapshot() == {}


# ── Formatting ───────────────────────────────────────────────────────────────
def test_report_sorts_by_total_descending():
    t = Timings()
    t.record("small", 0.05, count=10)
    t.record("big",   1.20, count=5)
    t.record("mid",   0.60, count=3)
    rpt = format_timing_report(t.snapshot())
    # The order in the textual report should be big, mid, small.
    pos_big   = rpt.index("big")
    pos_mid   = rpt.index("mid")
    pos_small = rpt.index("small")
    assert pos_big < pos_mid < pos_small


def test_report_includes_avg_and_percent():
    t = Timings()
    t.record("a", 0.40, count=2)   # avg 0.20s
    t.record("b", 0.60, count=3)   # avg 0.20s; total 1.0s ⇒ a=40%, b=60%
    rpt = format_timing_report(t.snapshot())
    # Both buckets surface, avg of 0.20s renders as "200" (formatter
    # picks ms for sub-1s totals), and the % column shows 40 / 60.
    assert "a" in rpt and "b" in rpt
    assert "200" in rpt   # 0.20s → "200.00ms"
    assert "60.0%" in rpt
    assert "40.0%" in rpt


def test_report_handles_zero_total():
    """Empty Timings shouldn't crash the formatter."""
    rpt = format_timing_report({})
    assert "no timing data" in rpt.lower() or rpt.strip() != ""


def test_report_handles_zero_count_bucket_gracefully():
    """A bucket with count=0 (theoretically possible if record(0,0)
    is ever called) shouldn't divide-by-zero."""
    t = Timings()
    t.record("idle", 0.0, count=0)
    rpt = format_timing_report(t.snapshot())
    assert "idle" in rpt
