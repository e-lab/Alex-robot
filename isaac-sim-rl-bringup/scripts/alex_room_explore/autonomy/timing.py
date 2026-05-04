"""End-of-run performance instrumentation.

A tiny accumulator + report formatter for wall-clock time-per-bucket.
Used at the top-level script to wrap hot paths (SAM3, ONNX, planner,
physics step, etc.) so the user sees where the run actually spent time.

Pure Python, no Isaac / CUDA / numpy required. Memory readers
(``rss_bytes``, ``cuda_peak_bytes``) shell out to ``psutil`` / ``torch``
which are imported lazily so the module still loads in slim environments.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


# ── Timings accumulator ──────────────────────────────────────────────────────
class Timings:
    """Per-bucket wall-clock accumulator.

    Usage::

        T = Timings()
        with T.time("sam3.set_image"):
            run_sam3_inference()

        for prompt in prompts:
            with T.time("sam3.set_text_prompt"):
                run_sam3_text(prompt)

        # Or, if an external timer (e.g. CUDA event) reports its own duration:
        T.record("cuda.kernel", elapsed_seconds, count=1)

    ``snapshot()`` returns a JSON-friendly view; ``format_timing_report``
    renders it as a column-aligned text table.
    """

    __slots__ = ("_total_s", "_count")

    def __init__(self) -> None:
        self._total_s: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    @contextmanager
    def time(self, bucket: str) -> Iterator[None]:
        """Time the body of a ``with`` block and add it to ``bucket``.

        ``perf_counter()`` is used for wall-clock (monotonic, sub-µs).
        Exceptions still record the elapsed time before propagating, so
        failed paths still show up in the report.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(bucket, time.perf_counter() - t0, count=1)

    def record(self, bucket: str, seconds: float, *, count: int = 1) -> None:
        """Manually add ``seconds`` and ``count`` to ``bucket``.

        Useful for timers whose duration is computed externally — e.g.
        CUDA events, where ``end.elapsed_time(start)`` already gives
        milliseconds, or ad-hoc post-hoc accounting where you compute
        the duration yourself.
        """
        self._total_s[bucket] = self._total_s.get(bucket, 0.0) + float(seconds)
        self._count[bucket] = self._count.get(bucket, 0) + int(count)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a ``{bucket: {"total_s": x, "count": n}}`` dict.

        Returned data is a plain copy; callers can mutate freely.
        """
        out: Dict[str, Dict[str, float]] = {}
        for bucket, total in self._total_s.items():
            out[bucket] = {
                "total_s": float(total),
                "count": int(self._count.get(bucket, 0)),
            }
        return out


# ── Report formatter ─────────────────────────────────────────────────────────
def format_timing_report(snapshot: Dict[str, Dict[str, float]]) -> str:
    """Render a timing snapshot as a human-readable text block.

    Sorted by total time descending. Columns: bucket, calls, total,
    avg-per-call, percent of grand total.
    """
    if not snapshot:
        return "(no timing data)\n"

    rows = sorted(
        snapshot.items(),
        key=lambda kv: kv[1].get("total_s", 0.0),
        reverse=True,
    )
    grand_total = sum(v.get("total_s", 0.0) for v in snapshot.values())

    name_w = max(20, max(len(k) for k in snapshot.keys()))
    header = (
        f"{'bucket':<{name_w}}  "
        f"{'calls':>8}  "
        f"{'total':>10}  "
        f"{'avg':>10}  "
        f"{'%total':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for bucket, vals in rows:
        total = float(vals.get("total_s", 0.0))
        count = int(vals.get("count", 0))
        avg = (total / count) if count > 0 else 0.0
        pct = (100.0 * total / grand_total) if grand_total > 0 else 0.0
        lines.append(
            f"{bucket:<{name_w}}  "
            f"{count:>8d}  "
            f"{_fmt_seconds(total):>10}  "
            f"{_fmt_seconds(avg):>10}  "
            f"{pct:>6.1f}%"
        )
    lines.append(sep)
    lines.append(
        f"{'total':<{name_w}}  "
        f"{'':>8}  "
        f"{_fmt_seconds(grand_total):>10}"
    )
    return "\n".join(lines) + "\n"


def _fmt_seconds(s: float) -> str:
    """Compact time format: seconds for >= 1, ms for >= 1e-3, µs for less."""
    if s >= 1.0:
        return f"{s:.3f}s"
    if s >= 1e-3:
        return f"{s * 1000.0:.2f}ms"
    return f"{s * 1e6:.1f}µs"


# ── Memory helpers ───────────────────────────────────────────────────────────
def rss_bytes() -> Optional[int]:
    """Resident set size in bytes for the current process, or None if
    psutil isn't installed.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:                              # pragma: no cover
        return None
    return int(psutil.Process().memory_info().rss)


def cuda_peak_bytes() -> "tuple[Optional[int], Optional[int]]":
    """Return ``(allocated_peak, reserved_peak)`` in bytes for the
    default CUDA device, or ``(None, None)`` if torch / CUDA isn't
    available.

    Note: ``torch.cuda.max_memory_*`` reports peak since the last
    ``reset_peak_memory_stats`` call (or program start, whichever was
    most recent).
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:                              # pragma: no cover
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return (
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def format_memory_report() -> str:
    """One-block human-readable memory snapshot."""
    rss = rss_bytes()
    cuda_alloc, cuda_reserv = cuda_peak_bytes()
    lines = []
    if rss is not None:
        lines.append(f"  peak RSS:                {rss / (1024 * 1024):8.1f} MB")
    else:                                            # pragma: no cover
        lines.append("  peak RSS:                <psutil unavailable>")
    if cuda_alloc is not None:
        lines.append(f"  peak GPU allocated:      {cuda_alloc / (1024 * 1024):8.1f} MB")
        lines.append(f"  peak GPU reserved:       {cuda_reserv / (1024 * 1024):8.1f} MB")
    else:
        lines.append("  GPU memory:              <CUDA unavailable>")
    return "\n".join(lines) + "\n"


__all__ = [
    "Timings",
    "format_timing_report",
    "format_memory_report",
    "rss_bytes",
    "cuda_peak_bytes",
]
