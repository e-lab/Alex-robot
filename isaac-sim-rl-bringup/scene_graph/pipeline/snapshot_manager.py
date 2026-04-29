"""Periodic snapshot writer for the online scene-graph driver.

Manifest contract (see Plan/online_graph_plan.md §5):

    <output_dir>/
      manifest.json                # consumer reads this only
      snapshots/
        snapshot_0000050.json      # full graph at frame 50
        snapshot_0000050_objects.npz
        snapshot_0000100.json
        ...
      final/
        final.json                 # written by finalise(...)
        ...

Atomicity:
  - Per-snapshot files are immutable, frame-stamped. Concurrent readers
    of `snapshot_NNN.json` are safe — that path is written-once.
  - `manifest.json` is the only file that gets rewritten in place.
    Updates use tmp + fsync + os.replace so a concurrent reader never
    observes a torn manifest.
  - Snapshot pairs (JSON + sidecar npz) are written via
    `scene_graph.graph.serialize.save`, which writes the sidecar first,
    then atomically replaces the JSON. Order: sidecar → snapshot JSON
    → manifest. A reader who sees the manifest pointing at frame N can
    safely open the sidecar named after frame N.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Optional

from ..graph.scene_graph import SceneGraph
from ..graph import serialize


MANIFEST_SCHEMA_VERSION = 1
SNAPSHOTS_SUBDIR = "snapshots"
FINAL_SUBDIR = "final"


@dataclass
class SnapshotConfig:
    """Trigger thresholds — snapshot when EITHER condition is met."""
    every_n_frames: int = 50
    every_n_seconds: float = 5.0


class SnapshotManager:
    """Drives snapshot cadence + manifest updates for one online run.

    Typical use::

        sm = SnapshotManager(out_dir="/tmp/run", scene="replica_room0")
        sm.start()                                  # writes initial manifest
        for frame_idx, ... in stream:
            process_one_frame(sg, ...)
            sm.maybe_snapshot(sg, frame_idx)        # cadence-gated
        sm.snapshot(sg, frame_idx, force=True)      # final pre-finalise dump
        sm.finalise(sg, final_artefacts={"final.ply": ply_path, ...})
    """

    def __init__(self, out_dir: str | pathlib.Path,
                 scene: str,
                 config: Optional[SnapshotConfig] = None):
        self.out_dir = pathlib.Path(out_dir)
        self.snapshots_dir = self.out_dir / SNAPSHOTS_SUBDIR
        self.final_dir = self.out_dir / FINAL_SUBDIR
        self.manifest_path = self.out_dir / "manifest.json"
        self.scene = scene
        self.cfg = config or SnapshotConfig()

        self._t_started: Optional[float] = None
        self._last_snapshot_t: float = 0.0
        self._last_snapshot_frame: int = -1
        self._latest_snapshot_rel: Optional[str] = None
        self._latest_snapshot_at: Optional[str] = None
        self._latest_snapshot_frame: Optional[int] = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        """Create directories + initial manifest with state='running'."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self._t_started = time.time()
        self._write_manifest(state="running",
                             frames_processed=0,
                             final_rel=None)

    # ── snapshots ───────────────────────────────────────────────────────────
    def maybe_snapshot(self, sg: SceneGraph, frame_idx: int) -> bool:
        """Cadence-gated snapshot. Returns True iff a snapshot was taken."""
        if self._t_started is None:
            raise RuntimeError("SnapshotManager.start() must be called first")
        now = time.time()
        frames_since = frame_idx - self._last_snapshot_frame
        seconds_since = now - self._last_snapshot_t
        if (frames_since >= self.cfg.every_n_frames
                or seconds_since >= self.cfg.every_n_seconds):
            self.snapshot(sg, frame_idx, force=True)
            return True
        return False

    def snapshot(self, sg: SceneGraph, frame_idx: int,
                 force: bool = False) -> pathlib.Path:
        """Write a snapshot at this frame. Returns the snapshot JSON path."""
        if self._t_started is None:
            raise RuntimeError("SnapshotManager.start() must be called first")
        stem = f"snapshot_{frame_idx:07d}"
        snap_path = self.snapshots_dir / f"{stem}.json"
        # serialize.save() handles atomicity and the per-object-points
        # sidecar (writes <stem>_objects.npz alongside the JSON).
        serialize.save(sg, str(snap_path))
        rel = snap_path.relative_to(self.out_dir).as_posix()
        self._latest_snapshot_rel = rel
        self._latest_snapshot_frame = frame_idx
        self._latest_snapshot_at = _utc_iso(time.time())
        self._last_snapshot_t = time.time()
        self._last_snapshot_frame = frame_idx
        self._write_manifest(state="running",
                             frames_processed=frame_idx + 1,
                             final_rel=None)
        return snap_path

    # ── finalisation ────────────────────────────────────────────────────────
    def mark_finalising(self, frames_processed: Optional[int] = None) -> None:
        """Flip exploration_state to 'finalising' so consumers know the
        latest snapshot is the last `running` one until finalise() lands.

        If `frames_processed` is None, use the most recent snapshot's
        frame index (or 0 if no snapshot yet). Callers that don't track
        the loop's frame counter independently can pass None.
        """
        if frames_processed is None:
            frames_processed = (
                self._last_snapshot_frame + 1
                if self._last_snapshot_frame >= 0 else 0
            )
        self._write_manifest(state="finalising",
                             frames_processed=frames_processed,
                             final_rel=None)

    def finalise(self, sg: SceneGraph,
                 frames_processed: Optional[int] = None) -> pathlib.Path:
        """Write the offline-finalised graph + flip state to 'finalised'.

        Caller is responsible for having already populated walls / rooms /
        places / occupancy on `sg` and dropped any extra artefacts (PLY,
        spatiallm.txt, occupancy.npz) into `self.final_dir` before calling.
        """
        if frames_processed is None:
            frames_processed = (
                self._last_snapshot_frame + 1
                if self._last_snapshot_frame >= 0 else 0
            )
        final_json = self.final_dir / "final.json"
        serialize.save(sg, str(final_json))
        final_rel = final_json.relative_to(self.out_dir).as_posix()
        self._write_manifest(state="finalised",
                             frames_processed=frames_processed,
                             final_rel=final_rel)
        return final_json

    def mark_stopped(self, frames_processed: Optional[int] = None) -> None:
        """Run ended without finalisation (crash/abort/user halt)."""
        if frames_processed is None:
            frames_processed = (
                self._last_snapshot_frame + 1
                if self._last_snapshot_frame >= 0 else 0
            )
        self._write_manifest(state="stopped",
                             frames_processed=frames_processed,
                             final_rel=None)

    # ── manifest write ──────────────────────────────────────────────────────
    def _write_manifest(self, state: str, frames_processed: int,
                        final_rel: Optional[str]) -> None:
        assert self._t_started is not None
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "scene": self.scene,
            "started_at": _utc_iso(self._t_started),
            "latest_snapshot": self._latest_snapshot_rel,
            "latest_snapshot_frame": self._latest_snapshot_frame,
            "latest_snapshot_at": self._latest_snapshot_at,
            "frames_processed": int(frames_processed),
            "exploration_state": state,
            "final": final_rel,
        }
        text = json.dumps(payload, indent=2)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(self.manifest_path))


def _utc_iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


__all__ = ["SnapshotManager", "SnapshotConfig", "MANIFEST_SCHEMA_VERSION"]
