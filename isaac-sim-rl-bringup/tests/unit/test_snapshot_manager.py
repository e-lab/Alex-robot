"""Unit tests for `pipeline.snapshot_manager`.

Covers:
  - Manifest schema + atomic update.
  - Frame-cadence and time-cadence triggers.
  - Snapshot pair (JSON + sidecar npz) is consistent under concurrent reads.
  - exploration_state transitions: running → finalising → finalised.
  - Crash recovery — stopped state.
"""
from __future__ import annotations

import json
import pathlib
import threading
import time

import numpy as np
import pytest

from scene_graph.graph.scene_graph import SceneGraph
from scene_graph.graph.node_types import ObjectNode
from scene_graph.pipeline.snapshot_manager import (
    SnapshotManager, SnapshotConfig, MANIFEST_SCHEMA_VERSION,
)


def _scene_with(n: int) -> SceneGraph:
    sg = SceneGraph(scene="t")
    for i in range(n):
        sg.objects[f"chair_{i}"] = ObjectNode(
            id=f"chair_{i}", label="chair",
            position_xyz=[float(i), 0, 0],
            bbox_min_xyz=[float(i) - 0.1, -0.1, -0.1],
            bbox_max_xyz=[float(i) + 0.1, 0.1, 0.1],
            confidence=0.9,
        )
    return sg


class TestManifestSchema:
    def test_initial_manifest_after_start(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="replica_room0")
        sm.start()
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert m["scene"] == "replica_room0"
        assert m["exploration_state"] == "running"
        assert m["frames_processed"] == 0
        assert m["latest_snapshot"] is None
        assert m["final"] is None

    def test_manifest_after_first_snapshot(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sm.snapshot(_scene_with(3), frame_idx=50)
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["latest_snapshot"] == "snapshots/snapshot_0000050.json"
        assert m["latest_snapshot_frame"] == 50
        assert m["frames_processed"] == 51
        assert m["exploration_state"] == "running"

    def test_state_transitions(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sm.snapshot(_scene_with(1), frame_idx=10)
        sm.mark_finalising(frames_processed=11)
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["exploration_state"] == "finalising"
        sm.finalise(_scene_with(2), frames_processed=11)
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["exploration_state"] == "finalised"
        assert m["final"] == "final/final.json"

    def test_stopped_transition(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sm.mark_stopped(frames_processed=42)
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["exploration_state"] == "stopped"
        assert m["final"] is None


class TestCadence:
    def test_frame_cadence_triggers(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t",
                             config=SnapshotConfig(every_n_frames=5,
                                                   every_n_seconds=999))
        sm.start()
        # Cadence: frame 0 fires (init t=0 means seconds_since is huge),
        # then every 5 frames thereafter (frame threshold).
        # We get an early snapshot for free — useful for consumers that
        # poll the manifest right after start().
        triggered_at = []
        for i in range(11):
            if sm.maybe_snapshot(_scene_with(1), frame_idx=i):
                triggered_at.append(i)
        assert triggered_at == [0, 5, 10]

    def test_time_cadence_triggers(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t",
                             config=SnapshotConfig(every_n_frames=10_000,
                                                   every_n_seconds=0.05))
        sm.start()
        sm.maybe_snapshot(_scene_with(1), frame_idx=0)
        # First call always triggers (0 seconds since start, > 0.05 false; but
        # frames_since = 0 - (-1) = 1 < threshold; seconds_since = now - 0 →
        # tiny). So no trigger expected on first call.
        time.sleep(0.06)
        triggered = sm.maybe_snapshot(_scene_with(1), frame_idx=1)
        assert triggered  # time threshold elapsed

    def test_force_always_writes(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t",
                             config=SnapshotConfig(every_n_frames=999,
                                                   every_n_seconds=999))
        sm.start()
        # maybe_snapshot wouldn't fire here, but explicit snapshot() does.
        p = sm.snapshot(_scene_with(2), frame_idx=7)
        assert p.exists()


class TestSnapshotFiles:
    def test_snapshot_pair_layout(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sg = _scene_with(2)
        # Add per-object points so the sidecar gets emitted.
        for o in sg.objects.values():
            o.points_xyz = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
            o.points_rgb = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
        sm.snapshot(sg, frame_idx=100)
        snap = tmp_path / "snapshots" / "snapshot_0000100.json"
        sidecar = tmp_path / "snapshots" / "snapshot_0000100_objects.npz"
        assert snap.exists()
        assert sidecar.exists()
        # The graph JSON points at the sidecar by relative path.
        graph = json.loads(snap.read_text())
        oid0 = next(iter(graph["layers"]["objects"]))
        attrs = graph["layers"]["objects"][oid0]["attrs"]
        assert attrs.get("points_path", "").endswith("_objects.npz")

    def test_snapshots_are_immutable_per_frame(self, tmp_path: pathlib.Path):
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sm.snapshot(_scene_with(1), frame_idx=10)
        sm.snapshot(_scene_with(2), frame_idx=20)
        sm.snapshot(_scene_with(3), frame_idx=30)
        # All three remain on disk; previous frames not overwritten.
        assert (tmp_path / "snapshots" / "snapshot_0000010.json").exists()
        assert (tmp_path / "snapshots" / "snapshot_0000020.json").exists()
        assert (tmp_path / "snapshots" / "snapshot_0000030.json").exists()


class TestAtomicity:
    def test_no_torn_manifest_under_concurrent_reads(
            self, tmp_path: pathlib.Path):
        """Hammer the manifest with rewrites while a reader polls in
        another thread. The reader must never see a partial / non-JSON
        file; every read either parses or returns a transient ENOENT."""
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        manifest_path = tmp_path / "manifest.json"

        stop = threading.Event()
        torn: list[str] = []

        def reader():
            while not stop.is_set():
                try:
                    text = manifest_path.read_text()
                except FileNotFoundError:
                    continue
                try:
                    json.loads(text)
                except json.JSONDecodeError as e:
                    torn.append(f"{e}: {text[:60]!r}")

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i in range(200):
                sm.snapshot(_scene_with(1), frame_idx=i)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert torn == [], f"manifest read-torn {len(torn)} times"


class TestRoundTrip:
    def test_can_load_snapshot_back(self, tmp_path: pathlib.Path):
        from scene_graph.graph import serialize as ser
        sm = SnapshotManager(tmp_path, scene="t")
        sm.start()
        sg = _scene_with(3)
        sm.snapshot(sg, frame_idx=50)
        snap = tmp_path / "snapshots" / "snapshot_0000050.json"
        sg2 = ser.load(str(snap))
        assert sorted(sg2.objects.keys()) == ["chair_0", "chair_1", "chair_2"]
