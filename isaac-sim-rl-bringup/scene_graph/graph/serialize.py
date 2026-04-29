"""JSON (de)serialisation of a SceneGraph.

Kept as free functions — `SceneGraph` and node dataclasses stay pure data.
The JSON shape matches the schema in `docs/scene_graph_spec.md`.

Top-level:

    {
      "meta": {
        "scene": str,              # e.g. "hallway"
        "scan_complete": bool,
        "pose_frames": int,
        "vocabulary": list[str],
        "timestamp": str,          # ISO-8601 UTC, set at save time
        "schema_version": int,     # bumped on breaking changes
      },
      "layers": {
        "mesh":     {...} | null,                        # Layer 1 (Phase 6)
        "objects":  {"<label>_<N>": ObjectNode, ...},    # Layer 2
        "places":   {"<id>":       PlaceNode, ...},      # Layer 3
        "rooms":    {"<id>":       RoomNode,  ...},      # Layer 4
        "building": BuildingNode | null,                 # Layer 5
      },
      "robot_path": list[[x, y, yaw_deg], ...],
    }

Loader accepts older flat-layout files ({"objects": {...}, "robot_path":
[...]}) for one-way migration.
"""

from __future__ import annotations
import json
import pathlib
import time
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

import numpy as np

from .scene_graph import SceneGraph
from .node_types import (
    MeshLayer, ObjectNode, WallNode, PlaceNode, RoomNode, BuildingNode,
)

SCHEMA_VERSION = 4   # bumped: objects carry attrs["points_path"] (Phase 5.6)


# ── Dataclass ↔ dict helpers ────────────────────────────────────────────────
# Fields that hold raw image or bulk-array data and must never appear in the
# JSON — they persist as sidecar files instead (see _save_best_view_sidecars
# and _save_object_point_sidecars).
_FIELDS_EXCLUDED_FROM_JSON = {"best_view_bgr", "points_xyz", "points_rgb"}


def _node_to_dict(node: Any) -> Dict[str, Any]:
    d = asdict(node)
    # numpy → list (concept_embedding mostly)
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            d[k] = v.astype(np.float32).tolist()
    # Drop image buffers — they ride as sidecar PNGs.
    for k in _FIELDS_EXCLUDED_FROM_JSON:
        d.pop(k, None)
    return d


def _dict_to_node(cls, d: Dict[str, Any]):
    """Construct `cls(**d)` defensively — ignore any unknown keys that a
    newer saver might have written out (forward-compat)."""
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in known}
    return cls(**kwargs)


# ── Public: save / to_dict / load ───────────────────────────────────────────
def to_dict(sg: SceneGraph) -> Dict[str, Any]:
    return {
        "meta": {
            "scene": sg.scene,
            "scan_complete": sg.scan_complete,
            "pose_frames": sg.pose_frames,
            "vocabulary": sg.vocabulary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": SCHEMA_VERSION,
        },
        "layers": {
            "mesh":     _node_to_dict(sg.mesh) if sg.mesh is not None else None,
            "objects":  {oid: _node_to_dict(o) for oid, o in sg.objects.items()},
            "walls":    {wid: _node_to_dict(w) for wid, w in sg.walls.items()},
            "places":   {pid: _node_to_dict(p) for pid, p in sg.places.items()},
            "rooms":    {rid: _node_to_dict(r) for rid, r in sg.rooms.items()},
            "building": _node_to_dict(sg.building) if sg.building else None,
        },
        "robot_path": sg.robot_path,
    }


def save(sg: SceneGraph, path: str) -> None:
    """Persist the scene graph to disk.

    Order:
      1. Best-view PNG sidecars (Phase 5.5) into <stem>_objects/.
      2. Per-object point sidecar (Phase 5.6) → <stem>_objects.npz.
      3. Top-level JSON, written atomically (tmp + fsync + os.replace).

    Sidecars are written before the JSON so any reference paths in the
    JSON point at existing files even if the process dies between writes.
    The JSON write is atomic so a concurrent reader (e.g. a planner
    polling for the latest snapshot in the online driver) never sees a
    half-written graph.
    """
    import os
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _save_best_view_sidecars(sg, p)
    _save_object_point_sidecar(sg, p)
    payload = json.dumps(to_dict(sg), indent=2)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def _save_best_view_sidecars(sg: SceneGraph, json_path: pathlib.Path) -> int:
    """Write each object's stored RGB crop to <stem>_objects/<oid>.png and
    record its relative path on `obj.attrs['best_view_path']`. Skips
    objects that never captured a crop. Returns count of files written.
    """
    out_dir = json_path.with_name(json_path.stem + "_objects")
    n_written = 0
    any_has_view = any(
        getattr(o, "best_view_bgr", None) is not None
        for o in sg.objects.values()
    )
    if not any_has_view:
        return 0

    from PIL import Image as _PIL
    out_dir.mkdir(parents=True, exist_ok=True)
    for oid, obj in sg.objects.items():
        crop = getattr(obj, "best_view_bgr", None)
        if crop is None:
            continue
        # Sanitise the id for a filesystem-safe filename.
        safe_id = oid.replace("/", "_").replace(" ", "_")
        rel_path = f"{out_dir.name}/{safe_id}.png"
        abs_path = out_dir / f"{safe_id}.png"
        _PIL.fromarray(crop.astype(np.uint8)).save(str(abs_path))
        obj.attrs["best_view_path"] = rel_path
        n_written += 1
    return n_written


def _save_object_point_sidecar(sg: SceneGraph,
                               json_path: pathlib.Path) -> int:
    """Dump per-object segmented point clouds to one `<stem>_objects.npz`.

    Keys in the npz:
        "<object_id>__xyz"  → (N, 3) float32
        "<object_id>__rgb"  → (N, 3) uint8

    Each object with a non-empty `points_xyz` also gets
    `attrs["points_path"] = "<stem>_objects.npz"` so downstream consumers
    know where to find it. Returns the number of objects written.
    """
    to_write: Dict[str, np.ndarray] = {}
    written_ids = []
    for oid, obj in sg.objects.items():
        pts = getattr(obj, "points_xyz", None)
        if pts is None or pts.size == 0:
            continue
        rgb = getattr(obj, "points_rgb", None)
        if rgb is None or rgb.size == 0:
            rgb = np.full((pts.shape[0], 3), 128, dtype=np.uint8)
        safe_id = oid.replace("/", "_").replace(" ", "_")
        to_write[f"{safe_id}__xyz"] = pts.astype(np.float32, copy=False)
        to_write[f"{safe_id}__rgb"] = rgb.astype(np.uint8, copy=False)
        written_ids.append((oid, safe_id))

    if not to_write:
        return 0

    sidecar = json_path.with_name(json_path.stem + "_objects.npz")
    np.savez_compressed(str(sidecar), **to_write)
    rel = sidecar.name
    for oid, safe_id in written_ids:
        sg.objects[oid].attrs["points_path"] = rel
        sg.objects[oid].attrs["points_npz_keys"] = [
            f"{safe_id}__xyz", f"{safe_id}__rgb",
        ]
    return len(written_ids)


def get_object_points(sg: SceneGraph, object_id: str,
                      json_path: str) -> Optional[tuple]:
    """Load per-object (xyz, rgb) arrays from the sidecar npz.

    Returns (points[N,3] float32, rgb[N,3] uint8) or None if the object
    has no stored cloud (old fixture, or empty detection).
    """
    obj = sg.objects.get(object_id)
    if obj is None:
        return None
    # Prefer in-memory if still around (right after running the pipeline).
    if getattr(obj, "points_xyz", None) is not None:
        pts = obj.points_xyz
        rgb = obj.points_rgb if obj.points_rgb is not None \
            else np.full((pts.shape[0], 3), 128, dtype=np.uint8)
        return pts, rgb
    rel = obj.attrs.get("points_path")
    if not rel:
        return None
    abs_path = pathlib.Path(json_path).parent / rel
    if not abs_path.exists():
        return None
    keys = obj.attrs.get("points_npz_keys") or []
    if len(keys) != 2:
        return None
    with np.load(str(abs_path)) as z:
        if keys[0] not in z.files or keys[1] not in z.files:
            return None
        return z[keys[0]].astype(np.float32), z[keys[1]].astype(np.uint8)


def get_best_view(sg: SceneGraph, object_id: str,
                  json_path: str) -> Optional[np.ndarray]:
    """Load the best-view crop for an object as an (H, W, 3) uint8 array.
    Returns None if the object has no stored view or the sidecar is
    missing. `json_path` anchors the `attrs['best_view_path']` relative
    resolution.
    """
    obj = sg.objects.get(object_id)
    if obj is None:
        return None
    # Prefer the in-memory copy (useful when we just ran the pipeline).
    if getattr(obj, "best_view_bgr", None) is not None:
        return obj.best_view_bgr
    rel = obj.attrs.get("best_view_path")
    if not rel:
        return None
    from PIL import Image as _PIL
    abs_path = pathlib.Path(json_path).parent / rel
    if not abs_path.exists():
        return None
    img = _PIL.open(str(abs_path)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load(path: str) -> SceneGraph:
    """Load a SceneGraph JSON. Accepts both the current layered layout and
    the pre-Phase-0 flat layout ({"objects": {...}, "robot_path": [...]})."""
    raw = json.loads(pathlib.Path(path).read_text())

    # Detect legacy flat layout (no `meta` / no `layers`).
    is_legacy = "meta" not in raw and "layers" not in raw
    if is_legacy:
        raw = _migrate_legacy(raw)

    meta = raw.get("meta", {})
    sg = SceneGraph(
        scene=meta.get("scene", ""),
        scan_complete=meta.get("scan_complete", False),
        pose_frames=meta.get("pose_frames", 0),
        vocabulary=meta.get("vocabulary", []),
        robot_path=raw.get("robot_path", []),
    )
    layers = raw.get("layers", {})

    m = layers.get("mesh")
    if m is not None:
        sg.mesh = _dict_to_node(MeshLayer, m)

    for oid, od in layers.get("objects", {}).items():
        if od.get("concept_embedding") is not None:
            od["concept_embedding"] = np.asarray(
                od["concept_embedding"], dtype=np.float32)
        sg.objects[oid] = _dict_to_node(ObjectNode, od)

    for wid, wd in layers.get("walls", {}).items():
        sg.walls[wid] = _dict_to_node(WallNode, wd)

    for pid, pd in layers.get("places", {}).items():
        sg.places[pid] = _dict_to_node(PlaceNode, pd)

    for rid, rd in layers.get("rooms", {}).items():
        sg.rooms[rid] = _dict_to_node(RoomNode, rd)

    b = layers.get("building")
    if b is not None:
        sg.building = _dict_to_node(BuildingNode, b)

    return sg


# ── Legacy migration ────────────────────────────────────────────────────────
def _migrate_legacy(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the pre-Phase-0 flat JSON layout to the current schema.

    Old:  {"room_id": str, "objects": {...}, "robot_path": [...], ...}
    New:  {"meta": {...}, "layers": {"objects": {...}, ...}, "robot_path": ...}
    """
    objs = {}
    for oid, o in raw.get("objects", {}).items():
        # legacy dict rarely included `id` — inject from the dict key.
        o.setdefault("id", oid)
        # legacy dict had `_n_obs` not `n_observations`
        if "_n_obs" in o and "n_observations" not in o:
            o["n_observations"] = o.pop("_n_obs")
        # legacy had `bbox_area_px` per-detection; drop (no analogue)
        o.pop("bbox_area_px", None)
        # ensure required 3D fields; if bbox missing, default ±0.1 m
        if "bbox_min_xyz" not in o or "bbox_max_xyz" not in o:
            c = np.asarray(o.get("position_xyz", [0, 0, 0]), dtype=np.float32)
            o["bbox_min_xyz"] = (c - 0.1).tolist()
            o["bbox_max_xyz"] = (c + 0.1).tolist()
        objs[oid] = o

    return {
        "meta": {
            "scene": raw.get("room_id", ""),
            "scan_complete": raw.get("scan_complete", False),
            "pose_frames": 0,
            "vocabulary": [],
            "schema_version": 0,   # migrated from legacy
        },
        "layers": {
            "mesh": None,
            "objects": objs,
            "places": {},
            "rooms": {},
            "building": None,
        },
        "robot_path": raw.get("robot_path", []),
    }


__all__ = ["SCHEMA_VERSION", "to_dict", "save", "load",
           "get_best_view", "get_object_points"]
