"""Layer 2 — Object layer builder.

Thin wrapper that takes a list of per-frame RawDetections + camera pose +
intrinsics + depth, unprojects each mask to 3D, computes an AABB, and asks
`association/merge.py::insert_or_merge` to add/merge into the scene graph.

Phase 0 subtasks P0.5 – P0.9 will add embedding + IoU-based merge rules on
top of the current simple centroid merge.
"""

from __future__ import annotations
from typing import List

import numpy as np

from ..detection.sam3_detector import RawDetection
from ..geometry.unprojection import pixel_grid_to_world
from ..geometry.bbox import points_to_aabb, aabb_center
from ..graph.scene_graph import SceneGraph
from ..association.merge import insert_or_merge


def ingest_detections(sg: SceneGraph,
                      dets: List[RawDetection],
                      depth: np.ndarray,
                      K: np.ndarray,
                      cam_pos: np.ndarray,
                      cam_quat_wxyz: np.ndarray,
                      tick: int,
                      downsample_stride: int = 4,
                      max_depth: float = 10.0,
                      rgb: np.ndarray = None) -> List[str]:
    """Project each detection's mask pixels to world, build AABB, merge.

    When `rgb` is supplied, every detection also competes to become the
    parent object's "best view" crop — whichever detection has the
    highest SAM 3 score across all frames gets its cropped RGB stored
    on the ObjectNode for later sidecar export (Phase 5.5).

    Returns the list of resulting object IDs (one per detection).
    """
    if depth.ndim == 3:
        depth = depth[..., 0]

    out_ids: List[str] = []
    for d in dets:
        # Stride-sample depth-pixels inside the mask only
        ys, xs = np.where(d.mask)
        if len(xs) == 0:
            continue
        # Subsample for speed — full mask can be 20k+ pixels.
        if downsample_stride > 1:
            keep = np.arange(0, len(xs), downsample_stride)
            xs, ys = xs[keep], ys[keep]
        # Collect valid depths
        z = depth[ys, xs]
        valid = np.isfinite(z) & (z > 0.05) & (z < max_depth)
        if not valid.any():
            continue
        xs, ys, z = xs[valid], ys[valid], z[valid]

        # Unproject
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (xs - cx) * z / fx
        y_cam = -(ys - cy) * z / fy
        z_cam = -z
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        from ..geometry.unprojection import quat_wxyz_to_rot
        R = quat_wxyz_to_rot(cam_quat_wxyz)
        pts_world = pts_cam @ R.T + cam_pos.astype(np.float32)

        # Sample RGB at the same masked pixels so each 3-D point keeps a
        # colour for Option-B viz. Missing rgb → write uint8 zeros; later
        # consumers treat them as neutral grey.
        if rgb is not None and rgb.ndim == 3 and rgb.shape[2] >= 3:
            det_rgb = rgb[ys, xs, :3].astype(np.uint8)
        else:
            det_rgb = np.full((xs.shape[0], 3), 128, dtype=np.uint8)

        mn, mx = points_to_aabb(pts_world)
        ctr = aabb_center(mn, mx)

        obj_id = insert_or_merge(
            sg,
            det_label=d.label,
            det_xyz=ctr,
            det_bbox_min=mn,
            det_bbox_max=mx,
            det_score=d.score,
            det_embedding=d.embedding,
            det_track_id=None,
            tick=tick,
            det_points_xyz=pts_world.astype(np.float32),
            det_points_rgb=det_rgb,
        )
        out_ids.append(obj_id)

        # Phase 5.5 — best-view snapshot. A detection becomes the stored
        # crop iff its SAM 3 score beats whatever's already saved.
        if rgb is not None and obj_id is not None:
            _update_best_view(sg, obj_id, rgb, d, tick)

    return out_ids


def _update_best_view(sg: SceneGraph, obj_id: str,
                      rgb: np.ndarray, d: RawDetection,
                      tick: int, margin_px: int = 8) -> None:
    """Crop RGB around the detection's 2D bbox (plus a small margin) and
    store it on the object if this detection has the highest score so far.
    """
    # Lookup works for both promoted objects and pending candidates.
    obj = sg.objects.get(obj_id) or sg.pending.get(obj_id)
    if obj is None:
        return
    prev_score = obj.attrs.get("best_view_score", -1.0)
    if float(d.score) <= float(prev_score):
        return

    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = d.bbox_xyxy.astype(np.int32).tolist()
    x0 = max(0, x0 - margin_px); y0 = max(0, y0 - margin_px)
    x1 = min(W, x1 + margin_px); y1 = min(H, y1 + margin_px)
    if x1 <= x0 or y1 <= y0:
        return

    crop = rgb[y0:y1, x0:x1].copy()      # decouple from frame buffer
    obj.best_view_bgr = crop
    obj.attrs["best_view_score"] = float(d.score)
    obj.attrs["best_view_tick"] = int(tick)


__all__ = ["ingest_detections"]
