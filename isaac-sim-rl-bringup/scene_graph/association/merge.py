"""Merge a fresh detection into an existing ObjectNode, or create a new one.

Phase-0 implementation lives here — see Plan/scene_graph_plan.md §5 Phase 0
for the rules. This module purposely owns *all* mutation of ObjectNode and
SceneGraph's `objects` dict, mirroring ConceptGraphs' `slam/utils.py`
containing every `merge_obj2_into_obj1` call-site.

Phase-0 status: only the naive distance-based merge is wired up right now,
matching the current behaviour. The IoU / cosine / cross-label rules land
in the Phase 0 subtasks (P0.3 – P0.9).
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from ..graph.node_types import ObjectNode
from ..graph.scene_graph import SceneGraph
from . import dedup_rules


def _running_average(old: np.ndarray, new: np.ndarray, n: int) -> np.ndarray:
    """Welford-style running mean of a 3-vector."""
    return (old * n + new) / (n + 1)


def _merge_would_oversize(obj_mn: np.ndarray, obj_mx: np.ndarray,
                          det_mn: np.ndarray, det_mx: np.ndarray,
                          label: str) -> bool:
    """True if unioning the detection AABB into the object would push the
    *horizontal footprint* past `dedup_rules.max_extent(label)`. We check
    max(extent_x, extent_y) rather than all three axes because vertical
    extent (object height) is naturally larger for tall-thin objects like
    doors and wouldn't be a runaway-growth signal."""
    new_mn = np.minimum(obj_mn, det_mn)
    new_mx = np.maximum(obj_mx, det_mx)
    ext = new_mx - new_mn
    horizontal = float(max(ext[0], ext[1]))
    vertical   = float(ext[2])
    cap = dedup_rules.max_extent(label)
    # Horizontal dominates the sanity check; vertical only fails if it's
    # more than twice the cap (catches absurd cases like "a 10 m tall door").
    if horizontal > cap:
        return True
    if vertical > cap * 2.0:
        return True
    return False


def _containment_iou(a_min, a_max, b_min, b_max) -> float:
    """Intersection-over-smaller-volume. A huge bbox absorbing a tiny one
    still only scores 1.0 when the tiny one is fully inside it — but
    standard IoU would score low because the union is dominated by the
    huge bbox. This ratio is more honest for "is this a re-observation"."""
    from ..geometry.bbox import aabb_volume
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = aabb_volume(inter_min, inter_max)
    if inter <= 0.0:
        return 0.0
    vol_a = aabb_volume(a_min, a_max)
    vol_b = aabb_volume(b_min, b_max)
    smaller = min(vol_a, vol_b)
    return inter / max(smaller, 1e-9)


def merge_detection_into_object(det_label: str,
                                det_xyz: np.ndarray,
                                det_bbox_min: np.ndarray,
                                det_bbox_max: np.ndarray,
                                det_score: float,
                                det_embedding: Optional[np.ndarray],
                                det_track_id: Optional[str],
                                obj: ObjectNode,
                                tick: int,
                                det_points_xyz: Optional[np.ndarray] = None,
                                det_points_rgb: Optional[np.ndarray] = None,
                                point_voxel_size: float = 0.02,
                                point_max: int = 5000) -> None:
    """Fuse a fresh detection into an existing object (in place).

    When `det_points_xyz`/`det_points_rgb` are provided, they are merged
    into `obj.points_xyz`/`obj.points_rgb` using voxel-downsample + cap
    (see `geometry.pointcloud.merge_and_cap`). Bounded memory per object.
    """
    n = max(obj.n_observations, 1)
    old_pos = np.asarray(obj.position_xyz)
    obj.position_xyz = _running_average(old_pos, det_xyz, n).tolist()

    old_min = np.asarray(obj.bbox_min_xyz)
    old_max = np.asarray(obj.bbox_max_xyz)
    obj.bbox_min_xyz = np.minimum(old_min, det_bbox_min).astype(np.float32).tolist()
    obj.bbox_max_xyz = np.maximum(old_max, det_bbox_max).astype(np.float32).tolist()

    obj.confidence = max(obj.confidence, det_score)
    obj.last_seen_tick = tick
    obj.n_observations = n + 1

    # Keep the highest-score embedding rather than averaging — simpler and
    # preserves discriminability; revisit in Phase 0 if dedup suffers.
    if det_embedding is not None and (obj.concept_embedding is None or det_score >= obj.confidence):
        obj.concept_embedding = det_embedding

    if det_track_id and det_track_id not in obj.source_tracks:
        obj.source_tracks.append(det_track_id)

    if det_points_xyz is not None and det_points_xyz.size:
        from ..geometry.pointcloud import merge_and_cap
        obj.points_xyz, obj.points_rgb = merge_and_cap(
            obj.points_xyz, obj.points_rgb,
            det_points_xyz, det_points_rgb,
            voxel_size=point_voxel_size,
            max_points=point_max,
        )


def _find_nearby_in_bucket(bucket, det_label: str, det_xyz: np.ndarray,
                           radius: float,
                           det_embedding: Optional[np.ndarray] = None,
                           det_bbox_min: Optional[np.ndarray] = None,
                           det_bbox_max: Optional[np.ndarray] = None,
                           stretch: float = 2.0,
                           enforce_size_cap: bool = True) -> Optional[str]:
    """Return the id of the closest same-label entry in `bucket` within radius.

    Three-tier rule (P0.5 + P0.7):
      1. Strong geometric: distance < radius                            → merge
      2. Bbox IoU:         IoU(det, obj) ≥ BBOX_IOU_THRESHOLD            → merge (P0.7)
      3. Weak geometric + visual:
           distance < radius * stretch AND cosine ≥ threshold            → merge
    This fixes Pattern B at two levels:
      * Real AABB overlap catches doors seen from both sides with drifting
        centroids but overlapping physical extent.
      * Embedding-boosted stretch catches the same case when the bboxes
        are imperfect (one mask much smaller than the other).
    """
    from ..detection.embeddings import cosine as _cos
    from ..geometry.bbox import aabb_iou

    best_id: Optional[str] = None
    best_dist = float("inf")
    has_det_bbox = det_bbox_min is not None and det_bbox_max is not None

    for obj_id, obj in bucket.items():
        if obj.label != det_label:
            continue
        dist = float(np.linalg.norm(det_xyz - np.asarray(obj.position_xyz)))
        obj_mn = np.asarray(obj.bbox_min_xyz, dtype=np.float32)
        obj_mx = np.asarray(obj.bbox_max_xyz, dtype=np.float32)

        matched = False
        # 1. Strong geometric
        if dist < radius:
            matched = True
        # 2. Bbox match (only when the detection carries a real AABB, not
        #    the shim's ±0.1 m placeholder). We require:
        #      - Containment-IoU ≥ threshold  (smaller mostly inside larger),
        #        which is a fairer signal than raw IoU when the existing
        #        bbox is large.
        if not matched and has_det_bbox:
            det_mn = det_bbox_min.astype(np.float32)
            det_mx = det_bbox_max.astype(np.float32)
            cov = _containment_iou(det_mn, det_mx, obj_mn, obj_mx)
            if cov >= dedup_rules.BBOX_IOU_THRESHOLD:
                matched = True
        # 3. Weak geometric + visual
        if not matched and dist < radius * stretch:
            if (det_embedding is not None and obj.concept_embedding is not None
                    and _cos(det_embedding, obj.concept_embedding)
                        >= dedup_rules.EMBEDDING_COSINE_THRESHOLD):
                matched = True

        # Size-cap guard — reject any would-be merge that inflates the
        # object past its per-label max extent. This is what stops a
        # single 'door' node from ballooning to 5 m across the hallway
        # by absorbing unrelated doors. Disabled for `pending` matching
        # so tentative candidates can fuse freely; the size cap is still
        # enforced at promotion + by clip_oversized_bboxes post-pass.
        if matched and has_det_bbox and enforce_size_cap:
            if _merge_would_oversize(
                obj_mn, obj_mx,
                det_bbox_min.astype(np.float32), det_bbox_max.astype(np.float32),
                det_label,
            ):
                matched = False

        if matched and dist < best_dist:
            best_id, best_dist = obj_id, dist

    return best_id


def _find_cross_label_nearest(bucket, det_xyz: np.ndarray,
                              radius: float) -> Optional[str]:
    """Return the id of the nearest *any-label* entry in `bucket` within radius."""
    best_id: Optional[str] = None
    best_dist = float("inf")
    for obj_id, obj in bucket.items():
        dist = float(np.linalg.norm(det_xyz - np.asarray(obj.position_xyz)))
        if dist < radius and dist < best_dist:
            best_id, best_dist = obj_id, dist
    return best_id


def insert_or_merge(sg: SceneGraph,
                    det_label: str,
                    det_xyz: np.ndarray,
                    det_bbox_min: np.ndarray,
                    det_bbox_max: np.ndarray,
                    det_score: float,
                    det_embedding: Optional[np.ndarray],
                    det_track_id: Optional[str],
                    tick: int,
                    det_points_xyz: Optional[np.ndarray] = None,
                    det_points_rgb: Optional[np.ndarray] = None) -> str:
    """Insert a detection into the scene graph. May land in `pending` if it
    hasn't been seen MIN_SIGHTINGS times yet. Returns the final id (which
    is `pending_N` while the candidate is below the threshold, or
    `<label>_N` once promoted).

    Order of ops (P0.1 + P0.2):
      1. Try to merge into an existing PROMOTED object (sg.objects)
      2. Else try to merge into an existing PENDING candidate (sg.pending)
      3. Else create a new pending entry.
    A pending entry that reaches MIN_SIGHTINGS is promoted: its id is
    renamed to `<label>_N` and it moves from sg.pending → sg.objects.
    """
    radius = dedup_rules.merge_radius(det_label)

    # 1. Already promoted?
    promoted_id = _find_nearby_in_bucket(
        sg.objects, det_label, det_xyz, radius,
        det_embedding=det_embedding,
        det_bbox_min=det_bbox_min, det_bbox_max=det_bbox_max,
    )
    if promoted_id is not None:
        merge_detection_into_object(
            det_label, det_xyz, det_bbox_min, det_bbox_max,
            det_score, det_embedding, det_track_id,
            sg.objects[promoted_id], tick,
            det_points_xyz=det_points_xyz,
            det_points_rgb=det_points_rgb,
        )
        return promoted_id

    # 2. Match an existing pending candidate? Size cap is disabled here —
    # tentative candidates may legitimately have oversized bboxes from
    # early noisy unprojections; we want them to fuse so one survives
    # the min-sightings gate instead of five all being dropped.
    pending_id = _find_nearby_in_bucket(
        sg.pending, det_label, det_xyz, radius,
        det_embedding=det_embedding,
        det_bbox_min=det_bbox_min, det_bbox_max=det_bbox_max,
        enforce_size_cap=False,
    )
    if pending_id is not None:
        obj = sg.pending[pending_id]
        merge_detection_into_object(
            det_label, det_xyz, det_bbox_min, det_bbox_max,
            det_score, det_embedding, det_track_id,
            obj, tick,
            det_points_xyz=det_points_xyz,
            det_points_rgb=det_points_rgb,
        )
        # Promotion check: once the candidate has been observed enough,
        # give it a real label_N id and move it out of `pending`.
        if obj.n_observations >= dedup_rules.MIN_SIGHTINGS:
            new_id = sg.new_object_id(det_label)
            obj.id = new_id
            sg.objects[new_id] = obj
            del sg.pending[pending_id]
            return new_id
        return pending_id

    # 3. P0.3 — CROSS-LABEL match: same spot, different label.
    #    If SAM3 returned two prompts for the same physical object, collapse
    #    them here rather than creating duplicate entries. Scan promoted
    #    objects first (more trusted) then pending.
    cross_radius = dedup_rules.CROSS_LABEL_MERGE_RADIUS_M
    x_promoted = _find_cross_label_nearest(sg.objects, det_xyz, cross_radius)
    if x_promoted is not None:
        existing = sg.objects[x_promoted]
        # Relabel decision MUST happen before merge_detection_into_object,
        # which updates confidence to max(old, new) in place.
        old_conf = existing.confidence
        merge_detection_into_object(
            existing.label, det_xyz, det_bbox_min, det_bbox_max,
            det_score, det_embedding, det_track_id,
            existing, tick,
            det_points_xyz=det_points_xyz,
            det_points_rgb=det_points_rgb,
        )
        if det_score > old_conf and det_label != existing.label:
            existing.label = det_label
        return x_promoted

    x_pending = _find_cross_label_nearest(sg.pending, det_xyz, cross_radius)
    if x_pending is not None:
        existing = sg.pending[x_pending]
        old_conf = existing.confidence
        merge_detection_into_object(
            existing.label, det_xyz, det_bbox_min, det_bbox_max,
            det_score, det_embedding, det_track_id,
            existing, tick,
            det_points_xyz=det_points_xyz,
            det_points_rgb=det_points_rgb,
        )
        if det_score > old_conf and det_label != existing.label:
            existing.label = det_label
        # Promotion still gated by MIN_SIGHTINGS.
        if existing.n_observations >= dedup_rules.MIN_SIGHTINGS:
            new_id = sg.new_object_id(existing.label)
            existing.id = new_id
            sg.objects[new_id] = existing
            del sg.pending[x_pending]
            return new_id
        return x_pending

    # 4. Brand-new candidate → pending
    new_id = sg.new_pending_id()
    # Seed points with the detection's unprojected masked pixels, voxel-
    # downsampled so even a 20k-pixel mask becomes a bounded contribution.
    init_pts = None
    init_rgb = None
    if det_points_xyz is not None and det_points_xyz.size:
        from ..geometry.pointcloud import merge_and_cap
        init_pts, init_rgb = merge_and_cap(
            None, None, det_points_xyz, det_points_rgb,
            voxel_size=0.02, max_points=5000,
        )
    sg.pending[new_id] = ObjectNode(
        id=new_id,
        label=det_label,
        position_xyz=det_xyz.astype(np.float32).tolist(),
        bbox_min_xyz=det_bbox_min.astype(np.float32).tolist(),
        bbox_max_xyz=det_bbox_max.astype(np.float32).tolist(),
        confidence=det_score,
        n_observations=1,
        first_seen_tick=tick,
        last_seen_tick=tick,
        concept_embedding=det_embedding,
        source_tracks=[det_track_id] if det_track_id else [],
        points_xyz=init_pts,
        points_rgb=init_rgb,
    )
    # A single-sighting threshold (MIN_SIGHTINGS=1) is unusual but valid
    # — promote immediately so the test suite can exercise that mode.
    if dedup_rules.MIN_SIGHTINGS <= 1:
        promoted = sg.new_object_id(det_label)
        sg.pending[new_id].id = promoted
        sg.objects[promoted] = sg.pending.pop(new_id)
        return promoted
    return new_id


def collapse_cross_label(sg: SceneGraph,
                         radius: Optional[float] = None) -> int:
    """Walk every pair of promoted objects; collapse any two within
    `radius` of each other regardless of label. Returns the number of
    collapses performed.

    Called at save time as a safety net — catches late cross-label
    collisions that the online path missed (e.g. because the two
    detections promoted on the same tick so neither saw the other in
    sg.objects yet).

    Rule: keep the higher-confidence node, absorb the lower one, relabel
    to the higher-confidence label.
    """
    if radius is None:
        radius = dedup_rules.CROSS_LABEL_MERGE_RADIUS_M
    collapsed = 0

    # Repeated sweep: one pass can leave pairs collapsed into a third
    # node; iterate until stable (cheap in practice — N^2 on a small N).
    while True:
        ids = sorted(sg.objects.keys(),
                     key=lambda i: -sg.objects[i].confidence)  # high conf first
        did_collapse = False
        for i, id_a in enumerate(ids):
            if id_a not in sg.objects:
                continue
            a = sg.objects[id_a]
            for id_b in ids[i + 1:]:
                if id_b not in sg.objects:
                    continue
                b = sg.objects[id_b]
                if id_a == id_b:
                    continue
                dist = float(np.linalg.norm(
                    np.asarray(a.position_xyz) - np.asarray(b.position_xyz)))
                if dist >= radius:
                    continue
                # Absorb b into a. a already has higher confidence by sort order.
                n_a = max(a.n_observations, 1)
                n_b = max(b.n_observations, 1)
                old_a = np.asarray(a.position_xyz)
                old_b = np.asarray(b.position_xyz)
                a.position_xyz = (
                    (old_a * n_a + old_b * n_b) / (n_a + n_b)
                ).astype(np.float32).tolist()
                a.bbox_min_xyz = np.minimum(
                    a.bbox_min_xyz, b.bbox_min_xyz).astype(np.float32).tolist()
                a.bbox_max_xyz = np.maximum(
                    a.bbox_max_xyz, b.bbox_max_xyz).astype(np.float32).tolist()
                a.n_observations = n_a + n_b
                a.first_seen_tick = min(a.first_seen_tick, b.first_seen_tick)
                a.last_seen_tick = max(a.last_seen_tick, b.last_seen_tick)
                for t in b.source_tracks:
                    if t not in a.source_tracks:
                        a.source_tracks.append(t)
                # Fuse per-object point clouds if either side has them.
                if a.points_xyz is not None or b.points_xyz is not None:
                    from ..geometry.pointcloud import merge_and_cap
                    a.points_xyz, a.points_rgb = merge_and_cap(
                        a.points_xyz, a.points_rgb,
                        b.points_xyz, b.points_rgb,
                        voxel_size=0.02, max_points=5000,
                    )
                del sg.objects[id_b]
                collapsed += 1
                did_collapse = True
        if not did_collapse:
            break
    return collapsed


def collapse_overlapping_same_label(
    sg: SceneGraph,
    containment_threshold: float = 0.5,
    require_embedding_cosine: Optional[float] = None,
) -> int:
    """Fuse same-label promoted objects whose AABBs overlap significantly.

    Pairwise rule:
      - same label
      - `_containment_iou(a, b) >= containment_threshold`
        i.e. at least `threshold` of the smaller bbox's volume lies inside
        the larger one.
      - (optional) cos(embedding_a, embedding_b) >= `require_embedding_cosine`
        if both have embeddings set.

    Greedy absorption: for each label, sort objects by AABB volume DESC so
    big bboxes absorb smaller ones consistently. Keep the bigger/more-
    observed id; merge the smaller into it (union bbox, weighted centroid,
    sum observations, keep higher-confidence embedding). Parent-room and
    parent-wall back-references pointing at the absorbed id are rewritten
    to the survivor.

    Called between `collapse_cross_label` and `clip_oversized_bboxes`:
    cross-label handles same-xyz different-label, this handles same-label
    different-xyz-but-overlapping-bboxes (the "two cubes, one actually
    inside the other" case). Size-cap is enforced afterwards, so a
    runaway-merged chair gets clipped, not dropped.

    Returns the number of collapses performed.
    """
    from ..geometry.bbox import aabb_volume
    from ..detection.embeddings import cosine as _cos

    collapsed = 0

    # Bucket by label once; we merge within buckets.
    def _bucket_by_label() -> Dict[str, list]:
        out: Dict[str, list] = {}
        for oid, o in sg.objects.items():
            out.setdefault(o.label, []).append(oid)
        return out

    while True:
        did_collapse = False
        buckets = _bucket_by_label()
        for label, ids in buckets.items():
            if len(ids) < 2:
                continue
            # Biggest volume first so the largest AABB keeps its id.
            ids.sort(
                key=lambda oid: -aabb_volume(
                    np.asarray(sg.objects[oid].bbox_min_xyz, dtype=np.float32),
                    np.asarray(sg.objects[oid].bbox_max_xyz, dtype=np.float32),
                ),
            )
            for i, id_a in enumerate(ids):
                if id_a not in sg.objects:
                    continue
                a = sg.objects[id_a]
                a_mn = np.asarray(a.bbox_min_xyz, dtype=np.float32)
                a_mx = np.asarray(a.bbox_max_xyz, dtype=np.float32)
                for id_b in ids[i + 1:]:
                    if id_b not in sg.objects:
                        continue
                    b = sg.objects[id_b]
                    b_mn = np.asarray(b.bbox_min_xyz, dtype=np.float32)
                    b_mx = np.asarray(b.bbox_max_xyz, dtype=np.float32)
                    cov = _containment_iou(a_mn, a_mx, b_mn, b_mx)
                    if cov < containment_threshold:
                        continue
                    # Optional visual guard.
                    if (require_embedding_cosine is not None
                            and a.concept_embedding is not None
                            and b.concept_embedding is not None):
                        if _cos(a.concept_embedding,
                                b.concept_embedding) < require_embedding_cosine:
                            continue
                    # Merge b into a.
                    n_a = max(a.n_observations, 1)
                    n_b = max(b.n_observations, 1)
                    old_a = np.asarray(a.position_xyz)
                    old_b = np.asarray(b.position_xyz)
                    a.position_xyz = (
                        (old_a * n_a + old_b * n_b) / (n_a + n_b)
                    ).astype(np.float32).tolist()
                    a.bbox_min_xyz = np.minimum(a_mn, b_mn).astype(
                        np.float32).tolist()
                    a.bbox_max_xyz = np.maximum(a_mx, b_mx).astype(
                        np.float32).tolist()
                    a.confidence = max(a.confidence, b.confidence)
                    a.n_observations = n_a + n_b
                    a.first_seen_tick = min(a.first_seen_tick,
                                            b.first_seen_tick)
                    a.last_seen_tick = max(a.last_seen_tick, b.last_seen_tick)
                    for t in b.source_tracks:
                        if t not in a.source_tracks:
                            a.source_tracks.append(t)
                    # Prefer higher-confidence embedding.
                    if b.concept_embedding is not None and (
                        a.concept_embedding is None
                        or b.confidence > a.confidence
                    ):
                        a.concept_embedding = b.concept_embedding
                    # Fuse per-object point clouds if either side has them.
                    if a.points_xyz is not None or b.points_xyz is not None:
                        from ..geometry.pointcloud import merge_and_cap
                        a.points_xyz, a.points_rgb = merge_and_cap(
                            a.points_xyz, a.points_rgb,
                            b.points_xyz, b.points_rgb,
                            voxel_size=0.02, max_points=5000,
                        )
                    # Rewrite back-refs that pointed at id_b.
                    _rewrite_object_backrefs(sg, old_id=id_b, new_id=id_a)
                    del sg.objects[id_b]
                    collapsed += 1
                    did_collapse = True
                    # Update a's cached bounds for subsequent iterations
                    # within this same outer pass.
                    a_mn = np.asarray(a.bbox_min_xyz, dtype=np.float32)
                    a_mx = np.asarray(a.bbox_max_xyz, dtype=np.float32)
        if not did_collapse:
            break
    return collapsed


def _rewrite_object_backrefs(sg: SceneGraph, old_id: str, new_id: str) -> None:
    """When an ObjectNode is absorbed, fix any forward/back-refs that
    name it by id. Keeps the graph internally consistent after a collapse.
    """
    # Walls that listed the absorbed door as a child.
    for w in sg.walls.values():
        if old_id in w.door_ids:
            w.door_ids = [did for did in w.door_ids if did != old_id]
            if new_id not in w.door_ids:
                w.door_ids.append(new_id)
    # Rooms that listed the absorbed object.
    for r in sg.rooms.values():
        if old_id in r.object_ids:
            r.object_ids = [oid for oid in r.object_ids if oid != old_id]
            if new_id not in r.object_ids:
                r.object_ids.append(new_id)


def clip_oversized_bboxes(sg: SceneGraph, drop_factor: float = 2.0,
                          verbose: bool = False,
                          min_n_observations: int = 1) -> tuple:
    """Post-pass that enforces the per-label size cap on promoted bboxes.

    Two-tier response:
      - Mildly oversized  (extent ≤ cap × drop_factor): SHRINK the bbox
        to the cap, centred on the object's centroid. Preserves the
        detection — it just reports a more honest footprint.
      - Badly oversized   (extent > cap × drop_factor): DROP the object
        as a probable runaway-merge.

    `min_n_observations` skips objects observed fewer than that many
    times. The default 1 preserves the offline behaviour (consider
    every promoted object). The online cleanup loop passes a higher
    value (e.g. 10) so an early-stage candidate isn't prematurely
    dropped before it has had a chance to grow into a sensible-sized
    AABB across more frames.

    When `verbose=True`, prints the label + extent + n_observations of
    every dropped object so callers can tell what was lost.

    Returns (n_clipped, n_dropped).
    """
    clipped = 0
    drop_ids: list[str] = []
    dropped_log: list[str] = []
    for obj_id, obj in sg.objects.items():
        if obj.n_observations < min_n_observations:
            continue
        mn = np.asarray(obj.bbox_min_xyz, dtype=np.float32)
        mx = np.asarray(obj.bbox_max_xyz, dtype=np.float32)
        ext = mx - mn
        cap = dedup_rules.max_extent(obj.label)
        horizontal = max(float(ext[0]), float(ext[1]))
        vertical = float(ext[2])
        # Drop when really out of range (> drop_factor × cap).
        if horizontal > cap * drop_factor or vertical > cap * 2.0 * drop_factor:
            drop_ids.append(obj_id)
            if verbose:
                dropped_log.append(
                    f"  drop {obj_id!r:22s} label={obj.label!r:14s} "
                    f"extent=({ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f}) m "
                    f"(cap={cap:.2f}, drop>{cap * drop_factor:.2f})  "
                    f"n_obs={obj.n_observations}")
            continue
        # Otherwise, clip each axis to the cap (or 2*cap vertically),
        # centred on the object's position.
        if horizontal > cap or vertical > cap * 2.0:
            ctr = np.asarray(obj.position_xyz, dtype=np.float32)
            per_axis_cap = np.array([cap, cap, cap * 2.0], dtype=np.float32)
            half = np.minimum(ext * 0.5, per_axis_cap * 0.5).astype(np.float32)
            obj.bbox_min_xyz = (ctr - half).astype(np.float32).tolist()
            obj.bbox_max_xyz = (ctr + half).astype(np.float32).tolist()
            clipped += 1
    if verbose and dropped_log:
        print("[clip_oversized_bboxes] runaway merges dropped:")
        for line in dropped_log:
            print(line)
    for oid in drop_ids:
        del sg.objects[oid]
    return clipped, len(drop_ids)


# Back-compat — older call sites (shim, replay test) still reference the
# old name; point it at the new behaviour. Returns count of dropped only.
def drop_oversized_objects(sg: SceneGraph) -> int:
    _, n_drop = clip_oversized_bboxes(sg)
    return n_drop


def run_periodic_cleanup(
    sg: SceneGraph,
    min_n_observations_for_clip: int = 10,
    verbose: bool = False,
) -> dict:
    """Online cleanup: same three passes the offline driver runs at the
    end of the SAM 3 loop, but called *during* the loop on a cadence so
    a live consumer (planner / VLM / Rerun viewer) sees a sane object
    count instead of a wall of pending duplicates.

    Order matches the offline driver (post-SAM3 cleanup block):

      1. `collapse_cross_label`   — fuse same-spot different-label pairs.
      2. `clip_oversized_bboxes`  — shrink mild-oversized; drop runaways.
                                    Gated by `min_n_observations_for_clip`
                                    so early-stage candidates with a
                                    transiently-noisy AABB don't get
                                    prematurely culled.
      3. `collapse_overlapping_same_label` — fuse same-label overlapping
                                             pairs.

    Returns a dict of per-pass counts so callers can decide whether to
    log. `verbose` is forwarded to `clip_oversized_bboxes` only — its
    drop log is the only useful one mid-run; the others are quiet.
    """
    n_cross = collapse_cross_label(sg)
    n_clipped, n_dropped = clip_oversized_bboxes(
        sg,
        min_n_observations=min_n_observations_for_clip,
        verbose=verbose,
    )
    n_overlap = collapse_overlapping_same_label(sg)
    return {
        "cross_label": n_cross,
        "clipped": n_clipped,
        "dropped": n_dropped,
        "overlap": n_overlap,
    }


__all__ = ["merge_detection_into_object", "insert_or_merge",
           "collapse_cross_label", "collapse_overlapping_same_label",
           "clip_oversized_bboxes", "drop_oversized_objects",
           "run_periodic_cleanup"]
