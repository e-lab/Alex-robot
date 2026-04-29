"""Per-tick driver for the scene-graph pipeline.

Mirrors ConceptGraphs' `slam/cfslam_pipeline_batch.py::main`: one plain
function that reads `cfg`, calls free functions in order, passes state
(the SceneGraph) through as an argument.

Phases add more steps but the shape stays the same:

    rgb, depth, pose ─► detect (SAM3) ─► embed (mask-pool) ─► unproject
      ─► object_layer.ingest_detections ─► (Phase 2) place_layer.update
      ─► (Phase 3) room_layer.update ─► (Phase 4) viz.log ─► (periodic) save
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from ..detection.sam3_detector import RawDetection, detect_all
from ..detection.embeddings import fill_embeddings
from ..graph.scene_graph import SceneGraph
from ..layers import object_layer


def process_one_frame(sg: SceneGraph,
                      rgb: np.ndarray,
                      depth: np.ndarray,
                      K: np.ndarray,
                      cam_pos: np.ndarray,
                      cam_quat_wxyz: np.ndarray,
                      tick: int,
                      sam3_processor,
                      prompts: List[str],
                      conf_threshold: float = 0.3) -> List[RawDetection]:
    """One pass of the perception pipeline. Returns the raw detections so
    the caller can reuse them for Rerun / debugging without re-running SAM 3.

    Each detection carries a 256-d mask-pooled SAM 3 embedding (when the
    backbone features are reachable on the state) — used later in the
    association step for visual similarity.
    """
    dets, state = detect_all(rgb, prompts, sam3_processor,
                             conf_threshold=conf_threshold)
    if not dets:
        return dets

    # Mask-pool SAM 3's cached vision features → 256-d embedding per det.
    # Silent no-op if the backbone output isn't present (no extra model).
    try:
        fill_embeddings(state, dets)
    except Exception as e:
        # Printing on every frame would be spammy; let the caller decide
        # whether to surface this.
        pass

    object_layer.ingest_detections(
        sg, dets, depth=depth, K=K,
        cam_pos=cam_pos, cam_quat_wxyz=cam_quat_wxyz,
        tick=tick, rgb=rgb,     # rgb enables Phase 5.5 best-view snapshot
    )
    return dets


__all__ = ["process_one_frame"]
