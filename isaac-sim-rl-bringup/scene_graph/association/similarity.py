"""Spatial + visual similarity scores between a new detection and existing objects.

Mirrors ConceptGraphs' `slam/mapping.py::compute_spatial_similarities` +
`compute_visual_similarities` split, but collapsed into two small functions.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from ..geometry.bbox import aabb_iou
from ..graph.node_types import ObjectNode
from ..detection.sam3_detector import RawDetection
from ..detection.embeddings import cosine


def spatial_score(det_bbox_min: np.ndarray,
                  det_bbox_max: np.ndarray,
                  obj: ObjectNode) -> float:
    """3D AABB IoU between a fresh detection bbox and an existing object."""
    return aabb_iou(det_bbox_min, det_bbox_max,
                    np.asarray(obj.bbox_min_xyz), np.asarray(obj.bbox_max_xyz))


def visual_score(det_embedding: Optional[np.ndarray], obj: ObjectNode) -> float:
    """Cosine similarity between detection embedding and stored object embedding."""
    return cosine(det_embedding, obj.concept_embedding)


def centroid_distance(det_centroid: np.ndarray, obj: ObjectNode) -> float:
    return float(np.linalg.norm(det_centroid - np.asarray(obj.position_xyz)))


__all__ = ["spatial_score", "visual_score", "centroid_distance"]
