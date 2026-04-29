"""Mask-pooled SAM 3 vision features as per-detection embeddings.

SAM 3's image backbone caches dense features on its inference `state` after
`set_image`. We resize each binary mask to the feature-map resolution and
average-pool the features under the mask → one 256-d (or whatever the
backbone outputs) vector per detection.

This is our cheapest replacement for CLIP/DINOv2: no second model loaded.
See Plan/scene_graph_plan.md §3.4 for rationale.
"""

from __future__ import annotations
from typing import List

import numpy as np

from .sam3_detector import RawDetection


def fill_embeddings(state, dets: List[RawDetection]) -> None:
    """Mask-pool cached vision features for each detection in-place.

    Expects SAM 3's inference state dict as returned by
    `sam3_processor.set_image`. Robust to either a tensor or missing key —
    if we can't find the feature map, leave `embedding=None` on every det
    (Phase 0 will gate the embedding-cosine check on this).
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return

    feat = _extract_feature_map(state)
    if feat is None:
        return

    # feat: (1, C, H_f, W_f) or (C, H_f, W_f)
    if feat.ndim == 3:
        feat = feat.unsqueeze(0)
    _, _, H_f, W_f = feat.shape
    feat = feat[0]  # (C, H_f, W_f)

    for d in dets:
        m = d.mask
        m_t = torch.from_numpy(m.astype(np.float32))[None, None]      # (1,1,H,W)
        m_small = F.interpolate(m_t, size=(H_f, W_f), mode="nearest")[0, 0] > 0.5
        if not m_small.any():
            continue
        pooled = feat[:, m_small].mean(dim=-1)                         # (C,)
        # L2-normalise so cosine similarity is a dot product
        pooled = pooled / (pooled.norm() + 1e-8)
        d.embedding = pooled.detach().cpu().numpy().astype(np.float32)


def _extract_feature_map(state):
    """Return the dense vision features tensor from SAM 3's state dict.

    In the current SAM 3 repo, `Sam3Processor.set_image()` stores its
    backbone output at `state["backbone_out"]["vision_features"]` with
    shape (1, 256, 72, 72). We probe both the nested path and several
    legacy flat keys so older / forked SAM versions still work.
    """
    if state is None or not hasattr(state, "get"):
        return None

    # Preferred path (SAM 3, 2025 repo layout).
    bb = state.get("backbone_out")
    if isinstance(bb, dict):
        v = bb.get("vision_features")
        if v is not None:
            return v
        # Some forks only expose the FPN pyramid — use the finest level.
        fpn = bb.get("backbone_fpn")
        if isinstance(fpn, (list, tuple)) and len(fpn) > 0:
            return fpn[0]

    # Legacy / alternative layouts.
    for k in ("vision_features", "image_embeddings",
              "image_embed", "image_features"):
        v = state.get(k)
        if v is not None:
            return v
    return None


def cosine(a: "np.ndarray | None", b: "np.ndarray | None") -> float:
    """Cosine similarity for pre-normalised vectors. Returns 0 if either is None."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


__all__ = ["fill_embeddings", "cosine"]
