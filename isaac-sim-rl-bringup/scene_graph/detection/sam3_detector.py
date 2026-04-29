"""SAM 3 concept-driven (PCS) segmentation wrapper.

Strategy 1 from the plan: run SAM 3 once per tick with a fixed concept
vocabulary; return per-detection label + mask + score + bbox.

We do not instantiate SAM 3 in this module's import — the caller loads the
model once and passes the `processor` in. This keeps Isaac / Kit startup
ordering decoupled from the scene-graph package.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RawDetection:
    """One SAM 3 mask + metadata from a single frame, pre-association."""
    label: str
    score: float
    mask: np.ndarray             # (H, W) bool
    bbox_xyxy: np.ndarray        # (4,) float — pixel coords
    # Optional; filled by detection/embeddings.py if enabled
    embedding: Optional[np.ndarray] = None


def detect_all(rgb: np.ndarray,
               prompts: List[str],
               sam3_processor,
               conf_threshold: float = 0.3):
    """Run SAM 3 for every prompt on one RGB frame. Returns (dets, state)
    where `state` is SAM 3's inference dict (holds cached vision features
    for the embedding pooler) and `dets` is a list of RawDetection.

    Reuses the `Sam3Processor.set_image` + per-prompt `set_text_prompt` flow.
    SAM 3 caches `backbone_out.vision_features` on `state` after set_image,
    so per-prompt runs are fast and we can extract per-mask embeddings from
    that feature map without a second network pass.

    `embedding` on each returned RawDetection is left None — fill via
    `detection.embeddings.fill_embeddings(state, dets)`.
    """
    from PIL import Image as _PIL

    img = _PIL.fromarray(rgb.astype(np.uint8))
    state = sam3_processor.set_image(img)

    dets: List[RawDetection] = []
    for prompt in prompts:
        out = sam3_processor.set_text_prompt(state=state, prompt=prompt)
        masks, boxes, scores = out.get("masks"), out.get("boxes"), out.get("scores")
        if masks is None or len(masks) == 0:
            continue
        for i, s in enumerate(scores):
            s = float(s)
            if s < conf_threshold:
                continue
            m = masks[i]
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            b = boxes[i]
            if hasattr(b, "cpu"):
                b = b.cpu().numpy()
            dets.append(RawDetection(
                label=prompt,
                score=s,
                mask=(m > 0.5),
                bbox_xyxy=np.asarray(b).reshape(-1)[:4].astype(np.float32),
            ))
    return dets, state


__all__ = ["RawDetection", "detect_all"]
