"""Loco-X live-run entry point.

Hydra @main composes ``conf/config.yaml`` and dispatches into either:

1. The Phase 1-4 demo (``scripts/alex_room_explore/alex_onnx_walking_policy.py``)
   when ``agent.enabled=false`` — the runtime delegates to the existing
   script unchanged.
2. The Loco-X agent loop (LA-5) when ``agent.enabled=true`` — currently
   raises ``NotImplementedError`` until LA-5 lands.

The split keeps Phase 1-4 reachable from the same CLI so we can A/B
behaviours side-by-side via Hydra overrides.

Run examples:
    python -m loco_x.cli.run scene=room
    python -m loco_x.cli.run scene=room occupancy=heightmap
    python -m loco_x.cli.run scene=room agent=stdin
"""
from __future__ import annotations

import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if not cfg.agent.get("enabled", False):
        print(
            "[loco_x] agent.enabled=false — Phase 1-4 only. The Hydra root\n"
            "[loco_x] composed here is informational; for the live demo,\n"
            "[loco_x] still launch via:\n"
            "[loco_x]   ./isaaclab.sh -p scripts/alex_room_explore/"
            "alex_onnx_walking_policy.py ..."
        )
        return

    raise NotImplementedError(
        "Loco-X agent loop is not yet wired (LA-5 outstanding). For now "
        "run Phase 1-4 directly or pass agent=disabled."
    )


if __name__ == "__main__":
    main()
