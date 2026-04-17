# Alex Isaac Sim RL Bring-up

Isaac Sim scripts for running the Alex humanoid (walking ONNX policy) and a
simpler camera-robot harness used to prototype perception + autonomy. Both
share the same Isaac Sim build, the same room USDs, and the same Hydra
config tree.

## Folder structure

```
isaac-sim-rl-bringup/
├── README.md                          ← this file (index)
├── configs/                           ← Hydra config groups + schema
│   ├── cam_room_explore.yaml          ← root composition
│   ├── schema.py                      ← dataclass schemas
│   ├── scene/                         ← room, hallway, groundplane
│   ├── detector/                      ← disabled, sam3
│   ├── rerun/                         ← disabled, full
│   └── autonomy/                      ← manual, approach
├── scripts/
│   ├── alex_room_explore/             ← walking ONNX policy in Isaac Sim
│   │   ├── alex_onnx_walking_policy.py
│   │   └── README.md
│   └── cam_room_explore/              ← camera-robot perception/autonomy harness
│       ├── cam_room_explore_isaac.py
│       └── README.md
├── models/                            ← ONNX policy weights (not in git)
├── docs/                              ← integration notes, diagrams
└── images/                            ← output videos
```

## Scripts

- **[scripts/alex_room_explore/](scripts/alex_room_explore/README.md)** — run
  the walking ONNX policy on Alex. Keyboard teleop, optional ithor kitchen
  scene. This is the real robot.

- **[scripts/cam_room_explore/](scripts/cam_room_explore/README.md)** —
  free-flying camera with SAM3 open-vocabulary detection, world-frame point
  cloud, Rerun dashboard, scene graph, prompt-driven autonomous
  search/approach FSM. Hydra-configured. This is the perception harness we
  port onto Alex once logic is right.

## One-time setup

### Isaac Sim 5.1
Built output at `~/alex/repository-group/IsaacSim/_build/linux-x86_64/release/`.

```bash
pip install isaacsim --extra-index-url https://pypi.nvidia.com
# OR build from source:
git clone <isaac-sim-source-repo> ~/alex/repository-group/IsaacSim
cd ~/alex/repository-group/IsaacSim && ./build.sh
```

### IsaacLab (IHMC fork — includes Alex robot config)
```bash
git clone https://github.com/ihmcrobotics/IsaacLab.git ~/alex/repository-group/IsaacLab
cd ~/alex/repository-group/IsaacLab && ./isaaclab.sh --install

# Symlink Isaac Sim build into IsaacLab
ln -s ~/alex/repository-group/IsaacSim/_build/linux-x86_64/release \
      ~/alex/repository-group/IsaacLab/_isaac_sim
```

### Extra Python packages (into IsaacLab's bundled Kit python)
```bash
cd ~/alex/repository-group/IsaacLab
./isaaclab.sh -p -m pip install onnxruntime rerun-sdk hydra-core
```

**For `cam_room_explore`:** SAM3 must be cloned and installed into Kit python.
See [scripts/cam_room_explore/README.md](scripts/cam_room_explore/README.md).

### Verify Alex robot config loads
```bash
cd ~/alex/repository-group/IsaacLab
./isaaclab.sh -p -c "from isaaclab_assets.ihmc.robots.alex import alex; print('OK')"
```

## Troubleshooting

- **`isaaclab_assets` import error** — use the IHMC fork, not upstream IsaacLab
- **ONNX model not found** — copy `policy.onnx` from the lab machine into
  `models/2026-03-17_23-20-27_flatfeet/`
- **`_isaac_sim` symlink missing** — see IsaacLab setup above
