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
Built output at `~/pathtoFolder/IsaacSim/_build/linux-x86_64/release/`.

```bash
pip install isaacsim --extra-index-url https://pypi.nvidia.com
# OR build from source:
git clone <isaac-sim-source-repo> ~/pathtoFolder/IsaacSim
cd ~/pathtoFolder/IsaacSim && ./build.sh
```

### IsaacLab (IHMC fork — includes Alex robot config)
```bash
git clone https://github.com/ihmcrobotics/IsaacLab.git ~/pathtoFolder/IsaacLab
cd ~/pathtoFolder/IsaacLab && ./isaaclab.sh --install

# Symlink Isaac Sim build into IsaacLab
ln -s ~/pathtoFolder/IsaacSim/_build/linux-x86_64/release \
      ~/pathtoFolder/IsaacLab/_isaac_sim
```

### Extra Python packages (into IsaacLab's bundled Kit python)
```bash
cd ~/pathtoFolder/IsaacLab
./isaaclab.sh -p -m pip install onnxruntime rerun-sdk hydra-core
```

Optional (needed if you use `detector=sam3` or `yolo=ihmc`):
```bash
./isaaclab.sh -p -m pip install ultralytics onnx          # YOLO
```

### SAM3 (needed for `detector=sam3` in either script)

SAM3 is a text-promptable segmentation model from Meta — required to run
open-vocabulary detection on the head camera. Access-gated on HuggingFace,
so **request access first:** https://huggingface.co/facebook/sam3

Install the SAM3 package into IsaacLab's Kit python:
```bash
# 1. Clone SAM3 (we use ~/E-Lab/Spring2026/repos/sam3)
git clone https://github.com/facebookresearch/sam3.git ~/pathtoFolder/sam3

# 2. Install into Kit python (editable — path stays linked)
cd ~/pathtoFolder/IsaacLab
./isaaclab.sh -p -m pip install -e ~/pathtoFolder/sam3
./isaaclab.sh -p -m pip install decord pycocotools    # transitive deps

# 3. Log in to HuggingFace (one-off; saves a token under ~/.huggingface)
./isaaclab.sh -p -m pip install huggingface_hub
./isaaclab.sh -p -c "from huggingface_hub import login; login()"
# paste your HF access token when prompted
```

The SAM3 checkpoint (~3.3 GB) auto-downloads to `~/.cache/huggingface/hub/
models--facebook--sam3/` on first use. After that, `detector=sam3` works
offline. Verify:
```bash
./isaaclab.sh -p -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

If pip downgrades torch/numpy/nvidia-cu* wheels and Isaac crashes with
`undefined symbol` errors, uninstall the mismatched versions so Isaac's
bundled `torch 2.7.0+cu128` resolves:
```bash
./isaaclab.sh -p -m pip uninstall -y torch triton \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
```

### Verify Alex robot config loads
```bash
cd ~/pathtoFolder/IsaacLab
./isaaclab.sh -p -c "from isaaclab_assets.ihmc.robots.alex import alex; print('OK')"
```

## Troubleshooting

- **`isaaclab_assets` import error** — use the IHMC fork, not upstream IsaacLab
- **ONNX model not found** — copy `policy.onnx` from the E-Lab folder into
  `models/2026-03-17_23-20-27_flatfeet/`
- **`_isaac_sim` symlink missing** — see IsaacLab setup above
- **SAM3 `HFValidationError`** / `401 unauthorized` — you haven't been granted
  access on HuggingFace or haven't logged in; see SAM3 install section
- **Isaac crashes with `libcusparse.so ... undefined symbol __nvJitLinkCreate`
  after installing SAM3 / ultralytics** — pip pulled CUDA 12.6 wheels that
  conflict with Isaac's bundled CUDA 12.8; uninstall the mismatched wheels
  (see SAM3 section)
