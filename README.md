# Alex robot

Developing software and control for IHMC Alex humanoid robot.

![](images/alex.png)


## Scenes

Alex scenes are in dir `alex-scenes/`. Load `scene_alex_v1_full_body_mjx_room1.xml` to see Alex in a room.

Room scenes from https://github.com/allenai/molmospaces.


## Install

### Requirements
- Python 3.11
- NVIDIA GPU (required for mjlab / GPU-accelerated training)
- CUDA 12.x

### 1. Clone the repo

```bash
git clone <repo-url>
cd Alex-robot
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `mujoco`, `gymnasium` — simulation & RL framework
- `stable-baselines3`, `torch` — RL training
- `opencv-python`, `mediapipe`, `ultralytics` — computer vision
- `numpy`, `scipy` — scientific computing
- `openai` — LLM brain controller

### 4. Install mjlab

[mjlab](https://github.com/e-lab/mjlab) is the e-Lab GPU-accelerated RL training framework:

```bash
pip install mjlab
```

### 5. Set up OpenAI API key (optional)

Required only for the LLM brain controller in demos:

```bash
export OPENAI_API_KEY=your_key_here
```

### Notes on packages not available via pip

The following must be installed separately if needed:
- **Isaac Lab** — see https://isaac-sim.github.io/IsaacLab/
- **Omniverse / Isaac Sim** — installed via NVIDIA Isaac Sim installer
- **warp-lang** — installed automatically as a dependency of mjlab (`pip install warp-lang`)
- **mujoco-warp** — installed automatically as a dependency of mjlab


## Run

###  MuJoCo

You can run alex model by dragging / dropping [this file](scenes/alex-scenes/scene_alex_v1_full_body_mjx_room1.xml) into MuJoCo. Or any file in that directory `scenes/alex-scenes/`

OR you can run with:

`python -m mujoco.viewer --mjcf Alex-robot/scenes/alex-scenes/scene_alex_v1_full_body_mjx_ec1.xml`

You will get this:

![](images/alex_room1.png)


### RL learning

We can train VERY efficiently with [mjlab](https://github.com/e-lab/mjlab). Use this fork to train RL algorithms on your NVIDIA GPU PC. See readme there.


For pure MuJoCo RL training, see [this file](training/README.md).


## References

[IHMC Alex](https://www.ihmc.us/news20251119/) 