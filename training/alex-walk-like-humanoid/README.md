# Alex Humanoid-like Walking Training

This directory contains a PPO implementation to train the Alex robot (`alex_v1`) to walk using a reward function and observation space similar to the standard MuJoCo Humanoid (`Humanoid-v5`).

## How it works

- **Environment**: `AlexHumanoidWalkEnv` loads the full body Alex model and defines an observation space including torso position, orientation, and all joint states.
- **Reward**: 
  - `reward_forward`: 1.25 * x_velocity
  - `reward_healthy`: +5.0 per step if the robot is upright and at the correct height.
  - `reward_ctrl`: Penalty for excessive joint torques.
- **Algorithm**: PPO from Stable Baselines3 with `SubprocVecEnv` (8 parallel environments).

## Usage

### Training
To start training:
```bash
python training/alex-walk-like-humanoid/train_ppo.py
```
- Logs and models are saved in `rl_models/`.
- Training progress can be monitored via TensorBoard.

### TensorBoard
```bash
tensorboard --logdir training/alex-walk-like-humanoid/rl_models/tensorboard
```

### Evaluation
To visualize the trained robot:
```bash
mjpython training/alex-walk-like-humanoid/eval_ppo.py
```
*The evaluation uses the standard MuJoCo passive viewer.*

## Prerequisites
Ensure `stable-baselines3` and `mujoco` are installed.
```bash
pip install stable-baselines3 mujoco gymnasium
```
