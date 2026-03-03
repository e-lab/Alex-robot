# Alex robot RL models

Developing RL control for IHMC Alex humanoid robot.




## Alex Stand

**STATUS: WORKS**


Train:

`python train.py --timesteps 10000000 --n-envs 8`

Evaluate:
`python eval.py`

TensorBoard:
`tensorboard --logdir rl_models/tensorboard`



## Alex walking

**STATUS: WORKS**

Loads the trained stand pose if available for curriculum.


1. Test initial untrained starting point gait for learning:

`mjpython test-gait.py`

2. Start training to refine walking:
`python train.py`

3. Evaluate (after training):
`python eval.py`


## Alex walk like humanoid

Since the gymnasium mujoco env humanoid learn to walk more easily, we made a similar env for Alex.

**STATUS: WORKS episodes up to 300 steps**

Train:
`python train.py`


Tensorboard:
`tensorboard --logdir rl_models/tensorboard`

Eval:
`python eval.py`



## Alex room explore

Runs a walking tracking policy trained in mjlab "Mjlab-Velocity-Flat-Alex-V1", with checkpoint from [here](https://wandb.ai/culurciello/mjlab/runs/ib5nrc31/).

Run:

```bash
mjpython play_alex_room.py
```

You can control the robot walking with manual "twist" commands:

  - ↑ / ↓: increase/decrease forward velocity (lin_vel_x)
  - ← / →: increase/decrease yaw rate (ang_vel_z)
  - CMD + ← / CMD + →: increase/decrease strafe (lin_vel_y)
  - Delete (or Backspace): reset all twist commands to zero




## humanoid walk

**STATUS: WORKS**

using the built in gynasium MuJoCo humanoid-v5 env to learn to walk

Training

```bash
python train.py
```
- Logs and models are saved in `rl_models/`.
- The script uses `SubprocVecEnv` for multi-threaded training and `VecNormalize` for observation/reward normalization.

Tensorboard - monitor training progress:
```bash
tensorboard --logdir rl_models/tensorboard
```

Evaluation

```bash
python eval.py
```

   
#### Notes:
   
- Reference Gait: The robot follows a periodic sinusoidal target (a "fixed set of steps") for its legs. The RL policy learns to provide residuals (offsets) to this gait to maintain balance and optimize forward velocity.

- Symmetry: The right leg automatically mirrors the left leg's motion with a 180-degree (0.5) phase shift.

- Minimum Actuation: Only the pitch axes of the legs are active: hip_y, knee, and ankle_y. All other joints are held at the stable STAND_PREP pose.

- Reward Function: Encourages forward velocity (vx) while penalizing lateral drift, vertical height deviations, and excessive control effort.


