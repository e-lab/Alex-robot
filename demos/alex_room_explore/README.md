# Alex room explore

Requires [mjlab from e-Lab](https://github.com/e-lab/mjlab) with IHMC robot models and environments.

If installing [mjlab from pip](https://mujocolab.github.io/mjlab/main/source/installation.html) with:

```bash
pip install mjlab
```

copy the robot and envs over:

- `mjlab/src/mjlab/asset_zoo/robots/alex_V1_description`

- `mjlab/src/mjlab/tasks/tracking/config/alex`

- `mjlab/src/mjlab/tasks/velocity/config/alex`

to the proper install location of your mjlab.


Runs a walking tracking policy trained in mjlab "Mjlab-Velocity-Flat-Alex-V1", with checkpoint from [here](https://wandb.ai/culurciello/mjlab/runs/ib5nrc31/).

Run:

```bash
cd training/alex-room-explore/
mjpython play_alex_room.py
```

Run to explore automatically for obejcts:

```bash
mjpython alex_room_explore.py --prompt oven
```

You can control the robot walking with manual "twist" commands:

  - ↑ / ↓: increase/decrease forward velocity (lin_vel_x)
  - ← / →: increase/decrease yaw rate (ang_vel_z)
  - CMD + ← / CMD + →: increase/decrease strafe (lin_vel_y)
  - Delete (or Backspace): reset all twist commands to zero


Tests:

```bash
cd training/alex-room-explore/
mjpython tests/test_alex_room.py
```

