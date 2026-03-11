# Alex room explore

![](demo.png)

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


Runs a walking policy for Alex in the room scene, with the same manual and prompt-driven exploration flow as `demos/cam_room_explore`, but using Alex locomotion instead of the camera robot.

Run:

```bash
cd demos/alex_room_explore/
mjpython run.py
```

Run to explore automatically for obejcts:

```bash
mjpython run.py --prompt oven
```

You can control the robot walking with manual commands:

- `Up` / `Down`: forward / backward
- `Left` / `Right`: turn left / right
- `Cmd` + `Left` / `Right`: strafe left / right


Tests:

```bash
cd training/alex-room-explore/
mjpython tests/test_alex_room.py
```
