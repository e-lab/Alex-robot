# Alex robot RL models

Developing RL control for IHMC Alex humanoid robot.




## Alex Stand

**STATUS: WORKS**


Train:

`python training/alex-stand/alex-stand-ppo.py --timesteps 10000000 --n-envs 8`

Demo:

`mjpython training/alex-stand/alex-stand-ppo.py --eval training/alex-stand/rl_models/best/best_model --vec-norm training/alex-stand/rl_models/vec_normalize_final.pkl --episodes 20`


TensorBoard:

`tensorboard --logdir training/alex-stand/rl_models/tensorboard`


## Alex walking

**STATUS: initial gait good example**

1. Test initial untrained srating poit gait for learning:

`mjpython training/alex-walking/alex-walking-test-gait.py`

2. Start training:
`python training/alex-walking/alex-walking-ppo.py`

3. Evaluate (after training):
`mjpython training/alex-walking/alex-walking-ppo.py --eval`

   
#### Notes:
   
- Reference Gait: The robot follows a periodic sinusoidal target (a "fixed set of steps") for its legs. The RL policy learns to provide residuals (offsets) to this gait to maintain balance and optimize forward velocity.

- Symmetry: The right leg automatically mirrors the left leg's motion with a 180-degree (0.5) phase shift.

- Minimum Actuation: Only the pitch axes of the legs are active: hip_y, knee, and ankle_y. All other joints are held at the stable STAND_PREP pose.

- Reward Function: Encourages forward velocity (vx) while penalizing lateral drift, vertical height deviations, and excessive control effort.


## Alex Stand IHMC IsaacLab

**STATUS: FAILS**

A few routines trained by IHMC on IsaacLab.

`https://github.com/ihmcrobotics/alex/tree/develop/src/main/resources/rl_models`

We want to see if we can make these models work here in MuJoCo also.

