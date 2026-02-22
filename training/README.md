# Alex robot RL models

Developing RL control for IHMC Alex humanoid robot.




## Alex Stand

**STATUS: WORKS**


Train:

`python training/alex-stand/alex_stand_ppo.py --timesteps 10000000 --n-envs 8`

Demo:

`mjpython training/alex-stand/alex_stand_ppo.py --eval training/alex-stand/rl_models/best/best_model --vec-norm training/alex-stand/rl_models/vec_normalize_final.pkl --episodes 20`


TensorBoard:

`tensorboard --logdir training/alex-stand/rl_models/tensorboard`


## Alex walking

**STATUS: FAILS - initial gait good example,  best epi lenght ~40**

Loads the trained stand pose if available for curriculum.


1. Test initial untrained starting point gait for learning:

`mjpython training/alex-walking/alex-walking-test-gait.py`

2. Start training to refine walking:
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


## Notes

### IsaacSim Lab notes from IHMC


1- Bent-knee initial pose vs near-upright pose
    - alex-stand: hip_y=0.05, knee=0.10 — nearly straight legs, CoM directly over feet, trivially stable
    - alex-stand-isaac: hip_y=-0.772, knee=1.419 — deep squat, very narrow balance basin, large active torques required just to hold still

    The bent-knee pose requires significant continuous muscle effort to maintain. An untrained policy outputting random actions immediately collapses. The near-upright pose is naturally stable even with zero torques.

2- GPU envs

    - The fundamental problem is that the isaacsimlab configuration was designed for 4096 GPU envs in parallel. It tolerates
  aggressive randomization because the sheer volume of environments guarantees enough "good starts." With only 8 CPU envs,
  you need gentle randomization, a stable initial pose, and no distractions (pushes, full action space) until the policy has
  learned the basics.

  -   The bottleneck is CPU physics simulation, not the GPU. SB3 uses the GPU only for tiny NN forward/backward passes
  ([128,128,128] is trivially small). A RTX 3090 GPU would sit at <5% utilization.