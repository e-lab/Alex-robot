# Alex robot RL models

Developing RL control for IHMC Alex humanoid robot.




## Alex Stand

Train:

`python training/alex-stand/alex-stand-ppo.py --timesteps 10000000 --n-envs 8`

Demo:

`mjpython training/alex-stand/alex-stand-ppo.py --eval training/rl_models/best/best_model --vec-norm rl_models/vec_normalize_final.pkl --episodes 20`


TensorBoard:

`tensorboard --logdir training/alex-stand/rl_models/tensorboard`


## Alex Stand IHMC IsaacLab

A few routines trained by IHMC on IsaacLab.

`https://github.com/ihmcrobotics/alex/tree/develop/src/main/resources/rl_models`

We want to see if we can make these models work here in MuJoCo also.

