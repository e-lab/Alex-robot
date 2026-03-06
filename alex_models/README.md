# Alex robot models


from: https://github.com/ihmcrobotics/ihmc-alex-sdk/tree/develop

Note:

Had to fix the model for mujoco: alex-models/alex_V1_description/mjcf/alex_v1_full_body_mjx.xml because of error: inertial must have positive eigenvalues line 234 pelvis_link id 1


## alex_sensors.py

Implements aobservation-group compatible layer as per [mujoco mjlab](https://mujocolab.github.io/mjlab/main/source/observations.html) specs.
