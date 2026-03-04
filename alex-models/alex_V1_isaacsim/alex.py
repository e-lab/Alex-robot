"""Configuration for IHMC robots. 

Robot config module from: https://github.com/ihmcrobotics/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/ihmc/robots/alex/alex.py

The following configurations are available:

* :obj:`ALEXANDER_V1`: 

The current configurations we have here are : 
- a full body which includes all the links on alex except the hands 
- a nub forearm which replaces the forearm assembly with a single carbon fiber cilinder, effectively removing any actuated joint below the elbow
- a hanging which is based of the nub but includes extra unactuated links to allow for hanging the robot in the air during sim

TODO:  Discuss with team whether each configuration of Alex lives in here or if they have different files
       We have 3 options:
       1- All alex configurations live in this file 
       2- All alex configurations for a specific version of alex live in 1 file 
       3- Each configuration of alex has its own file 
"""

from __future__ import annotations

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ALEX_V1_FULLBODY_MINIMALCOLLISIONS_URDF = "../alex_V1_description/rl_urdf/alex_v1.rlModel_fullBody_robotAccurate_torsoFootCollisions.urdf"
ALEX_V1_FULLBODY_FULLCOLLISIONS_URDF = "../alex_V1_description/rl_urdf/alex_v1.rlModel_fullBody_robotAccurate_fullCollisions.urdf"
ALEX_V1_NUBFOREARMS_MINIMALCOLLISIONS_URDF = "../alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf"
ALEX_V1_NUBFOREARMS_LOWERBODYCOLLISIONS_URDF = "../alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_legCollisions.urdf"
ALEX_V1_NUBFOREARMS_FULLCOLLISIONS_URDF = "../alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_fullCollisions.urdf"

ALEX_V1_NUBFOREARMS_MINIMALCOLLISIONS_HANGING_URDF = "../alex-models/alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions_hanging.urdf"
ALEX_V1_NUBFOREARMS_FULLCOLLISIONS_HANGING_URDF = "../alex-models/alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_fullCollisions_hanging.urdf"

ALEX_JOINT_SCALE = 0.3

CONTROL_DT = 0.02
SIM_DT = 0.005
MIN_DELAY_DT = 0.004    # minimum possible delay
MAX_DELAY_DT = 0.008    # double minimum delay    

EFFORT_LIMIT_115 = 217.2
EFFORT_LIMIT_85 = 160.7
EFFORT_LIMIT_76 = 96.8
EFFORT_LIMIT_68 = 70.5
EFFORT_LIMIT_S = 25.0
EFFORT_LIMIT_ANKLE_Y = EFFORT_LIMIT_76 * 2.0
EFFORT_LIMIT_ANKLE_X = EFFORT_LIMIT_76 * 1.5

VELOCITY_LIMIT_115 = 9.3
VELOCITY_LIMIT_85 = 10.38
VELOCITY_LIMIT_76 = 9.72
VELOCITY_LIMIT_68 = 10.59
VELOCITY_LIMIT_S = 17.3
VELOCITY_LIMIT_ANKLE_Y = VELOCITY_LIMIT_76
VELOCITY_LIMIT_ANKLE_X = VELOCITY_LIMIT_76

STIFFNESS_85_HIP_X = EFFORT_LIMIT_85 / 2.0
STIFFNESS_68_HIP_Z = EFFORT_LIMIT_68
STIFFNESS_115_HIP_Y = EFFORT_LIMIT_115 / 2.0
STIFFNESS_115_KNEE = EFFORT_LIMIT_115 / 2.0
STIFFNESS_76_ANKLE_Y = EFFORT_LIMIT_ANKLE_Y / 2.0
STIFFNESS_76_ANKLE_X = EFFORT_LIMIT_ANKLE_X / 2.0
STIFFNESS_85_SPINE_Z = EFFORT_LIMIT_85 / 2.0
STIFFNESS_S_NECK_Z = 5.0
STIFFNESS_S_NECK_Y = 5.0
STIFFNESS_85_SHOULDER_Y = EFFORT_LIMIT_85 / 6.0
STIFFNESS_85_SHOULDER_X = EFFORT_LIMIT_85 / 6.0
STIFFNESS_68_SHOULDER_Z = EFFORT_LIMIT_68 / 3.0
STIFFNESS_68_ELBOW_Y = EFFORT_LIMIT_68 / 3.0
STIFFNESS_S_WRIST_Z = 5.0
STIFFNESS_S_WRIST_X = 5.0
STIFFNESS_S_GRIPPER_Z = 0.0

DAMPING_85_HIP_X = EFFORT_LIMIT_85 / 20.0
DAMPING_68_HIP_Z = EFFORT_LIMIT_68 / 10.0
DAMPING_115_HIP_Y = EFFORT_LIMIT_115 / 20.0
DAMPING_115_KNEE = EFFORT_LIMIT_115 / 20.0
DAMPING_76_ANKLE_Y = EFFORT_LIMIT_ANKLE_Y / 20.0
DAMPING_76_ANKLE_X = EFFORT_LIMIT_ANKLE_X / 20.0
DAMPING_85_SPINE_Z = EFFORT_LIMIT_85 / 20.0
DAMPING_S_NECK_Z = 1.0
DAMPING_S_NECK_Y = 1.0
DAMPING_85_SHOULDER_Y = 8.0
DAMPING_85_SHOULDER_X = 8.0
DAMPING_68_SHOULDER_Z = 4.0
DAMPING_68_ELBOW_Y = 4.0
DAMPING_S_WRIST_Z = 1.0
DAMPING_S_WRIST_X = 1.0
DAMPING_S_GRIPPER_Z = 0.0

# For the previous settings with lower armature, use ARMATURE_SCALE=0.3, ARMATURE_ANKLE_SCALE=2.0
ARMATURE_SCALE = 1.0
ARMATURE_ANKLE_SCALE = 1.0

ARMATURE_85 = 0.062 * ARMATURE_SCALE
ARMATURE_68 = 0.020 * ARMATURE_SCALE
ARMATURE_115= 0.167 * ARMATURE_SCALE
ARMATURE_76 = 0.037 * ARMATURE_SCALE
ARMATURE_ANKLE_X = ARMATURE_76 * ARMATURE_ANKLE_SCALE
ARMATURE_ANKLE_Y = ARMATURE_76 * ARMATURE_ANKLE_SCALE
ARMATURE_S = 0.005 * ARMATURE_SCALE


ALEX_V1_FULLBODY_DEFAULT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path="",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*HIP_X", ".*HIP_Z", ".*HIP_Y", ".*KNEE_Y", ".*ANKLE_Y", ".*ANKLE_X"],
            stiffness={
                ".*HIP_X": STIFFNESS_85_HIP_X,
                ".*HIP_Z": STIFFNESS_68_HIP_Z,
                ".*HIP_Y": STIFFNESS_115_HIP_Y,
                ".*KNEE_Y": STIFFNESS_115_KNEE,
                ".*ANKLE_Y": STIFFNESS_76_ANKLE_Y,
                ".*ANKLE_X": STIFFNESS_76_ANKLE_X,
            },
            damping={
                ".*HIP_X": DAMPING_85_HIP_X,
                ".*HIP_Z": DAMPING_68_HIP_Z,
                ".*HIP_Y": DAMPING_115_HIP_Y,
                ".*KNEE_Y": DAMPING_115_KNEE,
                ".*ANKLE_Y": DAMPING_76_ANKLE_Y,
                ".*ANKLE_X": DAMPING_76_ANKLE_X,
            },
            velocity_limit_sim={
                ".*HIP_X": VELOCITY_LIMIT_85,
                ".*HIP_Z": VELOCITY_LIMIT_68,
                ".*HIP_Y": VELOCITY_LIMIT_115,
                ".*KNEE_Y": VELOCITY_LIMIT_115,
                ".*ANKLE_Y": VELOCITY_LIMIT_ANKLE_Y,
                ".*ANKLE_X": VELOCITY_LIMIT_ANKLE_X,
            },
            armature={
                ".*HIP_X": ARMATURE_85,
                ".*HIP_Z": ARMATURE_68,
                ".*HIP_Y": ARMATURE_115,
                ".*KNEE_Y": ARMATURE_115,
                ".*ANKLE_Y": ARMATURE_ANKLE_Y,
                ".*ANKLE_X": ARMATURE_ANKLE_X,
            },
            effort_limit_sim={
                ".*HIP_X": EFFORT_LIMIT_85,
                ".*HIP_Z": EFFORT_LIMIT_68,
                ".*HIP_Y": EFFORT_LIMIT_115,
                ".*KNEE_Y": EFFORT_LIMIT_115,
                ".*ANKLE_Y": EFFORT_LIMIT_ANKLE_Y,
                ".*ANKLE_X": EFFORT_LIMIT_ANKLE_X,
            }
        ),

        "torso": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=["SPINE_Z", "NECK_Z", "NECK_Y"],
            stiffness={
                "SPINE_Z": STIFFNESS_85_SPINE_Z,
                "NECK_Z": STIFFNESS_S_NECK_Z,
                "NECK_Y": STIFFNESS_S_NECK_Y,
            },
            damping={
                "SPINE_Z": DAMPING_85_SPINE_Z,
                "NECK_Z": DAMPING_S_NECK_Z,
                "NECK_Y": DAMPING_S_NECK_Y,
            },
            velocity_limit_sim={
                "SPINE_Z": VELOCITY_LIMIT_85,
                "NECK_Z": VELOCITY_LIMIT_S,
                "NECK_Y": VELOCITY_LIMIT_S,
            },
            armature={
                "SPINE_Z": ARMATURE_85,
                "NECK_Z": ARMATURE_S,
                "NECK_Y": ARMATURE_S,
            },
            effort_limit_sim={
                "SPINE_Z": EFFORT_LIMIT_85,
                "NECK_Z": EFFORT_LIMIT_S,
                "NECK_Y": EFFORT_LIMIT_S,
            }
        ),

        "arms": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*SHOULDER_Y",".*SHOULDER_X",".*SHOULDER_Z",".*ELBOW_Y",".*WRIST_Z",".*WRIST_X",".*GRIPPER_Z"],
            stiffness={
                ".*SHOULDER_Y": STIFFNESS_85_SHOULDER_Y,
                ".*SHOULDER_X": STIFFNESS_85_SHOULDER_X,
                ".*SHOULDER_Z": STIFFNESS_68_SHOULDER_Z,
                ".*ELBOW_Y": STIFFNESS_68_ELBOW_Y,
                ".*WRIST_Z": STIFFNESS_S_WRIST_Z,
                ".*WRIST_X": STIFFNESS_S_WRIST_X,
                ".*GRIPPER_Z": STIFFNESS_S_GRIPPER_Z,
            },
            damping={
                ".*SHOULDER_Y": DAMPING_85_SHOULDER_Y,
                ".*SHOULDER_X": DAMPING_85_SHOULDER_X,
                ".*SHOULDER_Z": DAMPING_68_SHOULDER_Z,
                ".*ELBOW_Y": DAMPING_68_ELBOW_Y,
                ".*WRIST_Z": DAMPING_S_WRIST_Z,
                ".*WRIST_X": DAMPING_S_WRIST_X,
                ".*GRIPPER_Z": DAMPING_S_GRIPPER_Z,
            },
            velocity_limit_sim={
                ".*SHOULDER_Y": VELOCITY_LIMIT_85,
                ".*SHOULDER_X": VELOCITY_LIMIT_85,
                ".*SHOULDER_Z": VELOCITY_LIMIT_68,
                ".*ELBOW_Y": VELOCITY_LIMIT_68,
                ".*WRIST_Z": VELOCITY_LIMIT_S,
                ".*WRIST_X": VELOCITY_LIMIT_S,
                ".*GRIPPER_Z": VELOCITY_LIMIT_S,
            },
            armature={
                ".*SHOULDER_Y": ARMATURE_85,
                ".*SHOULDER_X": ARMATURE_85,
                ".*SHOULDER_Z": ARMATURE_68,
                ".*ELBOW_Y": ARMATURE_68,
                ".*WRIST_Z": ARMATURE_S,
                ".*WRIST_X": ARMATURE_S,
                ".*GRIPPER_Z": ARMATURE_S,
            },
            effort_limit_sim={
                ".*SHOULDER_Y": EFFORT_LIMIT_85,
                ".*SHOULDER_X": EFFORT_LIMIT_85,
                ".*SHOULDER_Z": EFFORT_LIMIT_68,
                ".*ELBOW_Y": EFFORT_LIMIT_68,
                ".*WRIST_Z": EFFORT_LIMIT_S,
                ".*WRIST_X": EFFORT_LIMIT_S,
                ".*GRIPPER_Z": EFFORT_LIMIT_S,
            }
        ),
    },   
)




ALEX_V1_NUBS_DEFAULT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path="",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*HIP_X", ".*HIP_Z", ".*HIP_Y", ".*KNEE_Y", ".*ANKLE_Y", ".*ANKLE_X"],
            stiffness={
                ".*HIP_X": STIFFNESS_85_HIP_X,
                ".*HIP_Z": STIFFNESS_68_HIP_Z,
                ".*HIP_Y": STIFFNESS_115_HIP_Y,
                ".*KNEE_Y": STIFFNESS_115_KNEE,
                ".*ANKLE_Y": STIFFNESS_76_ANKLE_Y,
                ".*ANKLE_X": STIFFNESS_76_ANKLE_X,
            },
            damping={
                ".*HIP_X": DAMPING_85_HIP_X,
                ".*HIP_Z": DAMPING_68_HIP_Z,
                ".*HIP_Y": DAMPING_115_HIP_Y,
                ".*KNEE_Y": DAMPING_115_KNEE,
                ".*ANKLE_Y": DAMPING_76_ANKLE_Y,
                ".*ANKLE_X": DAMPING_76_ANKLE_X,
            },
            velocity_limit_sim={
                ".*HIP_X": VELOCITY_LIMIT_85,
                ".*HIP_Z": VELOCITY_LIMIT_68,
                ".*HIP_Y": VELOCITY_LIMIT_115,
                ".*KNEE_Y": VELOCITY_LIMIT_115,
                ".*ANKLE_Y": VELOCITY_LIMIT_ANKLE_Y,
                ".*ANKLE_X": VELOCITY_LIMIT_ANKLE_X,
            },
            armature={
                ".*HIP_X": ARMATURE_85,
                ".*HIP_Z": ARMATURE_68,
                ".*HIP_Y": ARMATURE_115,
                ".*KNEE_Y": ARMATURE_115,
                ".*ANKLE_Y": ARMATURE_ANKLE_Y,
                ".*ANKLE_X": ARMATURE_ANKLE_X,
            },
            effort_limit_sim={
                ".*HIP_X": EFFORT_LIMIT_85,
                ".*HIP_Z": EFFORT_LIMIT_68,
                ".*HIP_Y": EFFORT_LIMIT_115,
                ".*KNEE_Y": EFFORT_LIMIT_115,
                ".*ANKLE_Y": EFFORT_LIMIT_ANKLE_Y,
                ".*ANKLE_X": EFFORT_LIMIT_ANKLE_X,
            }
        ),

        "torso": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=["SPINE_Z", "NECK_Z", "NECK_Y"],
            stiffness={
                "SPINE_Z": STIFFNESS_85_SPINE_Z,
                "NECK_Z": STIFFNESS_S_NECK_Z,
                "NECK_Y": STIFFNESS_S_NECK_Y,
            },
            damping={
                "SPINE_Z": DAMPING_85_SPINE_Z,
                "NECK_Z": DAMPING_S_NECK_Z,
                "NECK_Y": DAMPING_S_NECK_Y,
            },
            velocity_limit_sim={
                "SPINE_Z": VELOCITY_LIMIT_85,
                "NECK_Z": VELOCITY_LIMIT_S,
                "NECK_Y": VELOCITY_LIMIT_S,
            },
            armature={
                "SPINE_Z": ARMATURE_85,
                "NECK_Z": ARMATURE_S,
                "NECK_Y": ARMATURE_S,
            },
            effort_limit_sim={
                "SPINE_Z": EFFORT_LIMIT_85,
                "NECK_Z": EFFORT_LIMIT_S,
                "NECK_Y": EFFORT_LIMIT_S,
            }
        ),

        "arms": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*SHOULDER_Y",".*SHOULDER_X",".*SHOULDER_Z",".*ELBOW_Y"],
            stiffness={
                ".*SHOULDER_Y": STIFFNESS_85_SHOULDER_Y,
                ".*SHOULDER_X": STIFFNESS_85_SHOULDER_X,
                ".*SHOULDER_Z": STIFFNESS_68_SHOULDER_Z,
                ".*ELBOW_Y": STIFFNESS_68_ELBOW_Y,
            },
            damping={
                ".*SHOULDER_Y": DAMPING_85_SHOULDER_Y,
                ".*SHOULDER_X": DAMPING_85_SHOULDER_X,
                ".*SHOULDER_Z": DAMPING_68_SHOULDER_Z,
                ".*ELBOW_Y": DAMPING_68_ELBOW_Y,
            },
            velocity_limit_sim={
                ".*SHOULDER_Y": VELOCITY_LIMIT_85,
                ".*SHOULDER_X": VELOCITY_LIMIT_85,
                ".*SHOULDER_Z": VELOCITY_LIMIT_68,
                ".*ELBOW_Y": VELOCITY_LIMIT_68,
            },
            armature={
                ".*SHOULDER_Y": ARMATURE_85,
                ".*SHOULDER_X": ARMATURE_85,
                ".*SHOULDER_Z": ARMATURE_68,
                ".*ELBOW_Y": ARMATURE_68,
            },
            effort_limit_sim={
                ".*SHOULDER_Y": EFFORT_LIMIT_85,
                ".*SHOULDER_X": EFFORT_LIMIT_85,
                ".*SHOULDER_Z": EFFORT_LIMIT_68,
                ".*ELBOW_Y": EFFORT_LIMIT_68,
            }
        ),
    },   
)


VARIABLE_ACTION_SCALE = {}
for a in ALEX_V1_NUBS_DEFAULT_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            VARIABLE_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

pass