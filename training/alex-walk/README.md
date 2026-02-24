# Alex Humanoid-v5 Training Integration

## Request
Make `train.py` work with the Alex robot model at:

`alex-models/alex_V1_description/mjcf/alex_v1_full_body_mjx.xml`

while keeping behavior as close as possible to `Humanoid-v5` training, and using the same effective training scripts.

Additional constraint requested:
- Use Humanoid-style torque motor control.
- Only move the Humanoid-analog motors.
- Fix all other Alex motors/joints.

## What Was Done

### 1. New trainable Alex XML created
A new derived model was generated:

`alex-models/alex_V1_description/mjcf/alex_v1_humanoid_train.xml`

This file is intended for RL training (source CAD-like XML is left intact).

### 2. MuJoCo compatibility fixed
The original Alex MJCF failed to compile due to invalid inertia definitions.
The derived training XML now compiles in MuJoCo.

Changes applied in the training XML:
- Inertia values transformed/sanitized for MuJoCo.
- `balanceinertia="true"` enabled on `<compiler>`.
- Floor plane added for locomotion training.
- Physics options aligned closer to Humanoid-v5 style (`RK4`, `dt=0.003`, `iterations=50`, `solver=PGS`).

### 3. Humanoid-style torque motor control applied
Actuators were replaced with MuJoCo `motor` actuators using:
- `ctrllimited="true"`
- `ctrlrange="-0.4 0.4"`
- Humanoid-like gear grouping (torso/legs/arms)

### 4. Only Humanoid-analog joints are actuated
Active (actuated) joints in Alex training XML:
- `spine_z`
- `left_hip_x`, `left_hip_z`, `left_hip_y`, `left_knee_y`
- `right_hip_x`, `right_hip_z`, `right_hip_y`, `right_knee_y`
- `left_shoulder_y`, `left_shoulder_x`, `left_elbow_y`
- `right_shoulder_y`, `right_shoulder_x`, `right_elbow_y`

Total active actuators: **15**.

### 5. Non-selected joints fixed
Non-humanoid joints are locked by setting `range="0 0"` (with autolimits enabled), including:
- ankles
- shoulder z joints
- wrists
- neck

### 6. Training/eval scripts wired to Alex XML
`train.py` and `eval.py` now use:
- `ENV_NAME = "Humanoid-v5"`
- `xml_file=<absolute path to alex_v1_humanoid_train.xml>`

This keeps Gymnasium Humanoid-v5 environment/reward semantics while swapping in Alex morphology.

### 7. Reproducible build script added
Transformation script:

`tools/build_alex_humanoid_train_xml.py`

This script regenerates the train XML from the source Alex XML.

## Important Note on 17 vs 15 Actuators
Humanoid-v5 has 17 actuators, but this Alex model provides one spine DOF (not three abdomen DOFs like Humanoid).
With the “only humanoid-like joints” and “fix everything else” constraints, the closest mapping is **15 actuators**.

If exact 17-action compatibility is required, the model would need structural kinematic changes (e.g., adding torso DOFs) beyond simple actuator remapping.

## Files Changed
- `train.py`
- `eval.py`
- `alex-models/alex_V1_description/mjcf/alex_v1_full_body_mjx.xml` (syntax fix)
- `alex-models/alex_V1_description/mjcf/alex_v1_humanoid_train.xml` (new)
- `tools/build_alex_humanoid_train_xml.py` (new)
- `PLAN.md`

## How To Run


### 1) Train
```bash
python train.py
```

Outputs are saved under:
- `rl_models/checkpoints/`
- `rl_models/best/`
- `rl_models/ppo_humanoid_final.zip`
- `rl_models/vec_normalize_final.pkl`

### 2) tensorboard

`tensorboard --logdir rl_models/tensorboard`


### 3) Evaluate
```bash
python eval.py --episodes 5
```

Headless evaluation:
```bash
python eval.py --episodes 5 --no-render
```

## Expected Runtime Behavior
- Observation size differs from stock Humanoid because Alex has different body/joint counts.
- Action space is `(15,)` for this Alex-humanoid mapping.
- Existing PPO setup remains unchanged aside from XML override.
