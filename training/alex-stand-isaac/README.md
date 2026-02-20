# training/alex-stand-isaac/

MuJoCo training script that ports the `isaacsimlab/` configuration into
a standalone PPO script, mirroring `training/alex-stand/alex-stand-ppo.py`
as the base structure.

---

## Files

| File | Purpose |
|------|---------|
| `alex-stand-isaac-ppo.py` | **Main training script** — MuJoCo env + SB3 PPO with IsaacLab config |
| `alex.py` | Robot hardware constants (actuators, limits, timing) from `isaacsimlab/alex.py` |
| `alex_stand_env_cfg.py` | IsaacLab `@configclass` env config (requires IsaacLab installed) |
| `rsl_rl_ppo_cfg.py` | IsaacLab RSL-RL runner config (requires IsaacLab installed) |

`alex-stand-isaac-ppo.py` runs with only MuJoCo + stable-baselines3 and
does not require Isaac Sim or IsaacLab.

---

## Usage

```bash
# Train
python training/alex-stand-isaac/alex-stand-isaac-ppo.py

# Evaluate
python training/alex-stand-isaac/alex-stand-isaac-ppo.py \
    --eval rl_models/best/best_model \
    --vec-norm rl_models/vec_normalize_final.pkl

# TensorBoard
tensorboard --logdir training/alex-stand-isaac/rl_models/tensorboard
```

---

## Changes from `training/alex-stand/alex-stand-ppo.py`

Every meaningful difference is listed below, traced back to the
`isaacsimlab/` config it came from.

### 1. Initial standing pose

**Source:** `isaacsimlab/alex_stand_env_cfg.py` → `init_state.joint_pos`

| Joint | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` |
|-------|---------------------|---------------------------|
| `hip_y` | +0.05 rad (near-upright) | −0.772 rad (≈ −44°, bent knee) |
| `knee` | +0.10 rad (nearly straight) | +1.419 rad (≈ +81°, bent) |
| `ankle_y` | −0.03 rad | −0.634 rad (≈ −36°) |
| `shoulder_y` | +0.30 rad | +0.15 rad |
| `elbow` | −0.80 rad | −0.50 rad |

The IsaacLab pose is a deeper squat, matching the production standing
config used for GPU-parallel training.

### 2. Timing & decimation

**Source:** `isaacsimlab/alex.py` → `CONTROL_DT`, `SIM_DT`

| Aspect | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` |
|--------|---------------------|---------------------------|
| Sim timestep | 5 ms (from MJCF) | 5 ms (from MJCF) |
| Control rate | 200 Hz (1 sim step / step) | **50 Hz** (4 sim steps / step) |
| `decimation` | 1 | **4** |
| Episode length | 5000 steps (25 s) | 1000 steps (20 s) |

Running 4 physics steps per policy step matches IsaacLab's
`CONTROL_DT=0.02 / SIM_DT=0.005 = 4`.

### 3. Action delay buffer

**Source:** `isaacsimlab/alex.py` → `DelayedPDActuatorCfg(min_delay=0, max_delay=2)`

```
MIN_DELAY_STEPS = floor(0.004 / 0.005) = 0
MAX_DELAY_STEPS = ceil( 0.008 / 0.005) = 2
```

Each episode the policy's desired joint positions are delayed by a
uniformly sampled 0–2 sim steps (0–10 ms) before being written to
`data.ctrl`. This simulates real actuator communication latency and
improves sim-to-real transfer.

Not present in `alex-stand-ppo.py`.

### 4. Observations

**Source:** `isaacsimlab/walk_flat_env_cfg.py` → `AlexObservationsCfg.PolicyCfg`

| Component | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` |
|-----------|---------------------|---------------------------|
| Orientation | Quaternion wxyz (4D) | **Gravity projection** (3D) in pelvis frame |
| Angular velocity | World frame | **Body frame** (R^T @ ω_world) |
| Height command | None | **Target height** (sampled 0.65–0.95 m) |
| Pelvis z | Absolute | Absolute |
| Joint positions | Offset from stand pose | Offset from stand pose |
| Joint velocities | Raw | Raw |
| Previous action | Yes | Yes |
| **Total dims** | 92 | ~95 (depends on joint count) |

Gravity projection `R^T @ [0,0,-1]` is the IsaacLab `projected_gravity`
observation term. It replaces the quaternion: rotation-invariant, no
discontinuities, compact (3D).

Angular velocity expressed in body frame matches IsaacLab's
`base_ang_vel` observation (IsaacLab reports this in the base frame).

### 5. Reward structure

**Source:** `isaacsimlab/walk_flat_env_cfg.py` → `AlexRewards`,
`isaacsimlab/alex_stand_env_cfg.py` → body/joint name overrides

| Reward term | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` | IsaacLab weight |
|-------------|---------------------|---------------------------|-----------------|
| Height (exp, std=0.05) | Gaussian z, w=5.0 | `W_HEIGHT=3.0` | 3.0 |
| Flat orientation (exp, std=0.2) | `W_UPRIGHT=3.0 × rot_z_z` | `W_ORIENTATION=2.0` | 2.0 |
| Alive bonus | `W_ALIVE=1.0` | `W_ALIVE=0.5` | — |
| Action rate L2 | `−W_SMOOTH=−0.50` | `W_ACTION_RATE=−0.10` | −0.1 |
| Lin vel Z L2 | `−W_LIN_VEL=−0.20 × ‖v‖²` | `W_LIN_VEL_Z=−0.10 × vz²` | −0.1 |
| Ang vel XY L2 | `−W_ANG_VEL=−0.20 × ‖ω‖²` | `W_ANG_VEL_XY=−0.20 × (ωx²+ωy²)` | −0.2 |
| Joint velocity L2 | `−W_JOINT_VEL=−0.05` | `W_JOINT_VEL=−5e-6` | −5e-6 |
| Joint torque L2 | Not present | `W_JOINT_TORQUE=−1e-5` | −1e-5 |
| Arms deviation L1 | Combined pose `−W_POSE=−0.50` | `W_JOINT_ARMS=−0.60` | −0.6 |
| Torso deviation L1 | Combined pose | `W_JOINT_TORSO=−0.60` | −0.6 |
| Hip yaw deviation L1 | Combined pose | `W_JOINT_HIP_YAW=−1.50` | −1.5 |
| Ankle roll torque L2 | Not present | `W_ANKLE_TORQUE=−2e-5` | −2e-5 |
| Foot sliding | Not present | `W_FOOT_SLIDE=−0.50` | present |
| Undesired contacts | Not present | `W_UNDESIRED_CONT=−0.10` | −0.1 |
| Fall penalty | −10.0 | −200.0 (`death` term) | −200 |

Key changes:
- **Height reward** shape changed from raw Gaussian to exponential with
  std=0.05 (IsaacLab `track_lin_pos_z_exp`) centered on a **sampled
  target height** (not fixed at init height).
- **Orientation** uses gravity projection error squared (IsaacLab
  `flat_orientation_exp` std=0.2) instead of raw `rot[2,2]`.
- **Pose tracking** split into per-group L1 terms (arms, torso, hip yaw)
  instead of a single combined L2 term.
- **Foot sliding** and **undesired contacts** added from IsaacLab.
- **Fall penalty** increased from −10 to −200 to match IsaacLab's
  `death` term weight.

### 6. Domain randomisation

**Source:** `isaacsimlab/alex_stand_env_cfg.py` → `events`

| Event | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` | IsaacLab config |
|-------|---------------------|---------------------------|-----------------|
| Joint reset range | ±0.03 rad | **±0.70 rad** | `position_range=(-0.7, 0.7)` |
| Base position | ±0.01 m z, ±0.08 rad yaw | **±0.5 m x/y, ±π yaw** | `reset_base` event |
| Mass randomisation | None | **±2 kg on torso** | `add_base_mass (-2,2)` |
| Push disturbances | None | **±0.7 m/s x/y, ±0.2 rad/s yaw every 2–4 s** | `push_robot` event |

### 7. Height command

**Source:** `isaacsimlab/alex_stand_env_cfg.py` → `commands.base_height`

`alex-stand-ppo.py` always targets the fixed init height.
`alex-stand-isaac-ppo.py` samples a random target height from
`[0.65, 0.95]` m at each episode reset (matching `lin_pos_z` range)
and includes it as an observation. The height reward is computed
relative to this sampled target.

### 8. PPO hyperparameters

**Source:** `isaacsimlab/rsl_rl_ppo_cfg.py` → `AlexStandingEnvPPORunnerCfg`

| Hyperparameter | `alex-stand-ppo.py` | `alex-stand-isaac-ppo.py` | RSL-RL config |
|----------------|---------------------|---------------------------|---------------|
| Network | [256, 256] 2 layers | **[128, 128, 128] 3 layers** | [128,128,128] |
| Learning rate | 3e-4 fixed | **1e-3 fixed** | 1e-3 adaptive |
| n_epochs | 10 | **5** | num_learning_epochs=5 |
| Mini-batches | ~1 (batch=512) | **4** | num_mini_batches=4 |
| Entropy coeff | 0.005 | **0.01** | entropy_coef=0.01 |
| Value loss coeff | 0.5 | **1.0** | value_loss_coef=1.0 |
| Gradient clip | 0.5 | **1.0** | max_grad_norm=1.0 |
| gamma | 0.99 | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 | 0.95 |
| clip_range | 0.2 | 0.2 | clip_param=0.2 |

`n_steps=512` (vs RSL-RL's 24) compensates for using 8 CPU envs instead
of thousands of GPU envs, keeping the total per-iteration batch size
reasonable.

### 9. What was NOT changed

- MuJoCo scene file (`scenes/alex-scenes/scene_alex_v1_train.xml`)
- Actuator type (position actuators, kp/kd from MJCF)
- Action scale: `ACTION_SCALE = 0.30` rad = `ALEX_JOINT_SCALE`
- SB3 VecNormalize (obs + reward normalisation)
- Checkpoint and evaluation callbacks structure
- `_find_floor_height()` helper (auto-calibrate init pelvis z)
