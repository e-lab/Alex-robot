# PLAN: Make `train.py` Work With Alex While Staying Humanoid-v5 Compatible

## Goal
Train an Alex-looking robot with the same PPO training flow and hyperparameters currently used for `Humanoid-v5`, while keeping environment behavior as close as possible to Humanoid-v5 so existing scripts remain effective.

## What We Found (Current State)
- `train.py` and `eval.py` currently hardcode `ENV_NAME = "Humanoid-v5"` and do not pass `xml_file` overrides.
- This repo contains a local `humanoid_v5.py` implementation, but training currently uses Gymnasium’s registered `Humanoid-v5` env.
- Gymnasium `Humanoid-v5` supports custom MuJoCo XML via `xml_file`.
- Alex MJCF files (`alex_v1_full_body_mjx.xml` and scene variants) currently fail MuJoCo compilation:
  - Error: `inertia must have positive eigenvalues` (first failure at `pelvis_link`, then additional links after partial fixes).
- The Alex model is structurally different from Humanoid-v5:
  - Alex has 27 actuators and position actuators.
  - Humanoid-v5 reference model uses 17 torque motor actuators.
  - Physics options differ (`Euler`, `dt=0.002` vs Humanoid `RK4`, `dt=0.003`).

## Design Intent (To Match "Humanoid-v5 but Looks Like Alex")
1. Keep Humanoid-v5 task/reward/termination semantics.
2. Use an Alex-based MJCF that compiles in MuJoCo and is trainable by the same PPO scripts.
3. Align control topology and dynamics assumptions to Humanoid-v5 as closely as practical.
4. Minimize script changes: ideally keep current commands and training pipeline intact.

## Implementation Plan

### Phase 1: Build a MuJoCo-valid Alex training XML (hard blocker first)
1. Create a new derived training XML (do not overwrite source CAD MJCF), e.g. `alex-models/.../mjcf/alex_v1_humanoid_train.xml`.
2. Fix inertial definitions so MuJoCo compiles cleanly for all bodies.
3. Add/ensure scene essentials required for locomotion training:
- floor plane with contact/friction
- stable camera/light defaults
- deterministic start keyframe (optional but recommended)
4. Add a compile smoke test script/command that loads XML with `mujoco.MjModel.from_xml_path` and fails fast if invalid.

Acceptance gate:
- XML compiles with no MuJoCo warnings/errors.

### Phase 2: Align Alex model to Humanoid-v5 control assumptions
1. Match actuator intent to Humanoid-v5:
- prefer torque-like control interface (`motor`) over position servo for PPO policy outputs
- set control ranges and gearing comparable to Humanoid scale
2. Decide DOF strategy (primary design decision):
- Recommended: map to Humanoid-like 17 controlled DOFs (keep extra joints fixed/passive)
- Alternative: keep full Alex control set and retune PPO (less similar, higher risk)
3. Preserve humanoid-style locomotion structure for reward signal quality:
- stable pelvis/root freejoint
- foot-ground contacts suitable for walking
- optional hip-knee tendons if needed to mirror Humanoid observation/info semantics
4. Tune healthy height range/start pose for Alex proportions so early episodes are not immediately unhealthy.

Acceptance gate:
- Environment resets and steps stably for 1k+ random actions without NaNs/explosions.
- Action/observation spaces are consistent and stationary across resets.

### Phase 3: Integrate with existing training/eval scripts with minimal disruption
1. Keep `ENV_NAME` as `Humanoid-v5`.
2. Pass `env_kwargs={"xml_file": <alex_training_xml_abs_or_repo_path>}` in both training and eval environment creation.
3. Keep PPO hyperparameters initially identical to current settings.
4. Ensure VecNormalize save/load paths remain unchanged and compatible.

Acceptance gate:
- `train.py` starts learning end-to-end with Alex XML and writes checkpoints.
- `eval.py` can load and run trained model with same XML config.

### Phase 4: Similarity hardening to Humanoid-v5 behavior
1. Compare Alex-vs-Humanoid diagnostics from short runs:
- reward term magnitudes (`reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`)
- episode length distribution and unhealthy termination rates
- COM forward velocity range
2. Adjust only model/env-level knobs first (actuator scales, friction, damping, healthy_z_range).
3. Change PPO hyperparameters only if model-level alignment is insufficient.

Acceptance gate:
- Training curves are stable and non-degenerate by early milestones.
- Agent learns forward locomotion behavior comparable in pattern to Humanoid-v5 baseline.

### Phase 5: Verification and handoff
1. Run short smoke training (e.g., 100k-300k steps) for regression checks.
2. Run a longer training segment to confirm sustained learning.
3. Document exact XML and env settings used so training is reproducible.

Acceptance gate:
- Reproducible run commands and artifacts (`best_model.zip`, final checkpoint, VecNormalize stats) exist and evaluate successfully.

## Risks and Mitigations
- Invalid CAD-derived inertias cause immediate failure.
  - Mitigation: dedicated inertia sanitization + compile gate before any RL run.
- Control mismatch (position actuators vs Humanoid torque assumptions) can collapse training.
  - Mitigation: convert/control-map to Humanoid-like motor interface.
- Extra Alex DOFs can expand action space and destabilize PPO.
  - Mitigation: start with Humanoid-like 17 controlled DOFs.

## Deliverables After Implementation (Next Step, Not Done Yet)
- MuJoCo-valid Alex training XML aligned to Humanoid-v5 assumptions.
- Updated `train.py` and `eval.py` that run Humanoid-v5 with Alex XML via `xml_file`.
- Validation notes: compile check, env smoke test, short training sanity results.
