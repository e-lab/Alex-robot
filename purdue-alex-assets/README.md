# IHMC Alex robot assets

This repository provides simulator-neutral IHMC Alex robot descriptions and a
reusable Isaac Lab Python package. It owns reusable assets, default actuator and
collision configuration, sensor/tool frames, and the measured Purdue Alex003
reference setup. Task scenes, controllers, policies, calibration, datasets,
and training outputs belong in consuming projects.

## Available assets

| Asset | Maintained variants | Status |
| --- | --- | --- |
| Alex V2 | `standard`, `forearm_convex`, `full_convex` | Portable URDFs; Isaac Lab PhysX/TGS validated |
| Alex Purdue + SAKE EZGripper | `source`, `full_convex` | Default fixed-base Purdue configuration; simulation ready |
| Alex Purdue + WSG32/UMI v1 | `source`, `full_convex` | Opt-in Golden Robot profile; simulation-training ready |
| LEAP Hand V1 | Independent left and right URDFs | Isaac Lab validated; Alex wrist transforms are not calibrated |
| ZED X Mini Wide | Pinned external Stereolabs USD | Opt-in Isaac Lab RGB/depth integration |
| RealSense | D405 and D435 | Tracked portable URDFs and opt-in Isaac Lab RGB/depth integration |
| Purdue Alex003 pedestal | Standalone URDF and reference scene | Measured collision geometry with approximate visuals |

Canonical robot files:

- Alex V2: `assets/robots/alex_v2/urdf/alex_v2.urdf`
- Alex Purdue: `assets/robots/alex_purdue/urdf/baseline/alex_purdue_full_convex.urdf`
- Alex Purdue WSG32/UMI: `assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf`

## Layout

- `assets/robots/`: complete robot descriptions and robot-owned geometry.
- `assets/end_effectors/`: SAKE, WSG32/UMI, and LEAP Hand assets.
- `assets/sensors/`: RealSense descriptions and pinned ZED dependency data.
- `assets/platforms/`: Purdue pedestal assets.
- `measurements.yaml`: physical observations for the Purdue Alex003 setup.
- `src/ihmc_alex_isaaclab/`: reusable Isaac Lab Python package.
- `scripts/generate/`: deterministic asset generators and dependency preparation.
- `scripts/validate/`: deterministic and GPU-capable validation entry points.
- `tests/`: asset and configuration contracts.
- `build/`: ignored generated USD, validation output, and external dependencies.

Each physical component owns its source and derived geometry. Complete robot
profiles reference those assets through relative paths. The old root-level
`urdf/` and `meshes/` layout is preserved only by the
`v0.1-alex-purdue-baseline` tag; current consumers must use the paths above or
the public factories.

## Install and use

The package targets Isaac Sim `6.0.1`, Isaac Lab `release/3.0.0-beta2`, and
Python 3.12. Install it with the Python selected by Isaac Lab:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e \
  /path/to/Alex
```

Create independent robot configurations from the public factories:

```python
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
from ihmc_alex_isaaclab.robots.alex_v2 import make_alex_v2_cfg

alex_v2 = make_alex_v2_cfg(fix_base=False, variant="standard")
alex_purdue = make_alex_purdue_cfg(fix_base=True, variant="full_convex")
alex_wsg = make_alex_purdue_cfg(
    fix_base=True,
    variant="full_convex",
    end_effector="wsg32_umi_v1",
)
```

Import each public API from its owning module:

```python
from ihmc_alex_isaaclab.end_effectors.leap_hand_v1 import (
    author_leap_hand_v1_mount,
    make_leap_hand_v1_cfg,
)
from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import make_wsg32_umi_v1_cfg
from ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal import (
    make_purdue_alex003_pedestal_cfg,
)
from ihmc_alex_isaaclab.scenes.purdue_alex003 import (
    make_purdue_alex003_reference_scene_cfg,
)
from ihmc_alex_isaaclab.sensors.realsense import (
    author_realsense_mount,
    make_realsense_d405_cfgs,
    make_realsense_d435_cfgs,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini import (
    author_alex_purdue_zed_x_mini_mount,
    make_zed_x_mini_cfgs,
)
```

Mount helpers require concrete parent prims and run after scene construction,
before `simulation.reset()`. LEAP and RealSense mounts require explicit
parent-local poses; this repository does not invent physical extrinsics.

Prepare the pinned external ZED asset when that integration is needed:

```bash
python3 scripts/generate/prepare_zed_x_mini.py
```

## Sources of truth

- Upstream fragments below asset `source/` directories remain immutable.
- Generated Purdue baselines and portable sensor/hand URDFs must be changed
  through their generators, never edited directly.
- `dependency.toml` and `SOURCE_SHA256SUMS` files record pinned revisions and
  exact imported bytes.
- `measurements.yaml` is the physical reference record. It contains only
  observed setup facts; loaders derive geometry and alignment from those
  measurements and the canonical URDFs. Simulation parameters and validation
  results do not belong there.
- URDF and source meshes are authoritative. Generated USD stays below the
  ignored `build/` tree.
- PhysX/TGS on `cuda:0` is the simulation-readiness authority. CPU fallback is
  not an equivalent readiness result.

## Validation

Run the complete deterministic validation after every change:

```bash
make test
```

Use the GPU gate for a changed simulation component and the release gate for
all maintained components:

```bash
make test-gpu COMPONENT=realsense
make test-release
```

`make test` runs Ruff, formatting, every deterministic generator check, and the
complete test suite. Valid GPU components are `alex_v2`, `alex_purdue`,
`wsg32`, `leap_hand`, `realsense`, `zed_x_mini`, and `alex003`.

Any failed or missing gate result is `NO-GO` for simulation readiness.

## Known gaps

- Measure the installed Alex-to-LEAP wrist transforms before creating a
  calibrated robot profile or making sim-to-real claims.
- Calibrate the WSG32 mount, TCP, finger mass, and contact parameters only when
  a consumer requires a physically calibrated digital twin or sim-to-real.
- Add new adapters, finger sets, variants, sensors, or measurement records only
  from identified hardware, explicit provenance, and a concrete consumer need.
## Credit

Developed and maintained by Patrizio Acquadro as part of his research work at
Purdue University eLab.

## Licensing and provenance

Repository-level IHMC assets use the Apache License 2.0 in [LICENSE](LICENSE).
The Python package uses [BSD 3-Clause](LICENSE-BSD-3-Clause). Imported assets
retain their upstream terms and notices, including the conservative LEAP Hand
classification. See [THIRD_PARTY.md](THIRD_PARTY.md) for the complete source,
revision, scope, and license inventory.
