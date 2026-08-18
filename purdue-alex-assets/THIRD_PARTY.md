# Third-party assets

Only sources currently used by this repository are listed. Exact imported-file hashes and generation details remain in each asset's `dependency.toml` and `SOURCE_SHA256SUMS` files.

| Component | Upstream and pinned revision | License | Local scope and notice |
| --- | --- | --- | --- |
| IHMC Alex V2 and Purdue | [`ihmcrobotics/ihmc-alex-sdk`](https://github.com/ihmcrobotics/ihmc-alex-sdk): V2 release `0.4.0` at `0789e4d70a20128b5aad5f7071b13d94abe746fd`; Purdue at `6e25f0cdc853172b4bc773dcabd1646e52e5dcc8` | Apache-2.0, root `LICENSE` | Alex URDF/mesh assets and Purdue pedestal and sensor frames; portable profiles and convex meshes are local derivatives |
| SAKE EZGripper | `ihmc_hands_ros2` submodule at `0a9fe5d2a47995f841854ca430c027bdf88bc22c` | Upstream terms | Dual EZGripper Gen2 fragments and OBJ/MTL meshes plus local convex collision derivatives |
| WEISS WSG 32 and UMI v1 | [`si-machines/wsg_32_description`](https://github.com/si-machines/wsg_32_description) at `8db8edb21d526967a5eebf0599fc4520ff2a0b51`; [`actuated-umi/actuated-umi-gripper`](https://github.com/actuated-umi/actuated-umi-gripper) at `8e9e9572a9021aab0a06f049f4bce96fe4de8ab4` | BSD-3-Clause and MIT; copied below `assets/end_effectors/weiss_wsg32/source/` | Immutable WSG inputs and selected UMI v1 finger; the qualified URDF and convex colliders are local derivatives |
| Stereolabs ZED X Mini | [`stereolabs/zed-isaac-sim`](https://github.com/stereolabs/zed-isaac-sim) at `504b183117560d19060fdeb6e531bec0ef510e0d` | Upstream terms | No Stereolabs files are distributed; the pinned USD is prepared under ignored `build/dependencies/` |
| RealSense D405/D435 | [`realsenseai/realsense-ros`](https://github.com/realsenseai/realsense-ros) tag `4.58.3`, commit `60c850958d651130fc2cc3d10efb37ff5be93da5`; build-only Xacro tag `2.1.1` at `390772abfe1e068f54aed674ce43873229a7db4e` | Apache-2.0, copied at `assets/sensors/realsense/LICENSE` | Official description inputs and portable URDFs; the D435 mesh contains only the documented Isaac material-name compatibility edit |
| LEAP Hand V1 | [`LEAP_Hand_Sim`](https://github.com/leap-hand/LEAP_Hand_Sim) at `150bc3d4b61fd6619193ba5a8ef209f3609ced89`; Isaac reference `c576e6a83183ba504aef8f57015f2899b799db1a`; API reference `b0d00c881d0119077b2cab771a6de44f5aaec904` | Conservatively classified `CC BY-NC-SA`, version unspecified; Sim repository MIT text retained | Official left/right URDFs and referenced STL meshes; see `assets/end_effectors/leap_hand_v1/LICENSE_ASSET_NOTICE.md` |
| Isaac Lab package | Local `src/ihmc_alex_isaaclab/` | BSD-3-Clause, root `LICENSE-BSD-3-Clause` | Generic reusable configuration and validation code |

LEAP's Sim repository root license and official CAD licensing statement conflict. This repository applies the more restrictive CAD classification without inventing a version; the complete rationale and retained MIT text remain beside the asset.
