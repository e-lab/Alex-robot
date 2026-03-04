# alex_V1 IsaacSim MJCF

Import this file in IsaacSim MJCF importer:
- `alex_v1_full_body_isaacsim.xml`

## Conversion notes
This MJCF variant is derived from:
- `../alex_V1_description/mjcf/alex_v1_full_body_mjx.xml`

Changes made for IsaacSim MJCF importer compatibility:
- `compiler eulerseq` changed from `XYZ` to intrinsic `xyz`.
- Mesh file paths updated from `../meshes/...` to `../alex_V1_description/meshes/...` so they resolve from this folder.
- Collision geoms that relied on class defaults now have explicit primitive definitions (`type`, `size`, `fromto`, etc.) to avoid importer fallback-to-sphere behavior.
- Base collision default now has an explicit tiny sphere primitive to avoid "Could not determine geometry type" warnings on class default geoms.

## Known non-blocking warning
If IsaacSim logs a missing icon PNG for the importer extension (`icoFileMJCF.png`), that is an extension UI asset warning and does not affect model import fidelity.
