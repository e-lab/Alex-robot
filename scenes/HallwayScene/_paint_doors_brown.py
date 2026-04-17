"""One-off editor: paint door panels brown in Hallway.usdc.

Adds a UsdPreviewSurface 'BrownDoor' material under /Looks and binds it to every
`.../DoorObject/Door/Cylinder_001` mesh (the door panel, not the frame).
Safe to run multiple times — reuses an existing material at that path.

Run once:
    python3 _paint_doors_brown.py
"""

from __future__ import annotations
import pathlib
from pxr import Usd, UsdShade, Sdf, Gf

USD_PATH = pathlib.Path(__file__).resolve().parent / "Hallway.usdc"
MAT_PATH = "/Looks/BrownDoor"
# Saddle-brown UsdPreviewSurface
COLOUR    = Gf.Vec3f(0.45, 0.26, 0.13)
ROUGHNESS = 0.7
METALLIC  = 0.0


def main() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))

    # Ensure /Looks exists (it already does in this file, but be safe).
    if not stage.GetPrimAtPath("/Looks").IsValid():
        stage.DefinePrim("/Looks", "Scope")

    # Create or reuse the material + shader.
    mat_prim = stage.DefinePrim(MAT_PATH, "Material")
    mat = UsdShade.Material(mat_prim)

    shader_path = Sdf.Path(MAT_PATH).AppendChild("PBR")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(COLOUR)
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float  ).Set(ROUGHNESS)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float  ).Set(METALLIC)

    # Wire shader to the material's surface output
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Bind to every door panel mesh (not frames).
    bound = 0
    for door_name in ("DoorObject", "DoorObject_01", "DoorObject_02", "DoorObject_03"):
        mesh_path = f"/{door_name}/DoorObject/Door/Cylinder_001"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            print(f"  WARN: mesh not found: {mesh_path}")
            continue
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)
        bound += 1

    stage.GetRootLayer().Save()
    print(f"Painted {bound} door panels brown. Saved {USD_PATH}")


if __name__ == "__main__":
    main()
