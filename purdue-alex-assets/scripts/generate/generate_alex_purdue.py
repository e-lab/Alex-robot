#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the portable Alex Purdue URDF profiles and convex colliders."""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPOSITORY_ROOT / "assets"
MEASUREMENTS_PATH = REPOSITORY_ROOT / "measurements.yaml"
ALEX_PURDUE_ROOT = ASSETS_ROOT / "robots" / "alex_purdue"
SAKE_ROOT = ASSETS_ROOT / "end_effectors" / "sake_ezgripper"
WSG_ROOT = ASSETS_ROOT / "end_effectors" / "weiss_wsg32"
PEDESTAL_ROOT = ASSETS_ROOT / "platforms" / "purdue_pedestal"

FRAGMENT_DIR = ALEX_PURDUE_ROOT / "urdf" / "fragments"
BASELINE_DIR = ALEX_PURDUE_ROOT / "urdf" / "baseline"
ALEX_PURDUE_MESH_DIR = ALEX_PURDUE_ROOT / "meshes"
ALEX_PURDUE_COLLISION_DIR = ALEX_PURDUE_MESH_DIR / "collision"
SAKE_MESH_DIR = SAKE_ROOT / "meshes"
SAKE_COLLISION_DIR = SAKE_MESH_DIR / "collision"
PEDESTAL_MESH_DIR = PEDESTAL_ROOT / "meshes"
PEDESTAL_COLLISION_DIR = PEDESTAL_MESH_DIR / "collision"

SOURCE_PROFILE = BASELINE_DIR / "alex_purdue_full.urdf"
CONVEX_PROFILE = BASELINE_DIR / "alex_purdue_full_convex.urdf"
WSG_SOURCE_PROFILE = BASELINE_DIR / "alex_purdue_wsg32_umi_v1.urdf"
WSG_CONVEX_PROFILE = BASELINE_DIR / "alex_purdue_wsg32_umi_v1_full_convex.urdf"
WSG_STANDALONE_URDF = WSG_ROOT / "urdf" / "wsg32_umi_v1.urdf"

FRAGMENTS = (
    FRAGMENT_DIR / "alex_purdue.headTorso.urdf",
    FRAGMENT_DIR / "alex_purdue.leftUpperArm.urdf",
    FRAGMENT_DIR / "alex_purdue.leftForearm.urdf",
    FRAGMENT_DIR / "alex_purdue.leftEZGripperAdapter.urdf",
    SAKE_ROOT / "urdf" / "left_ezgripper_gen2.urdf",
    FRAGMENT_DIR / "alex_purdue.rightUpperArm.urdf",
    FRAGMENT_DIR / "alex_purdue.rightForearm.urdf",
    FRAGMENT_DIR / "alex_purdue.rightEZGripperAdapter.urdf",
    SAKE_ROOT / "urdf" / "right_ezgripper_gen2.urdf",
)

ROBOT_FRAGMENTS = (
    FRAGMENT_DIR / "alex_purdue.headTorso.urdf",
    FRAGMENT_DIR / "alex_purdue.leftUpperArm.urdf",
    FRAGMENT_DIR / "alex_purdue.leftForearm.urdf",
    FRAGMENT_DIR / "alex_purdue.rightUpperArm.urdf",
    FRAGMENT_DIR / "alex_purdue.rightForearm.urdf",
)

PACKAGE_MAPPINGS = {
    "package://alex_purdue_description/meshes/": "../../meshes/",
    "package://ezGripper/ezgripper_gen2/": "../../../../end_effectors/sake_ezgripper/meshes/",
}


@dataclass(frozen=True)
class Collider:
    source: Path
    output: Path
    urdf_reference: str


def _wsg_adapter_axial_length_m() -> float:
    record = yaml.safe_load(MEASUREMENTS_PATH.read_text(encoding="utf-8"))
    try:
        value = record["end_effectors"]["weiss_wsg32_068"]["alex_wrist_adapter"][
            "axial_length_m"
        ]["value"]
    except (KeyError, TypeError) as error:
        raise ValueError("measurements.yaml has no WSG adapter axial length") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
        raise ValueError("WSG adapter axial length must be a positive number")
    return float(value)


def _collider(
    source_dir: Path,
    collision_dir: Path,
    urdf_collision_dir: str,
    source_relative_path: str,
) -> Collider:
    filename = f"{Path(source_relative_path).stem}_convex.stl"
    return Collider(
        source=source_dir / source_relative_path,
        output=collision_dir / filename,
        urdf_reference=f"{urdf_collision_dir}/{filename}",
    )


def _alex_purdue_collider(source_relative_path: str) -> Collider:
    return _collider(
        ALEX_PURDUE_MESH_DIR,
        ALEX_PURDUE_COLLISION_DIR,
        "../../meshes/collision",
        source_relative_path,
    )


def _sake_collider(source_relative_path: str) -> Collider:
    return _collider(
        SAKE_MESH_DIR,
        SAKE_COLLISION_DIR,
        "../../../../end_effectors/sake_ezgripper/meshes/collision",
        source_relative_path,
    )


def _pedestal_collider(source_relative_path: str) -> Collider:
    return _collider(
        PEDESTAL_MESH_DIR,
        PEDESTAL_COLLISION_DIR,
        "../../../../platforms/purdue_pedestal/meshes/collision",
        source_relative_path,
    )


PHYSICAL_COLLIDERS = {
    "PEDESTAL_LINK": _pedestal_collider("pedestal_purdue.obj"),
    "TORSO_LINK": _alex_purdue_collider("torso_head/torso_purdue.obj"),
    "NECK_Z_LINK": _alex_purdue_collider("torso_head/neck_purdue.obj"),
    "LEFT_SHOULDER_Y_LINK": _alex_purdue_collider("left_arm/l_shoulder_y_purdue.obj"),
    "LEFT_SHOULDER_X_LINK": _alex_purdue_collider("left_arm/l_shoulder_x_purdue.obj"),
    "LEFT_SHOULDER_Z_LINK": _alex_purdue_collider("left_arm/l_shoulder_z_purdue.obj"),
    "LEFT_ELBOW_Y_LINK": _alex_purdue_collider("left_arm/l_elbow_y_purdue.obj"),
    "LEFT_WRIST_Z_LINK": _alex_purdue_collider("left_arm/l_wrist_z_purdue.obj"),
    "LEFT_WRIST_X_LINK": _alex_purdue_collider("left_arm/l_wrist_x_purdue.obj"),
    "LEFT_GRIPPER_Y_LINK": _alex_purdue_collider("left_arm/l_gripper_y_purdue.obj"),
    "RIGHT_SHOULDER_Y_LINK": _alex_purdue_collider("right_arm/r_shoulder_y_purdue.obj"),
    "RIGHT_SHOULDER_X_LINK": _alex_purdue_collider("right_arm/r_shoulder_x_purdue.obj"),
    "RIGHT_SHOULDER_Z_LINK": _alex_purdue_collider("right_arm/r_shoulder_z_purdue.obj"),
    "RIGHT_ELBOW_Y_LINK": _alex_purdue_collider("right_arm/r_elbow_y_purdue.obj"),
    "RIGHT_WRIST_Z_LINK": _alex_purdue_collider("right_arm/r_wrist_z_purdue.obj"),
    "RIGHT_WRIST_X_LINK": _alex_purdue_collider("right_arm/r_wrist_x_purdue.obj"),
    "RIGHT_GRIPPER_Y_LINK": _alex_purdue_collider("right_arm/r_gripper_y_purdue.obj"),
}

GRIPPER_COLLIDERS = {
    "left_ezgripper_palm_link": _sake_collider("SAKE_Palm_Dual_Gen2.obj"),
    "left_ezgripper_finger_l1_1": _sake_collider("SAKE_Finger_L1_Gen2.obj"),
    "left_ezgripper_finger_l1_2": _sake_collider("SAKE_Finger_L1_Gen2.obj"),
    "left_ezgripper_finger_l2_1": _sake_collider("SAKE_Finger_L2_Gen2.obj"),
    "left_ezgripper_finger_l2_2": _sake_collider("SAKE_Finger_L2_Gen2.obj"),
    "right_ezgripper_palm_link": _sake_collider("SAKE_Palm_Dual_Gen2.obj"),
    "right_ezgripper_finger_l1_1": _sake_collider("SAKE_Finger_L1_Gen2.obj"),
    "right_ezgripper_finger_l1_2": _sake_collider("SAKE_Finger_L1_Gen2.obj"),
    "right_ezgripper_finger_l2_1": _sake_collider("SAKE_Finger_L2_Gen2.obj"),
    "right_ezgripper_finger_l2_2": _sake_collider("SAKE_Finger_L2_Gen2.obj"),
}

TCP_X_M = 0.160804


def _parser() -> ET.XMLParser:
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def _load_fragment(path: Path) -> ET.Element:
    if not path.is_file():
        raise FileNotFoundError(f"missing Alex Purdue source fragment: {path}")
    root = ET.parse(path, parser=_parser()).getroot()
    if root.tag != "robot":
        raise ValueError(f"unexpected URDF root in {path}: {root.tag!r}")
    return root


def _rewrite_mesh_reference(filename: str) -> str:
    for package_prefix, relative_prefix in PACKAGE_MAPPINGS.items():
        if filename.startswith(package_prefix):
            return relative_prefix + filename.removeprefix(package_prefix)
    raise ValueError(f"unmapped mesh reference in Purdue source: {filename!r}")


def _remove_legacy_zed_visual(root: ET.Element) -> None:
    """Keep the frozen ZED frame while excluding its obsolete visual downstream."""

    link = root.find("link[@name='HEAD_ZED_X_MINI_LINK']")
    if link is None:
        raise ValueError("the expected upstream HEAD_ZED_X_MINI_LINK was not found")
    visuals = link.findall("visual")
    if len(visuals) != 1:
        raise ValueError(
            "the frozen upstream HEAD_ZED_X_MINI_LINK must contain exactly one visual"
        )
    mesh = visuals[0].find("geometry/mesh")
    expected = "package://alex_V1_description/meshes/ZEDXMini.obj"
    if mesh is None or mesh.get("filename") != expected:
        raise ValueError("the expected frozen upstream ZED X Mini visual was not found")
    link.remove(visuals[0])


def _add_tcp(root: ET.Element, side: str) -> None:
    side_lower = side.lower()
    palm_link = f"{side_lower}_ezgripper_palm_link"
    tcp_link = f"{side}_EZGRIPPER_TCP_LINK"
    joint = ET.SubElement(
        root, "joint", {"name": f"{side}_EZGRIPPER_TCP", "type": "fixed"}
    )
    ET.SubElement(joint, "origin", {"xyz": f"{TCP_X_M:.6f} 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": palm_link})
    ET.SubElement(joint, "child", {"link": tcp_link})
    ET.SubElement(root, "link", {"name": tcp_link})


def _validate_tree(
    root: ET.Element, *, expected_links: int = 41, expected_joints: int = 40
) -> None:
    links = [link.get("name", "") for link in root.findall("link")]
    joints = [joint.get("name", "") for joint in root.findall("joint")]
    if len(links) != expected_links or len(set(links)) != expected_links:
        raise ValueError(
            f"unexpected or duplicate Purdue links: count={len(links)}, unique={len(set(links))}"
        )
    if len(joints) != expected_joints or len(set(joints)) != expected_joints:
        raise ValueError(
            f"unexpected or duplicate Purdue joints: count={len(joints)}, unique={len(set(joints))}"
        )

    children = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')!r} is missing parent or child")
        if parent.get("link") not in links or child.get("link") not in links:
            raise ValueError(f"joint {joint.get('name')!r} references an unknown link")
        children.append(child.get("link", ""))
    roots = set(links) - set(children)
    if roots != {"PEDESTAL_LINK"}:
        raise ValueError(f"unexpected Purdue root links: {sorted(roots)}")

    package_references = [
        mesh.get("filename", "")
        for mesh in root.findall(".//mesh")
        if mesh.get("filename", "").startswith(("package://", "/"))
    ]
    if package_references:
        raise ValueError(f"non-portable mesh references remain: {package_references}")


def _build_source_profile() -> ET.Element:
    roots = [_load_fragment(path) for path in FRAGMENTS]
    root = roots[0]
    root.set("name", "AlexPurdue")
    for fragment in roots[1:]:
        root.extend(list(fragment))

    for element in list(root):
        if element.tag in {"gazebo", "transmission"}:
            root.remove(element)

    _remove_legacy_zed_visual(root)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("mesh element has no filename")
        mesh.set("filename", _rewrite_mesh_reference(filename))

    pedestal = root.find("link[@name='PEDESTAL_LINK']/visual/geometry/mesh")
    expected_pedestal_reference = "../../meshes/torso_head/pedestal_purdue.obj"
    if pedestal is None or pedestal.get("filename") != expected_pedestal_reference:
        raise ValueError("the expected upstream pedestal mesh reference was not found")
    pedestal.set(
        "filename",
        "../../../../platforms/purdue_pedestal/meshes/pedestal_purdue.obj",
    )

    right_gripper = root.find("link[@name='RIGHT_GRIPPER_Y_LINK']/visual/geometry/mesh")
    expected_wrong_reference = "../../meshes/left_arm/l_gripper_y_purdue.obj"
    if (
        right_gripper is None
        or right_gripper.get("filename") != expected_wrong_reference
    ):
        raise ValueError(
            "the expected upstream right-gripper mesh defect was not found"
        )
    right_gripper.set("filename", "../../meshes/right_arm/r_gripper_y_purdue.obj")

    _add_tcp(root, "LEFT")
    _add_tcp(root, "RIGHT")

    provenance = ET.Comment(
        " Derived from IHMC Alex SDK 6e25f0c. Vanilla fragments remain separate; "
        "this portable merge removes Gazebo/transmission blocks, rewrites mesh URIs, "
        "removes the obsolete ZED visual while preserving its frame, corrects the right "
        "terminal mesh reference, and adds nominal EZGripper TCP frames. "
    )
    root.insert(0, provenance)
    _validate_tree(root)
    return root


def _rewrite_wsg_mesh_reference(filename: str) -> str:
    path = (WSG_STANDALONE_URDF.parent / filename).resolve()
    try:
        relative = path.relative_to(WSG_ROOT)
    except ValueError as error:
        raise ValueError(f"WSG mesh escapes its asset root: {filename!r}") from error
    if not path.is_file():
        raise FileNotFoundError(f"missing qualified WSG mesh: {path}")
    return f"../../../../end_effectors/weiss_wsg32/{relative.as_posix()}"


def _append_wsg(
    root: ET.Element,
    standalone: ET.Element,
    *,
    side: str,
    include_materials: bool,
    adapter_axial_length_m: float,
) -> None:
    side_lower = side.lower()
    prefix = f"{side_lower}_"
    if include_materials:
        for material in standalone.findall("material"):
            root.append(copy.deepcopy(material))

    for source_link in standalone.findall("link"):
        link = copy.deepcopy(source_link)
        link.set("name", prefix + link.get("name", ""))
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                raise ValueError("qualified WSG mesh element has no filename")
            mesh.set("filename", _rewrite_wsg_mesh_reference(filename))
        root.append(link)

    for source_joint in standalone.findall("joint"):
        joint = copy.deepcopy(source_joint)
        joint.set("name", prefix + joint.get("name", ""))
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"incomplete qualified WSG joint: {joint.get('name')!r}")
        parent.set("link", prefix + parent.get("link", ""))
        child.set("link", prefix + child.get("link", ""))
        mimic = joint.find("mimic")
        if mimic is not None:
            mimic.set("joint", prefix + mimic.get("joint", ""))
        root.append(joint)

    mount = ET.SubElement(
        root,
        "joint",
        {"name": f"{side}_WSG32_MOUNT", "type": "fixed"},
    )
    ET.SubElement(
        mount,
        "origin",
        {
            "xyz": f"0 0 {-adapter_axial_length_m:.12g}",
            "rpy": "3.1415926536 1.5707963268 0",
        },
    )
    ET.SubElement(mount, "parent", {"link": f"{side}_GRIPPER_Y_LINK"})
    ET.SubElement(mount, "child", {"link": f"{prefix}WSG32_BASE_LINK"})


def _build_wsg_source_profile() -> ET.Element:
    adapter_axial_length_m = _wsg_adapter_axial_length_m()
    roots = [_load_fragment(path) for path in ROBOT_FRAGMENTS]
    root = roots[0]
    root.set("name", "AlexPurdue")
    for fragment in roots[1:]:
        root.extend(list(fragment))

    for element in list(root):
        if element.tag in {"gazebo", "transmission"}:
            root.remove(element)
    _remove_legacy_zed_visual(root)
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("mesh element has no filename")
        mesh.set("filename", _rewrite_mesh_reference(filename))

    pedestal = root.find("link[@name='PEDESTAL_LINK']/visual/geometry/mesh")
    expected_pedestal_reference = "../../meshes/torso_head/pedestal_purdue.obj"
    if pedestal is None or pedestal.get("filename") != expected_pedestal_reference:
        raise ValueError("the expected upstream pedestal mesh reference was not found")
    pedestal.set(
        "filename",
        "../../../../platforms/purdue_pedestal/meshes/pedestal_purdue.obj",
    )
    right_gripper = root.find("link[@name='RIGHT_GRIPPER_Y_LINK']/visual/geometry/mesh")
    expected_wrong_reference = "../../meshes/left_arm/l_gripper_y_purdue.obj"
    if (
        right_gripper is None
        or right_gripper.get("filename") != expected_wrong_reference
    ):
        raise ValueError(
            "the expected upstream right-gripper mesh defect was not found"
        )
    right_gripper.set("filename", "../../meshes/right_arm/r_gripper_y_purdue.obj")

    standalone = _load_fragment(WSG_STANDALONE_URDF)
    _append_wsg(
        root,
        standalone,
        side="LEFT",
        include_materials=True,
        adapter_axial_length_m=adapter_axial_length_m,
    )
    _append_wsg(
        root,
        standalone,
        side="RIGHT",
        include_materials=False,
        adapter_axial_length_m=adapter_axial_length_m,
    )
    root.insert(
        0,
        ET.Comment(
            " Alex Purdue WSG32 + UMI v1 simulation-training profile. The explicit "
            f"{adapter_axial_length_m:.3f} m axial mount component and "
            "model-reference transforms are "
            "accepted for simulation; physical calibration is optional and only "
            "required for calibrated sim-to-real claims. "
        ),
    )
    _validate_tree(root, expected_links=43, expected_joints=42)
    return root


def _build_convex_profile(source_root: ET.Element) -> ET.Element:
    root = copy.deepcopy(source_root)
    for link_name, collider in PHYSICAL_COLLIDERS.items():
        link = root.find(f"link[@name='{link_name}']")
        if link is None or link.find("collision") is not None:
            raise ValueError(f"unexpected existing collision state for {link_name}")
        collision = ET.SubElement(link, "collision", {"name": f"{link_name}_CONVEX"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": collider.urdf_reference})

    for link_name, collider in GRIPPER_COLLIDERS.items():
        mesh = root.find(f"link[@name='{link_name}']/collision/geometry/mesh")
        if mesh is None:
            raise ValueError(f"missing upstream EZGripper collision for {link_name}")
        mesh.set("filename", collider.urdf_reference)

    if len(root.findall(".//collision")) != 28:
        raise ValueError(
            "the full convex Purdue profile must contain exactly 28 collision records"
        )
    _validate_tree(root)
    return root


def _build_wsg_convex_profile(source_root: ET.Element) -> ET.Element:
    root = copy.deepcopy(source_root)
    for link_name, collider in PHYSICAL_COLLIDERS.items():
        link = root.find(f"link[@name='{link_name}']")
        if link is None or link.find("collision") is not None:
            raise ValueError(f"unexpected existing collision state for {link_name}")
        collision = ET.SubElement(link, "collision", {"name": f"{link_name}_CONVEX"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": collider.urdf_reference})
    if len(root.findall(".//collision")) != 32:
        raise ValueError(
            "the WSG full-convex profile must contain exactly 32 collision records"
        )
    _validate_tree(root, expected_links=43, expected_joints=42)
    return root


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return (
        ET.tostring(
            root, encoding="utf-8", xml_declaration=True, short_empty_elements=True
        )
        + b"\n"
    )


def _unique_colliders() -> tuple[Collider, ...]:
    by_output = {
        collider.output: collider
        for collider in (*PHYSICAL_COLLIDERS.values(), *GRIPPER_COLLIDERS.values())
    }
    return tuple(by_output[path] for path in sorted(by_output))


def _convex_stl_bytes(source_path: Path) -> bytes:
    try:
        import trimesh
        from trimesh.exchange.stl import export_stl
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "convex generation requires trimesh; run this script with the Isaac Lab Python"
        ) from error

    mesh = trimesh.load(source_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"failed to load a non-empty mesh from {source_path}")
    hull = mesh.convex_hull
    if not hull.is_convex or not hull.is_watertight:
        raise ValueError(f"generated hull is not convex and watertight: {source_path}")
    return export_stl(hull)


def _write_or_check(path: Path, content: bytes, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def generate(*, check: bool) -> None:
    source_root = _build_source_profile()
    convex_root = _build_convex_profile(source_root)
    wsg_source_root = _build_wsg_source_profile()
    wsg_convex_root = _build_wsg_convex_profile(wsg_source_root)
    _write_or_check(SOURCE_PROFILE, _xml_bytes(source_root), check=check)
    _write_or_check(CONVEX_PROFILE, _xml_bytes(convex_root), check=check)
    _write_or_check(WSG_SOURCE_PROFILE, _xml_bytes(wsg_source_root), check=check)
    _write_or_check(WSG_CONVEX_PROFILE, _xml_bytes(wsg_convex_root), check=check)

    colliders = _unique_colliders()
    for collider in colliders:
        _write_or_check(
            collider.output,
            _convex_stl_bytes(collider.source),
            check=check,
        )

    mode = "validated" if check else "generated"
    print(f"PASS: {mode} 4 Purdue URDF profiles and {len(colliders)} convex hulls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if tracked generated artifacts are stale",
    )
    args = parser.parse_args()
    try:
        generate(check=args.check)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
