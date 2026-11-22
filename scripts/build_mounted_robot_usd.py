"""Build a mounted-robot USD: a robot arm USD on a moving base USD as ONE articulation (no simulator needed).

PhysX and Newton both simulate the arm and the base as one articulation, so the arm's weight and reactions load
the base exactly (the reference in ``docs/moving_base_and_student.md``). The arm stays its own USD: to mount another
arm, add an entry to ``MOUNTS`` (arm USD, its root body, the mount pose) and run this script again.

Layout of the output (all references relative, so the file works on any PC):

- the default prim references the **arm** USD, so the arm's prim paths (sensors, cameras, the ``Physics`` variant
  set) stay the same as for the arm alone: ``{ENV}/Robot/Geometry/...``
- ``Base``: references the base USD, placed so the base's articulation root body sits at the default prim origin
  (the asset's ``init_state`` is the base's root pose)
- the arm root body is moved to the mount pose and loses its articulation root (the base's root
  is the one root) and ``MountJoint`` fixes it to the base's mount body (inside the articulation)
- collisions between the arm root bodies that rest on the base and the base mount body are filtered

Usage (from the repository root):

    uv run python scripts/build_mounted_robot_usd.py            # all entries of MOUNTS
    uv run python scripts/build_mounted_robot_usd.py --name ur5_on_stewart
"""

import argparse
import math
import os
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drone_grasp_rl.assets import DRONE_GRASP_RL_ASSETS_DIR  # noqa: E402
from drone_grasp_rl.assets.bases import (  # noqa: E402
    HUSKY_CARRIAGE_BODY,
    HUSKY_RAIL_USD_PATH,
    HUSKY_ROBOT_MOUNT_POS,
    HUSKY_ROBOT_MOUNT_YAW_DEG,
    STEWART_TOP_BODY,
)
from drone_grasp_rl.assets.robots import BODY_PRIM_PATHS, UR5_RHP12_USD_PATH  # noqa: E402
from drone_grasp_rl.assets.stewart import STEWART_USD_PATH  # noqa: E402

OUT_DIR = DRONE_GRASP_RL_ASSETS_DIR / "mounted"

_UR5 = {
    "arm_usd": UR5_RHP12_USD_PATH,
    "arm_root_body": BODY_PRIM_PATHS["robot_mount"],  # Geometry/robot_mount
    # arm bodies resting on the base (no collision with the base mount body), as before on PhysX
    "filtered_arm_bodies": ["Geometry/robot_mount", "Geometry/robot_mount/base_link/base_link_inertia"],
}
MOUNTS = {
    "ur5_on_stewart": {
        **_UR5,
        "base_usd": STEWART_USD_PATH,
        "base_mount_body": STEWART_TOP_BODY,  # rotation center of the top plate
        "mount_pos": (0.0, 0.0, 0.0),
        "mount_yaw_deg": 0.0,
    },
    "ur5_on_husky_rail": {
        **_UR5,
        "base_usd": HUSKY_RAIL_USD_PATH,
        "base_mount_body": HUSKY_CARRIAGE_BODY,  # Husky base_link
        "mount_pos": HUSKY_ROBOT_MOUNT_POS,
        "mount_yaw_deg": HUSKY_ROBOT_MOUNT_YAW_DEG,
    },
}
"""Mounted robots: output name -> arm and base USDs, the arm root body, the base mount body and the arm root's pose
in the mount body's frame (position [m], yaw [deg])."""


def _local_to_default(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    """Pose of ``path`` relative to the stage's default prim (USD row-vector matrices)."""
    cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(path)
    assert prim, f"prim not found: {path}"
    return cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()


def _root_body(stage: Usd.Stage) -> str:
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1, f"expected one articulation root, found {[str(p.GetPath()) for p in roots]}"
    return str(roots[0].GetPath())


def _set_pose(prim, matrix: Gf.Matrix4d):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))


def build(name: str, spec: dict) -> Path:
    out = OUT_DIR / f"{name}.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    # base: root body and the mount body's rest pose, relative to the base's default prim
    base = Usd.Stage.Open(str(spec["base_usd"]))
    base_default = str(base.GetDefaultPrim().GetPath())
    base_root = _root_body(base)
    root_in_base = _local_to_default(base, base_root)
    mount_body_in_base = _local_to_default(base, f"{base_default}/{spec['base_mount_body']}")
    # arm: the root body's parent, relative to the arm's default prim (the root body gets the mount pose)
    arm = Usd.Stage.Open(str(spec["arm_usd"]))
    arm_default = str(arm.GetDefaultPrim().GetPath())
    parent = str(arm.GetPrimAtPath(f"{arm_default}/{spec['arm_root_body']}").GetParent().GetPath())
    parent_in_arm = _local_to_default(arm, parent)

    half = math.radians(spec["mount_yaw_deg"]) / 2
    mount_rot = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))
    mount_in_body = Gf.Matrix4d().SetTransform(Gf.Rotation(mount_rot), Gf.Vec3d(*spec["mount_pos"]))
    # frames relative to the output's default prim, whose origin is the base root body
    base_xform = root_in_base.GetInverse()  # places the base so its root body is at the origin
    arm_root_pose = mount_in_body * mount_body_in_base * base_xform
    arm_root_local = arm_root_pose * parent_in_arm.GetInverse()

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("comment", f"Generated by scripts/build_mounted_robot_usd.py ({name}). Do not edit.")
    root = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(spec["arm_usd"], out.parent))
    base_prim = stage.DefinePrim("/robot/Base", "Xform")
    base_prim.GetReferences().AddReference(os.path.relpath(spec["base_usd"], out.parent))
    _set_pose(base_prim, base_xform)
    arm_root = stage.GetPrimAtPath(f"/robot/{spec['arm_root_body']}")
    assert arm_root, "arm root body not found (is the arm's physics payload loaded?)"
    _set_pose(arm_root, arm_root_local)
    arm_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    arm_root.RemoveAppliedSchema("NewtonArticulationRootAPI")

    mount_body = f"/robot/Base/{spec['base_mount_body']}"
    joint = UsdPhysics.FixedJoint.Define(stage, "/robot/MountJoint")
    joint.CreateBody0Rel().SetTargets([mount_body])
    joint.CreateBody1Rel().SetTargets([arm_root.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in spec["mount_pos"]]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(v) for v in (mount_rot.GetReal(), *mount_rot.GetImaginary())]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for body in spec["filtered_arm_bodies"]:
        prim = stage.GetPrimAtPath(f"/robot/{body}")
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(mount_body)

    # one articulation root: the base's
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [f"/robot/Base{base_root[len(base_default) :]}"], roots
    stage.GetRootLayer().Save()
    pose = _local_to_default(stage, str(arm_root.GetPath()))
    print(
        f"[mounted] wrote {out.relative_to(REPO)}: arm root at {tuple(round(v, 4) for v in pose.ExtractTranslation())}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", choices=sorted(MOUNTS), default=None)
    args = ap.parse_args()
    for name, spec in MOUNTS.items():
        if args.name in (None, name):
            build(name, spec)


if __name__ == "__main__":
    main()
