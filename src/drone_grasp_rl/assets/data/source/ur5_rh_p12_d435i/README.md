# UR5 + RH-P12-RN(A) + RealSense D435i — source description

This folder holds the source files the robot USD is converted from. With them, the
conversion can be repeated on any machine without ROS or the original repository.

## Contents

| Path | What | Origin | License |
|---|---|---|---|
| `urdf/ur5_rh_p12_d435i.urdf` | Flattened URDF (xacro output) | `data_collection_ur5_gripper` @ `50563c1` (2026-11-04), `isaacsim/ur5_rh_p12_description/out/ur5_rh_p12_d435i.urdf`, built from `urdf/ur5_rh_p12_d435i.urdf.xacro` with default args (`mount_variant:=t10`, `mount_thickness:=0.005`) | — |
| `meshes/ur5/{visual,collision}/` | UR5 (CB3) meshes | ROS 2 Jazzy package `ur_description` 3.5.1 (`share/ur_description/meshes/ur5/`) | BSD-3-Clause, see `LICENSES/ur_description_copyright.txt` |
| `meshes/realsense/d435.dae` | D435 camera body mesh | ROS 2 Jazzy package `realsense2_description` 4.58.4 | Apache-2.0, see `LICENSES/realsense2_description_copyright.txt` |
| `meshes/rh_p12_rn_a/*.stl` | ROBOTIS RH-P12-RN(A) gripper meshes | ROBOTIS-GIT/open_manipulator @ `d31000d` (`open_manipulator_description/meshes/rh_p12_rn_a/`), identical to ROBOTIS-GIT/RH-P12-RN @ `58d06e7` | Apache-2.0 |
| `meshes/mount/*.stl` | Sandwich wrist mount (variant t10) and its collision plate | `data_collection_ur5_gripper` (own design) | — |

## Changes from the original files

The only change is to the URDF: absolute mesh paths (`/opt/ros/jazzy/share/...` and
`/home/<user>/data_collection_ur5_gripper/...`) were rewritten to paths relative to the
URDF file (`../meshes/...`). Joints, links, inertias and limits are unchanged.

## Kinematic notes (from the original repository)

- Chain: `robot_mount -> base_link ... wrist_3_link -> flange -> tool0 -> wrist_mount -> rh_p12_rn_base`
  (plus `camera_bottom_screw_frame` on the mount).
- Gripper: `rh_r1_joint` is the driven joint. `rh_r2`, `rh_l1`, `rh_l2` mimic it
  (these three names have no `_joint` suffix).
- TCP: `rh_p12_rn_tcp` (tool0 + 125 mm) is massless, so it is a plain Xform, not a rigid body.
  Define the TCP in Isaac Lab as `rh_p12_rn_base` (tool0 + 15 mm) + 0.11 m along z (`body_offset`).

## Converted USD

`scripts/convert_assets.py` converts this URDF to
`assets/data/robots/ur5_rh_p12_d435i/ur5_rh_p12_d435i/ur5_rh_p12_d435i.usda` (committed, Git LFS):

```bash
uv run --extra isaacsim python scripts/convert_assets.py
```

- Options mirror the old Isaac Sim 6.1 import: no fixed-joint merging, no collision from visuals,
  convex-hull colliders, no self collision, no mesh merging. The base is not fixed at import time
  (spawn with `fix_root_link=True` for a fixed base, or attach to a moving base).
- Result: 16 rigid bodies (massless frames such as `tool0`, `flange` and the TCP become plain Xforms),
  10 revolute joints, `Physics` variants `physics` (default), `physx`, `mujoco`, `none`.
- Mimic: `rh_r2`, `rh_l1`, `rh_l2` carry `newton:mimicJoint -> rh_r1_joint`. In the `physx` variant
  the followers also have a position drive, so their gains must be 0 in the actuator config.
- The script strips the absolute repository path and the converter's random temp-folder name from
  the text outputs. Re-runs then differ only in the generation timestamp in `config.yaml` and in the
  binary encoding of `payloads/geometries.usd`.
