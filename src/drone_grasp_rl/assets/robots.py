# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 + ROBOTIS RH-P12-RN(A) gripper + RealSense D435i wrist camera.

Values follow ``data_collection_ur5_gripper`` (``isaacsim/config/drive_gains.yaml`` and
``robot_ur5_rh_p12.yaml``, tuned for PhysX at dt = 1/120 s). Units are radians: stiffness [N·m/rad],
damping [N·m·s/rad], effort [N·m], velocity [rad/s].
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from . import DRONE_GRASP_RL_ASSETS_DIR

UR5_RHP12_USD_PATH = DRONE_GRASP_RL_ASSETS_DIR / "robots/ur5_rh_p12_d435i/ur5_rh_p12_d435i/ur5_rh_p12_d435i.usda"
"""Robot USD converted by ``scripts/convert_assets.py``."""

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
"""Arm joints in dataset order (``state``/``action`` indices 0..5)."""

GRIPPER_JOINT_NAME = "rh_r1_joint"
"""The only driven gripper joint. 0 = open, :data:`GRIPPER_CLOSED_POS` = fully closed."""

GRIPPER_MIMIC_JOINT_NAMES = ["rh_r2", "rh_l1", "rh_l2"]
"""Mimic followers of :data:`GRIPPER_JOINT_NAME` (no ``_joint`` suffix in the URDF)."""

GRIPPER_CLOSED_POS = 1.1351
"""Fully closed gripper joint position [rad] (65.04 deg), raw command 1150."""

GRIPPER_PROFILE_VELOCITY = 0.516
"""Speed at which the gripper position target moves toward the goal [rad/s] (1.1351 rad / 2.2 s)."""

TCP_BODY_NAME = "rh_p12_rn_base"
"""Rigid body the TCP is attached to (the URDF TCP frame is massless, so it is not a body)."""

TCP_OFFSET_POS = (0.0, 0.0, 0.11)
"""TCP position in :data:`TCP_BODY_NAME` [m]: grasp point between the fingertips (tool0 + 125 mm)."""

FINGER_BODY_NAMES = ["rh_p12_rn_r1", "rh_p12_rn_r2", "rh_p12_rn_l1", "rh_p12_rn_l2"]
"""Finger links (contact sensors, friction material)."""

BASE_BODY_NAME = "robot_mount"
"""Articulation root (URDF root link). Its frame is the robot base frame used for the drone targets."""

_WRIST_3 = (
    "Geometry/robot_mount/base_link/base_link_inertia/shoulder_link/upper_arm_link/forearm_link"
    "/wrist_1_link/wrist_2_link/wrist_3_link"
)
_GRIPPER_BASE = f"{_WRIST_3}/flange/tool0/wrist_mount/gripper_bracket/rh_p12_rn_base"
BODY_PRIM_PATHS = {
    BASE_BODY_NAME: "Geometry/robot_mount",
    "wrist_3_link": _WRIST_3,
    TCP_BODY_NAME: _GRIPPER_BASE,
    "rh_p12_rn_r1": f"{_GRIPPER_BASE}/rh_p12_rn_r1",
    "rh_p12_rn_r2": f"{_GRIPPER_BASE}/rh_p12_rn_r1/rh_p12_rn_r2",
    "rh_p12_rn_l1": f"{_GRIPPER_BASE}/rh_p12_rn_l1",
    "rh_p12_rn_l2": f"{_GRIPPER_BASE}/rh_p12_rn_l1/rh_p12_rn_l2",
}
"""Body prim paths below the robot prim (sensors need the full nested path of the converted USD)."""

UNDER_DRONE_JOINT_POS = {
    "shoulder_pan_joint": -0.1889,
    "shoulder_lift_joint": -1.0776,
    "elbow_joint": 1.7304,
    "wrist_1_joint": -2.2242,
    "wrist_2_joint": 1.5700,
    "wrist_3_joint": -2.9566,
}
"""Episode start pose ``under_drone`` (t10 wrist mount): gripper pointing +z, TCP 0.305 m below the grasp."""

HOME_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}
"""UR home pose."""

UR5_RHP12_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(UR5_RHP12_USD_PATH),
        # PhysX payload by default (gains above are PhysX-tuned). Newton consumers select "mujoco".
        variants={"Physics": "physx"},
        # Fixed table mount. A moving base attaches the robot instead and sets this to False.
        fix_root_link=True,
        activate_contact_sensors=True,
        articulation_props=[
            # Self collision stays off for now: the forearm convex hull fills its concave side and
            # produced false self contacts in the old repo, which used a convex decomposition for it.
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={**UNDER_DRONE_JOINT_POS, GRIPPER_JOINT_NAME: 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness={
                "shoulder_pan_joint": 56502.0,
                "shoulder_lift_joint": 85488.0,
                "elbow_joint": 33469.3,
                "wrist_1_joint": 6782.0,
                "wrist_2_joint": 4849.2,
                "wrist_3_joint": 903.6,
            },
            damping={
                "shoulder_pan_joint": 1130.04,
                "shoulder_lift_joint": 1399.75,
                "elbow_joint": 439.7,
                "wrist_1_joint": 56.07,
                "wrist_2_joint": 39.94,
                "wrist_3_joint": 4.27,
            },
            joint_effort_limit={"shoulder_.*": 150.0, "elbow_joint": 150.0, "wrist_.*": 28.0},
            joint_velocity_limit=3.1416,
            armature=0.0,
        ),
        # Only the leader is driven; with gains on the followers PhysX would apply the effort twice.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME],
            stiffness=45.6,
            damping=1.3758,
            joint_effort_limit=2.28,
            # Not the profile speed: a 0.516 rad/s joint limit acts like viscous drag with the mimic
            # chain and the gripper cannot open. The profile is applied to the target instead.
            joint_velocity_limit=6.5,
            # Without armature only ~26 % of the effort limit reaches the fingers (solver convergence).
            armature=0.01,
            # The URDF gives the gripper joints 0.7 N·m·s/rad of passive damping, stored as newton:damping.
            # Only Newton reads it (the PhysX-tuned values above ran without it), and on the light distal
            # links it beat the soft mimic constraint: followers lagged the leader by up to 42 deg.
            viscous_friction=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_MIMIC_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            viscous_friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""UR5 + RH-P12-RN(A) fixed to the world at the spawn pose, starting in ``under_drone``."""
