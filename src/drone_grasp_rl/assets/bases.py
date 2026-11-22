# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving robot bases: the closed-loop Stewart platform (``assets/stewart.py``, ``scripts/build_stewart_usd.py``)
and the Husky A200 (visual, moved kinematically)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from . import stewart as st

STEWART_LEG_JOINTS = [f"leg_{i}" for i in range(6)]
"""Actuated prismatic joints (position 0 = center-pose length)."""
STEWART_TOP_BODY = "top_plate"
"""Top plate body; its origin is the rotation center, where the robot base mounts."""

STEWART_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Stewart",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(st.STEWART_USD_PATH),
        fix_root_link=True,
        articulation_props=[
            # loops closed by 5 maximal spherical joints: more solver iterations keep them tight
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"leg_.*": 0.0}),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=STEWART_LEG_JOINTS,
            stiffness=st.LEG_STIFFNESS,
            damping=st.LEG_DAMPING,
            joint_effort_limit=st.LEG_FORCE_LIMIT,
            joint_velocity_limit=st.LEG_SPEED_LIMIT,
            armature=st.LEG_ARMATURE,
        ),
        "passive": ImplicitActuatorCfg(joint_names_expr=["leg_.*_u.*"], stiffness=0.0, damping=0.0),
    },
)
"""Stewart platform on the floor at the environment origin, base plate fixed to the world."""


# --- Husky A200 (visual), moved kinematically like in the data collection repo ---
from isaaclab.assets import AssetBaseCfg  # noqa: E402

from . import DRONE_GRASP_RL_ASSETS_DIR  # noqa: E402

HUSKY_USD_PATH = DRONE_GRASP_RL_ASSETS_DIR / "bases" / "husky" / "husky.usda"
HUSKY_BASE_HEIGHT = 0.13228
"""Husky base_link above the floor [m] (URDF base_footprint_joint)."""
HUSKY_ROBOT_MOUNT_POS = (0.0812 + 0.3058, 0.0, 0.245 + 0.0065)
"""UR5 base in the Husky base_link frame [m]: top_plate_link plus Clearpath's ``ur5_arm_base`` offset (provisional)."""
HUSKY_ROBOT_MOUNT_YAW_DEG = -90.0
"""UR5 base yaw on the Husky (Clearpath Husky + UR5): the arm's front points to the Husky's right."""
HUSKY_TRAJECTORY = {"distance": 0.30, "max_speed": 0.05, "accel": 0.10, "dwell": 1.0}
"""Straight back-and-forth along the Husky's x axis (data collection repo ``base_husky.yaml``, provisional) [m, m/s,
m/s^2, s]."""

HUSKY_RAIL_USD_PATH = DRONE_GRASP_RL_ASSETS_DIR / "bases" / "husky" / "husky_rail.usda"
HUSKY_DRIVE_JOINT = "drive"
HUSKY_CARRIAGE_BODY = "carriage"

HUSKY_RAIL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/HuskyRail",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(HUSKY_RAIL_USD_PATH),
        fix_root_link=True,
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=False, sleep_threshold=0.0),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, HUSKY_BASE_HEIGHT), joint_pos={HUSKY_DRIVE_JOINT: 0.0}),
    actuators={
        # stiff position servo: the carriage follows the trajectory (kinematic in the data collection repo)
        "drive": ImplicitActuatorCfg(
            joint_names_expr=[HUSKY_DRIVE_JOINT], stiffness=1.0e6, damping=2.0e4, armature=50.0
        ),
    },
)
"""Husky as a carriage on a driven rail along its x axis (``scripts/build_husky_rail_usd.py``), base_link height."""

HUSKY_VISUAL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/HuskyRail/carriage/husky",
    spawn=sim_utils.UsdFileCfg(usd_path=str(HUSKY_USD_PATH), variants={"Physics": "none"}),
)
"""The Husky's meshes on the carriage (no physics: the ``none`` variant)."""
