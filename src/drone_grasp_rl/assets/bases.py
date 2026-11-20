# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving robot bases: the closed-loop Stewart platform (``assets/stewart.py``, ``scripts/build_stewart_usd.py``)."""

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
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
