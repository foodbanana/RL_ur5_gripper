# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3DR Iris drone at 0.75x scale, the grasp target.

The USD (``iris_0p75.usda``) references the unmodified Pegasus ``iris.usd`` and applies the
overrides from ``data_collection_ur5_gripper`` (``isaacsim/config/drone_iris.yaml``): scale, link
masses, recomputed inertia, locked rotor joints and zeroed initial velocities.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from . import DRONE_GRASP_RL_ASSETS_DIR

IRIS_USD_PATH = DRONE_GRASP_RL_ASSETS_DIR / "drones/iris/iris_0p75.usda"
"""Scaled Iris with the data-collection overrides."""

DRONE_BODY_NAME = "body"
"""Central rigid body: the grasp target and the link that receives the hover wrench."""

DRONE_MASS = 0.8
"""Total mass [kg] (body 0.78 + 4 rotors x 0.005)."""

DRONE_GRASP_WIDTH = 0.0813
"""Body width at the waist (drone x = 0) along drone y, where the fingers close [m]."""

ROTOR_CONSTANT = 8.54858e-6
"""Thrust per rotor F = k * omega^2 [N/(rad/s)^2] (Pegasus Iris)."""

ROLLING_MOMENT_COEFFICIENT = 1.0e-6
"""Rotor reaction torque c * omega^2 [N·m/(rad/s)^2]."""

MAX_ROTOR_VELOCITY = 1100.0
"""Maximum rotor speed [rad/s] (about 10.3 N per rotor)."""

LINEAR_DRAG = (0.50, 0.30, 0.0)
"""Body-frame linear drag coefficients [N·s/m]."""

IRIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Drone",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(IRIS_USD_PATH),
        activate_contact_sensors=True,
        articulation_props=[
            # Stay awake while hovering; a sleeping articulation stops reacting to small wrenches.
            PhysxArticulationCfg(sleep_threshold=0.0),
            NewtonArticulationCfg(),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    actuators={
        # Rotor joints are locked by their +-0.01 deg limits in the USD; no drive acts on them.
        "rotors": ImplicitActuatorCfg(joint_names_expr=["joint[0-3]"], stiffness=0.0, damping=0.0),
    },
)
"""Iris drone floating at 1 m. Thrust/hover is applied by the task as a body wrench."""
