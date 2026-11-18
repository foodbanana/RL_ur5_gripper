# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backends shared by the project's scripts and tasks.

Select one with ``physics=<name>`` on ``isaaclab train/play`` or ``--backend <name>`` in scripts.
"""

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

SIM_DT = 1.0 / 120.0
"""Physics step [s]. Matches data_collection_ur5_gripper (120 Hz); the arm and gripper gains are tuned for it."""

PHYSICS_VARIANT = {"isaacsim_physx": "physx", "physx": "physx", "newton_mjwarp": "mujoco"}
"""USD ``Physics`` variant to select on the converted robot for each backend."""


@configclass
class DroneGraspPhysicsCfg(PresetCfg):
    """Physics backends for drone_grasp_rl.

    PhysX is the reference: the gains come from the PhysX-based data collection repository.
    ``physx`` is an alias of ``isaacsim_physx`` so the automatic selector never falls back to OvPhysX,
    which this project does not install.

    MJWarp uses the ``implicitfast`` integrator. With the default ``euler`` integrator the high PhysX
    gains made the wrist joints oscillate (wrist_3 std 12.6 deg holding a pose); with ``implicitfast``
    the same gains hold within 0.03 deg, so no per-backend gains are needed.
    """

    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = isaacsim_physx
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast"),
        num_substeps=1,
    )
    default: PhysxCfg = isaacsim_physx
