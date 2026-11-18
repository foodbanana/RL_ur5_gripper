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

from drone_grasp_rl.assets.drones import IRIS_NEWTON_USD_PATH, IRIS_USD_PATH

SIM_DT = 1.0 / 120.0
"""Physics step [s]. Matches data_collection_ur5_gripper (120 Hz); the arm and gripper gains are tuned for it."""

PHYSICS_VARIANT = {"isaacsim_physx": "physx", "physx": "physx", "newton_mjwarp": "mujoco"}
"""USD ``Physics`` variant to select on the converted robot for each backend."""

DRONE_USD = {
    "isaacsim_physx": str(IRIS_USD_PATH),
    "physx": str(IRIS_USD_PATH),
    "newton_mjwarp": str(IRIS_NEWTON_USD_PATH),
}
"""Drone USD for each backend (original body collider for PhysX, fixed convex pieces for Newton)."""


@configclass
class DroneGraspPhysicsCfg(PresetCfg):
    """Physics backends for drone_grasp_rl.

    PhysX is the reference: the gains come from the PhysX-based data collection repository.
    ``physx`` is an alias of ``isaacsim_physx`` so the automatic selector never falls back to OvPhysX,
    which this project does not install.

    MJWarp settings, each measured with ``scripts/check_assets.py`` / ``scripts/check_task.py``:

    - ``implicitfast`` integrator: with the default ``euler`` the high PhysX gains made the wrist joints
      oscillate (wrist_3 std 12.6 deg holding a pose); ``implicitfast`` holds within 0.03 deg with the
      same gains, so no per-backend gains are needed.
    - ``nconmax``/``njmax``: the drone body alone has 30 convex pieces; with the default capacity the
      broadphase overflowed and dropped the finger-drone contacts.
    - elliptic friction cone, ``impratio`` 10 and 2 substeps (the Isaac Lab "dexterous manipulation"
      starting profile): with the defaults (pyramidal, 1, 1 substep) closing the gripper on the
      gravity-compensated drone squeezed it out upward. Now the grasp holds, with the drone 7 mm higher
      and the fingers at 28 deg instead of PhysX's 1 mm and 17 deg (softer MuJoCo contacts).
    """

    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = isaacsim_physx
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            nconmax=200,
            njmax=600,
            cone="elliptic",
            impratio=10.0,
        ),
        num_substeps=2,
    )
    default: PhysxCfg = isaacsim_physx
