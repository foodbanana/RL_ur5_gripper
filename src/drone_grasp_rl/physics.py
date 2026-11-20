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
    - elliptic friction cone and ``impratio`` 10 (the Isaac Lab "dexterous manipulation" starting profile): with
      the defaults (pyramidal, 1) closing the gripper on the gravity-compensated drone squeezed it out upward.
    - 4 substeps (480 Hz): MuJoCo keeps a constraint's time constant at least two solver steps, and the grasp
      needs 0.005 s (below). With 2 substeps and 0.01 s the scripted expert held 5 of 16.

    Grasp on Newton (step 3). The teacher trained on PhysX first held 3 of 128 on Newton and the scripted expert
    none; three MuJoCo defaults were too soft for a 25 N two-finger grasp of a 0.8 kg drone:

    - gripper mimic coupling (``scripts/convert_assets.py`` ``stiffen_gripper_mimic``): only ``rh_r1_joint`` is
      driven, and with the default equality the four finger joints drifted apart under load (42/32/81/72 deg
      instead of 16 deg each); the leader read as nearly closed, past the judge's empty-close angle
    - drone contacts (``assets/drones.py`` ``NEWTON_DRONE_CONTACT``): solref 0.005 s, solimp 0.95-0.99
    - rotation about the line between the fingers: one MuJoCo contact point per finger-hull pair lets the
      dropped drone pivot (11.7 deg vs PhysX 1.6 deg); ``condim`` 4 and torsional friction 0.02 m on the drone
      pieces (``iris_collision.usda``) stop it. ``enable_multiccd`` did not help (Newton makes the contacts).

    Result: the PhysX-trained teacher holds 95-97 % on Newton (mid hover), 91 % on big hover, the scripted
    expert 16 of 16; the gripper stops at 18.5 deg on the drone (PhysX 16.2 deg).
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
        num_substeps=4,
    )
    default: PhysxCfg = isaacsim_physx
