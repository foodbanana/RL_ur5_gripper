# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot arm on a moving base as one articulation (``scripts/build_mounted_robot_usd.py``).

The mounted USD references the arm USD as its default prim (arm prim paths, sensors and the ``Physics`` variant set
unchanged) and the base USD as ``Base``; a fixed joint inside the articulation holds the arm root on the base's
mount body. PhysX and Newton then simulate the arm's weight and reactions on the base the same way. Another arm:
add it to ``MOUNTS`` in the script and build its USD.
"""

from isaaclab.assets import ArticulationCfg

from . import DRONE_GRASP_RL_ASSETS_DIR

MOUNTED_USD_DIR = DRONE_GRASP_RL_ASSETS_DIR / "mounted"


def mounted_robot_cfg(robot_cfg: ArticulationCfg, base_cfg: ArticulationCfg, name: str) -> ArticulationCfg:
    """The arm ``robot_cfg`` on the base ``base_cfg`` from the mounted USD ``name`` (e.g. ``ur5_on_stewart``).

    Keeps the arm's prim path, spawn settings (variants, contact sensors, articulation properties) and initial joint
    positions; adds the base's actuators and initial joint positions. The root pose is the base's (its articulation
    root body is the mounted USD's origin), and the root is fixed to the world like the base's.
    """
    return robot_cfg.replace(
        spawn=robot_cfg.spawn.replace(usd_path=str(MOUNTED_USD_DIR / f"{name}.usda"), fix_root_link=True),
        init_state=robot_cfg.init_state.replace(
            pos=base_cfg.init_state.pos,
            rot=base_cfg.init_state.rot,
            joint_pos={**base_cfg.init_state.joint_pos, **robot_cfg.init_state.joint_pos},
        ),
        actuators={**robot_cfg.actuators, **base_cfg.actuators},
    )
