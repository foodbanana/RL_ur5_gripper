# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the drone grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


def _env_ids_tensor(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> torch.Tensor:
    """Event managers pass ``slice(None)`` for all environments (e.g. on the first reset)."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


class reset_drone_at_targets(ManagerTermBase):
    """Place the drone at one of the reference positions plus a uniform offset, at rest.

    The reference positions are given in the robot base frame (the robot articulation root), so they stay
    valid when the base moves or turns. The drone keeps the base orientation. The chosen reference index
    of each environment is kept in :attr:`target_index` for logging and dataset metadata.

    Params:
        targets: Reference drone positions in the robot base frame [m], shape (K, 3).
        offset: Half width of the uniform offset added on each axis [m].
        drone_cfg: The drone articulation.
        robot_cfg: The robot whose root frame is the base frame.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.targets = torch.tensor(cfg.params["targets"], dtype=torch.float32, device=env.device)
        self.target_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        targets: list[tuple[float, float, float]],
        offset: float,
        drone_cfg: SceneEntityCfg = SceneEntityCfg("drone"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        env_ids = _env_ids_tensor(env, env_ids)
        drone: Articulation = env.scene[drone_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        num = len(env_ids)

        index = torch.randint(len(self.targets), (num,), device=env.device)
        self.target_index[env_ids] = index
        pos_b = self.targets[index] + math_utils.sample_uniform(-offset, offset, (num, 3), env.device)

        base_pos = robot.data.root_pos_w.torch[env_ids]
        base_quat = robot.data.root_quat_w.torch[env_ids]
        pos_w = base_pos + math_utils.quat_apply(base_quat, pos_b)

        drone.write_root_pose_to_sim_index(root_pose=torch.cat([pos_w, base_quat], dim=-1), env_ids=env_ids)
        drone.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=env.device), env_ids=env_ids)


def hold_drone_in_place(
    env: ManagerBasedEnv, env_ids: torch.Tensor | slice, drone_cfg: SceneEntityCfg = SceneEntityCfg("drone")
):
    """Cancel gravity on every drone body with a permanent world-frame force.

    Stands in for the flight controller until the hover controller is added: the drone stays where the
    reset put it unless something pushes it. Each body gets its own weight, so the light rotors do not
    tilt the drone. The articulation reset clears permanent wrenches, so this must run on every reset
    (after the scene reset) and also picks up randomized masses.
    """
    env_ids = _env_ids_tensor(env, env_ids)
    drone: Articulation = env.scene[drone_cfg.name]
    mass = drone.data.body_mass.torch[env_ids]
    forces = torch.zeros(*mass.shape, 3, device=env.device)
    forces[..., 2] = -env.sim.cfg.gravity[2] * mass
    drone.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, body_ids=list(range(drone.num_bodies)), env_ids=env_ids, is_global=True
    )
