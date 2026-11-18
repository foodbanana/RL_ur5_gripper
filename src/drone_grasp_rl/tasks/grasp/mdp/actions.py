# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the drone grasp task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ProfiledBinaryJointPositionAction(BinaryJointPositionAction):
    """Open/close command whose position target moves toward the goal at a fixed speed.

    Like the real gripper's position profile: the command picks the open or close position, and the
    joint target sent to the drive moves toward it by ``profile_velocity * physics_dt`` every physics
    step. A step change of the target would snap the fingers shut much faster than the real gripper.
    """

    cfg: ProfiledBinaryJointPositionActionCfg

    def __init__(self, cfg: ProfiledBinaryJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._max_step = cfg.profile_velocity * env.physics_dt
        self._target = self._open_command.repeat(self.num_envs, 1)
        self._joint_ids_torch = wp.to_torch(self._joint_ids).long()

    @property
    def target(self) -> torch.Tensor:
        """Current joint position target sent to the drive, shape (num_envs, num_joints)."""
        return self._target

    def apply_actions(self):
        self._target += (self._processed_actions - self._target).clamp(-self._max_step, self._max_step)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | wp.array | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = slice(None)
        elif isinstance(env_ids, slice):
            ids = env_ids
        elif isinstance(env_ids, wp.array):
            ids = wp.to_torch(env_ids).long()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # The reset puts the joints at their default (open) position; the profile starts there.
        self._target[ids] = self._asset.data.default_joint_pos.torch[ids][:, self._joint_ids_torch]
        self._processed_actions = self._processed_actions.clone()
        self._processed_actions[ids] = self._target[ids]


@configclass
class ProfiledBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration of :class:`ProfiledBinaryJointPositionAction`."""

    class_type: type = ProfiledBinaryJointPositionAction

    profile_velocity: float = MISSING
    """Speed of the position target toward the open or close position [rad/s or m/s]."""
