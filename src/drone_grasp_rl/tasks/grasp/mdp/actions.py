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
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def _to_index(env_ids, device) -> torch.Tensor | slice:
    """Action terms get ``None``, a slice, a list, a tensor or a Warp array as ``env_ids``."""
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    if isinstance(env_ids, wp.array):
        return wp.to_torch(env_ids).long()
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


class InterpolatedRelativeJointPositionAction(ActionTerm):
    """Joint position goal that moves by a bounded step per policy step, reached by linear interpolation.

    Each policy step the action in [-1, 1] moves the goal by ``action * max_step`` [rad], clamped to the joint
    limits minus ``limit_margin``. Over the physics steps of that policy step the drive target moves linearly
    from the previous goal to the new one, like the data collection repo's 33 ms command interpolation and the
    real UR servoJ stream. So the joint speed stays below ``max_step / step_dt`` and the tracking error small,
    which is what the protective stop checks.

    :attr:`processed_actions` is the absolute goal, i.e. the joint target a dataset records as the action.
    """

    cfg: InterpolatedRelativeJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: InterpolatedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._joint_ids_torch = torch.tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._decimation = env.cfg.decimation
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids_torch]
        self._lower = limits[..., 0] + cfg.limit_margin
        self._upper = limits[..., 1] - cfg.limit_margin
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids_torch]
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._goal = default.clone()
        self._start = default.clone()
        self._target = default.clone()
        self._substep = torch.full((self.num_envs, 1), self._decimation, dtype=torch.float32, device=self.device)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._goal

    @property
    def target(self) -> torch.Tensor:
        """Drive target of the current physics step [rad], shape (num_envs, num_joints)."""
        return self._target

    @property
    def joint_ids(self) -> torch.Tensor:
        return self._joint_ids_torch

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._start = self._target.clone()
        step = actions.clamp(-1.0, 1.0) * self.cfg.max_step
        self._goal = torch.max(torch.min(self._goal + step, self._upper), self._lower)
        self._substep.zero_()

    def apply_actions(self):
        self._substep += 1.0
        u = (self._substep / self._decimation).clamp(max=1.0)
        self._target = self._start + u * (self._goal - self._start)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        ids = _to_index(env_ids, self.device)
        # The reset puts the joints at their default positions.
        default = self._asset.data.default_joint_pos.torch[ids][:, self._joint_ids_torch]
        self._raw_actions[ids] = 0.0
        self._goal[ids] = default
        self._start[ids] = default
        self._target[ids] = default
        self._substep[ids] = self._decimation


@configclass
class InterpolatedRelativeJointPositionActionCfg(ActionTermCfg):
    """Configuration of :class:`InterpolatedRelativeJointPositionAction`."""

    class_type: type = InterpolatedRelativeJointPositionAction

    joint_names: list[str] = MISSING
    """Joints to command, in action order."""

    max_step: float = MISSING
    """Goal change per policy step at action 1 [rad]."""

    limit_margin: float = 0.05
    """Distance the goal keeps from the soft joint limits [rad]."""


class ProfiledBinaryJointPositionAction(BinaryJointPositionAction):
    """Open/close command whose position target moves toward the goal at a fixed speed.

    Like the real gripper's position profile: the command picks the open or close position, and the
    joint target sent to the drive moves toward it by ``profile_velocity * physics_dt`` every physics
    step. A step change of the target would snap the fingers shut much faster than the real gripper.

    With ``latch_close``, the first close command holds until the episode ends (one grasp attempt per episode,
    as in the data collection protocol). Closing takes 0.6-2.2 s of continuous command; a policy exploring with
    independent noise per step flips the command every 33 ms and the fingers never close, so it could never
    discover a grasp. :attr:`closing` is the command in force.
    """

    cfg: ProfiledBinaryJointPositionActionCfg

    def __init__(self, cfg: ProfiledBinaryJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._max_step = cfg.profile_velocity * env.physics_dt
        self._target = self._open_command.repeat(self.num_envs, 1)
        self._joint_ids_torch = wp.to_torch(self._joint_ids).long()
        self._closing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def target(self) -> torch.Tensor:
        """Current joint position target sent to the drive, shape (num_envs, num_joints)."""
        return self._target

    @property
    def closing(self) -> torch.Tensor:
        """The close command is in force (after the latch), shape (num_envs,)."""
        return self._closing

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close = actions[:, 0] < 0
        self._closing = self._closing | close if self.cfg.latch_close else close
        self._processed_actions = torch.where(self._closing.unsqueeze(-1), self._close_command, self._open_command)

    def apply_actions(self):
        self._target += (self._processed_actions - self._target).clamp(-self._max_step, self._max_step)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | wp.array | None = None) -> None:
        super().reset(env_ids)
        ids = _to_index(env_ids, self.device)
        # The reset puts the joints at their default (open) position; the profile starts there.
        self._target[ids] = self._asset.data.default_joint_pos.torch[ids][:, self._joint_ids_torch]
        self._processed_actions = self._processed_actions.clone()
        self._processed_actions[ids] = self._target[ids]
        self._closing[ids] = False


@configclass
class ProfiledBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration of :class:`ProfiledBinaryJointPositionAction`."""

    class_type: type = ProfiledBinaryJointPositionAction

    profile_velocity: float = MISSING
    """Speed of the position target toward the open or close position [rad/s or m/s]."""

    latch_close: bool = False
    """Keep closing after the first close command until the episode ends."""
