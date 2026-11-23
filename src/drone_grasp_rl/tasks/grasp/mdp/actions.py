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

    ``delay_steps``: each episode the actions are applied a random whole number of policy steps late (command
    latency of the real arm on top of the interpolation; the data collection repo measured 47 ms from command to
    joint motion).

    ``smoothing``: exponential moving average of the applied action, ``a = s * a_new + (1 - s) * a_prev`` (1 = off).
    The trained policy dithered by about 0.1 per step around a held pose; filtering also takes out part of the
    exploration noise (OpenAI's in-hand manipulation smoothed its actions the same way).

    ``slow_distance``: within this TCP distance to the grasp point [m] the goal step shrinks in proportion to the
    distance, down to ``min_step_scale`` of ``max_step`` at the grasp point, so the arm cannot rush the drone (the
    policy came in at a median 27 cm/s over the last 5 cm) and moves slowly while it holds it.

    ``hold_after_grasp``: once the grasp judge (termination term ``judge_term``) has the grasp, the goal stops moving,
    as the scripted expert stops the arm when it closes. Measured on a trained policy: with the arm held still after
    the grasp the success rate stayed (98 % -> 95 %), the end-of-hold gripper tilt halved (6.6 -> 3.2 deg) and the
    joints moved at 0.4 deg/s instead of 7 deg/s. A teacher-side rule: a camera student sees the closed fingers too.
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
        max_delay = cfg.delay_steps[1]
        self._history = torch.zeros(self.num_envs, max_delay + 1, len(self._joint_ids), device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._smoothed = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

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
        self._history = torch.roll(self._history, 1, dims=1)
        self._history[:, 0] = actions.clamp(-1.0, 1.0)
        applied = self._history[torch.arange(self.num_envs, device=self.device), self._delay]
        if self.cfg.smoothing < 1.0:
            self._smoothed = self.cfg.smoothing * applied + (1.0 - self.cfg.smoothing) * self._smoothed
            applied = self._smoothed
        step = applied * self.cfg.max_step
        if self.cfg.hold_after_grasp:
            manager = getattr(self._env, "termination_manager", None)
            if manager is not None and self.cfg.judge_term in manager.active_terms:
                phase = manager.get_term_cfg(self.cfg.judge_term).func.phase
                holding = (phase == 1) | (phase == 2) | (phase == 3)
                step = step * (~holding).unsqueeze(-1)
        if self.cfg.slow_distance is not None:
            from .rewards import tcp_in_grasp_frame  # noqa: PLC0415 (rewards imports this module)

            distance = tcp_in_grasp_frame(self._env).norm(dim=-1, keepdim=True)
            scale = (distance / self.cfg.slow_distance).clamp(self.cfg.min_step_scale, 1.0)
            if self.cfg.slow_mode == "relative":
                step = self._limit_relative_speed(step, self.cfg.max_relative_speed * scale)
            else:
                step = step * scale
        self._goal = torch.max(torch.min(self._goal + step, self._upper), self._lower)
        self._substep.zero_()

    def _limit_relative_speed(self, step: torch.Tensor, v_max: torch.Tensor) -> torch.Tensor:
        """Change the joint goal step so the TCP moves at most ``v_max`` (N, 1) [m/s] relative to the drone.

        Predicted TCP velocity over the next policy step = the step's joint motion (TCP Jacobian) + the TCP velocity the
        moving base causes (measured TCP velocity minus the arm joints' share) - the drone velocity. If it is too fast,
        the linear TCP velocity is scaled down to ``v_max`` and the joint step corrected by damped least squares. On a
        still base this slows the approach like the joint scaling; on a moving base the arm may still follow the base.
        """
        from isaaclab.utils.math import quat_apply  # noqa: PLC0415

        from drone_grasp_rl.assets.robots import TCP_BODY_NAME, TCP_OFFSET_POS  # noqa: PLC0415

        from .observations import frame_lin_vel_w  # noqa: PLC0415

        robot, env = self._asset, self._env
        if not hasattr(self, "_tcp_body"):
            self._tcp_body = robot.body_names.index(TCP_BODY_NAME)
            self._jac_body = self._tcp_body - 1 if robot.is_fixed_base else self._tcp_body
            self._jac_cols = [j + robot.num_base_dofs for j in self._joint_ids]
            self._tcp_offset = torch.tensor(TCP_OFFSET_POS, device=self.device).repeat(self.num_envs, 1)
            self._eye3 = torch.eye(3, device=self.device)
        jac = robot.data.body_link_jacobian_w.torch[:, self._jac_body][:, :, self._jac_cols]
        r = quat_apply(robot.data.body_link_pose_w.torch[:, self._tcp_body, 3:7], self._tcp_offset)
        jv = jac[:, :3] - torch.linalg.cross(r.unsqueeze(-1).expand(-1, -1, jac.shape[-1]), jac[:, 3:], dim=1)
        dt = env.step_dt
        qd = robot.data.joint_vel.torch[:, self._joint_ids_torch]
        v_base = frame_lin_vel_w(env, "tcp") - (jv @ qd.unsqueeze(-1)).squeeze(-1)
        v_rel = (jv @ step.unsqueeze(-1)).squeeze(-1) / dt + v_base - env.scene["drone"].data.root_lin_vel_w.torch
        speed = v_rel.norm(dim=-1, keepdim=True)
        dv = v_rel * ((v_max / speed.clamp(min=1e-6)).clamp(max=1.0) - 1.0)
        a = jv @ jv.transpose(1, 2) + 0.05**2 * self._eye3
        dq = (jv.transpose(1, 2) @ torch.linalg.solve(a, dv.unsqueeze(-1))).squeeze(-1) * dt
        return (step + dq).clamp(-self.cfg.max_step, self.cfg.max_step)

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
        self._history[ids] = 0.0
        self._smoothed[ids] = 0.0
        num = self._delay[ids].shape[0]
        low, high = self.cfg.delay_steps
        self._delay[ids] = torch.randint(low, high + 1, (num,), device=self.device)


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

    delay_steps: tuple[int, int] = (0, 0)
    """Range of the action delay in policy steps, sampled per episode (inclusive)."""

    smoothing: float = 1.0
    """Weight of the new action in its exponential moving average (1 = no smoothing)."""

    slow_distance: float | None = None
    """TCP distance to the grasp point below which the goal step shrinks [m]; None turns it off."""

    min_step_scale: float = 1.0
    """Smallest fraction of ``max_step`` near the grasp point."""

    slow_mode: str = "joint"
    """``"joint"``: scale the joint goal step by the distance factor. ``"relative"``: limit the TCP speed relative to
    the drone to ``max_relative_speed`` times the factor (the base's motion is not slowed, see
    :meth:`InterpolatedRelativeJointPositionAction._limit_relative_speed`)."""

    max_relative_speed: float = 0.5
    """TCP speed limit relative to the drone at and beyond ``slow_distance`` [m/s] (``slow_mode="relative"``)."""

    hold_after_grasp: bool = False
    """Freeze the goal once the grasp judge has the grasp."""

    judge_term: str = "grasp_success"
    """Termination term holding the grasp judge (for ``hold_after_grasp``)."""


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
