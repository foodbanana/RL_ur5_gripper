# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone flight controller and hover motion, run as an action term without policy input.

The drone is not controlled by the policy; it flies on its own like the real one (PX4 in the data collection repo).
An action term is used because it runs on every physics step (``apply_actions``), before the scene writes to the
simulation, which a flight controller needs. Its action dimension is 0, so the policy's action space is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .actions import _to_index

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

HOVER_PRESETS = {
    "static": ((0.0, 0.0, 0.0), (0.0, 0.0)),
    "small": ((0.020, 0.020, 0.010), (0.2, 0.5)),
    "mid": ((0.030, 0.030, 0.015), (0.05, 0.3)),
    "big": ((0.050, 0.050, 0.020), (0.05, 0.3)),
}
"""Hover target motion of the data collection repo (``drone_iris.yaml``): summed amplitude per axis x, y, z [m] and
the frequency range of each sine [Hz]. ``mid`` is the default for data collection and evaluation there."""


class DroneHoverController(ActionTerm):
    """Geometric position/attitude controller that keeps the drone on its hover target, with motor stop.

    Each physics step:

    - target ``p(t) = p0 + sum_j (A/2) [sin(2 pi f_j t + phi_j) - sin(phi_j)]`` per axis with two sines (starts at
      ``p0``, the reset position), with ``A``, ``f_j`` from a preset sampled per reset and ``phi_j`` random;
    - desired force ``F = -Kp e_p - Kd e_v - Ki int(e_p) + m (g + a_ref)`` in the world frame;
    - thrust ``T = F . b3`` along the body z axis only (a quadrotor cannot push sideways), clamped to [0, T_max];
    - torque ``-Kr e_R - Kw omega`` toward the attitude whose z axis is along ``F`` with the reset yaw;
    - body-frame linear drag ``-D v``; optional random disturbance force (an Ornstein-Uhlenbeck process);
    - after the judge's motor stop: no thrust, no torque, no disturbance (drag stays).

    The force and torque go to the drone body as a permanent world-frame wrench. The rotors hang on the body, so
    their weight is carried through the joints. Gains are the Pegasus Iris geometric controller's (the data
    collection repo's alternative to PX4), scaled by mass (position) and inertia (attitude) to the 0.75x drone.
    """

    cfg: DroneHoverControllerCfg
    _asset: Articulation

    def __init__(self, cfg: DroneHoverControllerCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._dt = env.physics_dt
        self._body = self._asset.body_names.index(cfg.body_name)
        self._p0 = torch.zeros(n, 3, device=dev)
        self._yaw_axis = torch.zeros(n, 3, device=dev)
        self._yaw_axis[:, 0] = 1.0
        self._amp = torch.zeros(n, 3, 2, device=dev)
        self._freq = torch.zeros(n, 3, 2, device=dev)
        self._phase = torch.zeros(n, 3, 2, device=dev)
        self._time = torch.zeros(n, device=dev)
        self._integral = torch.zeros(n, 3, device=dev)
        self._disturbance = torch.zeros(n, 3, device=dev)
        self._preset_index = torch.zeros(n, dtype=torch.long, device=dev)
        self._ref_pos = torch.zeros(n, 3, device=dev)
        names = list(cfg.preset_weights)
        self.preset_names = names
        weights = torch.tensor([cfg.preset_weights[k] for k in names], device=dev)
        self._preset_probs = weights / weights.sum()
        self._preset_amp = torch.tensor([HOVER_PRESETS[k][0] for k in names], device=dev)
        self._preset_freq = torch.tensor([HOVER_PRESETS[k][1] for k in names], device=dev)
        self._drag = torch.tensor(cfg.linear_drag, device=dev)
        self._raw = torch.zeros(n, 0, device=dev)

    # -- action term interface (no policy input) --
    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor):
        pass

    # -- state --
    @property
    def reference_pos_w(self) -> torch.Tensor:
        """Current hover target of each drone in the world frame [m], shape (N, 3)."""
        return self._ref_pos

    @property
    def preset_index(self) -> torch.Tensor:
        """Index into :attr:`preset_names` of each environment's hover preset."""
        return self._preset_index

    def _motors_off(self) -> torch.Tensor:
        judge_term = self.cfg.judge_term
        if judge_term is None or judge_term not in self._env.termination_manager.active_terms:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._env.termination_manager.get_term_cfg(judge_term).func.was_killed

    def _reference(self, t: torch.Tensor):
        w = 2.0 * math.pi * self._freq
        arg = w * t.view(-1, 1, 1) + self._phase
        half = 0.5 * self._amp
        pos = self._p0 + (half * (torch.sin(arg) - torch.sin(self._phase))).sum(dim=-1)
        vel = (half * w * torch.cos(arg)).sum(dim=-1)
        acc = (-half * w * w * torch.sin(arg)).sum(dim=-1)
        return pos, vel, acc

    def _float(self):
        """Per-body gravity compensation: the drone floats exactly where it is put (no flight dynamics)."""
        mass = self._asset.data.body_mass.torch
        forces = torch.zeros(self.num_envs, self._asset.num_bodies, 3, device=self.device)
        forces[..., 2] = -self._env.sim.cfg.gravity[2] * mass * (~self._motors_off()).unsqueeze(-1)
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, is_global=True)

    def apply_actions(self):
        cfg = self.cfg
        data = self._asset.data
        self._time += self._dt
        if cfg.mode == "float":
            self._ref_pos = self._reference(self._time)[0]
            self._float()
            return
        ref_pos, ref_vel, ref_acc = self._reference(self._time)
        self._ref_pos = ref_pos
        pose = data.body_link_pose_w.torch[:, self._body]
        pos, quat = pose[:, :3], pose[:, 3:7]
        vel = data.body_link_lin_vel_w.torch[:, self._body]
        omega_w = data.body_link_ang_vel_w.torch[:, self._body]
        mass = data.body_mass.torch.sum(dim=-1, keepdim=True)
        rot = math_utils.matrix_from_quat(quat)
        b3 = rot[..., 2]

        # position loop
        e_p = pos - ref_pos
        e_v = vel - ref_vel
        self._integral = (self._integral + e_p * self._dt).clamp(-cfg.integral_limit, cfg.integral_limit)
        gravity = torch.tensor([0.0, 0.0, -self._env.sim.cfg.gravity[2]], device=self.device)
        force_des = -cfg.kp * e_p - cfg.kd * e_v - cfg.ki * self._integral + mass * (gravity + ref_acc)
        thrust = (force_des * b3).sum(dim=-1, keepdim=True).clamp(0.0, cfg.max_thrust)

        # attitude loop: z axis along the desired force, x axis toward the reset yaw
        b3_d = math_utils.normalize(force_des)
        b2_d = math_utils.normalize(torch.linalg.cross(b3_d, self._yaw_axis, dim=-1))
        b1_d = torch.linalg.cross(b2_d, b3_d, dim=-1)
        rot_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)
        e_rot_m = rot_d.transpose(1, 2) @ rot - rot.transpose(1, 2) @ rot_d
        e_rot = 0.5 * torch.stack([e_rot_m[:, 2, 1], e_rot_m[:, 0, 2], e_rot_m[:, 1, 0]], dim=-1)
        omega_b = (rot.transpose(1, 2) @ omega_w.unsqueeze(-1)).squeeze(-1)
        torque_b = -cfg.kr * e_rot - cfg.kw * omega_b

        # disturbance (OU process) and drag
        if cfg.disturbance_std > 0.0:
            decay = math.exp(-self._dt / cfg.disturbance_time)
            noise = torch.randn_like(self._disturbance) * cfg.disturbance_std * math.sqrt(1.0 - decay * decay)
            self._disturbance = decay * self._disturbance + noise
        vel_b = (rot.transpose(1, 2) @ vel.unsqueeze(-1)).squeeze(-1)
        drag_w = (rot @ (-self._drag * vel_b).unsqueeze(-1)).squeeze(-1)

        on = (~self._motors_off()).unsqueeze(-1).float()
        force_w = on * (thrust * b3 + self._disturbance) + drag_w
        torque_w = on * (rot @ torque_b.unsqueeze(-1)).squeeze(-1)
        forces = torch.zeros(self.num_envs, self._asset.num_bodies, 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[:, self._body] = force_w
        torques[:, self._body] = torque_w
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torques, is_global=True
        )

    def reset(self, env_ids=None) -> None:
        ids = _to_index(env_ids, self.device)
        num = self._time[ids].shape[0]
        if num == 0:
            return
        # the reset events already placed the drone; start the hover target there
        pose = self._asset.data.root_pose_w.torch[ids]
        self._p0[ids] = pose[:, :3]
        x_axis = math_utils.matrix_from_quat(pose[:, 3:7])[..., 0]
        x_axis[:, 2] = 0.0
        self._yaw_axis[ids] = math_utils.normalize(x_axis)
        preset = torch.multinomial(self._preset_probs, num, replacement=True)
        self._preset_index[ids] = preset
        amp = self._preset_amp[preset]  # (num, 3)
        fmin, fmax = self._preset_freq[preset, 0], self._preset_freq[preset, 1]
        self._amp[ids] = amp.unsqueeze(-1).expand(-1, -1, 2)
        u = torch.rand(num, 3, 2, device=self.device)
        self._freq[ids] = fmin.view(-1, 1, 1) + u * (fmax - fmin).view(-1, 1, 1)
        self._phase[ids] = 2.0 * math.pi * torch.rand(num, 3, 2, device=self.device)
        self._time[ids] = 0.0
        self._integral[ids] = 0.0
        self._disturbance[ids] = 0.0
        self._ref_pos[ids] = self._p0[ids]


@configclass
class DroneHoverControllerCfg(ActionTermCfg):
    """Configuration of :class:`DroneHoverController`."""

    class_type: type = DroneHoverController

    body_name: str = MISSING
    """Drone body that receives the thrust and torque."""

    mode: str = "fly"
    """``"fly"``: the flight controller. ``"float"``: per-body gravity compensation only, the drone stays where it is
    put and drifts freely when pushed (used by the task checks)."""

    preset_weights: dict[str, float] = {"mid": 1.0}
    """Relative probability of each hover preset (keys of :data:`HOVER_PRESETS`), sampled per reset."""

    judge_term: str | None = "grasp_success"
    """Termination term holding the grasp judge; its motor stop turns the controller off. None: never off."""

    kp: float = 5.33
    """Position gain [N/m] (Pegasus Iris 10 x 0.8/1.5 kg)."""

    kd: float = 4.53
    """Velocity gain [N s/m] (8.5 x 0.8/1.5)."""

    ki: float = 0.8
    """Integral gain [N/(m s)] (1.5 x 0.8/1.5)."""

    integral_limit: float = 0.5
    """Clamp of the position error integral [m s] (anti-windup while the gripper holds the drone)."""

    kr: float = 0.52
    """Attitude gain [N m/rad] (3.5 x about 0.15, the inertia ratio)."""

    kw: float = 0.075
    """Angular rate gain [N m s/rad] (0.5 x about 0.15)."""

    max_thrust: float = 41.0
    """Total thrust limit [N] (4 rotors x about 10.3 N at the maximum rotor speed)."""

    linear_drag: tuple[float, float, float] = (0.50, 0.30, 0.0)
    """Body-frame linear drag coefficients [N s/m] (Pegasus Iris)."""

    disturbance_std: float = 0.0
    """Standard deviation of the random disturbance force per axis [N]; 0 turns it off."""

    disturbance_time: float = 0.5
    """Correlation time of the disturbance force [s]."""


def drone_target_pos_b(env: ManagerBasedEnv, term_name: str = "drone_hover", reset_event: str = "reset_drone"):
    """Where the drone is supposed to be, in the robot base frame [m], shape (N, 3): the hover controller's target,
    or the reset position when there is no controller."""
    if term_name in env.action_manager.active_terms:
        robot = env.scene["robot"]
        target_w = env.action_manager.get_term(term_name).reference_pos_w
        return math_utils.quat_apply_inverse(robot.data.root_quat_w.torch, target_w - robot.data.root_pos_w.torch)
    return env.event_manager.get_term_cfg(reset_event).func.reset_pos_b
