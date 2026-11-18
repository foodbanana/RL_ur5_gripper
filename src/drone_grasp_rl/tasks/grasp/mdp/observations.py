# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the drone grasp task.

Poses come from the ``ee_frame`` frame transformer, whose source is the robot base, so every pose is in
the robot base frame (what the real robot can compute from its own base).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor, FrameTransformer


def frame_pos_b(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Position of a target frame of the frame transformer in its source (robot base) frame [m], shape (N, 3)."""
    sensor: FrameTransformer = env.scene[sensor_cfg.name]
    index = sensor.data.target_frame_names.index(frame_name)
    return sensor.data.target_pos_source.torch[:, index]


def frame_quat_b(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Orientation (x, y, z, w) of a target frame in the robot base frame, with w >= 0, shape (N, 4)."""
    sensor: FrameTransformer = env.scene[sensor_cfg.name]
    index = sensor.data.target_frame_names.index(frame_name)
    return math_utils.quat_unique(sensor.data.target_quat_source.torch[:, index])


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) of a rotation vector [rad], shape (N, 3) -> (N, 4)."""
    return math_utils.quat_from_angle_axis(rotvec.norm(dim=-1), rotvec)


class mocap_drone_pose_b(ManagerTermBase):
    """Drone pose in the robot base frame as a motion-capture system reports it, shape (N, 7): x, y, z, qx, qy, qz, qw.

    The real robot gets the drone pose from motion capture, so the policy sees it the same way. The model follows
    the mocap assumption of the data collection repo (1 mm / 0.3 deg noise, 100 Hz, 20 ms latency):

    - latency: ``latency`` plus the age of the newest mocap sample when the 30 Hz policy reads it, uniform in
      ``[0, sample_period)``. Applied by moving the true pose back in time with the current velocities, which is
      exact for the slow, smooth drone motion here.
    - calibration bias: a constant offset per episode (base-to-mocap extrinsic error), uniform within
      ``pos_bias`` [m] per axis and a rotation of up to ``rot_bias_deg``. Not measured in the data collection
      repo; the defaults are estimates.
    - noise: Gaussian per sample, ``pos_std`` [m] per axis and ``rot_std_deg`` per rotation axis.

    The noise is part of the sensor model, so it stays on when observation corruption is turned off for play.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pos_bias = torch.zeros(env.num_envs, 3, device=env.device)
        self._rot_bias = torch.zeros(env.num_envs, 4, device=env.device)
        self._rot_bias[:, 3] = 1.0

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        num = self._pos_bias[ids].shape[0]
        pos_bias = self.cfg.params.get("pos_bias", 0.0)
        rot_bias = math.radians(self.cfg.params.get("rot_bias_deg", 0.0))
        self._pos_bias[ids] = math_utils.sample_uniform(-pos_bias, pos_bias, (num, 3), self.device)
        # random axis, angle uniform in [0, rot_bias]
        axis = torch.randn(num, 3, device=self.device)
        angle = math_utils.sample_uniform(0.0, rot_bias, (num,), self.device)
        self._rot_bias[ids] = math_utils.quat_from_angle_axis(angle, axis)

    def __call__(
        self,
        env: ManagerBasedEnv,
        frame_name: str = "drone",
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        drone_cfg: SceneEntityCfg = SceneEntityCfg("drone"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        latency: float = 0.020,
        sample_period: float = 0.010,
        pos_std: float = 0.001,
        rot_std_deg: float = 0.3,
        pos_bias: float = 0.002,
        rot_bias_deg: float = 0.5,
    ) -> torch.Tensor:
        pos = frame_pos_b(env, frame_name, sensor_cfg)
        quat = env.scene[sensor_cfg.name].data.target_quat_source.torch[
            :, env.scene[sensor_cfg.name].data.target_frame_names.index(frame_name)
        ]
        # velocities of the drone in the base frame
        base_quat = env.scene[robot_cfg.name].data.root_quat_w.torch
        drone = env.scene[drone_cfg.name]
        lin_vel = math_utils.quat_apply_inverse(base_quat, drone.data.root_lin_vel_w.torch)
        ang_vel = math_utils.quat_apply_inverse(base_quat, drone.data.root_ang_vel_w.torch)
        # latency: the pose the drone had `delay` seconds ago
        delay = latency + sample_period * torch.rand(pos.shape[0], 1, device=pos.device)
        pos = pos - lin_vel * delay
        quat = math_utils.quat_mul(_quat_from_rotvec(-ang_vel * delay), quat)
        # calibration bias, then per-sample noise
        pos = pos + self._pos_bias + pos_std * torch.randn_like(pos)
        quat = math_utils.quat_mul(self._rot_bias, quat)
        quat = math_utils.quat_mul(_quat_from_rotvec(math.radians(rot_std_deg) * torch.randn_like(pos)), quat)
        return torch.cat([pos, math_utils.quat_unique(quat)], dim=-1)


def contact_forces(env: ManagerBasedEnv, sensor_names: list[str]) -> torch.Tensor:
    """Magnitude of the filtered contact force on each sensor body [N], shape (N, len(sensor_names)).

    Each sensor watches one finger link and is filtered to the drone, so this is the force between that
    finger link and the drone only.
    """
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        # (N, bodies, filters, 3) -> total over bodies and filters
        forces.append(sensor.data.force_matrix_w.torch.sum(dim=(1, 2)).norm(dim=-1))
    return torch.stack(forces, dim=-1)
