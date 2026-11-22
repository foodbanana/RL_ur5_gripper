# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the drone grasp task.

Poses are defined by the ``ee_frame`` frame transformer config (source = robot base, target frames ``tcp`` and
``drone``), so every pose is in the robot base frame (what the real robot can compute from its own base). They are
computed from the articulation body poses, not read from the sensor: on Newton the sensor data stays zero until the
first physics step, so the first observation after the initial reset would be wrong.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .frames import base_quat_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor


def _resolve_frame(env: ManagerBasedEnv, prim_path: str) -> tuple:
    """(articulation, body index) of the body a frame transformer prim path points at."""
    for asset in env.scene.articulations.values():
        root = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "")
        if prim_path.replace("{ENV_REGEX_NS}", "").startswith(root + "/"):
            return asset, asset.body_names.index(prim_path.rsplit("/", 1)[-1])
    raise ValueError(f"No articulation body for frame prim path '{prim_path}'.")


def frame_pose_w(
    env: ManagerBasedEnv, frame_name: str | None, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose of a target frame of the frame transformer config (``None``: its source frame), from the
    articulation body poses: position (N, 3) [m], quaternion (x, y, z, w) (N, 4)."""
    cache = env.__dict__.setdefault("_drone_grasp_frames", {})
    key = (sensor_cfg.name, frame_name)
    if key not in cache:
        cfg = env.scene[sensor_cfg.name].cfg
        if frame_name is None:
            prim_path, offset = cfg.prim_path, cfg.source_frame_offset
        else:
            frame = next(f for f in cfg.target_frames if f.name == frame_name)
            prim_path, offset = frame.prim_path, frame.offset
        asset, body = _resolve_frame(env, prim_path)
        pos = torch.tensor(offset.pos, device=env.device).repeat(env.num_envs, 1)
        rot = torch.tensor(offset.rot, device=env.device).repeat(env.num_envs, 1)
        cache[key] = (asset, body, pos, rot)
    asset, body, off_pos, off_rot = cache[key]
    pose = asset.data.body_link_pose_w.torch[:, body]
    return math_utils.combine_frame_transforms(pose[:, :3], pose[:, 3:7], off_pos, off_rot)


def frame_lin_vel_w(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """World-frame linear velocity of a target frame of the frame transformer config [m/s], shape (N, 3)."""
    frame_pose_w(env, frame_name, sensor_cfg)  # fills the cache
    asset, body, off_pos, _ = env.__dict__["_drone_grasp_frames"][(sensor_cfg.name, frame_name)]
    quat = asset.data.body_link_pose_w.torch[:, body, 3:7]
    lin = asset.data.body_link_lin_vel_w.torch[:, body]
    ang = asset.data.body_link_ang_vel_w.torch[:, body]
    return lin + torch.linalg.cross(ang, math_utils.quat_apply(quat, off_pos), dim=-1)


def frame_pose_b(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of a target frame in the source (robot base) frame: position (N, 3) [m], quaternion (x, y, z, w)."""
    base_pos, base_quat = frame_pose_w(env, None, sensor_cfg)
    pos, quat = frame_pose_w(env, frame_name, sensor_cfg)
    return math_utils.subtract_frame_transforms(base_pos, base_quat, pos, quat)


def frame_pos_b(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Position of a target frame in the robot base frame [m], shape (N, 3)."""
    return frame_pose_b(env, frame_name, sensor_cfg)[0]


def frame_quat_b(
    env: ManagerBasedEnv, frame_name: str, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Orientation (x, y, z, w) of a target frame in the robot base frame, with w >= 0, shape (N, 4)."""
    return math_utils.quat_unique(frame_pose_b(env, frame_name, sensor_cfg)[1])


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
    - dropout: occlusion of the markers. At ``dropout_rate`` events per second a dropout of ``dropout_duration`` [s]
      starts, during which the last reading is repeated (as in OpenAI's in-hand manipulation marker dropout).

    The noise is part of the sensor model, so it stays on when observation corruption is turned off for play.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pos_bias = torch.zeros(env.num_envs, 3, device=env.device)
        self._rot_bias = torch.zeros(env.num_envs, 4, device=env.device)
        self._rot_bias[:, 3] = 1.0
        self._last = torch.zeros(env.num_envs, 7, device=env.device)
        self._dropout_left = torch.zeros(env.num_envs, device=env.device)

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
        self._dropout_left[ids] = 0.0

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
        dropout_rate: float = 0.0,
        dropout_duration: tuple[float, float] = (0.1, 0.5),
    ) -> torch.Tensor:
        pos, quat = frame_pose_b(env, frame_name, sensor_cfg)
        # velocities of the drone in the base frame
        base_quat = base_quat_w(env.scene[robot_cfg.name])
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
        reading = torch.cat([pos, math_utils.quat_unique(quat)], dim=-1)
        if dropout_rate > 0.0:
            # the first call after a reset has no previous reading to repeat
            fresh = env.episode_length_buf <= 1
            start = (torch.rand_like(self._dropout_left) < dropout_rate * env.step_dt) & ~fresh
            start &= self._dropout_left <= 0.0
            duration = math_utils.sample_uniform(*dropout_duration, self._dropout_left.shape, self.device)
            self._dropout_left = torch.where(start, duration, self._dropout_left)
            reading = torch.where((self._dropout_left > 0.0).unsqueeze(-1), self._last, reading)
            self._dropout_left = (self._dropout_left - env.step_dt).clamp(min=0.0)
        self._last = reading.clone()
        return reading


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


# -- privileged observations of the teacher policy (simulation only) --


def drone_vel_b(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Exact drone linear [m/s] and angular [rad/s] velocity in the robot base frame, shape (N, 6)."""
    base_quat = base_quat_w(env.scene[robot_cfg.name])
    drone = env.scene["drone"]
    lin = math_utils.quat_apply_inverse(base_quat, drone.data.root_lin_vel_w.torch)
    ang = math_utils.quat_apply_inverse(base_quat, drone.data.root_ang_vel_w.torch)
    return torch.cat([lin, ang], dim=-1)


def drone_pos_in_tcp(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Exact drone body position in the TCP frame [m], shape (N, 3)."""
    tcp_pos, tcp_quat = frame_pose_w(env, "tcp", sensor_cfg)
    drone_pos, _ = frame_pose_w(env, "drone", sensor_cfg)
    return math_utils.quat_apply_inverse(tcp_quat, drone_pos - tcp_pos)


def gripper_closing(env: ManagerBasedEnv, gripper_action: str = "gripper_action") -> torch.Tensor:
    """1 once the (latched) close command is in force, shape (N, 1)."""
    return env.action_manager.get_term(gripper_action).closing.float().unsqueeze(-1)


def grasp_detected(env: ManagerBasedEnv, judge_term: str = "grasp_success") -> torch.Tensor:
    """1 while the judge has the drone grasped (also after the motor stop, which is not revealed), shape (N, 1).

    Something a camera student can also see (closed fingers around the drone); the motor stop is left out on
    purpose, since the student could not know it.
    """
    # the observation manager evaluates its terms once before the termination manager exists
    manager = getattr(env, "termination_manager", None)
    if manager is None or judge_term not in manager.active_terms:
        return torch.zeros(env.num_envs, 1, device=env.device)
    phase = manager.get_term_cfg(judge_term).func.phase
    return ((phase == 1) | (phase == 2) | (phase == 3)).float().unsqueeze(-1)
