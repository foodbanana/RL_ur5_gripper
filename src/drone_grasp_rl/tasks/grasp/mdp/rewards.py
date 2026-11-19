# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the drone grasp task.

The grasp phases come from the judge (:mod:`.judge`), which runs in the termination manager before the rewards of
the same step. Success and failure rewards use the built-in ``is_terminated_term`` on the judge's terminations.
Sources of each term and weight are listed in the task's reward table (``grasp_env_cfg.RewardsCfg``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .actions import _to_index
from .drone_control import drone_target_pos_b
from .judge import APPROACH, GRASPED, KILLED, get_judge
from .observations import frame_lin_vel_w, frame_pos_b, frame_pose_b, frame_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

GRASP_OFFSET = (0.0, -0.0013, -0.0045)
"""Planned TCP position in the drone body frame [m]: the waist center (data collection repo)."""


def tcp_in_grasp_frame(
    env: ManagerBasedRLEnv,
    grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """TCP position relative to the grasp point, in the drone's axes [m], shape (N, 3). z < 0 is below the drone."""
    drone_pos, drone_quat = frame_pose_w(env, "drone", sensor_cfg)
    tcp_pos, _ = frame_pose_w(env, "tcp", sensor_cfg)
    offset = torch.tensor(grasp_offset, device=env.device).expand_as(drone_pos)
    grasp = drone_pos + math_utils.quat_apply(drone_quat, offset)
    return math_utils.quat_apply_inverse(drone_quat, tcp_pos - grasp)


def approach_path_distance(rel: torch.Tensor, pre_distance: float, lateral_tolerance: float) -> torch.Tensor:
    """Remaining length of the intended path to the grasp point [m], shape (N,).

    The path goes to the pre-grasp point ``pre_distance`` below the grasp point, then straight up. Once the TCP is
    below the drone and within ``lateral_tolerance`` of the vertical line, the remaining path is the straight
    distance; otherwise it is the distance to the pre-grasp point plus ``pre_distance``. Approaching from the side
    or from above therefore never shortens the path.
    """
    lateral = rel[:, :2].norm(dim=-1)
    on_line = (lateral < lateral_tolerance) & (rel[:, 2] < 0.005)
    pre = rel.clone()
    pre[:, 2] += pre_distance
    return torch.where(on_line, rel.norm(dim=-1), pre.norm(dim=-1) + pre_distance)


class approach_progress(ManagerTermBase):
    """1 on every step that shortens the approach path below its episode best by ``min_improvement``.

    Progress reward as in Isaac Lab's core lift task (``position_command_progress``): already credited ground cannot
    be earned again by backing off. Only paid before the grasp. Logs ``Metrics/approach_best_mm``, the shortest
    remaining path reached in each finished episode.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best = torch.full((env.num_envs,), float("inf"), device=env.device)

    def reset(self, env_ids=None):
        ids = _to_index(env_ids, self.device)
        best = self.best[ids]
        best = best[torch.isfinite(best)]
        if best.numel() > 0:
            self._env.extras.setdefault("log", {})["Metrics/approach_best_mm"] = 1000.0 * best.mean().item()
        self.best[ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_improvement: float = 0.005,
        pre_distance: float = 0.10,
        lateral_tolerance: float = 0.02,
        grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
    ) -> torch.Tensor:
        path = approach_path_distance(tcp_in_grasp_frame(env, grasp_offset), pre_distance, lateral_tolerance)
        self.best = torch.where(torch.isinf(self.best), path, self.best)
        improved = (get_judge(env).phase == APPROACH) & (path < self.best - min_improvement)
        self.best = torch.where(improved, path, self.best)
        return improved.float()


def approach_fine(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    pre_distance: float = 0.10,
    lateral_tolerance: float = 0.02,
    grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
) -> torch.Tensor:
    """``1 - tanh(d / std)`` of the TCP distance ``d`` to the grasp point, while the TCP is on the vertical approach
    line below the drone and before the grasp, shape (N,).

    Dense shaping for the last centimeters (the coarse/fine pair of contrib lift): the progress reward alone left the
    policy about 4 cm below the waist, where sliding the open fingers past the drone needs millimeter centering.
    """
    rel = tcp_in_grasp_frame(env, grasp_offset)
    on_line = (rel[:, :2].norm(dim=-1) < lateral_tolerance) & (rel[:, 2] < 0.005)
    phase_ok = get_judge(env).phase == APPROACH
    return (1.0 - torch.tanh(rel.norm(dim=-1) / std)) * (on_line & phase_ok)


def approach_speed_excess(
    env: ManagerBasedRLEnv,
    base_speed: float = 0.05,
    speed_per_distance: float = 1.0,
    active_distance: float = 0.10,
    grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """TCP speed relative to the drone above ``base_speed + speed_per_distance * d`` [m/s], before the grasp, (N,).

    ``d`` is the TCP distance to the grasp point, so the allowed speed shrinks as the gripper closes in (5 cm/s at
    the grasp point, 15 cm/s at 10 cm), like a teleoperator slowing down. The trained policy reached the waist fast
    enough to look dangerous (median 27 cm/s in the last 5 cm) and overshot by about 12 mm. Only within
    ``active_distance``: charged along the whole approach, it taught the policy within 10 iterations to stop moving.
    """
    rel = tcp_in_grasp_frame(env, grasp_offset)
    drone_vel = env.scene["drone"].data.root_lin_vel_w.torch
    speed = (frame_lin_vel_w(env, "tcp", sensor_cfg) - drone_vel).norm(dim=-1)
    distance = rel.norm(dim=-1)
    limit = base_speed + speed_per_distance * distance
    active = (get_judge(env).phase == APPROACH) & (distance < active_distance)
    return (speed - limit).clamp(min=0.0) * active


def misalignment(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Gripper misalignment with the drone, 0 when aligned, up to 3, shape (N,).

    ``(1 - z_tcp . z_drone) + (1 - |y_tcp . y_drone|)``: approach axis (TCP z) along the drone z axis and closing
    axis (TCP y, the fingers open along it) along the drone y axis, either sign since the gripper is symmetric.
    A penalty rather than a reward, because the start pose is already aligned.
    """
    _, tcp_quat = frame_pose_w(env, "tcp", sensor_cfg)
    _, drone_quat = frame_pose_w(env, "drone", sensor_cfg)
    tcp_rot, drone_rot = math_utils.matrix_from_quat(tcp_quat), math_utils.matrix_from_quat(drone_quat)
    z_dot = (tcp_rot[..., 2] * drone_rot[..., 2]).sum(dim=-1)
    y_dot = (tcp_rot[..., 1] * drone_rot[..., 1]).sum(dim=-1)
    return (1.0 - z_dot) + (1.0 - y_dot.abs())


def gripper_tilt(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """``1 - cos`` of the angle between the TCP z axis (approach axis) and the robot base +z axis (up), shape (N,).

    The misalignment term compares the gripper with the drone, so after the grasp, when both turn together, it is
    zero; the trained policy then tilted the grasped drone by a median 39 deg. The hovering drone is upright, so
    this also holds before the grasp.
    """
    _, tcp_quat_b = frame_pose_b(env, "tcp", sensor_cfg)
    return 1.0 - math_utils.matrix_from_quat(tcp_quat_b)[..., 2, 2]


def drone_tilt(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """``1 - cos`` of the angle between the drone z axis and the robot base +z axis (up), shape (N,).

    Zero while the drone hovers upright; after the grasp it keeps the held drone upright, which the gripper tilt
    alone does not fully cover (the drone can turn in the fingers).
    """
    _, drone_quat_b = frame_pose_b(env, "drone", sensor_cfg)
    return 1.0 - math_utils.matrix_from_quat(drone_quat_b)[..., 2, 2]


def arm_motion_while_closing(
    env: ManagerBasedRLEnv,
    gate: str = "close",
    norm: str = "l2",
    arm_action: str = "arm_action",
    gripper_action: str = "gripper_action",
) -> torch.Tensor:
    """Squared arm action (goal step) from the close command until the episode ends, shape (N,).

    Keeps the arm still while the fingers close and hold, as the scripted expert does and like the "keep the hand
    still after contact" term of in-flight catching (Hu et al., 2023). Gated on the close command rather than on the
    judge's grasp: the policy kept rising while the fingers closed and tilted the gripper and the drone by ~30 deg
    before the grasp was even detected. ``gate="grasp"`` applies it only once the judge has the grasp, which leaves
    the policy free to correct the tilt the closing fingers cause.
    """
    actions = env.action_manager.get_term(arm_action).raw_actions.clamp(-1.0, 1.0)
    # "l1" also charges the small actions that made the arm shiver after the grasp (squares of ~0.1 are ~0)
    size = actions.abs().sum(dim=-1) if norm == "l1" else actions.square().sum(dim=-1)
    if gate == "grasp":
        phase = get_judge(env).phase
        active = (phase == GRASPED) | (phase == KILLED)
    else:
        active = env.action_manager.get_term(gripper_action).closing
    return size * active


def finger_contact(env: ManagerBasedRLEnv, min_force: float = 1.0) -> torch.Tensor:
    """1 while both finger sides press on the drone with more than ``min_force`` [N], until the motor stop."""
    judge = get_judge(env)
    right, left = judge.finger_forces()
    phase_ok = (judge.phase == APPROACH) | (judge.phase == GRASPED)
    return ((right > min_force) & (left > min_force) & phase_ok).float()


def empty_close(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.03,
    gripper_action: str = "gripper_action",
    grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
) -> torch.Tensor:
    """1 while the gripper is commanded closed before the grasp with the TCP farther than ``max_distance`` [m]
    from the grasp point (the judge's ``no_grasp_empty`` in the making)."""
    closing = env.action_manager.get_term(gripper_action).closing
    far = tcp_in_grasp_frame(env, grasp_offset).norm(dim=-1) > max_distance
    return (closing & far & (get_judge(env).phase == APPROACH)).float()


def close_near(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.015,
    lateral_tolerance: float = 0.01,
    gripper_action: str = "gripper_action",
    grasp_offset: tuple[float, float, float] = GRASP_OFFSET,
) -> torch.Tensor:
    """1 while the gripper is commanded closed with the TCP below the waist and within ``max_distance`` [m] of the
    grasp point, before the grasp (like the cabinet task's ``grasp_handle``).

    Without it the policy reached the waist but never tried to close: closing had only been penalized far away.
    """
    closing = env.action_manager.get_term(gripper_action).closing
    rel = tcp_in_grasp_frame(env, grasp_offset)
    near = (rel.norm(dim=-1) < max_distance) & (rel[:, :2].norm(dim=-1) < lateral_tolerance)
    return (closing & near & (get_judge(env).phase == APPROACH)).float()


def hold(env: ManagerBasedRLEnv, min_force: float = 1.0, max_settle: float = 0.015) -> torch.Tensor:
    """1 while, after the motor stop, both finger sides still press on the drone and it has settled at most
    ``max_settle`` [m] relative to the TCP (the judge's hold criterion)."""
    judge = get_judge(env)
    right, left = judge.finger_forces()
    rel_pos, _ = judge.drone_in_tcp()
    settled = (rel_pos - judge.ref_pos).norm(dim=-1) <= max_settle
    return ((judge.phase == KILLED) & (right > min_force) & (left > min_force) & settled).float()


def drone_push(env: ManagerBasedRLEnv, deadband: float = 0.04) -> torch.Tensor:
    """Distance of the drone from where it is supposed to be (hover target or reset position) beyond ``deadband``,
    before the grasp [m], shape (N,). The deadband is above the drone's own hover jitter (about 19 mm RMS, at most
    about 45 mm with the disturbance force), so only pushes count."""
    moved = (frame_pos_b(env, "drone") - drone_target_pos_b(env)).norm(dim=-1)
    return (moved - deadband).clamp(min=0.0) * (get_judge(env).phase == APPROACH)


class terminal_bonus(ManagerTermBase):
    """1 / step_dt on the step one of ``term_keys`` terminates the episode (time-outs excluded).

    The reward manager scales every term by the step interval; dividing it out makes the weight the exact amount
    paid once, like Isaac Lab's ``terminated_penalty``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_names = env.termination_manager.find_terms(cfg.params["term_keys"])

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str]) -> torch.Tensor:
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in self._term_names:
            done |= env.termination_manager.get_term(name)
        return (done & ~env.termination_manager.time_outs).float() / env.step_dt
