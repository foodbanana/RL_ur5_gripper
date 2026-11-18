# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp judge: the episode protocol and success criteria of the data collection repo (``grasp_judge.py``).

Phases per environment::

    APPROACH -> GRASPED -> (kill_delay) -> KILLED -> (hold_time) -> HELD        success
                   |                         |
                   +-------------------------+--> FAILED (dropped, slip, opened_early)

- GRASPED: the gripper is being closed and has stopped (mean angle change rate over ``stall_time`` below
  ``stall_velocity``) short of ``empty_angle`` (not an empty close) with more than ``min_force`` on each
  finger side.
- KILLED: ``kill_delay`` after the grasp the drone motors stop: the gravity-compensating wrench is removed and
  the drone's weight goes onto the gripper. The policy does not observe this, as the real drone controller
  does not know it was grasped.
- HELD: ``hold_time`` after the kill the drone has settled at most ``max_settle`` relative to the TCP, moved at
  most ``max_final_change`` in the last 0.5 s, and turned at most ``max_rotation_deg`` relative to the TCP.
- FAILED: opening the gripper after the grasp (``opened_early``), the drone more than ``drop_distance`` away from
  where it was grasped (``dropped``), or the hold criteria not met at the end (``slip``).

The judge runs as the first termination term (``grasp_success``), so the other terminations and the rewards of
the same step read its state through :func:`get_judge`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .actions import _to_index
from .observations import frame_pose_w

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg

APPROACH, GRASPED, KILLED, HELD, FAILED = 0, 1, 2, 3, 4
"""Judge phases."""

FAIL_REASONS = {1: "dropped", 2: "slip", 3: "opened_early"}
"""Failure reason codes stored in :attr:`grasp_judge.fail_reason`."""


def get_judge(env: ManagerBasedRLEnv, term_name: str = "grasp_success") -> grasp_judge:
    """The judge instance of the termination term ``term_name``."""
    return env.termination_manager.get_term_cfg(term_name).func


class grasp_judge(ManagerTermBase):
    """Tracks the grasp phases, stops the drone motors after the grasp, and returns True on success (HELD).

    Params:
        right_sensors, left_sensors: Drone-filtered contact sensors of the right and left finger links.
        gripper_joint: Driven gripper joint.
        gripper_action: Name of the gripper action term (action < 0 closes).
        frame_cfg: Frame transformer with the ``tcp`` and ``drone`` target frames.
        drone_cfg: The drone articulation.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.fail_reason = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_time = torch.zeros(n, device=dev)
        """Time since entering the current phase [s]."""
        self.empty_close = torch.zeros(n, dtype=torch.bool, device=dev)
        """The gripper closed past ``empty_angle`` before any grasp (metric only)."""
        self.ref_pos = torch.zeros(n, 3, device=dev)
        """Drone position in the TCP frame when the current phase began [m]."""
        self.ref_quat = torch.zeros(n, 4, device=dev)
        self.mid_pos = torch.zeros(n, 3, device=dev)
        """Drone position in the TCP frame 0.5 s before the end of the hold [m]."""
        self.mid_set = torch.zeros(n, dtype=torch.bool, device=dev)
        self.was_killed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_phase = torch.zeros(n, dtype=torch.long, device=dev)
        """Final phase of each environment's previous episode (kept across the reset, for logging and datasets)."""
        self.last_fail_reason = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_empty_close = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hold_metrics = torch.zeros(n, 3, device=dev)
        """At the end of the hold: settle [m], change in the last 0.5 s [m], rotation [rad] (kept across resets)."""
        self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
        """Entered HELD or FAILED during the last update."""
        steps = max(1, round(cfg.params.get("stall_time", 0.2) / env.step_dt))
        self.vel_history = torch.ones(n, steps, device=dev)
        self.prev_angle = torch.zeros(n, device=dev)

    def reset(self, env_ids=None):
        ids = _to_index(env_ids, self.device)
        # episode metrics of the environments that end now
        phase = self.phase[ids]
        if phase.numel() > 0:
            log = self._env.extras.setdefault("log", {})
            # every failure happens after a grasp, so FAILED also counts as grasped
            log["Judge/grasped"] = (phase >= GRASPED).float().mean().item()
            log["Judge/killed"] = self.was_killed[ids].float().mean().item()
            log["Judge/held"] = (phase == HELD).float().mean().item()
            log["Judge/empty_close"] = self.empty_close[ids].float().mean().item()
            for code, name in FAIL_REASONS.items():
                log[f"Judge/fail_{name}"] = (self.fail_reason[ids] == code).float().mean().item()
        self.last_phase[ids] = self.phase[ids]
        self.last_fail_reason[ids] = self.fail_reason[ids]
        self.last_empty_close[ids] = self.empty_close[ids]
        self.phase[ids] = APPROACH
        self.fail_reason[ids] = 0
        self.phase_time[ids] = 0.0
        self.empty_close[ids] = False
        self.mid_set[ids] = False
        self.was_killed[ids] = False
        self.finished[ids] = False
        self.vel_history[ids] = 1.0
        self.prev_angle[ids] = 0.0

    def drone_in_tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Drone body pose in the TCP frame: position [m] (N, 3), quaternion (x, y, z, w) (N, 4)."""
        frame_cfg = self.cfg.params.get("frame_cfg", SceneEntityCfg("ee_frame"))
        tcp_pos, tcp_quat = frame_pose_w(self._env, "tcp", frame_cfg)
        drone_pos, drone_quat = frame_pose_w(self._env, "drone", frame_cfg)
        return math_utils.subtract_frame_transforms(tcp_pos, tcp_quat, drone_pos, drone_quat)

    def finger_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Drone contact force on the right and left finger links [N], each (N,)."""

        def side(names):
            return sum(self._env.scene[s].data.force_matrix_w.torch.sum(dim=(1, 2)).norm(dim=-1) for s in names)

        return side(self.cfg.params["right_sensors"]), side(self.cfg.params["left_sensors"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right_sensors: list[str],
        left_sensors: list[str],
        gripper_joint: str,
        gripper_action: str = "gripper_action",
        frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        drone_cfg: SceneEntityCfg = SceneEntityCfg("drone"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        stall_time: float = 0.2,
        stall_velocity: float = 0.01,
        empty_angle_deg: float = 62.0,
        min_force: float = 1.0,
        kill_delay: float = 1.0,
        hold_time: float = 2.5,
        max_settle: float = 0.015,
        max_final_change: float = 0.0005,
        max_rotation_deg: float = 10.0,
        drop_distance: float = 0.03,
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        joint = robot.joint_names.index(gripper_joint)
        angle = robot.data.joint_pos.torch[:, joint]
        # Speed from the angle change, as in the old judge: the reported joint velocity of the stalled gripper
        # stays around 0.6 rad/s in PhysX while the angle does not move.
        speed = (angle - self.prev_angle).abs() / env.step_dt
        self.prev_angle = angle.clone()
        closing = env.action_manager.get_term(gripper_action).raw_actions[:, 0] < 0
        right, left = self.finger_forces()
        rel_pos, rel_quat = self.drone_in_tcp()
        empty_angle = math.radians(empty_angle_deg)

        self.vel_history = torch.roll(self.vel_history, 1, dims=1)
        self.vel_history[:, 0] = torch.where(closing, speed, torch.ones_like(speed))
        self.phase_time += env.step_dt
        self.finished[:] = False
        self.empty_close |= (self.phase == APPROACH) & closing & (angle > empty_angle)

        def enter(mask, phase):
            self.phase[mask] = phase
            self.phase_time[mask] = 0.0
            self.ref_pos[mask] = rel_pos[mask]
            self.ref_quat[mask] = rel_quat[mask]

        def fail(mask, code):
            mask = mask & (self.phase != FAILED) & (self.phase != HELD)
            self.fail_reason[mask] = code
            self.phase[mask] = FAILED
            self.finished |= mask

        # APPROACH -> GRASPED
        stalled = self.vel_history.mean(dim=1) < stall_velocity
        grasped = (
            (self.phase == APPROACH)
            & closing
            & stalled
            & (angle < empty_angle)
            & (right > min_force)
            & (left > min_force)
        )
        enter(grasped, GRASPED)

        holding = (self.phase == GRASPED) | (self.phase == KILLED)
        moved = (rel_pos - self.ref_pos).norm(dim=-1)
        fail(holding & ~closing, 3)
        fail(holding & (moved > drop_distance), 1)

        # GRASPED -> KILLED: stop the drone motors
        kill = (self.phase == GRASPED) & (self.phase_time >= kill_delay)
        if kill.any():
            drone: Articulation = env.scene[drone_cfg.name]
            ids = kill.nonzero(as_tuple=False).squeeze(-1)
            drone.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.zeros(len(ids), drone.num_bodies, 3, device=self.device),
                body_ids=list(range(drone.num_bodies)),
                env_ids=ids,
                is_global=True,
            )
            enter(kill, KILLED)
            self.was_killed |= kill

        # KILLED -> HELD or slip
        killed = self.phase == KILLED
        mid = killed & ~self.mid_set & (self.phase_time >= hold_time - 0.5)
        self.mid_pos[mid] = rel_pos[mid]
        self.mid_set |= mid
        done = killed & (self.phase_time >= hold_time)
        settle = (rel_pos - self.ref_pos).norm(dim=-1)
        change = (rel_pos - self.mid_pos).norm(dim=-1)
        rotation = math_utils.quat_error_magnitude(rel_quat, self.ref_quat)
        ok = (settle <= max_settle) & (change <= max_final_change) & (rotation <= math.radians(max_rotation_deg))
        self.hold_metrics[done] = torch.stack([settle, change, rotation], dim=-1)[done]
        held = done & ok
        self.phase[held] = HELD
        self.finished |= held
        fail(done & ~ok, 2)
        return held


def grasp_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The judge failed the grasp in this step (dropped, slip or opened early)."""
    judge = get_judge(env)
    return judge.finished & (judge.phase == FAILED)
