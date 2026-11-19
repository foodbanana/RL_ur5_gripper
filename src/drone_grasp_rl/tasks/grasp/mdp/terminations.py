# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failure terminations of the drone grasp task (the judge's own terminations are in :mod:`.judge`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .drone_control import drone_target_pos_b
from .judge import APPROACH, get_judge
from .observations import frame_pos_b

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def drone_knocked(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.10,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Before the grasp, the drone is more than ``max_distance`` [m] from where it is supposed to be (its hover
    target, or the reset position without a flight controller)."""
    moved = (frame_pos_b(env, "drone", sensor_cfg) - drone_target_pos_b(env)).norm(dim=-1)
    return (get_judge(env).phase == APPROACH) & (moved > max_distance)


def protective_stop(
    env: ManagerBasedRLEnv,
    arm_action: str = "arm_action",
    max_tracking_error_deg: float = 5.0,
    max_joint_speed_deg: float = 200.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The real UR would stop: an arm joint lags its drive target by more than ``max_tracking_error_deg`` or moves
    faster than ``max_joint_speed_deg`` per second (values of the data collection repo's protective stop).

    The arm action interpolates its goal, so in normal motion the error stays well below the limit; a large error
    means the arm is blocked, e.g. by the table or the drone. The repo's 150 N link contact limit is not modeled
    yet.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    term = env.action_manager.get_term(arm_action)
    ids = term.joint_ids
    error = (term.target - robot.data.joint_pos.torch[:, ids]).abs().max(dim=-1).values
    speed = robot.data.joint_vel.torch[:, ids].abs().max(dim=-1).values
    return (error > math.radians(max_tracking_error_deg)) | (speed > math.radians(max_joint_speed_deg))
