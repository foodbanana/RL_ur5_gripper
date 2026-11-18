# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the drone grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import frame_pos_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tcp_drone_distance_tanh(
    env: ManagerBasedRLEnv, std: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """``1 - tanh(d / std)`` of the distance ``d`` between the TCP and the drone body origin (the grasp point)."""
    distance = torch.norm(frame_pos_b(env, "drone", sensor_cfg) - frame_pos_b(env, "tcp", sensor_cfg), dim=-1)
    return 1.0 - torch.tanh(distance / std)
