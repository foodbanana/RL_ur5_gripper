# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the drone grasp task.

Poses come from the ``ee_frame`` frame transformer, whose source is the robot base, so every pose is in
the robot base frame (what the real robot can compute from its own base).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
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
