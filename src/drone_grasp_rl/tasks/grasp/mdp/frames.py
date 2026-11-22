# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot base frame: the arm's root body (``robot_mount``), not the articulation root.

On the table the arm's root body is the articulation root. On a moving base the arm and the base are one
articulation (``assets/mounted.py``) whose root is the base's (Stewart base plate, Husky rail); the robot base frame
(drone targets, observations in the base frame, base IMU, the pose sent to the drone) stays the arm's root body.
"""

from __future__ import annotations

import torch

from drone_grasp_rl.assets.robots import BASE_BODY_NAME

_INDEX: dict[int, int] = {}


def base_body_index(asset) -> int:
    """Index of the robot base body (:data:`BASE_BODY_NAME`) in ``asset``."""
    key = id(asset)
    if key not in _INDEX:
        _INDEX[key] = asset.body_names.index(BASE_BODY_NAME)
    return _INDEX[key]


def base_pos_w(asset) -> torch.Tensor:
    """Robot base position in the world [m], (N, 3)."""
    return asset.data.body_link_pose_w.torch[:, base_body_index(asset), :3]


def base_quat_w(asset) -> torch.Tensor:
    """Robot base orientation in the world (x, y, z, w), (N, 4)."""
    return asset.data.body_link_pose_w.torch[:, base_body_index(asset), 3:7]


def base_lin_vel_w(asset) -> torch.Tensor:
    """Robot base linear velocity in the world [m/s], (N, 3)."""
    return asset.data.body_link_lin_vel_w.torch[:, base_body_index(asset)]


def base_ang_vel_w(asset) -> torch.Tensor:
    """Robot base angular velocity in the world [rad/s], (N, 3)."""
    return asset.data.body_link_ang_vel_w.torch[:, base_body_index(asset)]
