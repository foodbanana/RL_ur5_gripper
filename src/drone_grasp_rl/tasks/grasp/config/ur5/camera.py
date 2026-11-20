# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist camera (D435i color) of the UR5 for VLA demonstrations: the only camera of the data set.

Values copied from the data collection repo (``isaacsim/config/wrist_camera.yaml``, real ``camera_info`` of the D435i
serial 843112074130, 640x480). The camera prim sits below ``wrist_camera_color_optical_frame`` of the robot USD (ROS
optical convention: +z forward, +y down). No third-view camera: on the Stewart and Husky platforms there is no place
for one, so all recordings use the wrist camera alone.

Intrinsics to USD as in the data collection repo: pixel size = horizontal aperture / width, focal length = fx x pixel
size, aperture offsets = (-(cx + 0.5 - width/2), cy + 0.5 - height/2) x pixel size (horizontal sign reversed, 0.5 =
pixel-center convention, checked there with markers). USD has one focal length, so fy is approximated by fx (0.07 %).
Isaac Lab's ``PinholeCameraCfg.from_intrinsic_matrix`` drops the offsets (cx is 12 px off center here), so the
attributes are set directly.
"""

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480
WRIST_CAMERA_INTRINSICS = {
    "fx": 618.5508422851562,
    "fy": 618.956298828125,
    "cx": 332.2762756347656,
    "cy": 239.38070678710938,
}
"""Real D435i color ``camera_info`` (plumb_bob distortion, all coefficients 0)."""
WRIST_CAMERA_HORIZONTAL_APERTURE = 20.955
"""Sensor width in the USD camera units; only the ratios matter."""
WRIST_CAMERA_CLIPPING = (0.01, 10.0)
"""[m] The fingers are a few cm from the camera."""
WRIST_CAMERA_FRAME = (
    "Geometry/robot_mount/base_link/base_link_inertia/shoulder_link/upper_arm_link/forearm_link/wrist_1_link"
    "/wrist_2_link/wrist_3_link/flange/tool0/wrist_mount/wrist_camera_mount/wrist_camera_bottom_screw_frame"
    "/wrist_camera_link/wrist_camera_color_frame/wrist_camera_color_optical_frame"
)
"""Optical frame below the robot prim (fixed links of the URDF kept as Xforms)."""


def wrist_camera_spawn_cfg() -> PinholeCameraCfg:
    """USD camera attributes from the real intrinsics."""
    w, h, k = WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_INTRINSICS
    pixel = WRIST_CAMERA_HORIZONTAL_APERTURE / w
    return PinholeCameraCfg(
        focal_length=k["fx"] * pixel,
        horizontal_aperture=WRIST_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=h * pixel,
        horizontal_aperture_offset=-(k["cx"] + 0.5 - w / 2) * pixel,
        vertical_aperture_offset=(k["cy"] + 0.5 - h / 2) * pixel,
        clipping_range=WRIST_CAMERA_CLIPPING,
    )


def wrist_camera_cfg(data_types: list[str] | None = None) -> CameraCfg:
    """Wrist camera sensor, rendered once per policy step (the scene's render interval)."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + WRIST_CAMERA_FRAME + "/wrist_cam",
        offset=CameraCfg.OffsetCfg(convention="ros"),  # the parent frame is the ROS optical frame
        spawn=wrist_camera_spawn_cfg(),
        data_types=data_types or ["rgb"],
        width=WRIST_CAMERA_WIDTH,
        height=WRIST_CAMERA_HEIGHT,
    )


# --- depth (D435i depth stream aligned to the color image) ---
# The real recordings had depth off (launch/cameras.launch.py: color only). To record it on the robot, enable the depth
# stream at 640x480 and align it to color (aligned_depth_to_color), so it matches the simulated depth pixel for pixel.
# The simulated depth is the camera's ``distance_to_image_plane`` (z in the optical frame, like RealSense depth), made
# real-like by :func:`realsense_depth_mm`.

DEPTH_BASELINE = 0.050
"""[m] D435i stereo baseline."""
DEPTH_FOCAL_PX = 337.0
"""Focal length of the 640x480 depth stream [px]: 87 deg horizontal field of view."""
DEPTH_MIN = 0.14
"""[m] Closest measurable depth at 640x480: focal x baseline / 126 disparities (= 0.134 m), rounded up. Closer pixels
read 0, as on the real camera; at the grasp the drone is about 0.11 m from the camera, so its depth is missing there."""
DEPTH_MAX = 10.0
"""[m] Farther pixels (and pixels that hit nothing) read 0."""
DEPTH_SUBPIXEL = 0.08
"""RMS disparity error [px]: depth noise std = subpixel x z^2 / (focal x baseline) (4.7 mm at 1 m)."""
DEPTH_EDGE_JUMP = 0.05
"""Pixels next to a depth jump larger than this fraction of their depth read 0 (stereo misses at object edges)."""


def realsense_depth_mm(depth_m, generator=None):
    """Real-like D435i depth from simulated depth [m] (N, H, W) or (N, H, W, 1) -> millimeters (N, H, W), int32.

    Out of range (closer than :data:`DEPTH_MIN`, farther than :data:`DEPTH_MAX`, no hit) and edge pixels read 0, the
    rest gets depth-dependent noise and 1 mm steps, like the camera's ``Z16`` format (save as uint16).
    """
    import torch

    z = depth_m.squeeze(-1) if depth_m.dim() == 4 else depth_m
    z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (z >= DEPTH_MIN) & (z <= DEPTH_MAX)
    # edges: a neighbor (4-connected) differs by more than the jump fraction
    pad = torch.nn.functional.pad(z.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    jump = torch.zeros_like(z, dtype=torch.bool)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = pad[:, dy : dy + z.shape[1], dx : dx + z.shape[2]]
        jump |= (nb - z).abs() > DEPTH_EDGE_JUMP * z
    valid &= ~jump
    sigma = DEPTH_SUBPIXEL * z * z / (DEPTH_FOCAL_PX * DEPTH_BASELINE)
    noisy = z + sigma * torch.randn(z.shape, device=z.device, generator=generator)
    mm = torch.where(valid, (noisy * 1000.0).round(), torch.zeros_like(z)).clamp(0, 65535)
    return mm.to(torch.int32)
