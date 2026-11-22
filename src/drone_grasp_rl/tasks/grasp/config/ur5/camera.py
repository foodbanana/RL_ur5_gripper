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

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from ...mdp.frames import base_quat_w
from ...mdp.observations import _quat_from_rotvec, frame_pose_b

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


# --- camera pose without rendering ---

MARKER_OFFSET = (0.0, 0.0, -0.045)
"""Candidate marker position in the drone body frame [m]: center of the underside (body bottom at about -0.050 m)."""


class WristCameraModel:
    """Pose of the wrist camera from the gripper base body and a fixed offset read from the robot USD (no rendering).

    The optical frame (ROS: +z forward, +y down) and the gripper base are rigidly linked; their relative transform is
    taken once from environment 0's USD prims.
    """

    def __init__(self, env):
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils

        from drone_grasp_rl.assets.robots import BODY_PRIM_PATHS, TCP_BODY_NAME

        self.env, self.k = env, WRIST_CAMERA_INTRINSICS
        stage = sim_utils.get_current_stage()
        root = "/World/envs/env_0/Robot/"
        cache = UsdGeom.XformCache()
        m_body = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root + BODY_PRIM_PATHS[TCP_BODY_NAME]))
        m_opt = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root + WRIST_CAMERA_FRAME))
        rel = torch.tensor([list(row) for row in (m_opt * m_body.GetInverse())], dtype=torch.float32)
        rel[:3, :3] /= rel[:3, :3].norm(dim=1, keepdim=True)  # drop the scale (USD row-vector matrices)
        self.p_rel = rel[3, :3].to(env.device)  # optical origin in the body frame
        self.r_rel = rel[:3, :3].T.contiguous().to(env.device)  # columns: optical axes in the body frame
        robot = env.scene["robot"]
        self.robot, self.body = robot, robot.body_names.index(TCP_BODY_NAME)

    def pose(self):
        """Camera position (N, 3) and rotation (N, 3, 3), columns = optical axes (x right, y down, z forward)."""
        import isaaclab.utils.math as math_utils

        pose = self.robot.data.body_link_pose_w.torch[:, self.body]
        r_body = math_utils.matrix_from_quat(pose[:, 3:7])
        return pose[:, :3] + (r_body @ self.p_rel), r_body @ self.r_rel

    def project(self, points_w):
        """Pixel (u, v), depth z and angle off the optical axis [deg] of world points, shape (N, 4)."""
        pos, rot = self.pose()
        p = torch.einsum("nji,nj->ni", rot, points_w - pos)
        u = self.k["fx"] * p[:, 0] / p[:, 2] + self.k["cx"]
        v = self.k["fy"] * p[:, 1] / p[:, 2] + self.k["cy"]
        off = torch.rad2deg(torch.atan2(p[:, :2].norm(dim=-1), p[:, 2]))
        return torch.stack([u, v, p[:, 2], off], dim=-1)

    def tcp_pixel(self):
        from drone_grasp_rl.tasks.grasp.mdp import frame_pose_w

        uv = self.project(frame_pose_w(self.env, "tcp")[0])[0]
        return f"({uv[0]:.1f}, {uv[1]:.1f})"

    def in_image(self, proj):
        w, h = WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT
        u, v, z = proj[:, 0], proj[:, 1], proj[:, 2]
        return (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    def view(self, dist):
        """Rows (distance [m], center in image, marker in image, center angle [deg], marker angle [deg])."""
        import isaaclab.utils.math as math_utils

        drone = self.env.scene["drone"].data
        center = drone.root_pos_w.torch
        marker = center + math_utils.quat_apply(
            drone.root_quat_w.torch, torch.tensor(MARKER_OFFSET, device=center.device).expand_as(center)
        )
        pc, pm = self.project(center), self.project(marker)
        return torch.stack([dist, self.in_image(pc).float(), self.in_image(pm).float(), pc[:, 3], pm[:, 3]], -1)

    def yaw_error(self):
        """Angle between the gripper's closing axis (TCP y) and the drone's y axis, either sign [deg], shape (N,)."""
        import isaaclab.utils.math as math_utils

        from drone_grasp_rl.tasks.grasp.mdp import frame_pose_w

        _, tcp_q = frame_pose_w(self.env, "tcp")
        y_tcp = math_utils.matrix_from_quat(tcp_q)[..., 1]
        y_drone = math_utils.matrix_from_quat(self.env.scene["drone"].data.root_quat_w.torch)[..., 1]
        return torch.rad2deg(torch.acos((y_tcp * y_drone).sum(-1).abs().clamp(max=1.0)))


# --- ArUco marker observation (step 2 of the real-deployment plan) ---

MARKER_LAYOUT = ((0.040, -0.002, 0.0), (0.020, 0.035, 0.0))
"""Markers on the drone underside: (black square side [m], x [m], y [m]) in the drone body frame, at
:data:`MARKER_Z`, facing down. ``scripts/study_aruco.py`` (teacher, 48 episodes, rendered wrist images, OpenCV): a
40 mm marker at the center for the far approach and a 20 mm one on the camera side (+x) for the last centimeters
found at least one marker in 95 % of the steps beyond 30 cm and 100 % from 30 cm to the end of the hold; the center
marker alone in 0 % within the last 5 cm (out of the view of the camera beside the gripper)."""
MARKER_Z = -0.0315
"""Marker plane in the drone body frame [m], just below the lowest flat part of the underside."""
MARKER_POS_ERR = 0.0027
"""Position error of one marker's pose: norm ~= MARKER_POS_ERR * distance^2 / side (median; fits all study points,
0.9 mm at 0.08 m to 22 mm at 0.41 m for 20 mm, 3 to 12 mm for 40 mm)."""
MARKER_NORMAL_ERR_DEG = {
    0.020: ((0.08, 0.7), (0.17, 2.2), (0.29, 7.5), (0.45, 8.0)),
    0.040: ((0.2, 0.8), (0.3, 1.6), (0.45, 1.8)),
}
"""Median error of the marker normal [deg] at camera distance [m] per marker side (study, linear in between)."""
MARKER_MISS_IN_VIEW = 0.03
"""Probability that a marker fully in the image is not detected in a frame (study: 0-3.5 %)."""


def _interp(table, d):
    """Piecewise-linear lookup of ``table`` ((x, y), ...) at ``d``, clamped at the ends."""
    xs = torch.tensor([t[0] for t in table], device=d.device)
    ys = torch.tensor([t[1] for t in table], device=d.device)
    d = d.clamp(float(xs[0]), float(xs[-1]))
    idx = torch.searchsorted(xs, d.contiguous()).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    return y0 + (y1 - y0) * (d - x0) / (x1 - x0)


class marker_drone_pose_b(ManagerTermBase):
    """Drone pose in the robot base frame as the wrist camera's ArUco detection reports it, shape (N, 8):
    x, y, z, qx, qy, qz, qw, age [s] of the reading (0 = fresh, clipped at ``max_age``).

    No rendering: each policy step, for each marker of :data:`MARKER_LAYOUT`, the camera model
    (:class:`WristCameraModel`) checks that all four corners are in the image in front of the camera; a visible
    marker is still missed with :data:`MARKER_MISS_IN_VIEW`. Of the detected markers the one with the smallest
    expected error (largest side / distance^2) gives the drone pose, with position noise
    ``MARKER_POS_ERR * d^2 / side`` (median of the norm; per axis std = median / 1.54) and rotation noise from
    :data:`MARKER_NORMAL_ERR_DEG` (per axis std = median / 1.18). Then, as in the mocap model: latency (``latency``
    plus the frame age, uniform within one camera period) and a constant hand-eye calibration bias per episode.
    With no marker detected the last reading is repeated and its age grows, as a
    robot-side estimator would hold it.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.cam = WristCameraModel(env)
        self.sides = torch.tensor([m[0] for m in MARKER_LAYOUT], device=dev)
        self.centers = torch.tensor([[m[1], m[2], MARKER_Z] for m in MARKER_LAYOUT], device=dev)
        half = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], device=dev, dtype=torch.float32) * 0.5
        corners = self.centers[:, None, :2] + half * self.sides[:, None, None]
        self.corners = torch.cat([corners, torch.full_like(corners[..., :1], MARKER_Z)], -1)  # (M, 4, 3)
        self._pos_bias = torch.zeros(n, 3, device=dev)
        self._rot_bias = torch.zeros(n, 4, device=dev)
        self._rot_bias[:, 3] = 1.0
        self._last = torch.zeros(n, 7, device=dev)
        self._age = torch.zeros(n, device=dev)
        self._valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.visible_rate = torch.zeros((), device=dev)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        num = self._pos_bias[ids].shape[0]
        pos_bias = self.cfg.params.get("pos_bias", 0.002)
        rot_bias = math.radians(self.cfg.params.get("rot_bias_deg", 0.5))
        self._pos_bias[ids] = math_utils.sample_uniform(-pos_bias, pos_bias, (num, 3), self.device)
        axis = torch.randn(num, 3, device=self.device)
        self._rot_bias[ids] = math_utils.quat_from_angle_axis(
            math_utils.sample_uniform(0.0, rot_bias, (num,), self.device), axis
        )
        self._valid[ids] = False
        self._age[ids] = 0.0

    def __call__(
        self,
        env,
        latency: float = 0.045,
        frame_period: float = 1.0 / 30.0,
        pos_bias: float = 0.002,
        rot_bias_deg: float = 0.5,
        max_age: float = 1.0,
    ) -> torch.Tensor:
        drone = env.scene["drone"].data
        d_pos, d_quat = drone.root_pos_w.torch, drone.root_quat_w.torch
        r_d = math_utils.matrix_from_quat(d_quat)
        cam_pos, cam_rot = self.cam.pose()
        corners_w = d_pos[:, None, None] + torch.einsum("nij,mcj->nmci", r_d, self.corners)
        pc = torch.einsum("nji,nmcj->nmci", cam_rot, corners_w - cam_pos[:, None, None])
        k = self.cam.k
        u = k["fx"] * pc[..., 0] / pc[..., 2] + k["cx"]
        v = k["fy"] * pc[..., 1] / pc[..., 2] + k["cy"]
        in_view = (
            (pc[..., 2] > 0.02) & (u >= 0) & (u < WRIST_CAMERA_WIDTH) & (v >= 0) & (v < WRIST_CAMERA_HEIGHT)
        ).all(-1)
        detected = in_view & (torch.rand_like(in_view, dtype=torch.float32) >= MARKER_MISS_IN_VIEW)
        dist = pc.mean(dim=2).norm(dim=-1)  # (N, M) camera to marker center
        quality = torch.where(detected, self.sides / dist.square(), torch.zeros_like(dist))
        best = quality.argmax(dim=-1)
        any_det = detected.any(dim=-1)
        side = self.sides[best]
        d = dist.gather(1, best[:, None]).squeeze(1)
        pos_std = MARKER_POS_ERR * d.square() / side / 1.54
        rot_med = torch.where(
            side < 0.03, _interp(MARKER_NORMAL_ERR_DEG[0.020], d), _interp(MARKER_NORMAL_ERR_DEG[0.040], d)
        )
        rot_std = torch.deg2rad(rot_med / 1.18)
        # true pose in the base frame, moved back by the latency
        pos, quat = frame_pose_b(env, "drone")
        base_quat = base_quat_w(env.scene["robot"])
        lin_vel = math_utils.quat_apply_inverse(base_quat, drone.root_lin_vel_w.torch)
        ang_vel = math_utils.quat_apply_inverse(base_quat, drone.root_ang_vel_w.torch)
        delay = latency + frame_period * torch.rand(pos.shape[0], 1, device=pos.device)
        pos = pos - lin_vel * delay
        quat = math_utils.quat_mul(_quat_from_rotvec(-ang_vel * delay), quat)
        pos = pos + self._pos_bias + pos_std[:, None] * torch.randn_like(pos)
        quat = math_utils.quat_mul(self._rot_bias, quat)
        quat = math_utils.quat_mul(_quat_from_rotvec(rot_std[:, None] * torch.randn_like(pos)), quat)
        reading = torch.cat([pos, math_utils.quat_unique(quat)], dim=-1)
        # first reading of an episode: take it even if not detected, so the hold has a start value (the real
        # robot starts with the drone in view; the age says how fresh it is)
        use = any_det | ~self._valid
        self._last = torch.where(use[:, None], reading, self._last)
        self._age = torch.where(any_det, torch.zeros_like(self._age), self._age + env.step_dt).clamp(max=max_age)
        self._valid |= True
        self.visible_rate = any_det.float().mean()
        return torch.cat([self._last, self._age[:, None]], dim=-1)
