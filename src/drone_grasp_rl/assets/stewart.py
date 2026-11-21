"""Stewart platform: geometry, inverse kinematics and the top-plate trajectory (closed-loop model).

Geometry and motion presets copied from the data collection repo (``isaacsim/config/base_stewart.yaml``,
``scripts/design_stewart.py``); attachment angles from daniel-s-ingram/stewart (MIT, ``src/ik.cpp``) scaled to these
radii. That repo moved the top plate kinematically and only drew the legs. Here the platform is a mechanism: six legs,
each a universal joint at the base plate, an actuated prismatic joint and a spherical joint at the top plate; the top
plate goes where the legs push it. A pose command goes through :func:`leg_lengths` (inverse kinematics) to the
prismatic position targets, as on a real Stewart platform controller.

The dimensions, masses and actuator limits are those of the sim-only design of the data collection repo or
assumptions (marked); replace them with the real platform's once it is measured, then rebuild the USD
(``scripts/build_stewart_usd.py``).

Frames: world z up, the platform's base frame on the floor at the base plate center, x = robot front. The top
plate's frame (rotation center = robot base mount) is at ``(x, y, HEIGHT + z)`` with rotation
Rz(yaw) Ry(pitch) Rx(roll).
"""

import math

import torch

from . import DRONE_GRASP_RL_ASSETS_DIR

HEIGHT = 1.0
"""Top surface of the top plate above the floor in the center pose [m]."""
BASE_RADIUS = 0.8
TOP_RADIUS = 0.6
BASE_PLATE = {"radius": 0.9, "thickness": 0.05}
TOP_PLATE = {"radius": 0.7, "thickness": 0.05}
BASE_ANGLES_DEG = (97.2, 82.8, -22.8, -37.2, -142.8, -157.2)
TOP_ANGLES_DEG = (142.8, 37.2, 22.8, -82.8, -97.2, 157.2)
LEG_MIN, LEG_MAX = 0.747, 1.395
"""Leg length range [m] between the joint centers (design_stewart.py: the big preset plus 3 cm margin)."""
LEG_CYLINDER_RADIUS, LEG_ROD_RADIUS = 0.045, 0.025

# masses and actuators: assumptions (no real platform yet)
TOP_PLATE_MASS = 20.0
LEG_CYLINDER_MASS, LEG_ROD_MASS, GIMBAL_MASS = 2.0, 1.0, 0.5
BASE_PLATE_MASS = 100.0
LEG_STIFFNESS = 2.0e5
"""Prismatic position drive [N/m] (assumed servo stiffness; 0.3 mm static error at ~65 N per leg)."""
LEG_DAMPING = 5.0e3
LEG_FORCE_LIMIT = 3000.0
"""[N] assumed electric linear actuator."""
LEG_SPEED_LIMIT = 0.5
"""[m/s] assumed."""
LEG_ARMATURE = 10.0
"""Reflected mass of the actuator's motor and screw [kg] (assumed; a ball-screw actuator's is of this order). Needed
on Newton: MuJoCo integrates the stiff position drive explicitly, and with only the 1 kg rod the legs oscillated."""

STEWART_USD_PATH = DRONE_GRASP_RL_ASSETS_DIR / "bases" / "stewart" / "stewart.usda"

AXES = ("x", "y", "z", "roll", "pitch", "yaw")
BASE_OMEGA = {"x": 0.9, "y": 0.5, "z": 0.9, "roll": 0.5, "pitch": 0.9, "yaw": 0.2}
"""[rad/s] per axis; each reset scales them by U(0.7, 1.3) with a random phase."""
BIG_AMPLITUDE = {"x": 0.15, "y": 0.10, "z": 0.15, "roll": 0.105, "pitch": 0.105, "yaw": 0.053}
"""[m] / [rad]."""
MOTION_PRESETS = {"static": 0.0, "small": 0.25, "mid": 0.5, "big": 1.0}
"""Scale of :data:`BIG_AMPLITUDE` per preset."""
RAMP_TIME = 5.0
"""[s] the data collection repo's fade-in with 10u^3 - 15u^4 + 6u^5 (continuous position, velocity, acceleration).
The grasp task uses 1 s (``StewartMotionCfg.ramp_time``): with 5 s the teacher grasped (in about 4 s) before the
platform really moved."""


def anchors(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Base anchors B (6, 3) in the platform base frame (top of the base plate) and top anchors P (6, 3) in the
    top-plate frame (bottom of the top plate)."""
    ab = torch.tensor([math.radians(a) for a in BASE_ANGLES_DEG], device=device)
    at = torch.tensor([math.radians(a) for a in TOP_ANGLES_DEG], device=device)
    b = torch.stack([BASE_RADIUS * ab.cos(), BASE_RADIUS * ab.sin(), torch.full_like(ab, BASE_PLATE["thickness"])], 1)
    p = torch.stack([TOP_RADIUS * at.cos(), TOP_RADIUS * at.sin(), torch.full_like(at, -TOP_PLATE["thickness"])], 1)
    return b, p


def rotation(rpy: torch.Tensor) -> torch.Tensor:
    """Rz(yaw) Ry(pitch) Rx(roll) for (..., 3) roll, pitch, yaw -> (..., 3, 3)."""
    r, p, y = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
            torch.stack([-sp, cp * sr, cp * cr], -1),
        ],
        -2,
    )


def leg_lengths(pose: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Inverse kinematics: top-plate pose (N, 6) = x, y, z offset from the center pose, roll, pitch, yaw ->
    leg lengths (N, 6) [m] between the joint centers."""
    center = pose[:, :3] + torch.tensor([0.0, 0.0, HEIGHT], device=pose.device)
    top = torch.einsum("nij,kj->nki", rotation(pose[:, 3:]), p) + center[:, None]
    return (top - b[None]).norm(dim=-1)


def nominal_leg_length() -> float:
    b, p = anchors()
    return float(leg_lengths(torch.zeros(1, 6), b, p)[0, 0])


def ramp(t: torch.Tensor, ramp_time: float = RAMP_TIME) -> torch.Tensor:
    u = (t / ramp_time).clamp(0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


class SineMotion:
    """Top-plate pose command of the data collection repo: per axis ``ramp(t) * scale * A_k * sin(w_k t + phi_k)``,
    ``w_k = c_k * BASE_OMEGA[k]``, ``c_k ~ U(0.7, 1.3)`` and ``phi_k ~ U(0, 2 pi)`` drawn per environment at reset."""

    def __init__(self, num_envs: int, device, generator: torch.Generator | None = None, ramp_time: float = RAMP_TIME):
        self.n, self.device, self.gen, self.ramp_time = num_envs, device, generator, ramp_time
        self.amp = torch.tensor([BIG_AMPLITUDE[k] for k in AXES], device=device)
        self.omega0 = torch.tensor([BASE_OMEGA[k] for k in AXES], device=device)
        self.scale = torch.zeros(num_envs, device=device)
        self.omega = self.omega0.repeat(num_envs, 1)
        self.phase = torch.zeros(num_envs, 6, device=device)

    def reset(self, env_ids: torch.Tensor, scale: torch.Tensor):
        num = len(env_ids)
        c = 0.7 + 0.6 * torch.rand(num, 6, device=self.device, generator=self.gen)
        self.omega[env_ids] = c * self.omega0
        self.phase[env_ids] = 2 * math.pi * torch.rand(num, 6, device=self.device, generator=self.gen)
        self.scale[env_ids] = scale

    def pose(self, t: torch.Tensor) -> torch.Tensor:
        """Pose command (N, 6) at time t (N,) [s] since the reset: x, y, z offsets [m], roll, pitch, yaw [rad]."""
        s = (ramp(t, self.ramp_time) * self.scale)[:, None]
        return s * self.amp * torch.sin(self.omega * t[:, None] + self.phase)

    def velocity(self, t: torch.Tensor, dt: float = 1e-3) -> torch.Tensor:
        return (self.pose(t + dt) - self.pose(t - dt)) / (2 * dt)
