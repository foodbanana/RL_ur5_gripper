# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving robot base: the Stewart platform's motion command, the robot mount, the base IMU and the motion curriculum.

The platform is the closed-loop mechanism of ``assets/stewart.py``. Its controller is not the policy: like the drone's
flight controller (``drone_control.py``), it runs as a 0-dimensional action term every physics step, sends the
6-axis sine pose command of the data collection repo through the inverse kinematics to the six leg actuators, and the
top plate (with the robot on it) moves where the legs push it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from drone_grasp_rl.assets import stewart as st

from .actions import _to_index

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, ObservationTermCfg


class StewartMotion(ActionTerm):
    """Top-plate pose command -> inverse kinematics -> leg position targets, every physics step (no policy input).

    Per reset each environment draws a motion preset (``preset_weights``) and the random frequency scales and phases
    of :class:`drone_grasp_rl.assets.stewart.SineMotion`; ``amplitude_scale`` (set by the curriculum) multiplies the
    preset's amplitude.
    """

    cfg: StewartMotionCfg
    _asset: Articulation

    def __init__(self, cfg: StewartMotionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._dt = env.physics_dt
        self._legs = [self._asset.joint_names.index(j) for j in cfg.leg_joint_names]
        self._b, self._p = st.anchors(dev)
        self._l0 = st.nominal_leg_length()
        self.preset_names = list(cfg.preset_weights)
        weights = torch.tensor([cfg.preset_weights[k] for k in self.preset_names], device=dev, dtype=torch.float32)
        self._preset_probs = weights / weights.sum()
        self._preset_scale = torch.tensor([st.MOTION_PRESETS[k] for k in self.preset_names], device=dev)
        self._preset_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.motion = st.SineMotion(n, dev, ramp_time=cfg.ramp_time)
        self.amplitude_scale = cfg.amplitude_scale
        self._time = torch.zeros(n, device=dev)
        self.command = torch.zeros(n, 6, device=dev)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    @property
    def preset_index(self) -> torch.Tensor:
        return self._preset_index

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        self._time += self._dt
        self.command = self.motion.pose(self._time) * self.amplitude_scale
        target = st.leg_lengths(self.command, self._b, self._p) - self._l0
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._legs)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _to_index(env_ids, self.device)
        if isinstance(ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[ids]
        num = ids.numel()
        if num == 0:
            return
        preset = torch.multinomial(self._preset_probs, num, replacement=True)
        self._preset_index[ids] = preset
        self.motion.reset(ids, self._preset_scale[preset])
        self._time[ids] = 0.0


@configclass
class StewartMotionCfg(ActionTermCfg):
    """Configuration of :class:`StewartMotion`."""

    class_type: type[ActionTerm] = StewartMotion
    asset_name: str = "stewart"
    leg_joint_names: list[str] = MISSING
    preset_weights: dict[str, float] = {"mid": 1.0}
    """Probability of each motion preset (``assets/stewart.py`` ``MOTION_PRESETS``) per reset."""
    amplitude_scale: float = 1.0
    """Multiplies every preset's amplitude (the curriculum ramps it from 0)."""
    ramp_time: float = 1.0
    """[s] fade-in after each reset (the data collection repo: 5 s). Each reset draws a new random phase, so the
    platform is in full motion, at a random point of it, after this time; with 5 s the grasp was over first."""


class HuskyMotion(ActionTerm):
    """Straight back-and-forth of the Husky along its x axis, every physics step (no policy input).

    The data collection repo moved the Husky kinematically (no wheel physics). Here a carriage on a stiff, driven
    rail (``assets/bases.py`` ``HUSKY_RAIL_CFG``) gets the position and velocity targets of a trapezoidal speed
    profile (``HUSKY_TRAJECTORY``: out ``distance``, dwell, back, dwell, ...). Each reset starts at the rail's
    center at a random point of the cycle (speed faded in over ``ramp_time``), heading forward or backward at
    random. ``amplitude_scale`` (curriculum) scales the path.
    """

    cfg: HuskyMotionCfg

    def __init__(self, cfg: HuskyMotionCfg, env: ManagerBasedEnv):
        from drone_grasp_rl.assets.bases import HUSKY_DRIVE_JOINT, HUSKY_TRAJECTORY

        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._dt = env.physics_dt
        tr = HUSKY_TRAJECTORY
        self._v, self._a, self._d, self._dwell = tr["max_speed"], tr["accel"], tr["distance"], tr["dwell"]
        self._t_acc = self._v / self._a
        self._leg = 2 * self._t_acc + (self._d - self._v * self._t_acc) / self._v
        self._period = 2 * (self._leg + self._dwell)
        self._time = torch.zeros(n, device=dev)
        self._sign = torch.ones(n, device=dev)
        self._phase = torch.zeros(n, device=dev)
        self._pos = torch.zeros(n, device=dev)
        self._joint = [self._asset.joint_names.index(HUSKY_DRIVE_JOINT)]
        self.amplitude_scale = cfg.amplitude_scale

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def _leg_profile(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance and speed along one leg (0 -> distance) at time tau in [0, leg]."""
        v, a, ta, leg = self._v, self._a, self._t_acc, self._leg
        rest = (leg - tau).clamp(min=0.0)
        s = torch.where(
            tau < ta,
            0.5 * a * tau**2,
            torch.where(tau < leg - ta, 0.5 * a * ta**2 + v * (tau - ta), self._d - 0.5 * a * rest**2),
        )
        speed = torch.where(tau < ta, a * tau, torch.where(tau < leg - ta, torch.full_like(tau, v), a * rest))
        return s, speed

    def apply_actions(self):
        self._time += self._dt
        t = torch.remainder(self._time + self._phase, self._period)
        out = t < self._leg
        back = (t >= self._leg + self._dwell) & (t < 2 * self._leg + self._dwell)
        _, v_out = self._leg_profile(t.clamp(max=self._leg))
        _, v_back = self._leg_profile((t - self._leg - self._dwell).clamp(0.0, self._leg))
        zero = torch.zeros_like(t)
        vel_1d = torch.where(out, v_out, torch.where(back, -v_back, zero))
        # start anywhere in the cycle: from rest at the reset position, the speed of the profile at that phase fades
        # in over ramp_time and the position integrates it (the path keeps its shape, shifted to the start)
        u = (self._time / self.cfg.ramp_time).clamp(0.0, 1.0)
        fade = 10 * u**3 - 15 * u**4 + 6 * u**5
        vel = self._sign * self.amplitude_scale * fade * vel_1d
        self._pos = self._pos + vel * self._dt
        self._asset.set_joint_position_target_index(target=self._pos[:, None], joint_ids=self._joint)
        self._asset.set_joint_velocity_target_index(target=vel[:, None], joint_ids=self._joint)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _to_index(env_ids, self.device)
        if isinstance(ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[ids]
        if ids.numel() == 0:
            return
        self._time[ids] = 0.0
        self._pos[ids] = 0.0
        self._phase[ids] = self._period * torch.rand(ids.numel(), device=self.device)
        self._sign[ids] = torch.where(torch.rand(ids.numel(), device=self.device) < 0.5, -1.0, 1.0)


@configclass
class HuskyMotionCfg(ActionTermCfg):
    """Configuration of :class:`HuskyMotion`."""

    class_type: type[ActionTerm] = HuskyMotion
    asset_name: str = "husky_rail"
    amplitude_scale: float = 1.0
    ramp_time: float = 1.0
    """[s] speed fade-in after each reset, which starts at a random point of the back-and-forth cycle."""


class base_imu(ManagerTermBase):
    """Robot base IMU in the base frame, shape (N, 6): angular velocity [rad/s] and specific force [m/s^2]
    (acceleration minus gravity, so a base at rest reads +9.81 on z), like the data set's ``observation.base_imu``.

    The acceleration is the finite difference of the base velocity over the policy step. Gaussian noise
    ``gyro_std`` / ``accel_std`` (an IMU's, assumed) is added; the fixed base of the data collection repo stored the
    constant ``[0, 0, 0, 0, 0, 9.81]``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, 3, device=env.device)
        self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._valid[_to_index(env_ids, self.device)] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gyro_std: float = 0.005,
        accel_std: float = 0.05,
    ) -> torch.Tensor:
        data = env.scene[asset_cfg.name].data
        quat = data.root_quat_w.torch
        vel = data.root_lin_vel_w.torch
        acc = torch.where(self._valid[:, None], (vel - self._prev) / env.step_dt, torch.zeros_like(vel))
        self._prev = vel.clone()
        self._valid[:] = True
        gravity = torch.tensor(env.sim.cfg.gravity, device=env.device)
        f_b = math_utils.quat_apply_inverse(quat, acc - gravity)
        w_b = math_utils.quat_apply_inverse(quat, data.root_ang_vel_w.torch)
        return torch.cat([w_b + gyro_std * torch.randn_like(w_b), f_b + accel_std * torch.randn_like(f_b)], dim=-1)


def attach_robot_to_base(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    base_body_path: str,
    robot_body_path: str,
    filtered_robot_bodies: Sequence[str] = (),
    joint_name: str = "RobotMount",
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_yaw_deg: float = 0.0,
):
    """Prestartup event: fix the robot's root body to the base's mount body with a joint outside both articulations
    (as the data collection repo's moving-base scene does; inside, USD Physics parsing would pull the base into the
    robot's articulation). The two bodies must already sit at the mount pose (``mount_pos``, ``mount_yaw_deg`` in the
    base body's frame). Collisions between the base and the robot bodies that rest on it are filtered.

    Paths are relative to each environment prim. Needs ``scene.replicate_physics = False`` (the joint is authored in
    every environment, after cloning).
    """
    from pxr import Gf, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for i in range(env.num_envs):
        root = f"/World/envs/env_{i}"
        base, robot = f"{root}/{base_body_path}", f"{root}/{robot_body_path}"
        if not stage.GetPrimAtPath(base) or not stage.GetPrimAtPath(robot):
            # Newton clones the environments from env_0 in its model builder; only env_0 is on the USD stage
            continue
        joint = UsdPhysics.FixedJoint.Define(stage, f"{root}/{joint_name}")
        joint.CreateBody0Rel().SetTargets([base])
        joint.CreateBody1Rel().SetTargets([robot])
        half = math.radians(mount_yaw_deg) / 2
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in mount_pos]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        for body in filtered_robot_bodies:
            prim = stage.GetPrimAtPath(f"{root}/{body}")
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(base)


class ramp_base_motion(ManagerTermBase):
    """Curriculum: raise the base motion's ``amplitude_scale`` from ``initial`` to ``final`` while the training
    success rate (judge HELD at reset, EMA) stays at least ``min_success``: each reset episode then adds
    ``(final - initial) / resets_to_full`` (about 245 resets per iteration with 2048 environments).

    Like the unitree / Isaac Lab ADR style: the platform moves more only when the policy copes.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success = 0.0
        self._scale = cfg.params.get("initial", 0.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str = "base_motion",
        initial: float = 0.0,
        final: float = 1.0,
        resets_to_full: float = 120000.0,
        min_success: float = 0.85,
        judge_term: str = "grasp_success",
    ):
        from .judge import HELD, get_judge

        ids = _to_index(env_ids, env.device)
        judge = get_judge(env)
        # the curriculum runs before the judge's reset: its phase is still the episode's outcome
        phase = judge.phase[ids]
        if phase.numel() > 0:
            held = (phase == HELD).float().mean().item()
            self._success = 0.95 * self._success + 0.05 * held
            if self._success >= min_success:
                self._scale = min(final, self._scale + (final - initial) * phase.numel() / resets_to_full)
        env.action_manager.get_term(term_name).amplitude_scale = self._scale
        return self._scale


def spawn_mount_joint(prim_path: str, cfg: MountJointCfg, translation=None, orientation=None, **kwargs):
    """Spawner (Newton): the robot mount joint in the source environment only, which Newton's model builder clones.

    Same joint as :func:`attach_robot_to_base` (PhysX path), plus ``MjcEqualityWeldAPI`` with a stiff ``mjc:solref``:
    Newton turns the excluded fixed joint into a MuJoCo WELD, whose default softness would let the robot sag.
    """
    from pxr import Gf, Sdf, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    root = prim_path.rsplit("/", 1)[0]
    base, robot = f"{root}/{cfg.base_body_path}", f"{root}/{cfg.robot_body_path}"
    joint = UsdPhysics.FixedJoint.Define(stage, prim_path)
    joint.CreateBody0Rel().SetTargets([base])
    joint.CreateBody1Rel().SetTargets([robot])
    half = math.radians(cfg.mount_yaw_deg) / 2
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in cfg.mount_pos]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    prim = joint.GetPrim()
    if cfg.stiff_weld:
        prim.AddAppliedSchema("MjcEqualityWeldAPI")
        prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.005, 1.0])
        prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.95, 0.99, 0.001, 0.5, 2.0])
    for body in cfg.filtered_robot_bodies:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{root}/{body}")).CreateFilteredPairsRel().AddTarget(
            base
        )
    return prim


@configclass
class MountJointCfg:
    """Spawn configuration of :func:`spawn_mount_joint` (paths relative to the environment prim)."""

    func: object = spawn_mount_joint
    base_body_path: str = MISSING
    robot_body_path: str = MISSING
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_yaw_deg: float = 0.0
    filtered_robot_bodies: list[str] = []
    stiff_weld: bool = False
    """Author ``MjcEqualityWeldAPI`` with a stiff solref (with it the robot fell over: the weld did not hold)."""
    visible: bool = True


def base_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Privileged moving-base state (teacher only), shape (N, 12): the robot base's linear and angular velocity in its
    own frame [m/s, rad/s], its displacement from the episode's start pose in the start frame's axes [m] and gravity
    in the base frame (tilt). The base IMU alone gives angular velocity and acceleration, not where the base is or how
    fast it goes; the teacher plateaued at about 81 % success on the moving Stewart platform without them."""
    data = env.scene[asset_cfg.name].data
    quat = data.root_quat_w.torch
    lin = math_utils.quat_apply_inverse(quat, data.root_lin_vel_w.torch)
    ang = math_utils.quat_apply_inverse(quat, data.root_ang_vel_w.torch)
    start = data.default_root_state.torch[:, :3] + env.scene.env_origins
    disp = data.root_pos_w.torch - start
    grav = math_utils.quat_apply_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(lin))
    return torch.cat([lin, ang, disp, grav], dim=-1)


class stewart_fk_state(ManagerTermBase):
    """Stewart top-plate pose and velocity from the measured leg lengths by forward kinematics, shape (N, 12):
    x, y, z offset [m] and roll, pitch, yaw [rad] from the center pose, then their rates per second.

    Deployable: the platform controller reads its six actuator encoders (``noise`` [m] per leg, assumed) and solves
    the forward kinematics (``assets/stewart.py``); rates by finite difference over the policy step. This is what the
    student gets in place of the teacher's privileged ``base_state``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._b, self._p = st.anchors(env.device)
        self._l0 = st.nominal_leg_length()
        asset = env.scene[cfg.params.get("asset_name", "stewart")]
        self._legs = [asset.joint_names.index(j) for j in cfg.params["leg_joint_names"]]
        self._pose = torch.zeros(env.num_envs, 6, device=env.device)
        self._prev = torch.zeros(env.num_envs, 6, device=env.device)
        self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = _to_index(env_ids, self.device)
        self._valid[ids] = False
        self._pose[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        leg_joint_names: list[str],
        asset_name: str = "stewart",
        noise: float = 1.0e-4,
    ) -> torch.Tensor:
        lengths = env.scene[asset_name].data.joint_pos.torch[:, self._legs] + self._l0
        lengths = torch.nan_to_num(lengths + noise * torch.randn_like(lengths), nan=self._l0)
        guess = torch.where(self._valid[:, None], self._pose, torch.zeros_like(self._pose))
        self._pose = st.forward_kinematics(lengths, self._b, self._p, guess=guess)
        rate = torch.where(self._valid[:, None], (self._pose - self._prev) / env.step_dt, torch.zeros_like(self._pose))
        self._prev = self._pose.clone()
        self._valid[:] = True
        return torch.cat([self._pose, rate], dim=-1)
