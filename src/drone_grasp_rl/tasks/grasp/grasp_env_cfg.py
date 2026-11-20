# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the drone grasp task: a robot arm on a table grasps a hovering drone from below.

The scene follows ``data_collection_ur5_gripper`` (``scene_drone.yaml``): a 2 x 1 m table whose top is the
robot base height, the robot base at the env origin raised to the table top, and the drone hovering above
the table in front of the robot (+x). Robot configs (``config/<robot>/``) fill in the robot, its sensors
and its actions.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from drone_grasp_rl.assets.drones import DRONE_BODY_NAME, IRIS_CFG
from drone_grasp_rl.physics import DRONE_USD, SIM_DT, DroneGraspPhysicsCfg

from . import mdp

TABLE_TOP_Z = 0.762
"""Table top height = robot base height [m]."""

DRONE_TARGETS_BASE = [
    (x, y, z)
    for z, xs in ((0.70, (0.45, 0.55, 0.65)), (0.80, (0.45, 0.55)))
    for x in xs
    for y in (-0.20, -0.10, 0.0, 0.10, 0.20)
]
"""The 25 reference drone positions in the robot base frame [m] (``scene_drone.yaml`` ``drone_targets_base``).

Chosen in the data collection repo so that the grasp and a pre-grasp 10 cm below it are reachable from
``under_drone`` with joint margin, also at the corners of the offset and hover-motion box.
"""

DRONE_TARGET_OFFSET = 0.03
"""Uniform offset on each axis around the reference position at reset [m]."""

DRONE_DEFAULT_POS_BASE = (0.60, 0.0, 0.738)
"""Default drone position in the robot base frame [m] (world z 1.50 in the data collection scene)."""


def contact_material(asset_cfg: SceneEntityCfg):
    """Startup event giving ``asset_cfg``'s bodies the contact material of the data collection repo.

    PhysX: 0.6 static / 0.5 dynamic friction, no restitution (``drive_gains.yaml``, finger links and object), each
    randomized by x0.7-1.3 per environment.
    Newton keeps its default (friction 1.0): PhysX friction values do not carry over numerically, and with 0.6 the
    closing fingers squeezed the floating drone out of the grasp.
    """
    return preset(
        default=EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": asset_cfg,
                # domain randomization: x0.7-1.3 around 0.6 / 0.5 (OpenAI in-hand manipulation range); the real
                # finger friction is not measured yet
                "static_friction_range": (0.42, 0.78),
                "dynamic_friction_range": (0.35, 0.65),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        ),
        newton_mjwarp=None,
    )


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Table, robot, drone and lights. The robot config adds the robot, ``ee_frame`` and contact sensors."""

    robot: ArticulationCfg = MISSING

    drone: ArticulationCfg = IRIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Drone",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(DRONE_DEFAULT_POS_BASE[0], DRONE_DEFAULT_POS_BASE[1], DRONE_DEFAULT_POS_BASE[2] + TABLE_TOP_Z)
        ),
    )

    ee_frame: FrameTransformerCfg = MISSING
    """Source = robot base; target frames ``tcp`` and ``drone``."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.04),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, TABLE_TOP_Z - 0.02)),
    )

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # The wrist camera needs light also when headless.
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1000.0))

    def __post_init__(self):
        # Original body collider on PhysX, fixed convex pieces on Newton.
        self.drone.spawn.usd_path = preset(default=DRONE_USD["isaacsim_physx"], **DRONE_USD)


@configclass
class ActionsCfg:
    """Arm and gripper action terms (set by the robot config) and the drone's own flight controller (no policy
    input)."""

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING
    # Training samples every hover preset; evaluation and play use "mid", the data collection default.
    drone_hover = mdp.DroneHoverControllerCfg(
        asset_name="drone",
        body_name=DRONE_BODY_NAME,
        preset_weights={"static": 1.0, "small": 1.0, "mid": 1.0, "big": 1.0},
        # domain randomization: 0.1 N gives 8.8-10 mm hover jitter per axis (19 mm RMS), as PX4 hovered in the data
        # collection repo (19.5 mm RMS, 8-10 mm per axis)
        disturbance_std=0.1,
    )


@configclass
class ObservationsCfg:
    """``policy`` is the teacher's input: what the real robot can measure (arm, gripper, mocap drone pose; the
    first 30 entries) followed by privileged simulator state (exact drone pose and velocity, drone in the TCP
    frame, finger contacts, gripper closed, grasp detected; 22 more). ``critic`` adds the exact drone pose and
    velocity and the contact forces once more (it was built for the 30-entry policy)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # small sensor noise (domain randomization)
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": MISSING}, noise=Unoise(n_min=-0.002, n_max=0.002)
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel, params={"asset_cfg": MISSING}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        gripper_pos = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": MISSING}, noise=Unoise(n_min=-0.005, n_max=0.005)
        )
        tcp_pos = ObsTerm(func=mdp.frame_pos_b, params={"frame_name": "tcp"}, noise=Unoise(n_min=-0.001, n_max=0.001))
        # The real robot gets the drone pose from motion capture: latency, calibration bias and noise.
        drone_pose = ObsTerm(func=mdp.mocap_drone_pose_b, params={"dropout_rate": 0.2})
        actions = ObsTerm(func=mdp.last_action)
        # -- privileged teacher observations (simulation only), after the 30 deployable ones above so a policy
        #    trained on those can be extended (scripts/expand_policy_inputs.py). Scaled to about unit size, since
        #    the extended observation normalizer keeps the new entries at mean 0 / std 1. Things a camera could
        #    also tell (no motor stop flag), so the teacher can later be distilled into an image policy.
        drone_pos_true = ObsTerm(func=mdp.frame_pos_b, params={"frame_name": "drone"})
        drone_quat_true = ObsTerm(func=mdp.frame_quat_b, params={"frame_name": "drone"})
        drone_vel = ObsTerm(func=mdp.drone_vel_b, scale=5.0)
        drone_in_tcp = ObsTerm(func=mdp.drone_pos_in_tcp, scale=10.0)
        finger_contact = ObsTerm(
            func=mdp.contact_forces, params={"sensor_names": MISSING}, clip=(0.0, 50.0), scale=0.05
        )
        gripper_closing = ObsTerm(func=mdp.gripper_closing)
        grasped = ObsTerm(func=mdp.grasp_detected)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        drone_pos_true = ObsTerm(func=mdp.frame_pos_b, params={"frame_name": "drone"})
        drone_quat_true = ObsTerm(func=mdp.frame_quat_b, params={"frame_name": "drone"})
        drone_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("drone")})
        drone_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("drone")})
        finger_contact = ObsTerm(func=mdp.contact_forces, params={"sensor_names": MISSING}, clip=(0.0, 50.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Startup events once; reset events in this order after the scene reset."""

    drone_material = contact_material(SceneEntityCfg("drone", body_names=[DRONE_BODY_NAME]))
    # domain randomization: drone body mass x0.9-1.1 (the flight controller reads the masses every step)
    drone_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("drone", body_names=[DRONE_BODY_NAME]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # Robot to its default (start) pose with matching joint targets, drone to its default state.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_drone = EventTerm(
        func=mdp.reset_drone_at_targets,
        mode="reset",
        params={"targets": DRONE_TARGETS_BASE, "offset": DRONE_TARGET_OFFSET},
    )


@configclass
class RewardsCfg:
    """Grasp rewards. The reward manager multiplies each term by the step interval (1/30 s), except the terminal
    bonuses, which pay their weight exactly once. Effective size per episode in brackets.

    Sources: progress reward from Isaac Lab core lift; staged weights and the alignment idea from the cabinet task;
    contact, hold and failure thresholds from the data collection repo's grasp judge; action/joint penalties from
    contrib lift. Weights without a reference (close_near, drone_push) start small.
    """

    # approach: 1 per 5 mm of new path progress, before the grasp [about 0.3 m / 5 mm * 5 / 30 = 10]
    approach = RewTerm(func=mdp.approach_progress, weight=5.0)
    # last centimeters: dense, while on the line below the drone [2 per second at the grasp point]
    approach_fine = RewTerm(func=mdp.approach_fine, weight=2.0)
    # gripper axes vs drone axes; a penalty since the start pose is already aligned [0 when aligned]
    misalign = RewTerm(func=mdp.misalignment, weight=-1.0)
    # gripper approach axis vs up, always (after the grasp the drone turns with the gripper, so misalign is blind)
    # -25: with -15 the randomized training let the end-of-hold tilt grow to a median 8-13 deg near the 15 deg limit
    gripper_tilt = RewTerm(func=mdp.gripper_tilt, weight=-25.0)
    # held drone tilted from up (zero while it hovers upright)
    drone_tilt = RewTerm(func=mdp.drone_tilt, weight=-10.0)
    # arm action once the judge has the grasp (gated on the close command instead, the policy could not correct the
    # tilt the closing fingers cause and success dropped from 82 % to 45 %). L1: with squares, the ~0.1 actions
    # that made the arm shiver after the grasp cost almost nothing.
    arm_still = RewTerm(func=mdp.arm_motion_while_closing, weight=0.0, params={"gate": "grasp", "norm": "l1"})
    # within 10 cm of the grasp point: speed above 5 cm/s + 1/s x distance (slow down when close)
    approach_speed = RewTerm(func=mdp.approach_speed_excess, weight=0.0)
    # drone sliding in the fingers while grasped, beyond 2 mm/s (the arm holds still then, so it shapes the grasp)
    drone_slip = RewTerm(func=mdp.drone_slip, weight=-1000.0)
    # arm joint accelerations (abrupt starts and stops)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": MISSING})
    # both finger sides on the drone, until the motor stop [about 2 per second, at most about 1.5 s]
    finger_contact = RewTerm(func=mdp.finger_contact, weight=2.0)
    # commanding close at the waist, before the grasp [about 2 per second for the ~0.5 s until contact]
    # (an empty_close penalty for closing far away taught the policy never to close; the judge still counts it)
    # only within 4 mm (6 mm before the slip term): with 15 mm the policy closed while still rising, squeezed the
    # drone and tilted the gripper
    close_near = RewTerm(func=mdp.close_near, weight=2.0, params={"max_distance": 0.004})
    # still holding after the motor stop [4 per second, 2.5 s = 10]
    hold = RewTerm(func=mdp.hold, weight=4.0)
    # judge says held [+20 once]; any failure termination [-5 once]
    success = RewTerm(func=mdp.terminal_bonus, weight=20.0, params={"term_keys": "grasp_success"})
    failure = RewTerm(
        func=mdp.terminal_bonus,
        weight=-5.0,
        params={"term_keys": ["grasp_failed", "drone_knocked", "protective_stop"]},
    )
    # pushing the floating drone before the grasp, per m beyond 5 mm
    drone_push = RewTerm(func=mdp.drone_push, weight=-20.0)
    # smooth motion (contrib lift values; MJWarp does not enforce joint velocity limits)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": MISSING})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": MISSING})


@configclass
class TerminationsCfg:
    """The judge must stay first: the other terms and the rewards of the same step read its state."""

    # success; the judge also stops the drone motors after the grasp. Sensor and joint names: robot config.
    grasp_success = DoneTerm(
        func=mdp.grasp_judge, params={"right_sensors": MISSING, "left_sensors": MISSING, "gripper_joint": MISSING}
    )
    grasp_failed = DoneTerm(func=mdp.grasp_failed)
    drone_knocked = DoneTerm(func=mdp.drone_knocked)
    protective_stop = DoneTerm(func=mdp.protective_stop)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Smoothness penalties ramped in over 24k env steps (1000 iterations of 24 steps) from the start of a run:
    added all at once to a trained policy they are a large, sudden change the critic has not seen. The final
    weights are the targets; the reward config holds the starting weights."""

    approach_speed = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={"term_name": "approach_speed", "initial_weight": 0.0, "final_weight": -10.0, "num_steps": 24000},
    )
    joint_acc = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={"term_name": "joint_acc", "initial_weight": 0.0, "final_weight": -1.0e-4, "num_steps": 24000},
    )
    action_rate = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={"term_name": "action_rate", "initial_weight": -0.001, "final_weight": -0.01, "num_steps": 24000},
    )
    arm_still = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={"term_name": "arm_still", "initial_weight": 0.0, "final_weight": -1.5, "num_steps": 24000},
    )


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Drone grasp environment. Physics 120 Hz, policy 30 Hz (decimation 4), as in the data collection repo."""

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        # approach about 3 s, close 2.2 s, kill delay 1 s, hold 2.5 s, plus slack
        self.episode_length_s = 12.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics = DroneGraspPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.8, -1.6, 1.8), lookat=(0.4, 0.0, 1.2))

    def disable_randomization(self):
        """Turn the domain randomization off (task checks, scripted demos, ``eval_policy.py --no_dr``)."""
        self.actions.drone_hover.disturbance_std = 0.0
        self.observations.policy.drone_pose.params["dropout_rate"] = 0.0
        self.observations.policy.enable_corruption = False
        arm = self.actions.arm_action
        if hasattr(arm, "delay_steps"):
            arm.delay_steps = (0, 0)
        for name in ("drone_mass", "arm_gains"):
            setattr(self.events, name, None)
        for name in ("drone_material", "finger_material"):
            term = getattr(self.events, name, None)
            # a resolved config holds the event term itself; an unresolved one the per-backend preset
            term = getattr(term, "default", term)
            if term is not None and hasattr(term, "params"):
                term.params["static_friction_range"] = (0.6, 0.6)
                term.params["dynamic_friction_range"] = (0.5, 0.5)

    def play_mode(self):
        super().play_mode()
        self.scene.ee_frame.debug_vis = True
        # play and evaluation use the real start distribution, without the near-grasp curriculum
        self.events.reset_drone.params["near_fraction"] = 0.0
        self.actions.drone_hover.preset_weights = {"mid": 1.0}
