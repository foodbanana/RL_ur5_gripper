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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from drone_grasp_rl.assets.drones import IRIS_CFG
from drone_grasp_rl.physics import SIM_DT, DroneGraspPhysicsCfg

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

    # The wrist and third-view cameras need light also when headless.
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


@configclass
class ActionsCfg:
    """Arm and gripper action terms, set by the robot config."""

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """``policy`` holds only what the real robot can measure (arm, gripper, mocap drone pose); ``critic`` adds
    simulator-only state (exact drone pose and velocity, finger contact forces)."""

    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": MISSING})
        arm_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": MISSING})
        gripper_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": MISSING})
        tcp_pos = ObsTerm(func=mdp.frame_pos_b, params={"frame_name": "tcp"})
        # The real robot gets the drone pose from motion capture: latency, calibration bias and noise.
        drone_pose = ObsTerm(func=mdp.mocap_drone_pose_b)
        actions = ObsTerm(func=mdp.last_action)

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
    """Reset events, applied in this order after the scene reset."""

    # Robot to its default (start) pose with matching joint targets, drone to its default state.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_drone = EventTerm(
        func=mdp.reset_drone_at_targets,
        mode="reset",
        params={"targets": DRONE_TARGETS_BASE, "offset": DRONE_TARGET_OFFSET},
    )

    hold_drone = EventTerm(func=mdp.hold_drone_in_place, mode="reset")


@configclass
class RewardsCfg:
    """Placeholder rewards so the task runs end to end. The grasp rewards come in the next step."""

    approach = RewTerm(func=mdp.tcp_drone_distance_tanh, weight=1.0, params={"std": 0.1})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Drone grasp environment. Physics 120 Hz, policy 30 Hz (decimation 4), as in the data collection repo."""

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics = DroneGraspPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.8, -1.6, 1.8), lookat=(0.4, 0.0, 1.2))

    def play_mode(self):
        super().play_mode()
        self.scene.ee_frame.debug_vis = True
