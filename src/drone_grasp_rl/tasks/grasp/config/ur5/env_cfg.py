# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 + RH-P12-RN(A) drone grasp environment, starting from ``under_drone`` below the drone."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import clone, configclass, replace

from isaaclab_tasks.utils import preset

from drone_grasp_rl.assets.drones import DRONE_BODY_NAME
from drone_grasp_rl.assets.robots import (
    ARM_JOINT_NAMES,
    BASE_BODY_NAME,
    BODY_PRIM_PATHS,
    FINGER_BODY_NAMES,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAME,
    GRIPPER_PROFILE_VELOCITY,
    TCP_BODY_NAME,
    TCP_OFFSET_POS,
    UNDER_DRONE_TCP_POS_B,
    UR5_RHP12_CFG,
)
from drone_grasp_rl.physics import PHYSICS_VARIANT

from ... import grasp_env_cfg as grasp
from ... import mdp

FRAME_MARKER_SMALL_CFG = clone(FRAME_MARKER_CFG)
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
"""Frame axes for ``ee_frame`` in play: a fifth of the default (0.5), the size the cabinet example uses."""

FINGER_CONTACT_SENSORS = [f"{name.removeprefix('rh_p12_rn_')}_contact" for name in FINGER_BODY_NAMES]
"""One drone-filtered contact sensor per finger link: ``r1_contact``, ``r2_contact``, ``l1_contact``, ``l2_contact``."""


def _robot_path(body_name: str) -> str:
    return "{ENV_REGEX_NS}/Robot/" + BODY_PRIM_PATHS[body_name]


@configclass
class UR5GraspSceneCfg(grasp.GraspSceneCfg):
    """Scene with the UR5 at the env origin on the table top."""

    robot: ArticulationCfg = UR5_RHP12_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR5_RHP12_CFG.init_state.replace(pos=(0.0, 0.0, grasp.TABLE_TOP_Z)),
    )

    ee_frame = FrameTransformerCfg(
        prim_path=_robot_path(BASE_BODY_NAME),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=_robot_path(TCP_BODY_NAME), name="tcp", offset=OffsetCfg(pos=TCP_OFFSET_POS)
            ),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Drone/" + DRONE_BODY_NAME, name="drone"),
        ],
        debug_vis=False,
        visualizer_cfg=replace(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/EEFrame"),
    )

    def __post_init__(self):
        super().__post_init__()
        # The converted robot carries one physics payload per engine.
        self.robot.spawn.variants["Physics"] = preset(default=PHYSICS_VARIANT["isaacsim_physx"], **PHYSICS_VARIANT)
        for body_name, sensor_name in zip(FINGER_BODY_NAMES, FINGER_CONTACT_SENSORS, strict=True):
            setattr(
                self,
                sensor_name,
                ContactSensorCfg(
                    prim_path=_robot_path(body_name),
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Drone/" + DRONE_BODY_NAME],
                ),
            )


@configclass
class UR5ActionsCfg(grasp.ActionsCfg):
    """Bounded arm joint goal steps and an open/close gripper command."""

    # Goal moves at most 0.05 rad per 30 Hz step (1.5 rad/s, the old IK's joint target speed limit), reached by
    # linear interpolation like the old repo's 33 ms command interpolation.
    # domain randomization: 0-1 extra policy steps (0-33 ms) of command delay per episode
    # Smoothing, slowing within 25 cm of the grasp point and holding still once grasped: the teacher's demonstrations
    # should be calm (before these: median 27 cm/s over the last 5 cm and 7 deg/s of arm motion while holding; with
    # them 14 cm/s and 0.1 deg/s at the same success rate).
    arm_action = mdp.InterpolatedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        max_step=0.05,
        delay_steps=(0, 1),
        smoothing=0.5,
        slow_distance=0.25,
        min_step_scale=0.1,
        hold_after_grasp=True,
    )
    # Only the leader joint is commanded; the three mimic joints follow it. The target moves at the real
    # gripper's profile speed (fully closed in 2.2 s); action < 0 closes and stays closed for the episode.
    gripper_action = mdp.ProfiledBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        open_command_expr={GRIPPER_JOINT_NAME: 0.0},
        close_command_expr={GRIPPER_JOINT_NAME: GRIPPER_CLOSED_POS},
        profile_velocity=GRIPPER_PROFILE_VELOCITY,
        # one grasp attempt per episode; also lets an exploring policy actually close the gripper
        latch_close=True,
    )


@configclass
class UR5GraspEnvCfg(grasp.GraspEnvCfg):
    """UR5 + RH-P12-RN(A) drone grasp."""

    scene: UR5GraspSceneCfg = UR5GraspSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    actions: UR5ActionsCfg = UR5ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        arm = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)
        self.observations.policy.arm_joint_pos.params["asset_cfg"] = arm
        self.observations.policy.arm_joint_vel.params["asset_cfg"] = arm
        self.rewards.joint_vel.params["asset_cfg"] = arm
        self.rewards.joint_acc.params["asset_cfg"] = arm
        self.observations.policy.finger_contact.params["sensor_names"] = FINGER_CONTACT_SENSORS
        self.rewards.joint_limits.params["asset_cfg"] = arm
        self.observations.policy.gripper_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[GRIPPER_JOINT_NAME]
        )
        self.observations.critic.finger_contact.params["sensor_names"] = FINGER_CONTACT_SENSORS
        self.events.finger_material = grasp.contact_material(SceneEntityCfg("robot", body_names=FINGER_BODY_NAMES))
        # domain randomization: arm drive stiffness and damping x0.8-1.25 (log-uniform) per environment
        self.events.arm_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
                "stiffness_distribution_params": (0.8, 1.25),
                "damping_distribution_params": (0.8, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        # Curriculum: half of the training resets put the drone 0-30 cm right above the start TCP, the rest use the
        # reference positions (play: none). 0-10 cm only taught the final rise and close; the approach from the
        # reference positions was lost until the range covered the full 30 cm.
        self.events.reset_drone.params.update(
            near_fraction=0.5,
            start_tcp_pos_b=UNDER_DRONE_TCP_POS_B,
            grasp_offset=mdp.GRASP_OFFSET,
            near_height_range=(0.0, 0.30),
        )
        self.terminations.grasp_success.params.update(
            right_sensors=[s for s in FINGER_CONTACT_SENSORS if s.startswith("r")],
            left_sensors=[s for s in FINGER_CONTACT_SENSORS if s.startswith("l")],
            gripper_joint=GRIPPER_JOINT_NAME,
        )
