# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 + RH-P12-RN(A) drone grasp environment, starting from ``under_drone`` below the drone."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

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
    UR5_RHP12_CFG,
)
from drone_grasp_rl.physics import PHYSICS_VARIANT

from ... import grasp_env_cfg as grasp
from ... import mdp

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
    """Arm joint position targets around ``under_drone`` and an open/close gripper command."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5, use_default_offset=True, preserve_order=True
    )
    # Only the leader joint is commanded; the three mimic joints follow it. The target moves at the real
    # gripper's profile speed (fully closed in 2.2 s); action < 0 closes, otherwise opens.
    gripper_action = mdp.ProfiledBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        open_command_expr={GRIPPER_JOINT_NAME: 0.0},
        close_command_expr={GRIPPER_JOINT_NAME: GRIPPER_CLOSED_POS},
        profile_velocity=GRIPPER_PROFILE_VELOCITY,
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
        self.observations.policy.gripper_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[GRIPPER_JOINT_NAME]
        )
        self.observations.critic.finger_contact.params["sensor_names"] = FINGER_CONTACT_SENSORS
