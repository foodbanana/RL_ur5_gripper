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


@configclass
class UR5GraspRecordEnvCfg(UR5GraspEnvCfg):
    """Drone grasp with the wrist camera, for recording VLA demonstrations with a trained teacher.

    Same task, observations and actions as :class:`UR5GraspEnvCfg` (the teacher checkpoint loads unchanged); the
    camera is a scene sensor read by ``scripts/record_demos.py``, not a policy input. Starts from the 25 reference
    positions only (no near-start curriculum), physics randomization on as in training.
    """

    def __post_init__(self):
        super().__post_init__()
        _setup_recording(self)


def _setup_recording(cfg):
    """Wrist camera, a ceiling and a key light for visual randomization, the 25 reference starts only."""
    from .camera import wrist_camera_cfg
    from .visuals import ceiling_cfg, key_light_cfg

    cfg.scene.wrist_cam = wrist_camera_cfg()
    # backdrop and shadows for the camera; randomized between recording rounds (visuals.randomize_visuals)
    cfg.scene.ceiling = ceiling_cfg()
    cfg.scene.key_light = key_light_cfg()
    cfg.scene.num_envs = 16
    # visual randomization edits per-environment USD prims; replicated environments are clones the renderer does
    # not read back from USD (a changed ceiling color did not show), as Isaac Lab's visual color terms also require
    cfg.scene.replicate_physics = False
    cfg.events.reset_drone.params["near_fraction"] = 0.0
    # no curricula: rewards do not matter while recording, and a moving base keeps its full amplitude (scale 1)
    cfg.curriculum = None


def _mount_robot(cfg, **params):
    """Fix the robot base to the moving base. PhysX: a prestartup event authors the joint in every environment
    (no replication). Newton: a spawner authors it in env_0, which Newton's builder clones (replication on)."""
    from isaaclab.assets import AssetBaseCfg

    cfg.events.attach_robot = preset(
        default=EventTerm(func=mdp.attach_robot_to_base, mode="prestartup", params=dict(params)),
        newton_mjwarp=None,
    )
    cfg.scene.robot_mount_joint = preset(
        default=None,
        newton_mjwarp=AssetBaseCfg(prim_path="{ENV_REGEX_NS}/RobotMount", spawn=mdp.MountJointCfg(**params)),
    )


@configclass
class UR5GraspStewartEnvCfg(UR5GraspEnvCfg):
    """The drone grasp with the UR5 on the closed-loop Stewart platform (``assets/stewart.py``), no table.

    - the robot base is fixed to the platform's top plate (rotation center = mount) by a joint authored per
      environment before the simulation starts (``mdp.attach_robot_to_base``); robot base 1.0 m above the floor,
      the drone at the same base-relative targets as on the table
    - ``base_motion``: the platform controller (pose command -> inverse kinematics -> leg actuators), presets of the
      data collection repo, amplitude ramped up by the ``base_motion`` curriculum while the success rate holds
    - the drone follows the base pose in the plane with latency and PX4-style limits (``follow_asset``): the robot
      sends its base pose, so the targets stay reachable
    - the policy additionally observes the base IMU (6, appended at the end of the policy group)
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        from drone_grasp_rl.assets import stewart as st
        from drone_grasp_rl.assets.bases import STEWART_CFG, STEWART_LEG_JOINTS, STEWART_TOP_BODY

        scene = self.scene
        scene.table = None
        # PhysX: the mount joint is authored in every environment (no replication). Newton clones the environments
        # from env_0 in its model builder (only env_0 is on the stage); without replication it found no bodies.
        scene.replicate_physics = preset(default=False, newton_mjwarp=True)
        scene.stewart = STEWART_CFG
        scene.robot = scene.robot.replace(
            spawn=scene.robot.spawn.replace(fix_root_link=False),
            init_state=scene.robot.init_state.replace(pos=(0.0, 0.0, st.HEIGHT)),
        )
        drone_pos = scene.drone.init_state.pos
        scene.drone = scene.drone.replace(
            init_state=scene.drone.init_state.replace(
                pos=(drone_pos[0], drone_pos[1], drone_pos[2] - grasp.TABLE_TOP_Z + st.HEIGHT)
            )
        )
        self.actions.base_motion = mdp.StewartMotionCfg(
            asset_name="stewart",
            leg_joint_names=STEWART_LEG_JOINTS,
            # mostly small and mid, a little big (user decision 2026-11-22: big kept to see it, about 11 %): with the
            # platform moving from the start, big (x, z +-15 cm, roll / pitch +-6 deg, the base up to 0.3 m/s)
            # pushed the drone away in most approaches; the drone follows the base in x and y only (not z)
            preset_weights={"small": 1.0, "mid": 1.0, "big": 0.25},
        )
        self.actions.drone_hover.follow_asset = "robot"
        mount = BODY_PRIM_PATHS[BASE_BODY_NAME]
        _mount_robot(
            self,
            base_body_path=f"Stewart/{STEWART_TOP_BODY}",
            robot_body_path=f"Robot/{mount}",
            filtered_robot_bodies=[f"Robot/{mount}", f"Robot/{mount}/base_link/base_link_inertia"],
        )
        self.observations.policy.base_imu = ObsTerm(func=mdp.base_imu)
        self.curriculum.base_motion = CurrTerm(func=mdp.ramp_base_motion, params={"initial": 0.0, "final": 1.0})


@configclass
class UR5GraspStudentEnvCfg(UR5GraspEnvCfg):
    """The fixed-base grasp with a ``student`` observation group for distilling the teacher into a deployable policy.

    The student sees only what the real robot can measure: arm joints, gripper position, TCP (forward kinematics),
    its own last action and gripper command, and the drone pose from the wrist camera's ArUco markers
    (``camera.marker_drone_pose_b``: detection by geometry, noise and latency from ``scripts/study_aruco.py``, the
    reading held with its age when no marker is seen), over the last ``STUDENT_HISTORY`` policy steps. The teacher
    keeps its ``policy`` group (52 entries with the simulator-only state).
    """

    def __post_init__(self):
        super().__post_init__()
        _add_student_obs(self)


STUDENT_HISTORY = 3


def _add_student_obs(cfg):
    """The deployable ``student`` group: the policy group without the simulator-only entries, the ArUco marker drone
    pose instead of mocap (base IMU kept on moving bases), over the last ``STUDENT_HISTORY`` policy steps."""
    import copy

    from isaaclab.managers import ObservationTermCfg as ObsTerm

    from .camera import marker_drone_pose_b

    student = copy.deepcopy(cfg.observations.policy)
    for name in ("drone_pos_true", "drone_quat_true", "drone_vel", "drone_in_tcp", "finger_contact", "grasped"):
        setattr(student, name, None)
    student.drone_pose = ObsTerm(func=marker_drone_pose_b)
    student.history_length = STUDENT_HISTORY
    student.flatten_history_dim = True
    cfg.observations.student = student


@configclass
class UR5GraspHuskyEnvCfg(UR5GraspEnvCfg):
    """The drone grasp with the UR5 on a Husky A200 that drives straight back and forth (no table).

    The Husky drives along its x axis like the data collection repo's kinematic Husky (``mdp.HuskyMotion``:
    0.30 m out and back at 0.05 m/s, provisional), as a carriage on a stiff driven rail that carries the Husky's
    look and, through a fixed joint, the robot (Clearpath's Husky + UR5 mount, yaw -90 deg). Drone following,
    base IMU and the motion curriculum as in :class:`UR5GraspStewartEnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()
        import math

        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        from drone_grasp_rl.assets.bases import (
            HUSKY_BASE_HEIGHT,
            HUSKY_RAIL_CFG,
            HUSKY_ROBOT_MOUNT_POS,
            HUSKY_ROBOT_MOUNT_YAW_DEG,
            HUSKY_VISUAL_CFG,
        )

        scene = self.scene
        scene.table = None
        scene.replicate_physics = preset(default=False, newton_mjwarp=True)  # see UR5GraspStewartEnvCfg
        scene.husky_rail = HUSKY_RAIL_CFG
        scene.husky = HUSKY_VISUAL_CFG
        yaw = math.radians(HUSKY_ROBOT_MOUNT_YAW_DEG)
        base = (HUSKY_ROBOT_MOUNT_POS[0], HUSKY_ROBOT_MOUNT_POS[1], HUSKY_BASE_HEIGHT + HUSKY_ROBOT_MOUNT_POS[2])
        scene.robot = scene.robot.replace(
            spawn=scene.robot.spawn.replace(fix_root_link=False),
            init_state=scene.robot.init_state.replace(pos=base, rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
        )
        # the drone at the same base-relative place as on the table
        d = scene.drone.init_state.pos
        d_b = (d[0], d[1], d[2] - grasp.TABLE_TOP_Z)
        c, s = math.cos(yaw), math.sin(yaw)
        scene.drone = scene.drone.replace(
            init_state=scene.drone.init_state.replace(
                pos=(base[0] + c * d_b[0] - s * d_b[1], base[1] + s * d_b[0] + c * d_b[1], base[2] + d_b[2])
            )
        )
        self.actions.base_motion = mdp.HuskyMotionCfg(asset_name="husky_rail")
        self.actions.drone_hover.follow_asset = "robot"
        mount = BODY_PRIM_PATHS[BASE_BODY_NAME]
        _mount_robot(
            self,
            base_body_path="HuskyRail/carriage",
            robot_body_path=f"Robot/{mount}",
            mount_pos=HUSKY_ROBOT_MOUNT_POS,
            mount_yaw_deg=HUSKY_ROBOT_MOUNT_YAW_DEG,
        )
        self.observations.policy.base_imu = ObsTerm(func=mdp.base_imu)
        self.curriculum.base_motion = CurrTerm(func=mdp.ramp_base_motion, params={"initial": 0.0, "final": 1.0})


@configclass
class UR5GraspStewartRecordEnvCfg(UR5GraspStewartEnvCfg):
    """:class:`UR5GraspStewartEnvCfg` with the recording setup (wrist camera, visual randomization), full motion."""

    def __post_init__(self):
        super().__post_init__()
        _setup_recording(self)


@configclass
class UR5GraspHuskyRecordEnvCfg(UR5GraspHuskyEnvCfg):
    """:class:`UR5GraspHuskyEnvCfg` with the recording setup (wrist camera, visual randomization), full motion."""

    def __post_init__(self):
        super().__post_init__()
        _setup_recording(self)


@configclass
class UR5GraspStewartStudentEnvCfg(UR5GraspStewartEnvCfg):
    """:class:`UR5GraspStewartEnvCfg` with the deployable ``student`` group (marker drone pose, base IMU)."""

    def __post_init__(self):
        super().__post_init__()
        _add_student_obs(self)


@configclass
class UR5GraspHuskyStudentEnvCfg(UR5GraspHuskyEnvCfg):
    """:class:`UR5GraspHuskyEnvCfg` with the deployable ``student`` group (marker drone pose, base IMU)."""

    def __post_init__(self):
        super().__post_init__()
        _add_student_obs(self)
