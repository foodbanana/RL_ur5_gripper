"""Check the robot and drone assets against the values measured in data_collection_ur5_gripper.

Runs one environment with a table, the UR5 + RH-P12-RN in ``under_drone`` and the Iris drone held in
the air by a gravity-compensating wrench, then prints PASS/FAIL for each check:

1. names     arm, gripper leader, mimic followers, TCP and finger bodies exist
2. tcp_fk    TCP position matches forward kinematics computed from the source URDF (< 2 mm)
3. arm_hold  arm holds ``under_drone`` (max tracking error < 0.2 deg after 2 s)
4. close     empty gripper closes to 62..65.1 deg (old repo: stops at about 64.2 deg)
5. mimic     followers track the leader during the close (max error < 1.5 deg)
6. open      gripper reopens below 2 deg
7. hover     drone with per-body gravity compensation drifts < 5 mm and tilts < 1 deg

Usage (from the repository root):

    uv run --extra isaacsim python scripts/check_assets.py --backend isaacsim_physx
    uv run python scripts/check_assets.py --backend newton_mjwarp
    # add --viz kit or --viz newton_gl to watch
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Check the drone_grasp_rl assets.")
parser.add_argument("--backend", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"])
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = args_cli.backend

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from drone_grasp_rl.assets import DRONE_GRASP_RL_ASSETS_DIR  # noqa: E402
from drone_grasp_rl.assets.drones import DRONE_MASS, IRIS_CFG  # noqa: E402
from drone_grasp_rl.assets.robots import (  # noqa: E402
    ARM_JOINT_NAMES,
    FINGER_BODY_NAMES,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAME,
    GRIPPER_MIMIC_JOINT_NAMES,
    GRIPPER_PROFILE_VELOCITY,
    TCP_BODY_NAME,
    TCP_OFFSET_POS,
    UNDER_DRONE_JOINT_POS,
    UR5_RHP12_CFG,
)
from drone_grasp_rl.physics import DRONE_USD, PHYSICS_VARIANT, SIM_DT, DroneGraspPhysicsCfg  # noqa: E402

TABLE_TOP_Z = 0.762
"""Table top height = robot base height (scene_drone.yaml)."""
DRONE_POS_BASE = (0.60, 0.0, 1.50 - TABLE_TOP_Z)
"""Default drone position in the robot base frame (scene_drone.yaml drone_pos, world z 1.50)."""
URDF_PATH = DRONE_GRASP_RL_ASSETS_DIR / "source/ur5_rh_p12_d435i/urdf/ur5_rh_p12_d435i.urdf"
URDF_TCP_LINK = "rh_p12_rn_tcp"

physics_cfg = getattr(DroneGraspPhysicsCfg(), args_cli.backend)
robot_cfg = replace(UR5_RHP12_CFG, prim_path="{ENV_REGEX_NS}/Robot")
robot_cfg.spawn.variants = {"Physics": PHYSICS_VARIANT[args_cli.backend]}
robot_cfg.init_state.pos = (0.0, 0.0, TABLE_TOP_Z)
drone_cfg = replace(IRIS_CFG, prim_path="{ENV_REGEX_NS}/Drone")
drone_cfg.spawn.usd_path = DRONE_USD[args_cli.backend]
drone_cfg.init_state.pos = (DRONE_POS_BASE[0], DRONE_POS_BASE[1], DRONE_POS_BASE[2] + TABLE_TOP_Z)


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.04),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, TABLE_TOP_Z - 0.02)),
    )
    robot = robot_cfg
    drone = drone_cfg


def urdf_fk(urdf_path: Path, joint_pos: dict[str, float], tip: str) -> np.ndarray:
    """Position of ``tip`` in the URDF root link frame, from the joint origins and axes."""
    joints = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        origin = j.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        axis = j.find("axis")
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"),
            parent=j.find("parent").get("link"),
            type=j.get("type"),
            xyz=xyz,
            rpy=rpy,
            axis=np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1.0, 0.0, 0.0]),
        )

    def rot_rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    def rot_axis(axis, angle):
        a = axis / np.linalg.norm(axis)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k

    chain, link = [], tip
    while link in joints:
        chain.append(joints[link])
        link = joints[link]["parent"]
    T = np.eye(4)
    for j in reversed(chain):
        step = np.eye(4)
        step[:3, :3] = rot_rpy(*j["rpy"])
        step[:3, 3] = j["xyz"]
        if j["type"] in ("revolute", "continuous"):
            q = np.eye(4)
            q[:3, :3] = rot_axis(j["axis"], joint_pos.get(j["name"], 0.0))
            step = step @ q
        T = T @ step
    return T[:3, 3]


def main():
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli):
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view([1.8, -1.6, 1.8], [0.4, 0.0, 1.2])
        scene = instantiate(CheckSceneCfg(num_envs=1, env_spacing=4.0))
        sim.reset()
        robot, drone = scene["robot"], scene["drone"]
        results = []

        def step(n: int, gripper_target: float | None = None):
            for _ in range(n):
                robot.actuators.target_command.set_position_index(value=arm_target, joint_ids=arm_ids)
                if gripper_target is not None:
                    robot.actuators.target_command.set_position_index(
                        value=torch.full((1, 1), gripper_target, device=sim.device), joint_ids=[g_id]
                    )
                # Each drone body carries its own weight, so the rotors add no torque (a hover
                # controller in the task replaces this).
                drone.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=hover_force, body_ids=drone_body_ids, is_global=True
                )
                scene.write_data_to_sim()
                sim.step()
                scene.update(SIM_DT)

        # 1. names
        missing = [
            n for n in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME] + GRIPPER_MIMIC_JOINT_NAMES if n not in robot.joint_names
        ]
        missing += [n for n in [TCP_BODY_NAME] + FINGER_BODY_NAMES if n not in robot.body_names]
        print(f"[check] backend={args_cli.backend} joints={robot.joint_names}")
        print(f"[check] bodies={robot.body_names}")
        results.append(("names", not missing, f"missing={missing}"))

        arm_ids = [robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
        g_id = robot.joint_names.index(GRIPPER_JOINT_NAME)
        m_ids = [robot.joint_names.index(n) for n in GRIPPER_MIMIC_JOINT_NAMES]
        tcp_id = robot.body_names.index(TCP_BODY_NAME)
        drone_body_ids = list(range(drone.num_bodies))
        hover_force = torch.zeros(1, drone.num_bodies, 3, device=sim.device)
        hover_force[0, :, 2] = drone.data.body_mass.torch[0] * 9.81
        total_mass = float(drone.data.body_mass.torch[0].sum())
        results.append(
            ("drone_mass", abs(total_mass - DRONE_MASS) < 1e-3, f"total={total_mass:.4f} kg body={drone.body_names}")
        )

        # Start from a clean state: sim.reset() may already have stepped without the hover force.
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_vel.torch))
        arm_target = robot.data.default_joint_pos.torch[:, arm_ids].clone()
        d_root = drone.data.default_root_pose.torch.clone()
        d_root[:, :3] += scene.env_origins
        drone.write_root_pose_to_sim_index(root_pose=d_root)
        drone.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(drone.data.default_root_vel.torch))
        scene.reset()

        # 2./3. hold under_drone for 2 s, then compare TCP with URDF FK and the arm tracking error
        step(240, gripper_target=0.0)
        root = robot.data.root_pose_w.torch[0]
        body = robot.data.body_pose_w.torch[0, tcp_id]
        tcp_w = (
            body[:3]
            + math_utils.quat_apply(body[3:7].unsqueeze(0), torch.tensor([TCP_OFFSET_POS], device=sim.device))[0]
        )
        tcp_b, _ = math_utils.subtract_frame_transforms(
            root[:3].unsqueeze(0), root[3:7].unsqueeze(0), tcp_w.unsqueeze(0)
        )
        tcp_b = tcp_b[0].cpu().numpy()
        tcp_urdf = urdf_fk(URDF_PATH, UNDER_DRONE_JOINT_POS, URDF_TCP_LINK)
        err_mm = float(np.linalg.norm(tcp_b - tcp_urdf) * 1000)
        results.append(
            ("tcp_fk", err_mm < 2.0, f"sim={np.round(tcp_b, 4)} urdf={np.round(tcp_urdf, 4)} err={err_mm:.2f} mm")
        )
        q = robot.data.joint_pos.torch[0]
        q_des = torch.tensor([UNDER_DRONE_JOINT_POS[n] for n in ARM_JOINT_NAMES], device=sim.device)
        hold_err = float(torch.rad2deg((q[arm_ids] - q_des).abs().max()))
        results.append(("arm_hold", hold_err < 0.2, f"max err={hold_err:.4f} deg"))

        # 4./5. close with the profile (target moves at 0.516 rad/s), then wait
        target, mimic_err, t_62 = 0.0, 0.0, None
        for k in range(int(4.0 / SIM_DT)):
            target = min(GRIPPER_CLOSED_POS, target + GRIPPER_PROFILE_VELOCITY * SIM_DT)
            step(1, gripper_target=target)
            qg = robot.data.joint_pos.torch[0]
            mimic_err = max(mimic_err, float((qg[m_ids] - qg[g_id]).abs().max()))
            if t_62 is None and math.degrees(float(qg[g_id])) >= 62.0:
                t_62 = (k + 1) * SIM_DT
        closed_deg = math.degrees(float(robot.data.joint_pos.torch[0, g_id]))
        results.append(("close", 62.0 <= closed_deg <= 65.1, f"final={closed_deg:.2f} deg, reached 62 deg at {t_62} s"))
        results.append(
            ("mimic", math.degrees(mimic_err) < 1.5, f"max |follower - leader|={math.degrees(mimic_err):.3f} deg")
        )

        # 6. open
        target = GRIPPER_CLOSED_POS
        for _ in range(int(4.0 / SIM_DT)):
            target = max(0.0, target - GRIPPER_PROFILE_VELOCITY * SIM_DT)
            step(1, gripper_target=target)
        open_deg = math.degrees(float(robot.data.joint_pos.torch[0, g_id]))
        results.append(("open", open_deg < 2.0, f"final={open_deg:.2f} deg"))

        # 7. hover: drift since spawn after all of the above (about 10 s with the compensating wrench)
        d_pose = drone.data.root_pose_w.torch[0]
        drift_mm = float(torch.linalg.norm(d_pose[:3] - d_root[0, :3]) * 1000)
        tilt_deg = math.degrees(
            float(
                math_utils.quat_error_magnitude(
                    d_pose[3:7].unsqueeze(0), torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=sim.device)
                )[0]
            )
        )
        results.append(("hover", drift_mm < 5.0 and tilt_deg < 1.0, f"drift={drift_mm:.2f} mm tilt={tilt_deg:.3f} deg"))

        print("[check] ---------------- results ----------------")
        for name, ok, info in results:
            print(f"[check] {'PASS' if ok else 'FAIL'}  {name:9s} {info}")
        print(f"[check] {sum(ok for _, ok, _ in results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
