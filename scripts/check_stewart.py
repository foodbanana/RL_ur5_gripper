"""Check the closed-loop Stewart platform: does the top plate follow the commanded pose through the legs?

Runs the platform alone (``assets/bases.py`` ``STEWART_CFG``), one environment per motion preset (static, small, mid,
big). Every physics step the pose command of the data collection repo's sine motion goes through the inverse
kinematics (``assets/stewart.py`` ``leg_lengths``) to the leg actuators. Prints per preset:

- top-plate tracking: position error [mm] and rotation error [deg] of the measured top plate against the command
  (RMS and max after the 5 s ramp), and the lag that best aligns them
- loop closure: the gap between each rod tip and its top-plate anchor (0 for a closed mechanism) [mm]
- legs: measured length range against the stroke, leg speed, actuator force

Usage (from the repository root):

    uv run --extra isaacsim python scripts/check_stewart.py
    uv run python scripts/check_stewart.py --backend newton_mjwarp
    # --payload 20 adds 20 kg on the top plate (robot stand-in); --viz kit to watch
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--backend", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"])
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument(
    "--hold", type=str, default=None, help="Constant pose command 'x,y,z,roll,pitch,yaw' [m, rad] for all envs."
)
parser.add_argument("--payload", type=float, default=20.0, help="Extra mass on the top plate [kg] (UR5 ~ 20 kg).")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = args_cli.backend

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from drone_grasp_rl.assets import stewart as st  # noqa: E402
from drone_grasp_rl.assets.bases import STEWART_CFG, STEWART_LEG_JOINTS, STEWART_TOP_BODY  # noqa: E402
from drone_grasp_rl.physics import SIM_DT, DroneGraspPhysicsCfg  # noqa: E402

PRESETS = list(st.MOTION_PRESETS)


@configclass
class StewartSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    stewart = STEWART_CFG


def main():
    physics_cfg = getattr(DroneGraspPhysicsCfg(), args_cli.backend)
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli):
        from isaaclab.scene import InteractiveScene  # loads pxr: only after the app starts

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view((3.5, -3.5, 2.5), (0.0, 0.0, 0.8))
        scene_cfg = StewartSceneCfg(num_envs=len(PRESETS), env_spacing=3.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot = scene["stewart"]
        dev = sim.device
        legs = [robot.joint_names.index(j) for j in STEWART_LEG_JOINTS]
        top = robot.body_names.index(STEWART_TOP_BODY)
        if args_cli.payload > 0:  # a heavier top plate stands in for the robot
            mass = robot.data.body_mass.torch[:, top : top + 1].clone() + args_cli.payload
            robot.set_masses_index(masses=mass, body_ids=[top])
        rods = [robot.body_names.index(f"leg_{i}_rod") for i in range(6)]
        b, p = (a.to(dev) for a in st.anchors(dev))
        l0 = st.nominal_leg_length()
        n = scene.num_envs
        motion = st.SineMotion(n, dev, torch.Generator(device=dev).manual_seed(0))
        motion.reset(torch.arange(n, device=dev), torch.tensor([st.MOTION_PRESETS[k] for k in PRESETS], device=dev))
        origin = scene.env_origins
        steps = int(args_cli.seconds / SIM_DT)
        hold = (
            None if args_cli.hold is None else torch.tensor([[float(v) for v in args_cli.hold.split(",")]], device=dev)
        )
        log = {k: [] for k in ("cmd", "pos", "quat", "gap", "len", "force")}
        for k in range(steps):
            t = torch.full((n,), k * SIM_DT, device=dev)
            cmd = motion.pose(t) if hold is None else hold.expand(n, 6)
            target = st.leg_lengths(cmd, b, p) - l0
            robot.set_joint_position_target_index(target=target, joint_ids=legs)
            scene.write_data_to_sim()
            sim.step()
            scene.update(SIM_DT)
            pose = robot.data.body_link_pose_w.torch
            log["cmd"].append(cmd)
            log["pos"].append(pose[:, top, :3] - origin)
            log["quat"].append(pose[:, top, 3:7].clone())
            # rod tip (z = l0 in the rod frame) vs top anchor
            rod_pose = pose[:, rods]
            tip = rod_pose[..., :3] + math_utils.quat_apply(
                rod_pose[..., 3:7], torch.tensor([0.0, 0.0, l0], device=dev).expand(n, 6, 3)
            )
            anchor = pose[:, top, None, :3] + math_utils.quat_apply(
                pose[:, top, None, 3:7].expand(n, 6, 4), p.expand(n, 6, 3)
            )
            log["gap"].append((tip - anchor).norm(dim=-1).max(dim=-1).values)
            log["len"].append(robot.data.joint_pos.torch[:, legs].clone() + l0)
            log["force"].append(robot.data.applied_torque.torch[:, legs].abs().max(dim=-1).values)
        cmd, pos, quat = (torch.stack(log[k], 1) for k in ("cmd", "pos", "quat"))
        gap, lens, force = (torch.stack(log[k], 1) for k in ("gap", "len", "force"))
        want_pos = cmd[..., :3] + torch.tensor([0.0, 0.0, st.HEIGHT], device=dev)
        want_rot = st.rotation(cmd[..., 3:])
        after = min(int(st.RAMP_TIME / SIM_DT), steps - 40)
        print(f"[stewart] backend {args_cli.backend}, payload {args_cli.payload} kg, {args_cli.seconds} s")
        if hold is not None:
            from isaaclab.utils.math import euler_xyz_from_quat

            r, pp, y = euler_xyz_from_quat(quat[0, -1:])
            offset = 1000 * (pos[0, -1] - torch.tensor([0, 0, st.HEIGHT], device=dev))
            print(
                f"[stewart] hold {args_cli.hold}: top plate offset {[round(float(v), 2) for v in offset]} mm, "
                f"roll/pitch/yaw {float(r):.4f} {float(pp):.4f} {float(y):.4f} rad"
            )
        for i, name in enumerate(PRESETS):
            pe = 1000 * (pos[i, after:] - want_pos[i, after:]).norm(dim=-1)
            # rotation error from the matrices: angle of R_measured^T R_command
            rel = math_utils.matrix_from_quat(quat[i, after:]).transpose(-1, -2) @ want_rot[i, after:]
            re = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)))
            # lag: shift of the command that best matches the measured position
            best = (0, float(pe.square().mean()))
            for lag in range(1, 31):
                e = (pos[i, after:] - want_pos[i, after - lag : pos.shape[1] - lag]).norm(dim=-1).square().mean()
                best = min(best, (lag, float(e)), key=lambda x: x[1])
            speed = (lens[i, 1:] - lens[i, :-1]).abs().max() / SIM_DT
            print(
                f"[stewart] {name:6s}: position error RMS {pe.square().mean().sqrt():.2f} mm (max {pe.max():.2f}), "
                f"rotation error RMS {re.square().mean().sqrt():.3f} deg (max {re.max():.3f}), "
                f"lag {1000 * best[0] * SIM_DT:.0f} ms; "
                f"loop gap max {1000 * gap[i].max():.3f} mm; legs {lens[i].min():.3f}..{lens[i].max():.3f} m "
                f"(stroke {st.LEG_MIN}..{st.LEG_MAX}), speed max {speed:.3f} m/s, force max {force[i].max():.0f} N"
            )


if __name__ == "__main__":
    main()
