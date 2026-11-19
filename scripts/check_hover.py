"""Check the drone's flight controller: hover tracking per preset and recovery after a push.

Runs ``DroneGraspRl-Grasp-UR5`` with the arm holding still and the gripper open, and prints per hover preset the
tracking error of the drone body to its hover target over 6 s, then the recovery of static-hover drones after a
0.3 m/s sideways velocity kick. Expected: static error ~0, moving presets a few mm (the controller tracks a known
target with feedforward), the kicked drone back within 5 mm in about 2 s.

Usage (from the repository root):

    uv run --extra isaacsim python scripts/check_hover.py physics=isaacsim_physx
    uv run python scripts/check_hover.py physics=newton_mjwarp
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import drone_grasp_rl.tasks  # noqa: F401

TASK = "DroneGraspRl-Grasp-UR5"
PRESETS = ["static", "small", "mid", "big"]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--disturbance", type=float, default=None, help="Override the disturbance force std [N].")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 60.0
    env_cfg.events.reset_drone.params["near_fraction"] = 0.0
    env_cfg.actions.drone_hover.preset_weights = dict.fromkeys(PRESETS, 1.0)
    for name in ("grasp_success", "grasp_failed", "drone_knocked", "protective_stop"):
        setattr(env_cfg.terminations, name, None)
    for name in env_cfg.rewards.__dataclass_fields__:
        if name not in ("action_rate", "joint_vel", "joint_limits"):
            setattr(env_cfg.rewards, name, None)
    env_cfg.curriculum = None  # it ramps reward terms that are turned off here
    env_cfg.actions.drone_hover.judge_term = None
    if args_cli.disturbance is not None:
        env_cfg.actions.drone_hover.disturbance_std = args_cli.disturbance
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        cleanup.callback(env.close)
        hover = env.action_manager.get_term("drone_hover")
        drone = env.scene["drone"]
        body = drone.body_names.index(hover.cfg.body_name)
        n, dev = env.num_envs, env.device
        act = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
        act[:, 6] = 1.0  # gripper open

        def error():
            return drone.data.body_link_pose_w.torch[:, body, :3] - hover.reference_pos_w

        env.reset()
        errs, tilts, axis = [], [], []
        with torch.inference_mode():
            for _ in range(int(6.0 / env.step_dt)):
                env.step(act)
                errs.append(error().norm(dim=-1))
                axis.append(error())
                z = drone.data.body_link_pose_w.torch[:, body, 3:7]
                tilts.append(torch.rad2deg(2 * torch.asin(z[:, :2].norm(dim=-1).clamp(max=1.0))))
        errs, tilts = torch.stack(errs), torch.stack(tilts)
        axis_err = torch.stack(axis)[int(1.0 / env.step_dt) :]  # (T, N, 3) after the first second
        names = [hover.preset_names[i] for i in hover.preset_index.tolist()]
        for preset in PRESETS:
            idx = [i for i, p in enumerate(names) if p == preset]
            if not idx:
                continue
            e = 1000 * errs[:, idx]
            print(
                f"[hover] {preset:6s} ({len(idx):2d} envs): "
                f"per-axis std {1000 * axis_err[:, idx].std(dim=0).mean():.2f} mm, "
                f"tracking error RMS {e.square().mean().sqrt():.2f} mm, "
                f"max {e.max():.2f} mm, max tilt {tilts[:, idx].max():.1f} deg"
            )
        static = torch.tensor([p == "static" for p in names], device=dev)
        if static.any():
            vel = drone.data.root_vel_w.torch.clone()
            vel[:, 1] += 0.3 * static.float()
            drone.write_root_velocity_to_sim_index(root_velocity=vel)
            peak, back = torch.zeros(n, device=dev), torch.full((n,), float("nan"), device=dev)
            with torch.inference_mode():
                for k in range(int(4.0 / env.step_dt)):
                    env.step(act)
                    e = error().norm(dim=-1)
                    peak = torch.maximum(peak, e)
                    settled = torch.isnan(back) & (e < 0.005) & (peak > 0.005)
                    back = torch.where(settled, torch.full_like(back, (k + 1) * env.step_dt), back)
            print(
                f"[hover] push 0.3 m/s (static envs): peak error {1000 * peak[static].max():.1f} mm, back within 5 mm "
                f"after {back[static].nanmean():.2f} s (max {back[static].nan_to_num(99).max():.2f} s)"
            )


if __name__ == "__main__":
    main()
