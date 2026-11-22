"""Render an outside view of a policy grasping the drone (robot, drone, moving base) to PNG frames, for reports.

Adds a world-fixed camera looking at environment 0 to a recording task (``...-Record``: wrist camera, ceiling and
lights already there), runs the policy deterministically and saves every policy step's image. Encode the frames with
``tools/lerobot_export/frames_to_mp4.py`` (H.264). No MoviePy needed (``isaaclab play --video`` requires it).

Usage (from the repository root):

    uv run --extra isaacsim python scripts/render_scene_video.py --task DroneGraspRl-Grasp-UR5-Stewart-Record \\
        --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt --steps 300 --out outputs/scene_frames/stewart
    # --eye / --lookat [m, relative to the environment origin]; --base_preset small|mid|big on moving bases
"""

import argparse
import contextlib
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import drone_grasp_rl.tasks  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=300, help="Policy steps (30 Hz).")
parser.add_argument("--eye", type=float, nargs=3, default=(2.0, -1.8, 2.0))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.4, 0.0, 1.2))
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--base_preset", type=str, default=None)
parser.add_argument("--hover", type=str, default="mid")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def look_at_quat(eye, target):
    """Quaternion (x, y, z, w) turning the world-convention camera (+x forward, +z up) toward the target."""
    dx, dy, dz = (t - e for t, e in zip(target, eye, strict=True))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))  # positive looks down
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    # q = Rz(yaw) * Ry(pitch)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def main():
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.actions.drone_hover.preset_weights = {args_cli.hover: 1.0}
    if args_cli.base_preset is not None:
        env_cfg.actions.base_motion.preset_weights = {args_cli.base_preset: 1.0}
    env_cfg.scene.scene_cam = CameraCfg(
        prim_path="/World/envs/env_0/SceneCam",
        offset=CameraCfg.OffsetCfg(
            pos=tuple(args_cli.eye), rot=look_at_quat(args_cli.eye, args_cli.lookat), convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 50.0)),
        width=args_cli.width,
        height=args_cli.height,
    )
    # the ceiling would hide the view from above; lights stay
    env_cfg.scene.ceiling = None
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    os.makedirs(args_cli.out, exist_ok=True)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        cleanup.callback(env.close)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(
            args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        cam = env.unwrapped.scene["scene_cam"]
        obs = env.get_observations()
        with torch.inference_mode():
            for k in range(args_cli.steps):
                obs, _, _, _ = env.step(policy(obs))
                rgb = cam.data.output["rgb"]
                rgb = getattr(rgb, "torch", rgb)[0, ..., :3].cpu().numpy()
                Image.fromarray(rgb).save(os.path.join(args_cli.out, f"{k:05d}.png"))
        print(f"[scene] {args_cli.steps} frames in {os.path.abspath(args_cli.out)}")


if __name__ == "__main__":
    main()
