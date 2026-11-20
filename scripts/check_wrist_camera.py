"""Check the wrist camera of the recording task: save images and the projection of the TCP and the drone.

Builds ``DroneGraspRl-Grasp-UR5-Record`` (task + wrist D435i camera), holds the arm at its start pose with the gripper
open for a few steps, then saves each environment's wrist image with markers drawn where the camera model projects:

- red: the TCP (between the fingertips). The data collection repo measured the TCP at about (332, 446) px in its wrist
  image, so the mount and the intrinsics agree when the red mark sits there, between the fingertips
- green: the drone body origin

and prints both pixel positions. With ``--depth`` it also saves the real-like D435i depth (``*_depth.png``, gray
0.1-3 m, black = no reading) and compares the depth at the drone's pixel with its true distance. With ``--checkpoint``
the teacher policy runs instead and images are saved every ``--every`` steps until the first episode ends (a strip
through approach, grasp and hold).

Usage (from the repository root):

    uv run --extra isaacsim python scripts/check_wrist_camera.py
    uv run --extra isaacsim python scripts/check_wrist_camera.py \
        --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
"""

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import drone_grasp_rl.tasks  # noqa: F401

TASK = "DroneGraspRl-Grasp-UR5-Record"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default=None, help="Run this teacher policy instead of holding still.")
parser.add_argument("--every", type=int, default=15, help="With --checkpoint: save an image every this many steps.")
parser.add_argument("--out", type=str, default="outputs/wrist_camera", help="Output folder for the PNG images.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--visual_dr", action="store_true", help="Apply one visual randomization before the images.")
parser.add_argument("--depth", action="store_true", help="Also save the real-like depth (color map) and check it.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def project(points_w, cam_pos_w, cam_quat_w_ros, intrinsics):
    """Pixel (u, v) and depth of world points in the ROS optical frame (+z forward, +y down), shape (N, 3)."""
    import isaaclab.utils.math as math_utils

    rot = math_utils.matrix_from_quat(cam_quat_w_ros)  # camera -> world
    p = torch.einsum("nji,nj->ni", rot, points_w - cam_pos_w)  # world -> camera
    u = intrinsics["fx"] * p[:, 0] / p[:, 2] + intrinsics["cx"]
    v = intrinsics["fy"] * p[:, 1] / p[:, 2] + intrinsics["cy"]
    return torch.stack([u, v, p[:, 2]], dim=-1)


def marks_for(i, uv_tcp, uv_drone):
    return [(uv_tcp[i].tolist(), (255, 0, 0)), (uv_drone[i].tolist(), (0, 255, 0))]


def save(img, marks, path):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for (u, v, z), color in marks:
        if z > 0:
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=color, width=2)
    im.save(path)


def main():
    from drone_grasp_rl.tasks.grasp.config.ur5.camera import WRIST_CAMERA_INTRINSICS
    from drone_grasp_rl.tasks.grasp.mdp import frame_pose_w

    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.wrist_cam.update_latest_camera_pose = True  # the camera moves with the wrist
    if args_cli.depth:
        env_cfg.scene.wrist_cam.data_types = ["rgb", "distance_to_image_plane"]
    env_cfg.seed = args_cli.seed
    if args_cli.checkpoint is None:
        env_cfg.disable_randomization()
        env_cfg.actions.drone_hover.preset_weights = {"static": 1.0}
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    os.makedirs(args_cli.out, exist_ok=True)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(TASK, cfg=env_cfg)
        cleanup.callback(env.close)
        base = env.unwrapped
        cam = base.scene["wrist_cam"]
        policy = None
        if args_cli.checkpoint:
            from isaaclab_rl.rsl_rl import (
                RslRlVecEnvWrapper,
                check_rsl_rl_version,
                create_rsl_rl_runner,
                handle_deprecated_rsl_rl_cfg,
            )

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(env, agent_cfg)
            runner.load(
                args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}
            )
            policy = runner.get_inference_policy(device=base.device)
            obs = env.get_observations()
        else:
            env.reset()
        if args_cli.visual_dr:
            import random

            from drone_grasp_rl.tasks.grasp.config.ur5.visuals import randomize_visuals

            look = randomize_visuals(base, random.Random(args_cli.seed))
            for i, e in enumerate(look["envs"]):
                print(
                    f"[camera] env {i}: ceiling color {[round(c, 2) for c in e['ceiling_color']]}, "
                    f"height {e['ceiling_height']:.2f} m"
                )
        hold = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
        hold[:, -1] = 1.0  # gripper open

        def snapshot(tag):
            tcp = frame_pose_w(base, "tcp")[0]
            drone = base.scene["drone"].data.root_pos_w.torch
            pos, quat = cam.data.pos_w, cam.data.quat_w_ros
            pos = getattr(pos, "torch", pos)
            quat = getattr(quat, "torch", quat)
            uv_tcp = project(tcp, pos, quat, WRIST_CAMERA_INTRINSICS)
            uv_drone = project(drone, pos, quat, WRIST_CAMERA_INTRINSICS)
            rgb = cam.data.output["rgb"]
            rgb = getattr(rgb, "torch", rgb)[..., :3].cpu().numpy()
            for i in range(base.num_envs):
                marks = [(uv_tcp[i].tolist(), (255, 0, 0)), (uv_drone[i].tolist(), (0, 255, 0))]
                save(rgb[i], marks, os.path.join(args_cli.out, f"env{i}_{tag}.png"))
            if args_cli.depth:
                from drone_grasp_rl.tasks.grasp.config.ur5.camera import realsense_depth_mm

                raw = cam.data.output["distance_to_image_plane"]
                raw = getattr(raw, "torch", raw)
                mm = realsense_depth_mm(raw).cpu().numpy()
                z = raw.squeeze(-1).cpu().numpy()
                for i in range(base.num_envs):
                    # gray = 0.1-3 m (near is bright), black = no reading
                    vis = np.where(mm[i] > 0, 255 - np.clip((mm[i] - 100) / 2900 * 230, 0, 230), 0).astype(np.uint8)
                    save(
                        np.stack([vis] * 3, -1),
                        marks_for(i, uv_tcp, uv_drone),
                        os.path.join(args_cli.out, f"env{i}_{tag}_depth.png"),
                    )
                u, v = int(round(float(uv_drone[0, 0]))), int(round(float(uv_drone[0, 1])))
                inside = 0 <= u < mm.shape[2] and 0 <= v < mm.shape[1]
                print(
                    f"[camera] {tag}: depth zero (no reading) {100 * (mm[0] == 0).mean():.1f}% of pixels (env 0); "
                    f"drone center depth along the optical axis {uv_drone[0, 2]:.3f} m"
                    + (f", rendered there {z[0, v, u]:.3f} m, real-like {mm[0, v, u]} mm" if inside else " (off image)")
                )
            top = rgb[:, :60].reshape(base.num_envs, -1).mean(axis=1)
            print(f"[camera] {tag}: mean value of the top 60 rows per env {[round(float(x)) for x in top]}")
            print(
                f"[camera] {tag}: image {tuple(rgb.shape[1:])}, mean value {rgb.mean():.1f}; TCP px (env 0) "
                f"({uv_tcp[0, 0]:.1f}, {uv_tcp[0, 1]:.1f}) depth {uv_tcp[0, 2]:.3f} m; drone px (env 0) "
                f"({uv_drone[0, 0]:.1f}, {uv_drone[0, 1]:.1f}) depth {uv_drone[0, 2]:.3f} m"
            )

        with torch.inference_mode():
            if policy is None:
                for _ in range(10):
                    env.step(hold)
                snapshot("start")
            else:
                step = 0
                while True:
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    step += 1
                    if bool(dones[0]):
                        break
                    if step % args_cli.every == 0:
                        snapshot(f"step{step:03d}")
        print(f"[camera] images in {os.path.abspath(args_cli.out)}")


if __name__ == "__main__":
    main()
