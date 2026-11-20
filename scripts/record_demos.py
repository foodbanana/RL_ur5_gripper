"""Record VLA demonstrations with the teacher policy: wrist camera + state + action, successful episodes only.

Runs ``DroneGraspRl-Grasp-UR5-Record`` (the grasp task with the wrist D435i camera) with a trained teacher and writes
each successful (HELD) episode as an intermediate folder in the data collection repo's format (``schema_version`` 2,
the same as ``lerobot_stage1_extract_bag.py`` makes from a ROS bag):

    <out>/<episode>/states.npy (T, 7)   arm joints [rad] + gripper position / closed position (0 = open, 0-1)
                    actions.npy (T, 7)  arm joint goals [rad] sent to the drives + gripper command {0, 1}
                    base_imu.npy (T, 6) fixed base: [0, 0, 0, 0, 0, 9.81]
                    timestamps.npy (T,) k / 25 s
                    wrist/000000.png    640x480 RGB
                    meta.json

``tools/lerobot_export`` turns the folders into a LeRobot v2.1 data set (one camera, ``observation.images.wrist``).

Timing: the policy runs at 30 Hz; the frames are put on a 25 Hz grid by taking the latest 30 Hz frame at each grid
time, as the bag extractor does with the 30 Hz camera topics (the data sets are 25 fps). A frame is (state, image) after
a policy step and the action the teacher sent in response: the arm goal after the action term's filter and slowdown,
like ``/joint_command`` on the real robot. The first frame after a reset is dropped (its image is rendered before the
reset). The teacher acts deterministically (mean action); physics randomization stays on as in training.

Rounds and visual randomization: all environments reset together and each one's first episode is recorded; when all
have finished, the look changes (``config/ur5/visuals.py``: lights, per-environment ceiling color and height, table
color) and the next round starts. Lights are global and the environments end at different times, so the look only
changes between rounds, never within a recorded episode. ``--no_visual_dr`` keeps the default look.

Usage (from the repository root):

    uv run --extra isaacsim python scripts/record_demos.py \
        --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt --num_episodes 10 --name first10
    # hover mix, e.g. --hover static:0.1,small:0.3,mid:0.4,big:0.2 ; Hydra overrides pass through
    #   (a teacher trained with other action settings needs them, e.g. env.actions.arm_action.slow_distance=0.30)
"""

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import gymnasium as gym
import numpy as np
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

TASK = "DroneGraspRl-Grasp-UR5-Record"
FPS = 25
BASE_IMU_CONST = [0.0, 0.0, 0.0, 0.0, 0.0, 9.81]
STATE_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper",
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", type=str, required=True, help="Teacher policy checkpoint.")
parser.add_argument("--num_episodes", type=int, default=10, help="Successful episodes to save.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--name", type=str, default=None, help="Recording name (folder below --out_root).")
parser.add_argument("--out_root", type=str, default="datasets/intermediate")
parser.add_argument("--hover", type=str, default="static:0.1,small:0.3,mid:0.4,big:0.2", help="Hover preset weights.")
parser.add_argument("--no_dr", action="store_true", help="Record without the physics randomization.")
parser.add_argument(
    "--no_visual_dr", action="store_true", help="Keep the default look (no light/ceiling/table changes)."
)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resample(n_frames: int, policy_dt: float) -> list[int]:
    """Latest 30 Hz frame index at each 25 Hz grid time within the episode."""
    duration = (n_frames - 1) * policy_dt
    n_out = int(math.floor(duration * FPS + 1e-9)) + 1
    return [min(n_frames - 1, int(math.floor(j / FPS / policy_dt + 1e-6))) for j in range(n_out)]


def main():
    from PIL import Image

    from drone_grasp_rl.assets.robots import GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAME
    from drone_grasp_rl.tasks.grasp.config.ur5 import camera
    from drone_grasp_rl.tasks.grasp.mdp.judge import FAIL_REASONS, HELD, KILLED, get_judge

    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args_cli.num_envs
    weights = {k: float(v) for k, v in (item.split(":") for item in args_cli.hover.split(","))}
    env_cfg.actions.drone_hover.preset_weights = weights
    if args_cli.no_dr:
        env_cfg.disable_randomization()
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    name = args_cli.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args_cli.out_root, name))
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[record] output folder exists and is not empty: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    commit = git_commit()

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        from drone_grasp_rl.tasks.grasp.config.ur5.visuals import randomize_visuals  # uses pxr: after the app starts

        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        cleanup.callback(env.close)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(
            args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        n, dt = base.num_envs, base.step_dt
        judge = get_judge(base)
        robot = base.scene["robot"]
        cam = base.scene["wrist_cam"]
        arm_term = base.action_manager.get_term("arm_action")
        grip_term = base.action_manager.get_term("gripper_action")
        hover = base.action_manager.get_term("drone_hover")
        reset_ev = base.event_manager.get_term_cfg("reset_drone").func
        arm_ids = arm_term.joint_ids
        grip_id = robot.find_joints(GRIPPER_JOINT_NAME)[0][0]

        # the command of each step, captured inside the step before a reset can clear it
        sent = {}
        arm_process, grip_process = arm_term.process_actions, grip_term.process_actions

        def arm_hook(actions):
            arm_process(actions)
            sent["arm"] = arm_term.processed_actions.clone()

        def grip_hook(actions):
            grip_process(actions)
            sent["grip"] = grip_term.closing.clone()

        arm_term.process_actions, grip_term.process_actions = arm_hook, grip_hook

        def observe():
            q = robot.data.joint_pos.torch
            state = torch.cat([q[:, arm_ids], (q[:, grip_id : grip_id + 1] / GRIPPER_CLOSED_POS).clamp(0, 1)], dim=-1)
            rgb = cam.data.output["rgb"]
            rgb = getattr(rgb, "torch", rgb)[..., :3]
            return state.cpu().numpy(), rgb.cpu().numpy()

        def episode_info(i):
            return {
                "hover_preset": hover.preset_names[int(hover.preset_index[i])],
                "drone_target_index": int(reset_ev.target_index[i]),
                "drone_reset_pos_base": [round(float(x), 5) for x in reset_ev.reset_pos_b[i]],
            }

        outcomes, saved, total_frames = Counter(), 0, 0
        writer = ThreadPoolExecutor(max_workers=8)
        jobs = []

        def save_episode(buf, idx, look):
            frames = resample(len(buf["states"]), dt)
            ep_name = f"{name}_ep{idx:04d}"
            ep_dir = os.path.join(out_dir, ep_name)
            os.makedirs(os.path.join(ep_dir, "wrist"), exist_ok=True)
            states = np.stack([buf["states"][k] for k in frames]).astype(np.float32)
            actions = np.stack([buf["actions"][k] for k in frames]).astype(np.float32)
            np.save(os.path.join(ep_dir, "states.npy"), states)
            np.save(os.path.join(ep_dir, "actions.npy"), actions)
            np.save(os.path.join(ep_dir, "base_imu.npy"), np.tile(np.float32(BASE_IMU_CONST), (len(frames), 1)))
            np.save(os.path.join(ep_dir, "timestamps.npy"), np.arange(len(frames), dtype=np.float64) / FPS)
            for j, k in enumerate(frames):
                jobs.append(
                    writer.submit(
                        lambda img, p: Image.fromarray(img).save(p, compress_level=3),
                        buf["images"][k],
                        os.path.join(ep_dir, "wrist", f"{j:06d}.png"),
                    )
                )
            kill = []
            if buf["kill_step"] is not None:
                t = buf["kill_step"] * dt
                kill = [{"t": round(t, 6), "t_from_start": round(t, 6), "frame": int(math.ceil(t * FPS - 1e-9))}]
            meta = {
                "schema_version": 2,
                "bag_name": ep_name,
                "mode": "sim_rl_teacher",
                "fps": float(FPS),
                "num_frames": len(frames),
                "t_start": 0.0,
                "t_end": round((len(frames) - 1) / FPS, 6),
                "state_dim": 7,
                "state_names": STATE_NAMES,
                "action_dim": 7,
                "action_names": STATE_NAMES,
                "arm_action": "command",
                "arm_action_source": "arm_action goal (filtered, slowed) sent to the joint drives, 30 Hz",
                "gripper_action_source": "gripper close command (latched), 30 Hz",
                "gripper_scale": "joint position / closed position (0 = open, 1 = closed); command {0, 1}",
                "base_imu": "const",
                "base_imu_names": ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"],
                "image_keys": ["wrist"],
                "image_size": [camera.WRIST_CAMERA_HEIGHT, camera.WRIST_CAMERA_WIDTH, 3],
                "cameras": {
                    "wrist": {
                        "device": "sim",
                        "prim": camera.WRIST_CAMERA_FRAME + "/wrist_cam",
                        "intrinsics": camera.WRIST_CAMERA_INTRINSICS,
                        "source": "drone_grasp_rl.tasks.grasp.config.ur5.camera",
                    }
                },
                "resample": {"from_hz": round(1.0 / dt, 3), "rule": "latest frame at each 25 Hz grid time"},
                "protective_stop": {"occurred": False},
                "drone_kill": kill,
                "episode": {
                    "episode": ep_name,
                    "mode": "sim_rl_teacher",
                    "task_name": "drone_grasp_rl_teacher",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "outcome": "held",
                    "teacher_checkpoint": os.path.relpath(os.path.abspath(args_cli.checkpoint)),
                    "seed": args_cli.seed,
                    "physics_randomization": not args_cli.no_dr,
                    "visual_randomization": look,
                    "hydra_overrides": hydra_args,
                    "git": commit,
                    **buf["info"],
                },
            }
            with open(os.path.join(ep_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            return len(frames)

        # Rounds: all environments reset together, each one's first episode is recorded, the rest of the round is
        # not (an environment that finishes early runs on until the slowest one ends). The look changes only between
        # rounds, so it never changes within a recorded episode.
        rng = random.Random(args_cli.seed)
        t0, rounds = time.time(), 0
        with torch.inference_mode():
            while saved < args_cli.num_episodes:
                look = None if args_cli.no_visual_dr else randomize_visuals(base, rng)
                obs, _ = env.reset()
                rounds += 1
                info_now = [episode_info(i) for i in range(n)]
                active = [True] * n
                buffers, pending = [None] * n, [None] * n  # the first frame after the reset is dropped
                while any(active) and saved < args_cli.num_episodes:
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    arm_cmd = sent["arm"].cpu().numpy()
                    grip_cmd = sent["grip"].float().cpu().numpy()
                    killed = (judge.phase == KILLED).cpu().numpy()
                    done = dones.bool().cpu().numpy()
                    states, images = observe()
                    for i in range(n):
                        if not active[i]:
                            continue
                        if pending[i] is not None:
                            if buffers[i] is None:
                                buffers[i] = {
                                    "states": [],
                                    "actions": [],
                                    "images": [],
                                    "kill_step": None,
                                    "info": info_now[i],
                                }
                            buf = buffers[i]
                            buf["states"].append(pending[i][0])
                            buf["images"].append(pending[i][1])
                            buf["actions"].append(np.concatenate([arm_cmd[i], grip_cmd[i : i + 1]]))
                            if buf["kill_step"] is None and killed[i]:
                                buf["kill_step"] = len(buf["states"])
                        if done[i]:
                            active[i] = False
                            held = int(judge.last_phase[i]) == HELD
                            reason = "held" if held else FAIL_REASONS.get(int(judge.last_fail_reason[i]), "other")
                            outcomes[reason] += 1
                            if held and buffers[i] is not None and saved < args_cli.num_episodes:
                                env_look = None
                                if look is not None:
                                    env_look = {k: v for k, v in look.items() if k != "envs"} | look["envs"][i]
                                total_frames += save_episode(buffers[i], saved, env_look)
                                saved += 1
                            buffers[i] = None
                        else:
                            pending[i] = (states[i], images[i])
                print(f"[record] round {rounds}: saved {saved}/{args_cli.num_episodes}, outcomes {dict(outcomes)}")
        for job in jobs:
            job.result()
        writer.shutdown()
        tried = sum(outcomes.values())
        print(
            f"[record] {saved} episodes, {total_frames} frames at {FPS} fps in {out_dir} "
            f"({time.time() - t0:.0f} s, {rounds} rounds); outcomes of {tried} recorded episodes: {dict(outcomes)}"
        )


if __name__ == "__main__":
    main()
