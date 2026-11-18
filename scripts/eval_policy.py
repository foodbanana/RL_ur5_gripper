"""Evaluate a trained drone grasp policy: one episode per environment, judge outcomes and approach geometry.

Prints, over all environments:

- the judge outcome counts (held, slip, dropped, opened_early, drone_knocked, protective_stop, time_out)
- the closest approach: the shortest remaining approach path and the TCP position relative to the grasp point at
  that moment (drone axes; z < 0 is below the drone), so a policy that stalls below the drone shows where
- how often the gripper was commanded closed and how close the TCP was when it first closed

Usage (from the repository root):

    uv run --extra isaacsim python scripts/eval_policy.py --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
    # add physics=newton_mjwarp for Newton, --viz kit to watch
"""

import argparse
import contextlib
import sys
from collections import Counter

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

TASK = "DroneGraspRl-Grasp-UR5"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def main():
    from drone_grasp_rl.tasks.grasp.mdp import approach_path_distance, tcp_in_grasp_frame
    from drone_grasp_rl.tasks.grasp.mdp.judge import FAIL_REASONS, HELD, get_judge

    torch.manual_seed(args_cli.seed)
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.reset_drone.params["near_fraction"] = 0.0  # the 25 reference positions only, no curriculum
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        cleanup.callback(env.close)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        n, dev = base.num_envs, base.device
        judge = get_judge(base)
        gripper = base.action_manager.get_term("gripper_action")

        done_once = torch.zeros(n, dtype=torch.bool, device=dev)
        best_path = torch.full((n,), float("inf"), device=dev)
        best_rel = torch.zeros(n, 3, device=dev)
        closed_at = torch.full((n,), float("nan"), device=dev)
        outcomes = [None] * n

        obs = env.get_observations()
        with torch.inference_mode():
            while not bool(done_once.all()):
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                live = ~done_once & ~dones.bool()
                rel = tcp_in_grasp_frame(base)
                path = approach_path_distance(rel, 0.10, 0.02)
                better = live & (path < best_path)
                best_path = torch.where(better, path, best_path)
                best_rel = torch.where(better.unsqueeze(-1), rel, best_rel)
                first_close = live & torch.isnan(closed_at) & gripper.closing
                closed_at = torch.where(first_close, rel.norm(dim=-1), closed_at)
                for i in (dones.bool() & ~done_once).nonzero(as_tuple=False).squeeze(-1).tolist():
                    phase, reason = int(judge.last_phase[i]), int(judge.last_fail_reason[i])
                    fired = [
                        k
                        for k in base.termination_manager.active_terms
                        if bool(base.termination_manager.get_term(k)[i])
                    ]
                    outcomes[i] = "held" if phase == HELD else FAIL_REASONS.get(reason, "+".join(fired) or "?")
                    done_once[i] = True

        print(f"[eval] checkpoint: {args_cli.checkpoint}")
        print(f"[eval] outcomes over {n} episodes: {dict(Counter(outcomes))}")
        mm = 1000 * best_rel
        print(
            f"[eval] closest approach path [mm]: mean {1000 * best_path.mean():.1f}, "
            f"median {1000 * best_path.median():.1f}, <10 mm in {(best_path < 0.01).float().mean() * 100:.0f}% of envs"
        )
        print(
            "[eval] TCP at closest approach, relative to the grasp point [mm] (drone axes): "
            f"x mean {mm[:, 0].mean():+.1f} (|x| {mm[:, 0].abs().mean():.1f}), "
            f"y mean {mm[:, 1].mean():+.1f} (|y| {mm[:, 1].abs().mean():.1f}), "
            f"z mean {mm[:, 2].mean():+.1f} (min {mm[:, 2].min():+.1f}, max {mm[:, 2].max():+.1f})"
        )
        closed = ~torch.isnan(closed_at)
        if closed.any():
            print(
                f"[eval] gripper closed in {closed.float().mean() * 100:.0f}% of episodes, "
                f"TCP distance at first close: median {1000 * closed_at[closed].median():.1f} mm"
            )
        else:
            print("[eval] gripper never closed")


if __name__ == "__main__":
    main()
