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
import math
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
parser.add_argument("--hover", type=str, default="mid", help="Hover preset: static, small, mid, big.")
parser.add_argument("--no_dr", action="store_true", help="Evaluate without the domain randomization.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def main():
    import isaaclab.utils.math as math_utils

    from drone_grasp_rl.tasks.grasp.mdp import (
        approach_path_distance,
        frame_lin_vel_w,
        frame_pose_b,
        frame_pose_w,
        tcp_in_grasp_frame,
    )
    from drone_grasp_rl.tasks.grasp.mdp.judge import APPROACH, FAIL_REASONS, HELD, get_judge

    torch.manual_seed(args_cli.seed)
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.reset_drone.params["near_fraction"] = 0.0  # the 25 reference positions only, no curriculum
    env_cfg.actions.drone_hover.preset_weights = {args_cli.hover: 1.0}
    if args_cli.no_dr:
        env_cfg.disable_randomization()
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        cleanup.callback(env.close)
        runner = create_rsl_rl_runner(env, agent_cfg)
        # actor and critic only: checkpoints made by expand_policy_inputs.py carry no optimizer state
        runner.load(
            args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        n, dev = base.num_envs, base.device
        judge = get_judge(base)
        gripper = base.action_manager.get_term("gripper_action")

        done_once = torch.zeros(n, dtype=torch.bool, device=dev)
        best_path = torch.full((n,), float("inf"), device=dev)
        best_rel = torch.zeros(n, 3, device=dev)
        closed_at = torch.full((n,), float("nan"), device=dev)
        tilt_max = torch.full((n,), float("nan"), device=dev)
        near_speed = torch.full((n,), float("nan"), device=dev)
        robot = base.scene["robot"]
        arm_ids = base.action_manager.get_term("arm_action").joint_ids
        prev_arm = torch.zeros(n, 6, device=dev)
        prev_q, prev_tcp = None, None
        holding_stats = []  # per live grasped env-step: max joint speed [rad/s], TCP speed [m/s], arm action change
        outcomes = [None] * n
        hold = torch.full((n, 4), float("nan"), device=dev)

        obs = env.get_observations()
        with torch.inference_mode():
            while not bool(done_once.all()):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                arm = actions[:, :6].clamp(-1.0, 1.0)
                policy.reset(dones)
                live = ~done_once & ~dones.bool()
                rel = tcp_in_grasp_frame(base)
                path = approach_path_distance(rel, 0.10, 0.02)
                better = live & (path < best_path)
                best_path = torch.where(better, path, best_path)
                best_rel = torch.where(better.unsqueeze(-1), rel, best_rel)
                _, tcp_quat_b = frame_pose_b(base, "tcp")
                tilt = torch.acos(math_utils.matrix_from_quat(tcp_quat_b)[..., 2, 2].clamp(-1.0, 1.0))
                after = live & (judge.phase != APPROACH)
                tilt_max = torch.where(after, torch.fmax(tilt_max, tilt), tilt_max)
                tcp_vel = frame_lin_vel_w(base, "tcp")
                rel_speed = (tcp_vel - base.scene["drone"].data.root_lin_vel_w.torch).norm(dim=-1)
                near = live & (judge.phase == APPROACH) & (rel.norm(dim=-1) < 0.05)
                near_speed = torch.where(near, torch.fmax(near_speed, rel_speed), near_speed)
                # speeds from position changes: in contact PhysX reports joint velocities of joints that do not move
                # (0.6 rad/s on the stalled gripper), so the reported velocities overstate the shivering
                q_now = robot.data.joint_pos.torch[:, arm_ids]
                tcp_now = frame_pose_w(base, "tcp")[0]
                holding = live & ((judge.phase == 1) | (judge.phase == 2)) & (prev_q is not None)
                if holding.any():
                    jspeed = (q_now - prev_q).abs().max(dim=-1).values / base.step_dt
                    tspeed = (tcp_now - prev_tcp).norm(dim=-1) / base.step_dt
                    change = (arm - prev_arm).norm(dim=-1)
                    holding_stats.append(torch.stack([jspeed, tspeed, change], dim=-1)[holding])
                prev_q, prev_tcp = q_now.clone(), tcp_now.clone()
                prev_arm = arm.clone()
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
                    if outcomes[i] in ("held", "slip"):
                        hold[i] = judge.hold_metrics[i]
                    done_once[i] = True

        dr = "off" if args_cli.no_dr else "on"
        print(f"[eval] checkpoint: {args_cli.checkpoint} (hover preset: {args_cli.hover}, randomization {dr})")
        print(f"[eval] outcomes over {n} episodes: {dict(Counter(outcomes))}")
        judged = ~torch.isnan(hold[:, 3])
        if judged.any():
            end_tilt = torch.rad2deg(hold[judged, 3])
            print(
                f"[eval] gripper tilt at the end of the hold [deg]: median {end_tilt.median():.1f}, "
                f"90th pct {end_tilt.quantile(0.9):.1f}, max {end_tilt.max():.1f}"
            )
        slips = torch.tensor([o == "slip" for o in outcomes], device=dev)
        if slips.any():
            m = hold[slips]
            print(
                f"[eval] slip causes ({int(slips.sum())}): settle > 15 mm {int((m[:, 0] > 0.015).sum())}, "
                f"last 0.5 s > 0.5 mm {int((m[:, 1] > 0.0005).sum())}, rotation > 10 deg "
                f"{int((m[:, 2] > math.radians(10)).sum())}, "
                f"gripper tilt > 10 deg {int((m[:, 3] > math.radians(10)).sum())}"
            )
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
        if (~torch.isnan(near_speed)).any():
            v = 1000 * near_speed[~torch.isnan(near_speed)]
            print(
                f"[eval] fastest TCP speed relative to the drone in the last 5 cm before the grasp [mm/s]: "
                f"median {v.median():.0f}, 90th pct {v.quantile(0.9):.0f}, max {v.max():.0f}"
            )
        if holding_stats:
            a = torch.cat(holding_stats)
            print(
                f"[eval] after the grasp: max joint speed mean {torch.rad2deg(a[:, 0]).mean():.2f} deg/s "
                f"(95th pct {torch.rad2deg(a[:, 0]).quantile(0.95):.2f}), "
                f"TCP speed mean {1000 * a[:, 1].mean():.1f} mm/s "
                f"(95th pct {1000 * a[:, 1].quantile(0.95):.1f}), arm action change mean {a[:, 2].mean():.3f}"
            )
        grasped = ~torch.isnan(tilt_max)
        if grasped.any():
            deg = torch.rad2deg(tilt_max[grasped])
            print(
                f"[eval] gripper tilt from up after the grasp (max per episode) [deg]: median {deg.median():.1f}, "
                f"90th pct {deg.quantile(0.9):.1f}, max {deg.max():.1f}, "
                f">10 deg in {(deg > 10).float().mean() * 100:.0f}%"
            )


if __name__ == "__main__":
    main()
