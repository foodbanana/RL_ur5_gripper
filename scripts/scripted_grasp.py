"""Scripted grasp of the hovering drone: checks the grasp judge and the task protocol before any training.

An IK state machine drives ``DroneGraspRl-Grasp-UR5`` through the same actions a policy would send:

1. PRE    move the TCP to the pre-grasp point 10 cm below the drone waist, gripper axis along the drone z axis
2. RISE   rise straight up to the grasp point (drone body origin + the planned TCP offset of the old repo)
3. CLOSE  hold the arm goal and close the gripper; the judge then detects the grasp, stops the drone motors
          after 1 s and decides success (held) or failure 2.5 s later

The expert uses the true drone pose (it is privileged, like a teleoperator who sees the drone). Each environment
runs one episode; the script prints the outcome per environment and the totals.

Usage (from the repository root):

    uv run --extra isaacsim python scripts/scripted_grasp.py physics=isaacsim_physx
    uv run python scripts/scripted_grasp.py physics=newton_mjwarp
    # add --viz kit / --viz newton_gl to watch
"""

import argparse
import contextlib
import sys
from collections import Counter

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import drone_grasp_rl.tasks  # noqa: F401

TASK = "DroneGraspRl-Grasp-UR5"
PRE_DISTANCE = 0.10
"""Pre-grasp point below the grasp point along the drone z axis [m]."""
GRASP_OFFSET = (0.0, -0.0013, -0.0045)
"""Planned TCP position in the drone body frame [m] (old repo, waist center)."""

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

PRE, RISE, CLOSE = 0, 1, 2


def main():
    import isaaclab.utils.math as math_utils

    from drone_grasp_rl.assets.robots import ARM_JOINT_NAMES, TCP_BODY_NAME, TCP_OFFSET_POS
    from drone_grasp_rl.tasks.grasp.mdp import frame_pose_w
    from drone_grasp_rl.tasks.grasp.mdp.judge import FAIL_REASONS, HELD, get_judge

    torch.manual_seed(args_cli.seed)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.reset_drone.params["near_fraction"] = 0.0  # the 25 reference positions only, no curriculum
    # a hovering drone: the expert freezes the arm while closing, which only works on a still target (as in the
    # data collection repo, where hover episodes were teleoperated)
    env_cfg.actions.drone_hover.preset_weights = {"static": 1.0}
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        cleanup.callback(env.close)
        n, dev = env.num_envs, env.device
        robot, drone = env.scene["robot"], env.scene["drone"]
        arm_term = env.action_manager.get_term("arm_action")
        judge = get_judge(env)
        reset_term = env.event_manager.get_term_cfg("reset_drone").func
        arm_ids = [robot.joint_names.index(j) for j in ARM_JOINT_NAMES]
        body = robot.body_names.index(TCP_BODY_NAME)
        jac_body = body - 1 if robot.is_fixed_base else body
        jac_cols = [j + robot.num_base_dofs for j in arm_ids]
        offset_b = torch.tensor(TCP_OFFSET_POS, device=dev).repeat(n, 1)
        grasp_offset = torch.tensor(GRASP_OFFSET, device=dev).repeat(n, 1)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)

        env.reset()
        # TCP orientation relative to the drone: the start pose holds the gripper the way it grasps a drone that
        # has the base orientation (grasp axis along the drone y axis, approach along +z).
        tcp_quat_w = frame_pose_w(env, "tcp")[1]
        tcp_in_drone = math_utils.quat_mul(math_utils.quat_inv(robot.data.root_quat_w.torch), tcp_quat_w)

        state = torch.full((n,), PRE, dtype=torch.long, device=dev)
        steps = torch.zeros(n, dtype=torch.long, device=dev)
        returns = torch.zeros(n, device=dev)
        done_once = torch.zeros(n, dtype=torch.bool, device=dev)
        outcomes = [None] * n
        targets = reset_term.target_index.clone()

        with torch.inference_mode():
            while not bool(done_once.all()):
                # targets in the world frame
                d_pos, d_quat = drone.data.root_pos_w.torch, drone.data.root_quat_w.torch
                grasp_w = d_pos + math_utils.quat_apply(d_quat, grasp_offset)
                pre_w = grasp_w - PRE_DISTANCE * math_utils.quat_apply(d_quat, up)
                quat_des = math_utils.quat_mul(d_quat, tcp_in_drone)
                tcp_pos, tcp_quat = frame_pose_w(env, "tcp")
                goal = torch.where((state == PRE).unsqueeze(-1), pre_w, grasp_w)
                pos_err, rot_err = math_utils.compute_pose_error(tcp_pos, tcp_quat, goal, quat_des)
                dist = pos_err.norm(dim=-1)
                state = torch.where((state == PRE) & (dist < 0.01), torch.full_like(state, RISE), state)
                state = torch.where((state == RISE) & (dist < 0.001), torch.full_like(state, CLOSE), state)
                # at most 1 cm of TCP travel per step (0.3 m/s), 5 mm while rising to the waist
                limit = torch.where(state == PRE, 0.01, 0.005).unsqueeze(-1)
                pos_err = pos_err * (limit / dist.clamp(min=1e-6).unsqueeze(-1)).clamp(max=1.0)

                # TCP Jacobian from the gripper base body Jacobian (world frame)
                jac = robot.data.body_link_jacobian_w.torch[:, jac_body, :, jac_cols]
                body_quat = robot.data.body_link_quat_w.torch[:, body]
                r = math_utils.quat_apply(body_quat, offset_b)
                jv = jac[:, :3] - torch.linalg.cross(r.unsqueeze(-1).expand(-1, -1, 6), jac[:, 3:], dim=1)
                j = torch.cat([jv, jac[:, 3:]], dim=1)
                err = torch.cat([pos_err, rot_err], dim=-1).unsqueeze(-1)
                lam = 0.05
                dq = (
                    j.transpose(1, 2)
                    @ torch.linalg.solve(j @ j.transpose(1, 2) + lam**2 * torch.eye(6, device=dev), err)
                ).squeeze(-1)
                # absolute joint goal = measured joints + dq, expressed as a step of the goal
                q = robot.data.joint_pos.torch[:, arm_ids]
                action = torch.zeros(n, 7, device=dev)
                action[:, :6] = ((q + dq - arm_term.processed_actions) / arm_term.cfg.max_step).clamp(-1.0, 1.0)
                # Once closing, hold the arm goal (the old demo also stopped tracking near the grasp): chasing the
                # grasped drone would drag it along.
                action[:, :6] *= (state != CLOSE).unsqueeze(-1)
                action[:, 6] = torch.where(state == CLOSE, -1.0, 1.0)

                _, reward, _, _, _ = env.step(action)
                returns += reward
                steps += 1
                for i in env.reset_buf.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if not done_once[i]:
                        phase, reason = int(judge.last_phase[i]), int(judge.last_fail_reason[i])
                        fired = [
                            k
                            for k in env.termination_manager.active_terms
                            if bool(env.termination_manager.get_term(k)[i])
                        ]
                        label = "held" if phase == HELD else FAIL_REASONS.get(reason, "+".join(fired) or "?")
                        if label in ("held", "slip"):
                            m = judge.hold_metrics[i]
                            label += (
                                f" (settle {1000 * m[0]:.1f} mm, last 0.5 s {1000 * m[1]:.2f} mm,"
                                f" rot {torch.rad2deg(m[2]):.1f} deg, gripper tilt {torch.rad2deg(m[3]):.1f} deg)"
                            )
                        outcomes[i] = (
                            int(targets[i]),
                            f"{label}, return {float(returns[i]):.1f}",
                            int(steps[i]) * env.step_dt,
                        )
                        done_once[i] = True
                    state[i] = PRE
                    steps[i] = 0
                    returns[i] = 0.0
                    targets[i] = reset_term.target_index[i]

        names = [
            f"x{round(100 * t[0])}_y{round(100 * t[1]):+d}_z{round(100 * t[2])}" for t in reset_term.targets.tolist()
        ]
        for i, (t, label, sec) in enumerate(outcomes):
            print(f"[scripted] env {i:2d} target {names[t]:14s} after {sec:5.2f} s -> {label}")
        count = Counter(label.split(" ")[0].rstrip(",") for _, label, _ in outcomes)
        held_times = [sec for _, label, sec in outcomes if label.startswith("held")]
        mean_time = sum(held_times) / max(1, len(held_times))
        print(f"[scripted] outcomes: {dict(count)}; held {count['held']}/{n}; mean episode {mean_time:.2f} s when held")


if __name__ == "__main__":
    main()
