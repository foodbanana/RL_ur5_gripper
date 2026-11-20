"""Check the drone grasp task before any training: reset geometry, action contract, sensors.

Runs ``DroneGraspRl-Grasp-UR5`` with scripted actions and prints PASS/FAIL per check:

1. targets   every drone starts within the offset of one of the 25 reference positions (base frame)
2. clear     no finger touches the drone after the reset
3. hold      with zero actions the arm stays in ``under_drone`` and the drone stays in place for 2 s
3b. mocap    the policy's drone pose differs from the true pose like the motion-capture model: per-axis
             noise about 1 mm, offset within the calibration bias, and while the drone moves at 0.1 m/s a
             lag of about 2.5 mm (20 ms latency + half of the 10 ms sample period)
4. arm       an arm action moves each joint goal by ``action * max_step`` per step; the joints follow the
             interpolated target within 4 deg at full speed (old repo 2.9 deg, protective stop 5 deg)
             and settle on the goal
5. close     a close command shuts the empty gripper past 62 deg; an open command opens it again
6. grasp     with the drone waist placed at the TCP, closing at the gripper profile speed stops before
             62 deg with both fingers on the drone (> 1 N per side) and the drone within 10 mm of the TCP:
             the contact sensors and the grasp geometry work

Usage (from the repository root):

    uv run --extra isaacsim python scripts/check_task.py physics=isaacsim_physx
    uv run python scripts/check_task.py physics=newton_mjwarp
    # add --viz kit / --viz newton_gl to watch
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

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def main():
    import isaaclab.utils.math as math_utils

    from drone_grasp_rl.assets.robots import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME
    from drone_grasp_rl.tasks.grasp.config.ur5.env_cfg import FINGER_CONTACT_SENSORS
    from drone_grasp_rl.tasks.grasp.grasp_env_cfg import DRONE_TARGET_OFFSET
    from drone_grasp_rl.tasks.grasp.mdp import frame_pos_b, frame_quat_b

    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.reset_drone.params["near_fraction"] = 0.0  # the 25 reference positions only, no curriculum
    env_cfg.actions.gripper_action.latch_close = False  # the close check opens the gripper again
    # the arm check measures the raw goal step
    env_cfg.actions.arm_action.smoothing = 1.0
    env_cfg.actions.arm_action.slow_distance = None
    env_cfg.actions.arm_action.hold_after_grasp = False
    # The drone only floats (no position control, no drag): the checks push it and measure the mocap lag on a drone
    # drifting at constant velocity. scripts/check_hover.py checks the flight controller.
    env_cfg.actions.drone_hover.preset_weights = {"static": 1.0}
    env_cfg.actions.drone_hover.mode = "float"
    env_cfg.disable_randomization()
    env_cfg.episode_length_s = 60.0  # no time-out resets during the checks
    # No success/failure resets either: the grasp check closes on a teleported drone. The judge, which also stops
    # the drone motors, is checked by scripts/scripted_grasp.py.
    for name in ("grasp_success", "grasp_failed", "drone_knocked", "protective_stop"):
        setattr(env_cfg.terminations, name, None)
    env_cfg.curriculum = None  # it ramps reward terms that are turned off here
    # the grasp rewards read the judge, which is off
    for name in (
        "approach",
        "approach_fine",
        "approach_speed",
        "arm_still",
        "drone_tilt",
        "finger_contact",
        "close_near",
        "hold",
        "success",
        "failure",
        "drone_push",
        "drone_slip",
    ):
        setattr(env_cfg.rewards, name, None)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    results = []

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        cleanup.callback(env.close)
        n, dev = env.num_envs, env.device
        robot, drone = env.scene["robot"], env.scene["drone"]
        arm_ids = [robot.joint_names.index(j) for j in ARM_JOINT_NAMES]
        grip_id = robot.joint_names.index(GRIPPER_JOINT_NAME)
        reset_term = env.event_manager.get_term_cfg("reset_drone").func
        arm_term = env.action_manager.get_term("arm_action")
        max_step = arm_term.cfg.max_step

        def tcp_b():
            return frame_pos_b(env, "tcp").clone()

        def drone_b():
            return frame_pos_b(env, "drone").clone()

        def contacts():  # (N, 4) force per finger link [N]: r1, r2, l1, l2
            return torch.stack(
                [env.scene[s].data.force_matrix_w.torch.sum(dim=(1, 2)).norm(dim=-1) for s in FINGER_CONTACT_SENSORS],
                dim=-1,
            )

        def run(action, steps):
            with torch.inference_mode():
                for _ in range(steps):
                    env.step(action)

        def grip_deg():
            return torch.rad2deg(robot.data.joint_pos.torch[:, grip_id])

        open_action = torch.zeros(n, 7, device=dev)
        close_action = open_action.clone()
        close_action[:, 6] = -1.0
        steps_per_s = round(1.0 / env.step_dt)

        env.reset()
        run(open_action, 1)

        # 1. reset positions
        err = drone_b() - reset_term.targets[reset_term.target_index]
        ok = bool((err.abs().max() <= DRONE_TARGET_OFFSET + 1e-3).item())
        results.append(
            (
                "targets",
                ok,
                f"max |offset|={1000 * err.abs().max():.1f} mm (limit {1000 * DRONE_TARGET_OFFSET:.0f}), "
                f"{len(set(reset_term.target_index.tolist()))} of {len(reset_term.targets)} targets used by {n} envs",
            )
        )
        # 2. nothing touches the drone
        c = contacts()
        gap = (drone_b() - tcp_b())[:, 2]
        results.append(
            (
                "clear",
                bool((c.max() < 1e-3).item()),
                f"max finger force={c.max():.3f} N, drone above TCP {100 * gap.min():.1f}..{100 * gap.max():.1f} cm",
            )
        )
        # 3. hold with zero actions
        tcp0, drone0, q0 = tcp_b(), drone_b(), robot.data.joint_pos.torch[:, arm_ids].clone()
        run(open_action, 2 * steps_per_s)
        tcp_d = 1000 * (tcp_b() - tcp0).norm(dim=-1).max()
        drone_d = 1000 * (drone_b() - drone0).norm(dim=-1).max()
        q_err = torch.rad2deg((robot.data.joint_pos.torch[:, arm_ids] - q0).abs().max())
        results.append(
            (
                "hold",
                bool(tcp_d < 1.0 and drone_d < 1.0 and q_err < 0.2),
                f"TCP drift={tcp_d:.2f} mm, drone drift={drone_d:.2f} mm, arm={q_err:.3f} deg",
            )
        )
        # 3b. mocap model of the policy's drone pose
        names = env.observation_manager.active_terms["policy"]
        dims = env.observation_manager.group_obs_term_dim["policy"]
        off = sum(d[0] for d in dims[: names.index("drone_pose")])

        def mocap_pos_err():  # (N, 3) observed - true drone position [m]
            return env.obs_buf["policy"][:, off : off + 3] - drone_b()

        rest = []
        for _ in range(steps_per_s):
            run(open_action, 1)
            rest.append(mocap_pos_err())
        rest = torch.stack(rest)  # (T, N, 3)
        noise_mm = 1000 * rest.std(dim=0).mean()
        bias_mm = 1000 * rest.mean(dim=0).abs().max()
        rot_err = torch.rad2deg(
            math_utils.quat_error_magnitude(env.obs_buf["policy"][:, off + 3 : off + 7], frame_quat_b(env, "drone"))
        ).max()
        drone.write_root_velocity_to_sim_index(
            root_velocity=torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1)
        )
        moving = []
        for _ in range(10):
            run(open_action, 1)
            moving.append(mocap_pos_err())
        drone.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
        lag_mm = -1000 * (torch.stack(moving).mean(dim=0) - rest.mean(dim=0))[:, 0].mean()
        ok = bool(0.7 < noise_mm < 1.3 and bias_mm < 2.5 and rot_err < 1.5 and 2.0 < lag_mm < 3.0)
        results.append(
            (
                "mocap",
                ok,
                f"noise={noise_mm:.2f} mm/axis, max bias={bias_mm:.2f} mm, max rot err={rot_err:.2f} deg, "
                f"lag at 0.1 m/s={lag_mm:.2f} mm (expect 2.5)",
            )
        )
        # 4. arm action contract: full action (1.0) on every arm joint for 2 steps moves the goal by 2 * max_step
        #    (0.1 rad, small enough to stay clear of the table and the drone) at the maximum joint speed; the joints
        #    must follow the interpolated target closely. Then back.
        act = open_action.clone()
        act[:, :6] = 1.0
        start = arm_term.processed_actions.clone()
        lag = []
        for _ in range(2):
            run(act, 1)
            lag.append((robot.data.joint_pos.torch[:, arm_ids] - arm_term.target).abs().max())
        goal_err = torch.rad2deg((arm_term.processed_actions - start - 2 * max_step).abs().max())
        run(open_action, steps_per_s // 2)
        a_err = torch.rad2deg((robot.data.joint_pos.torch[:, arm_ids] - arm_term.processed_actions).abs().max())
        lag_deg = torch.rad2deg(torch.stack(lag).max())
        run(-act, 2)
        results.append(
            (
                "arm",
                bool(goal_err < 1e-3 and a_err < 0.2 and lag_deg < 4.0 and grip_deg().abs().max() < 1.0),
                f"goal step error={goal_err:.4f} deg, settled error={a_err:.3f} deg, "
                f"max lag while moving={lag_deg:.2f} deg (protective stop at 5), "
                f"gripper={grip_deg().abs().max():.2f} deg",
            )
        )
        run(open_action, steps_per_s)
        # 5. empty close / open
        run(close_action, 3 * steps_per_s)
        closed = grip_deg().min()
        run(open_action, 3 * steps_per_s)
        reopened = grip_deg().abs().max()
        results.append(
            (
                "close",
                bool(closed > 62.0 and reopened < 2.0),
                f"closed min={closed:.2f} deg, reopened max={reopened:.2f} deg",
            )
        )
        # 6. grasp: drone waist at the TCP, at rest, same orientation as the base
        base_pos, base_quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        tcp_w = base_pos + math_utils.quat_apply(base_quat, tcp_b())
        drone.write_root_pose_to_sim_index(root_pose=torch.cat([tcp_w, base_quat], dim=-1))
        drone.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
        run(open_action, steps_per_s // 2)
        before = contacts().max()
        run(close_action, 3 * steps_per_s)
        c = contacts()
        right, left = c[:, :2].sum(dim=-1), c[:, 2:].sum(dim=-1)
        ang = grip_deg()
        shift = 1000 * (drone_b() - tcp_b())  # drone body origin relative to the TCP [mm]
        ok = bool(
            (right > 1.0).all() and (left > 1.0).all() and (ang < 62.0).all() and (shift.norm(dim=-1) < 10.0).all()
        )
        results.append(
            (
                "grasp",
                ok,
                f"contact before close={before:.2f} N; closed at {ang.min():.1f}..{ang.max():.1f} deg, "
                f"right {right.min():.1f}..{right.max():.1f} N, left {left.min():.1f}..{left.max():.1f} N, "
                f"drone moved (x, y, z) {shift[0, 0]:.1f}, {shift[0, 1]:.1f}, {shift[0, 2]:.1f} mm",
            )
        )

        for name, ok, msg in results:
            print(f"[check] {'PASS' if ok else 'FAIL'}  {name:8s}{msg}")
        print(f"[check] {sum(ok for _, ok, _ in results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
