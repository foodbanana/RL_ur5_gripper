"""ArUco marker study: which marker placement on the drone the wrist camera can still detect, by grasp phase.

For each candidate layout, real ArUco markers (DICT_4X4_50, white quiet zone) are put on the drone's underside in the
recording task (``DroneGraspRl-Grasp-UR5-Record``: wrist D435i color camera with the real intrinsics, visual
randomization between rounds). The teacher runs its usual episodes, and every policy step each environment's
640x480 image goes through OpenCV's ArUco detector. Reported per layout and per stage (TCP distance to the grasp
point while approaching, then holding):

- detection rate: at least one marker of the layout found
- why the others were missed: a corner outside the image (out of view) or inside the image but not found
  (occluded by fingers, propellers, too steep, too small or too large)
- position error of the detected marker center from ``solvePnP`` with the known size, against the true pose

The camera at the grasp sits about 57 mm toward the drone's +x and 109 mm below its center, looking up and 13 deg
toward the center (measured with the teacher); the flat underside is roughly x -20..38 mm, y -16..14 mm at
z = -31 mm (curving up to -28 mm over x -44..48, y -27..24). Hence candidates centered and shifted toward +x.

Usage (from the repository root):

    uv run --extra isaacsim python scripts/study_aruco.py \
        --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt --episodes 48
    # --layouts center40,side30 to run some layouts only
"""

import argparse
import contextlib
import math
import os
import random
import sys
from collections import defaultdict

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
DRONE_SCALE = 0.75
"""The drone USD scales /vehicle by 0.75; markers below the body are given 1/0.75 to come out at true size."""
MARKER_Z = -0.0315
"""Marker plane in the drone body frame [m], just below the lowest flat underside (-31 mm), facing down."""
LAYOUTS = {
    # name: [(aruco id, black square side [m], x [m], y [m]) in the drone body frame]
    "center40": [(0, 0.040, 0.0, 0.0)],
    "side30": [(1, 0.030, 0.030, 0.0)],
    "side20": [(2, 0.020, 0.035, 0.0)],
    "board4x20": [
        (3, 0.020, 0.010, -0.012),
        (4, 0.020, 0.010, 0.012),
        (5, 0.020, 0.035, -0.012),
        (6, 0.020, 0.035, 0.012),
    ],
    "board6x15": [
        (7 + 2 * i + j, 0.015, x, y) for i, x in enumerate((0.0, 0.019, 0.038)) for j, y in enumerate((-0.009, 0.009))
    ],
    # a large marker for the far approach and a small one on the camera side for the last centimeters
    "center40+side20": [(0, 0.040, -0.002, 0.0), (2, 0.020, 0.035, 0.0)],
}
STAGES = ["approach >30 cm", "approach 15-30 cm", "approach 5-15 cm", "approach <5 cm", "holding"]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=48, help="Episodes per layout.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--layouts", type=str, default=",".join(LAYOUTS))
parser.add_argument("--hover", type=str, default="static:0.1,small:0.3,mid:0.4,big:0.2")
parser.add_argument("--out", type=str, default="outputs/aruco")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

MODULES = 6  # 4x4 data + black border
QUIET = 1  # white modules around the black square


def write_marker_assets(out_dir: str, marker_id: int, size: float) -> str:
    """PNG texture and a one-quad USD (facing -z, true size in meters) for one marker; returns the USD path."""
    import cv2

    px = 60
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, px)
    pad = px // MODULES * QUIET
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    png = os.path.join(out_dir, f"aruco4x4_{marker_id}.png")
    cv2.imwrite(png, img)
    h = size * (MODULES + 2 * QUIET) / MODULES / 2  # half side including the quiet zone
    usd = os.path.join(out_dir, f"aruco4x4_{marker_id}_{round(1000 * size)}mm.usda")
    # Seen from below (-z) with u to +x and v to +y the texture reads mirrored; the detector handles any rotation,
    # and a mirrored marker would not decode, so the quad's st runs (1 - x) to keep it readable from below.
    with open(usd, "w") as f:
        f.write(f"""#usda 1.0
(
    defaultPrim = "marker"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "marker"
{{
    def Mesh "quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 3, 2, 1]
        point3f[] points = [({-h}, {-h}, 0), ({h}, {-h}, 0), ({h}, {h}, 0), ({-h}, {h}, 0)]
        texCoord2f[] primvars:st = [(1, 0), (0, 0), (0, 1), (1, 1)] (
            interpolation = "vertex"
        )
        uniform bool doubleSided = 1
        rel material:binding = </marker/Look>
    }}

    def Material "Look"
    {{
        token outputs:surface.connect = </marker/Look/Surface.outputs:surface>

        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </marker/Look/Texture.outputs:rgb>
            float inputs:roughness = 0.7
            token outputs:surface
        }}

        def Shader "Texture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./{os.path.basename(png)}@
            token inputs:sourceColorSpace = "sRGB"
            token inputs:wrapS = "clamp"
            token inputs:wrapT = "clamp"
            float2 inputs:st.connect = </marker/Look/Reader.outputs:result>
            float3 outputs:rgb
        }}

        def Shader "Reader"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
    }}
}}
""")
    return usd


def stage_of(dist: float, holding: bool) -> str | None:
    if holding:
        return STAGES[4]
    for name, lo in zip(STAGES[:4], (0.30, 0.15, 0.05, -1.0), strict=True):
        if dist >= lo:
            return name
    return None


def run_layout(name, markers, env_cfg, agent_cfg, out_dir):
    import cv2

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import AssetBaseCfg

    from drone_grasp_rl.tasks.grasp.config.ur5.camera import WRIST_CAMERA_INTRINSICS
    from drone_grasp_rl.tasks.grasp.config.ur5.visuals import randomize_visuals
    from drone_grasp_rl.tasks.grasp.mdp import tcp_in_grasp_frame
    from drone_grasp_rl.tasks.grasp.mdp.judge import APPROACH, GRASPED, KILLED, get_judge

    for marker_id, size, x, y in markers:
        usd = write_marker_assets(out_dir, marker_id, size)
        s = 1.0 / DRONE_SCALE
        setattr(
            env_cfg.scene,
            f"aruco_{marker_id}",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Drone/body/aruco_" + str(marker_id),
                spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(usd), scale=(s, s, s)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x * s, y * s, MARKER_Z * s)),
            ),
        )

    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(
            args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        n, dev = base.num_envs, base.device
        cam, judge = base.scene["wrist_cam"], get_judge(base)
        k = WRIST_CAMERA_INTRINSICS
        cam_mat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=np.float64)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
        )
        ids = [m[0] for m in markers]
        sizes = {m[0]: m[1] for m in markers}
        local = torch.tensor([[m[2], m[3], MARKER_Z] for m in markers], device=dev)  # true size frame (body)
        half = torch.tensor([[[-1, -1], [1, -1], [1, 1], [-1, 1]]], device=dev) * 0.5
        stats = defaultdict(lambda: defaultdict(float))  # stage -> counters
        errors = defaultdict(list)
        by_size = defaultdict(list)  # (size mm, stage) -> (position error mm, normal error deg, distance m)
        saved = 0
        rng = random.Random(args_cli.seed)
        episodes = 0
        with torch.inference_mode():
            while episodes < args_cli.episodes:
                randomize_visuals(base, rng)
                obs, _ = env.reset()
                active = torch.ones(n, dtype=torch.bool, device=dev)
                first = True
                while bool(active.any()):
                    obs, _, dones, _ = env.step(policy(obs))
                    done = dones.bool()
                    if first:  # the image of the first step after a reset shows the old scene
                        first = False
                        active &= ~done
                        continue
                    rgb = cam.data.output["rgb"]
                    rgb = getattr(rgb, "torch", rgb)[..., :3].cpu().numpy()
                    pos_c, quat_c = cam.data.pos_w, cam.data.quat_w_ros
                    pos_c, quat_c = getattr(pos_c, "torch", pos_c), getattr(quat_c, "torch", quat_c)
                    r_c = math_utils.matrix_from_quat(quat_c)  # columns: optical axes in the world
                    drone = base.scene["drone"].data
                    r_d = math_utils.matrix_from_quat(drone.root_quat_w.torch)
                    # marker centers and corners in the camera frame (true size, in meters)
                    centers_w = drone.root_pos_w.torch[:, None] + torch.einsum("nij,mj->nmi", r_d, local)
                    centers_c = torch.einsum("nji,nmj->nmi", r_c, centers_w - pos_c[:, None])
                    side = torch.tensor([m[1] for m in markers], device=dev)[:, None, None]
                    corner_body = local[:, None, :2] + half * side
                    corner_body = torch.cat([corner_body, torch.full_like(corner_body[..., :1], MARKER_Z)], -1)
                    corners_w = drone.root_pos_w.torch[:, None, None] + torch.einsum("nij,mcj->nmci", r_d, corner_body)
                    corners_c = torch.einsum("nji,nmcj->nmci", r_c, corners_w - pos_c[:, None, None])
                    u = k["fx"] * corners_c[..., 0] / corners_c[..., 2] + k["cx"]
                    v = k["fy"] * corners_c[..., 1] / corners_c[..., 2] + k["cy"]
                    in_view = ((corners_c[..., 2] > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)).all(-1)
                    dist = tcp_in_grasp_frame(base).norm(dim=-1)
                    phase = judge.phase
                    for i in range(n):
                        if not bool(active[i]) or bool(done[i]):
                            continue
                        holding = int(phase[i]) in (GRASPED, KILLED)
                        if int(phase[i]) != APPROACH and not holding:
                            continue
                        st = stage_of(float(dist[i]), holding)
                        corners, found, _ = detector.detectMarkers(cv2.cvtColor(rgb[i], cv2.COLOR_RGB2GRAY))
                        found_ids = set() if found is None else {int(f) for f in found.flatten()}
                        hit = [m for m in ids if m in found_ids]
                        s = stats[st]
                        s["steps"] += 1
                        s["detected"] += bool(hit)
                        s["any_in_view"] += bool(in_view[i].any())
                        for j, m in enumerate(ids):
                            s["marker_steps"] += 1
                            if m in found_ids:
                                s["marker_found"] += 1
                            elif not bool(in_view[i, j]):
                                s["missed_out_of_view"] += 1
                            else:
                                s["missed_in_view"] += 1
                        for c, f in zip(corners or [], [] if found is None else found.flatten(), strict=False):
                            f = int(f)
                            if f not in ids:
                                continue
                            h = sizes[f] / 2
                            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
                            ok, rvec, tvec = cv2.solvePnP(
                                obj, c.reshape(4, 2).astype(np.float64), cam_mat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
                            )
                            if ok:
                                true = centers_c[i, ids.index(f)].cpu().numpy()
                                # marker normal (out of the face, toward the camera) vs the drone's -z axis
                                normal = cv2.Rodrigues(rvec)[0][:, 2]
                                true_n = -(r_c[i].T @ r_d[i][:, 2]).cpu().numpy()
                                tilt = math.degrees(math.acos(float(np.clip(abs(normal @ true_n), -1.0, 1.0))))
                                depth = float(np.linalg.norm(true))
                                errors[st].append(1000 * float(np.linalg.norm(tvec.flatten() - true)))
                                by_size[(round(1000 * sizes[f]), st)].append(
                                    (1000 * float(np.linalg.norm(tvec.flatten() - true)), tilt, depth)
                                )
                        if saved < 6 and st == STAGES[3] and i == 0:
                            from PIL import Image

                            Image.fromarray(rgb[i]).save(os.path.join(out_dir, f"{name}_close_{saved}.png"))
                            saved += 1
                    newly = active & done
                    episodes += int(newly.sum())
                    active &= ~done
        return stats, errors, by_size
    finally:
        env.close()


def main():
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.wrist_cam.update_latest_camera_pose = True
    env_cfg.actions.drone_hover.preset_weights = {
        k: float(v) for k, v in (item.split(":") for item in args_cli.hover.split(","))
    }
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    os.makedirs(args_cli.out, exist_ok=True)
    names = args_cli.layouts.split(",")
    if len(names) != 1:
        raise SystemExit("[aruco] one layout per run (the scene is built once per process); use --layouts <name>")
    name = names[0]
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack():
        stats, errors, by_size = run_layout(name, LAYOUTS[name], env_cfg, agent_cfg, args_cli.out)
        sizes = "+".join(f"{round(1000 * m[1])}mm@({round(1000 * m[2])},{round(1000 * m[3])})" for m in LAYOUTS[name])
        print(f"[aruco] layout {name}: {sizes} (drone body frame, mm), {args_cli.episodes} episodes")
        for st in STAGES:
            s = stats.get(st)
            if not s or not s["steps"]:
                continue
            ms = max(s["marker_steps"], 1)
            e = errors.get(st, [])
            err = f"{np.median(e):.1f} / {np.quantile(e, 0.9):.1f}" if e else "-"
            print(
                f"[aruco]   {st:18s} steps {int(s['steps']):6d}: detected {100 * s['detected'] / s['steps']:5.1f}%"
                f" | per marker found {100 * s['marker_found'] / ms:5.1f}%, missed out of view "
                f"{100 * s['missed_out_of_view'] / ms:5.1f}%, missed in view {100 * s['missed_in_view'] / ms:5.1f}%"
                f" | position error median / 90th [mm] {err}"
            )
        for (size, st), rows in sorted(by_size.items()):
            a = np.array(rows)
            print(
                f"[aruco]   {size} mm marker, {st:18s} n {len(a):5d}: distance median {np.median(a[:, 2]):.3f} m,"
                f" position error median {np.median(a[:, 0]):.1f} mm, normal error median {np.median(a[:, 1]):.1f} deg"
                f" (90th {np.quantile(a[:, 1], 0.9):.1f})"
            )


if __name__ == "__main__":
    main()
