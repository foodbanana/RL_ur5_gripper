# drone_grasp_rl

Reinforcement-learning tasks for drone grasping, built as an installable
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0 task package (Isaac Sim 6.2, uv `src` layout).

> **Status:** the UR5 drone grasp task trains on PhysX. The current policy is a privileged teacher for VLA data:
> on the flying drone (mid hover) with domain randomization it grasps and holds it upright (gripper tilt <= 10 deg)
> in 93-96 % of episodes from the 25 reference starts, slows to about 14 cm/s before the grasp and holds the arm
> still afterwards.

Registered tasks:
- `DroneGraspRl-Grasp-UR5`: UR5 + ROBOTIS RH-P12-RN(A) grasps a hovering Iris drone from below
- `DroneGraspRl-Balance-Cartpole`: template example

<br>

## Requirements

- Ubuntu 24.04 (recommended) or 22.04, x86_64
- NVIDIA GPU (Turing or newer) with driver **580.65.06 or newer**
- **50 GB or more** of free disk space (Isaac Sim + PyTorch use about 25 GB, plus caches)
- [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 for you, so you don't need a system Python.

See [INSTALL.md](INSTALL.md) for the tested versions, the details behind each step, and troubleshooting.

<br>

## How to install?

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"

# 2. Clone
git clone https://github.com/foodbanana/RL_ur5_gripper.git drone_grasp_rl
cd drone_grasp_rl

# 3. Install the exact locked environment into ./.venv
uv sync --locked --extra isaacsim

# 4. Apply the Isaac Sim 6.2 patch (required)
bash scripts/fix_isaacsim_fastfinder.sh

# 5. Check that it works
uv run isaaclab list_envs --show_presets
uv run isaaclab train --rl_library rsl_rl --task DroneGraspRl-Balance-Cartpole
```

Training is working when the last log shows `Mean episode length: 300.00`.

> [!IMPORTANT]
> - Use a terminal where **ROS 2 is not sourced**. Otherwise `/opt/ros/.../setup.bash` mixes its Python paths into `.venv`.
> - **Always pass `--extra isaacsim` to `uv sync`.** Without it, uv removes Isaac Sim from the environment.
> - **Re-run `scripts/fix_isaacsim_fastfinder.sh`** whenever `.venv` is recreated or `uv sync` reinstalls Isaac Sim.

<br>

## Run the tasks

Two physics backends are available. For `DroneGraspRl-Grasp-UR5` **PhysX is the default** (the robot gains come
from a PhysX-tuned setup), so its commands need `--extra isaacsim`. Newton MJWarp runs without Isaac Sim when you
pass `physics=newton_mjwarp`. (The template `DroneGraspRl-Balance-Cartpole` defaults to Newton.)

```bash
TASK=DroneGraspRl-Grasp-UR5

# PhysX backend (default, Isaac Sim)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task $TASK
uv run --extra isaacsim isaaclab play  --rl_library rsl_rl --task $TASK --checkpoint latest --num_envs 16 --viz kit

# Newton backend
uv run isaaclab train --rl_library rsl_rl --task $TASK physics=newton_mjwarp
uv run isaaclab play  --rl_library rsl_rl --task $TASK --checkpoint latest --num_envs 16 \
    physics=newton_mjwarp --viz newton_gl

# Run without a trained policy
uv run --extra isaacsim isaaclab zero_agent   --task $TASK --num_envs 16 --viz kit
uv run                  isaaclab random_agent --task $TASK --num_envs 16 --viz newton_gl physics=newton_mjwarp

# Multi-GPU training and benchmarks
uv run --extra isaacsim isaaclab train_multigpu --rl_library rsl_rl --task $TASK --num_gpus 2
uv run --extra isaacsim isaaclab benchmark runtime  --task $TASK --num_envs 16 --num_steps 1000
uv run --extra isaacsim isaaclab benchmark training --rl_library rsl_rl --task $TASK --max_iterations 10
```

Checkpoints are written to `logs/rsl_rl/<experiment>/<timestamp>/`. `logs/` is git-ignored, so trained policies stay
on the machine that trained them. To share a policy, copy that run folder to the other machine's `logs/rsl_rl/<experiment>/`.

### Record VLA demonstrations

The trained teacher generates demonstrations in the LeRobot v2.1 schema of the data collection repo, with the wrist
camera only (`observation.images.wrist`, 640x480, 25 fps). Successful episodes only; lights, ceiling and table colors
change between recording rounds. See [docs/plan_vla_recording.md](docs/plan_vla_recording.md) (Korean).

```bash
uv run --extra isaacsim python scripts/record_demos.py \
    --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt --num_episodes 300 --num_envs 16 --name <name>
uv sync --project tools/lerobot_export --locked   # lerobot 0.3.3 in its own environment
uv run --project tools/lerobot_export python tools/lerobot_export/lerobot_merge_episodes_v21.py \
    --all --intermediate-dir datasets/intermediate/<name> --repo-id local/<name>
uv run --project tools/lerobot_export python tools/lerobot_export/inspect_dataset_v21.py \
    datasets/lerobot_v21/local/<name> --load
```

Add `--depth` to `record_demos.py` and `--with-depth` to the merge to keep a real-like D435i depth (16-bit mm,
0 = no reading) next to the data set in `depth/chunk-*/episode_*.npz`; lerobot 0.3.3 cannot store 16-bit
single-channel images, and without the flag the schema stays RGB only. `datasets/` is git-ignored.

Use `physics=<PRESET>` to select any preset listed by `isaaclab list_envs --show_presets`. To use other optional
features (`rerun`, `wandb`, `skrl`, ...), pass the matching extra from `pyproject.toml`, for example
`uv run --extra isaacsim --extra rerun isaaclab ... --viz rerun`.

<br>

## Project structure

The RL design (observations, actions, rewards, judge, training history) is documented in Korean in
[docs/rl_design.md](docs/rl_design.md).

```
src/drone_grasp_rl/
├── physics.py         # physics backends (PhysX default, Newton MJWarp) and the 120 Hz step
├── assets/            # robot and drone configs; data/ holds project-owned URDF, meshes and USD (Git LFS)
└── tasks/
    ├── grasp/         # drone grasp task family
    │   ├── grasp_env_cfg.py   # scene, observations, events, rewards shared by all robots
    │   ├── mdp/               # task terms (drone reset and hold, profiled gripper action, ...)
    │   └── config/ur5/        # UR5 scene, sensors, actions, registration, agent configs; wrist camera and
    │                          #   visual randomization of the recording task
    └── balance/       # template example (cart-pole)
scripts/
├── convert_assets.py        # URDF -> robot USD (rerun after changing the URDF or meshes)
├── build_drone_collision.py # drone body collider -> fixed convex pieces (iris_collision.usda)
├── check_assets.py          # robot and drone checks: TCP, drives, gripper, mimic, hover
├── check_task.py            # task checks: reset positions, mocap model, action contract, contact sensors, grasp
├── scripted_grasp.py        # IK expert runs the full episode protocol and reports the judge's outcomes
├── eval_policy.py           # trained checkpoint: judge outcomes and approach geometry on the 25 reference starts
├── check_hover.py           # drone flight controller: hover tracking per preset, recovery after a push
├── check_wrist_camera.py    # recording task: wrist camera images with the projected TCP and drone
├── record_demos.py          # teacher rollouts -> intermediate episode folders (wrist images, state, action)
└── fix_isaacsim_fastfinder.sh, list_envs.py
tools/lerobot_export/  # intermediate folders -> LeRobot v2.1 (own uv project, lerobot 0.3.3)
tests/                 # registration test
```

Check the assets and the task on both backends after changing an asset, a gain, or the scene:

```bash
uv run --extra isaacsim python scripts/check_assets.py --backend isaacsim_physx
uv run                  python scripts/check_assets.py --backend newton_mjwarp
uv run --extra isaacsim python scripts/check_task.py physics=isaacsim_physx
uv run                  python scripts/check_task.py physics=newton_mjwarp
uv run --extra isaacsim python scripts/scripted_grasp.py physics=isaacsim_physx
```

To add another robot, create a sibling of `config/ur5`. To add another task, create a sibling task-family
directory next to `grasp/`.

<br>

## Project assets

Keep project-owned USD files and related data under `src/drone_grasp_rl/assets/data`. The asset module exposes a
path that works from both an editable checkout and an installed wheel:

```python
from drone_grasp_rl.assets import DRONE_GRASP_RL_ASSETS_DIR

robot_usd_path = DRONE_GRASP_RL_ASSETS_DIR / "robots" / "my_robot.usd"
```

Pass `str(robot_usd_path)` to configuration fields that expect a string. USD and checkpoint files (`*.usd*`, `*.pt`,
`*.jit`, `*.hdf5`) are stored with Git LFS. **Install Git LFS before you add or clone such files:**

```bash
sudo apt install git-lfs && git lfs install
```

<br>

## Development

```bash
uv run pytest                          # registration test
uv run pre-commit run --all-files      # ruff, codespell, ...
uv add <package>                       # add a dependency
```

`uv add` updates `pyproject.toml` and `uv.lock`. **Commit both files** so everyone else gets the same environment
from `uv sync --locked --extra isaacsim`.

To configure the editor (VS Code with Pylance, or Cursor with basedpyright), run the following command and then select
the `.venv` interpreter. You can also run the `setup_python_env` task, but that task leaves out the `isaacsim` extra.

```bash
uv run --extra isaacsim isaaclab --editor
```

This command writes a git-ignored `pyrightconfig.json` that includes the Isaac Sim extensions and the project `src` root.
For an Isaac Sim binary install outside the project environment, add `--isaac_path <isaac-sim-path>`.

<br>

## Troubleshooting

| Symptom | Fix |
|---|---|
| About 40 extensions fail with `charset_normalizer.md.MessDetectorPlugin size changed ...` | Run `bash scripts/fix_isaacsim_fastfinder.sh` |
| `ROS2 Bridge startup failed` in the Isaac Sim console | Harmless when ROS 2 is not loaded; it does not affect training |
| `uv sync --locked` fails | `uv.lock` is out of date with `pyproject.toml`. Pull the latest commit, or ask whoever changed the dependencies to commit `uv.lock` |
| Pylance or basedpyright cannot resolve modules | Select the interpreter that ran `isaaclab --editor`, then reload the window. Edit `extraPaths` in `pyrightconfig.json` to add a missing extension or to reduce indexing memory |

See [INSTALL.md](INSTALL.md#troubleshooting) for more.
