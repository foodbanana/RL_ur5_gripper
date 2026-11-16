# drone_grasp_rl

Reinforcement-learning tasks for drone grasping, built as an installable
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0 task package (Isaac Sim 6.2, uv `src` layout).

> **Status:** only the template task is registered so far. Drone grasping tasks will be added under
> `src/drone_grasp_rl/tasks`.

Registered tasks:
- `DroneGraspRl-Balance-Cartpole`

## Requirements

- Ubuntu 24.04 (recommended) or 22.04, x86_64
- NVIDIA GPU (Turing or newer) with driver **580.65.06 or newer**
- **50 GB or more** of free disk space (Isaac Sim + PyTorch use about 25 GB, plus caches)
- [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 for you, so you don't need a system Python.

See [INSTALL.md](INSTALL.md) for the tested versions, the details behind each step, and troubleshooting.

## Quick start (new machine)

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

## Run the tasks

Two physics backends are available. Newton is the default and does not need Isaac Sim. PhysX runs through Isaac Sim.

```bash
TASK=DroneGraspRl-Balance-Cartpole

# Newton backend
uv run isaaclab train --rl_library rsl_rl --task $TASK
uv run isaaclab play  --rl_library rsl_rl --task $TASK --checkpoint latest --num_envs 16 --viz newton_gl

# PhysX backend (Isaac Sim)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task $TASK physics=isaacsim_physx
uv run --extra isaacsim isaaclab play  --rl_library rsl_rl --task $TASK --checkpoint latest --num_envs 16 \
    physics=isaacsim_physx --viz kit

# Run without a trained policy
uv run isaaclab zero_agent   --task $TASK --num_envs 16
uv run isaaclab random_agent --task $TASK --num_envs 16

# Multi-GPU training and benchmarks
uv run isaaclab train_multigpu --rl_library rsl_rl --task $TASK --num_gpus 2
uv run isaaclab benchmark runtime  --task $TASK --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library rsl_rl --task $TASK --max_iterations 10
```

Checkpoints are written to `logs/rsl_rl/<experiment>/<timestamp>/`. `logs/` is git-ignored, so trained policies stay
on the machine that trained them. To share a policy, copy that run folder to the other machine's `logs/rsl_rl/<experiment>/`.

Use `physics=<PRESET>` to select any preset listed by `isaaclab list_envs --show_presets`. To use other optional
features (`rerun`, `wandb`, `skrl`, ...), pass the matching extra from `pyproject.toml`, for example
`uv run --extra isaacsim --extra rerun isaaclab ... --viz rerun`.

## Project structure

```
src/drone_grasp_rl/
├── assets/            # asset configs; data/ holds project-owned USD files (Git LFS)
└── tasks/
    └── balance/       # one task family
        ├── mdp/       # task-wide MDP terms (rewards, observations, ...)
        └── config/
            └── cartpole/   # robot-specific scene, registration, agent configs
scripts/               # helper scripts (Isaac Sim patch, list_envs)
tests/                 # registration test
```

To add another robot, create a sibling of `config/cartpole`. To add another task, create a sibling task-family
directory next to `balance/`.

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| About 40 extensions fail with `charset_normalizer.md.MessDetectorPlugin size changed ...` | Run `bash scripts/fix_isaacsim_fastfinder.sh` |
| `ROS2 Bridge startup failed` in the Isaac Sim console | Harmless when ROS 2 is not loaded; it does not affect training |
| `uv sync --locked` fails | `uv.lock` is out of date with `pyproject.toml`. Pull the latest commit, or ask whoever changed the dependencies to commit `uv.lock` |
| Pylance or basedpyright cannot resolve modules | Select the interpreter that ran `isaaclab --editor`, then reload the window. Edit `extraPaths` in `pyrightconfig.json` to add a missing extension or to reduce indexing memory |

See [INSTALL.md](INSTALL.md#troubleshooting) for more.
