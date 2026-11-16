# Installation Guide (Isaac Lab 3.0.0 + Isaac Sim 6.2.0, uv)

How to reproduce this repository's RL environment on a fresh Ubuntu 24.04 machine.
All package versions are pinned in `pyproject.toml` + `uv.lock`, and the environment is installed only into the
project's `.venv` folder (no effect on the system Python or other projects).

## Tested configuration

| Item | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (x86_64) |
| GPU / driver | RTX 4070 Ti SUPER 16 GB / NVIDIA 595.x (CUDA 13.2) |
| uv | 0.12.17 |
| Python | 3.12 (downloaded and managed by uv) |
| Isaac Lab | 3.0.0 (wheel from NVIDIA's package index) |
| Isaac Sim | 6.2.0.0 (pip, `isaacsim` extra) |
| PyTorch | 2.11.0+cu130 |
| RL library | rsl-rl-lib 5.5.1 |

## 0. Prerequisites

- Ubuntu 24.04(recommended) or 22.04 , x86_64
- NVIDIA driver **580.65.06 or newer**, Turing or newer GPU (compute capability 7.5+)
- Disk: about 25 GB for the install (Isaac Sim ~17 GB + PyTorch, etc.) plus first-launch caches.
  **At least 50 GB free is recommended.**
- Internet access (pypi.nvidia.com, download.pytorch.org, pypi.org)

```bash
nvidia-smi                 # check the driver version
ldd --version | head -1    # GLIBC must be 2.35 or newer (24.04 ships 2.39)
df -h ~                    # check free disk space
```

## 1. Install basic tools

```bash
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin
source "$HOME/.local/bin/env"                       # or open a new terminal
uv --version
```

## 2. Clone the repository

```bash
mkdir -p ~/isaaclab_ws && cd ~/isaaclab_ws
git clone https://github.com/foodbanana/RL_ur5_gripper.git drone_grasp_rl
cd drone_grasp_rl
```

## 3. Install the environment (exactly as locked)

> Use a **new terminal where ROS 2 has not been sourced.**
> (`/opt/ros/jazzy/setup.bash` adds Python 3.12 package paths that can mix with `.venv`.)

```bash
uv sync --locked --extra isaacsim
```

- `--locked`: stops if the install would differ from `uv.lock`, so only the exact locked versions are installed.
- `--extra isaacsim`: includes Isaac Sim. **Always add it when running `uv sync`** (without it, Isaac Sim is removed).
- Downloading Isaac Sim and PyTorch takes a while.

## 4. Run the Isaac Sim patch script (required)

```bash
bash scripts/fix_isaacsim_fastfinder.sh
```

This works around an Isaac Sim 6.2 issue: Kit's FastFinder uses whichever of a same-named `.py` file and compiled
`.so` module it finds first. Depending on each machine's directory order, the issue may or may not appear; when it
does, about 40 extensions fail to load (`charset_normalizer.md.MessDetectorPlugin size changed ...`). The script only
renames those `.py` files so Kit loads the same file standard Python would, so it is safe to run on any machine.
To undo: `bash scripts/fix_isaacsim_fastfinder.sh --revert`

**Run the script again after recreating `.venv` or whenever `uv sync` reinstalls Isaac Sim packages.**

## 5. Verify the installation

```bash
# PyTorch + GPU
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.11.0+cu130 True

# Isaac Sim GUI (first launch can take 10+ minutes; EULA accepted automatically)
OMNI_KIT_ACCEPT_EULA=yes uv run --extra isaacsim isaacsim
#   Normal if the only console [Error] is "ROS2 Bridge startup failed" (ROS 2 environment not loaded)

# Registered tasks
uv run isaaclab list_envs --show_presets

# Newton backend: train -> play
uv run isaaclab train --rl_library rsl_rl --task DroneGraspRl-Balance-Cartpole
uv run isaaclab play  --rl_library rsl_rl --task DroneGraspRl-Balance-Cartpole --checkpoint latest --num_envs 16 --viz newton_gl

# PhysX backend (Isaac Sim): train -> play
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task DroneGraspRl-Balance-Cartpole physics=isaacsim_physx
uv run --extra isaacsim isaaclab play  --rl_library rsl_rl --task DroneGraspRl-Balance-Cartpole --checkpoint latest --num_envs 16 physics=isaacsim_physx --viz kit
```

Success: the last training log shows `Mean episode length: 300.00`, and the poles stay upright in the playback window.

## Common commands

```bash
uv sync --extra isaacsim                     # match the environment to uv.lock
uv add <package>                             # add a package (updates pyproject.toml + uv.lock)
uv run python my_script.py                   # run with the .venv Python (no activation needed)
rm -rf .venv && uv sync --locked --extra isaacsim && bash scripts/fix_isaacsim_fastfinder.sh   # recreate the environment
```

## Troubleshooting

| Symptom | Cause / action |
|---|---|
| `ROS2 Bridge startup failed` | Appears when running without a ROS 2 environment. Does not affect RL training |
