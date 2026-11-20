# LeRobot v2.1 export

Copied from `data_collection_ur5_gripper` (`lerobot_v21_common.py`, `lerobot_merge_episodes_v21.py`,
`inspect_dataset_v21.py`), with one camera (`observation.images.wrist`) instead of two.

Its own uv project (Python 3.10, `lerobot==0.3.3` = `CODEBASE_VERSION` v2.1, locked in `uv.lock`), apart from the Isaac
Lab `.venv`: lerobot pins its own torch and video libraries.

```bash
# from the repository root
uv sync --project tools/lerobot_export --locked
# intermediate folders written by scripts/record_demos.py -> data set
uv run --project tools/lerobot_export python tools/lerobot_export/lerobot_merge_episodes_v21.py \
    --all --intermediate-dir datasets/intermediate/<recording> --repo-id local/<name>
uv run --project tools/lerobot_export python tools/lerobot_export/inspect_dataset_v21.py datasets/lerobot_v21/local/<name> --load
```
