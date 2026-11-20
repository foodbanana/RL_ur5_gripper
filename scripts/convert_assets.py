"""Convert the source robot description in ``assets/data/source`` to USD.

The generated USD is committed (Git LFS), so training never converts at run time. Re-run this
script only when the source URDF or the options below change. It produces the same output on any
machine because it reads only files inside this repository.

Options mirror the Isaac Sim 6.1 import used by data_collection_ur5_gripper
(no fixed-joint merging, no collision from visuals, no self collision, no mesh merging,
base not fixed at import time; the scene decides how the base is held).

After the conversion the script stiffens the gripper's mimic coupling for Newton (see ``stiffen_gripper_mimic``);
``--fixups_only`` applies only that to the committed USD, without converting (no simulator needed).

Usage (from the repository root):

    uv run --extra isaacsim python scripts/convert_assets.py
    uv run python scripts/convert_assets.py --fixups_only
"""

import argparse
import re
import shutil
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "drone_grasp_rl" / "assets" / "data"
ROBOT_URDF = DATA_DIR / "source" / "ur5_rh_p12_d435i" / "urdf" / "ur5_rh_p12_d435i.urdf"
ROBOT_OUT = DATA_DIR / "robots" / "ur5_rh_p12_d435i"

parser = argparse.ArgumentParser(description="Convert the project's source assets to USD.")
parser.add_argument("--fixups_only", action="store_true", help="Only re-apply the post-conversion fixups.")
add_launcher_args(parser)
args_cli = parser.parse_args()
# The URDF importer runs inside Kit unless the standalone importers extra is installed.
args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def make_paths_relative(out_dir: Path) -> None:
    """Replace the absolute repository path the converter writes into text outputs.

    ``config.yaml`` records the input/output paths and USD layers carry a ``doc`` string with the
    composed stage path. Neither is needed at run time, but absolute paths break the rule that the
    repository holds no machine-specific paths.
    """
    root = str(REPO_ROOT) + "/"
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix in (".yaml", ".usda"):
            text = path.read_text()
            # Also drop the random suffix of the converter's temp folder so re-runs are byte-identical.
            new_text = re.sub(r"\.asset-transformer-[A-Za-z0-9_]+", ".asset-transformer", text.replace(root, ""))
            if new_text != text:
                path.write_text(new_text)


GRIPPER_FOLLOWERS = ("rh_r2", "rh_l1", "rh_l2")
MIMIC_SOLREF = (0.005, 1.0)
"""MuJoCo equality time constant [s] and damping ratio of the gripper mimic coupling (default 0.02, 1)."""
MIMIC_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
"""MuJoCo equality impedance (default 0.9, 0.95, ...): 0.99 lets the coupling carry the grasp force."""


def stiffen_gripper_mimic(out_dir: Path) -> None:
    """Author a stiffer MuJoCo equality on the gripper's mimic followers in the ``mujoco`` physics variant.

    Newton turns each ``NewtonMimicAPI`` follower into a MuJoCo joint equality with the default softness. Only the
    leader ``rh_r1_joint`` is driven, so the grasp force reaches the other three fingers through that coupling, and
    with the default the fingers drifted apart while squeezing the drone (joints 42 / 32 / 81 / 72 deg instead of
    16 deg each as on PhysX); the leader then read as nearly closed and the judge never saw a grasp.
    ``MjcEqualityJointAPI`` keeps the same mimic (``newton:mimicJoint``) and adds ``mjc:solref`` / ``mjc:solimp``.
    PhysX uses the ``physx`` variant and is unaffected.
    """
    layer = out_dir / "ur5_rh_p12_d435i" / "payloads" / "Physics" / "mujoco.usda"
    text = layer.read_text()
    solref = ", ".join(f"{v:g}" for v in MIMIC_SOLREF)
    solimp = ", ".join(f"{v:g}" for v in MIMIC_SOLIMP)
    for joint in GRIPPER_FOLLOWERS:
        # the block the converter writes for a follower joint (re-runs: a block already fixed is rewritten)
        block = re.compile(
            rf'( *)over "{joint}" \(\n( *)delete apiSchemas = (\[[^\]\n]*\])\n *prepend apiSchemas = \[[^\]\n]*\]\n'
            rf" *\)\n *\{{\n(?: *double\[\] mjc:sol(?:ref|imp) = \[[^\]\n]*\]\n)* *\}}"
        )
        match = block.search(text)
        if match is None:
            raise RuntimeError(f"{layer.name}: no over block for {joint}")
        ind, inner = match.group(1), match.group(2)
        new = (
            f'{ind}over "{joint}" (\n'
            f"{inner}delete apiSchemas = {match.group(3)}\n"
            f'{inner}prepend apiSchemas = ["MjcJointAPI", "MjcEqualityJointAPI"]\n'
            f"{ind})\n{ind}{{\n"
            f"{inner}double[] mjc:solref = [{solref}]\n"
            f"{inner}double[] mjc:solimp = [{solimp}]\n"
            f"{ind}}}"
        )
        text = text[: match.start()] + new + text[match.end() :]
    layer.write_text(text)
    print(f"[convert_assets] stiffer gripper mimic (solref {MIMIC_SOLREF}, solimp {MIMIC_SOLIMP}) in {layer.name}")


def main():
    if args_cli.fixups_only:
        stiffen_gripper_mimic(ROBOT_OUT)
        return
    if ROBOT_OUT.exists():
        # The converter writes "<name>_1" next to an existing output instead of replacing it.
        shutil.rmtree(ROBOT_OUT)

    robot_cfg = UrdfConverterCfg(
        asset_path=str(ROBOT_URDF),
        usd_dir=str(ROBOT_OUT),
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=False,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        self_collision=False,
        merge_mesh=False,
        robot_type="Manipulator",
        # Gains are set by the actuator groups in robots.py; the USD only needs position drives.
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="position"),
    )

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
        converter = UrdfConverter(robot_cfg)
        # Post-process inside the block: closing the Kit app on exit also ends the process.
        make_paths_relative(ROBOT_OUT)
        stiffen_gripper_mimic(ROBOT_OUT)
        print(f"[convert_assets] robot USD: {Path(converter.usd_path).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
