# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual randomization of the recording scene, applied between recording rounds.

The wrist camera mostly looks up: behind the drone it sees the ceiling, and the lighting sets how the drone's
underside and the fingers look. So the randomization changes, per environment, the ceiling's color, brightness and
height, and the table's color, and globally the dome light (brightness, tint) and a key light (direction, brightness,
tint) that casts shadows. The drone, the robot and the gripper keep their look (they are the real objects).

Lights are global and Isaac Lab's Replicator color terms update all environments at once, while the environments
reset at different times; :func:`randomize_visuals` is therefore called only when all environments reset together
(``scripts/record_demos.py`` records in rounds), so the look never changes within an episode.
"""

import random

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

# pxr is imported inside the functions: this module is imported while the task config is built, before the simulation
# app starts, and a pxr loaded that early clashes with the one Kit loads (symbol lookup errors in the RTX plugins).

CEILING_HEIGHT_RANGE = (2.2, 3.2)
"""[m] Ceiling height above the ground (a lab room)."""
CEILING_SIZE = 3.0
"""[m] Square ceiling tile per environment, as wide as the environment spacing (no gaps between tiles)."""
DOME_INTENSITY_RANGE = (300.0, 1500.0)
KEY_INTENSITY_RANGE = (0.0, 3000.0)
TINT_RANGE = (0.85, 1.0)
"""Each RGB channel of a light's color (warm to cool white)."""


def ceiling_cfg() -> AssetBaseCfg:
    """Per-environment ceiling tile (visual only, no collision), moved and recolored by :func:`randomize_visuals`."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(CEILING_SIZE, CEILING_SIZE, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 2.7)),
    )


def key_light_cfg() -> AssetBaseCfg:
    """Directional light with shadows, turned and dimmed by :func:`randomize_visuals`."""
    return AssetBaseCfg(prim_path="/World/KeyLight", spawn=sim_utils.DistantLightCfg(intensity=1000.0, angle=1.0))


def _set_color(stage, prim_path: str, rgb: tuple[float, float, float]):
    """Diffuse color of an OmniPBR material bound to the prim (created on first use).

    The RTX renderer keeps a UsdPreviewSurface as it was when loaded: changing its ``diffuseColor`` at run time does
    not show in the images (checked). An OmniPBR (MDL) material's ``diffuse_color_constant`` does, which is also how
    Isaac Lab's Replicator color randomization works.
    """
    from pxr import Gf, Sdf, UsdShade

    mat_path = f"{prim_path}/RandomLook"
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
    if not shader_prim:
        import carb.tokens

        mdl = carb.tokens.get_tokens_interface().resolve("${kit}/mdl/core/Base/OmniPBR.mdl")
        sim_utils.spawn_from_mdl_file(mat_path, sim_utils.MdlFileCfg(mdl_path=mdl))
        sim_utils.bind_visual_material(prim_path, mat_path)
        shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
    shader = UsdShade.Shader(shader_prim)
    inp = shader.GetInput("diffuse_color_constant") or shader.CreateInput(
        "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
    )
    inp.Set(Gf.Vec3f(*rgb))


def _set_height(stage, prim_path: str, z: float):
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            op.Set(type(t)(t[0], t[1], z))
            return
    raise RuntimeError(f"no translate op on {prim_path}")


def _tint(rng: random.Random) -> tuple[float, float, float]:
    return tuple(rng.uniform(*TINT_RANGE) for _ in range(3))


def randomize_visuals(env, rng: random.Random) -> dict:
    """Sample a new look for all environments; returns what was sampled (global lights, per-env ceiling and table)."""
    from pxr import Gf, UsdGeom, UsdLux

    stage = sim_utils.get_current_stage()
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    key = UsdLux.DistantLight(stage.GetPrimAtPath("/World/KeyLight"))
    look = {
        "dome_intensity": rng.uniform(*DOME_INTENSITY_RANGE),
        "dome_color": _tint(rng),
        "key_intensity": rng.uniform(*KEY_INTENSITY_RANGE),
        "key_color": _tint(rng),
        "key_elevation_deg": rng.uniform(20.0, 90.0),
        "key_azimuth_deg": rng.uniform(-180.0, 180.0),
        "envs": [],
    }
    dome.GetIntensityAttr().Set(look["dome_intensity"])
    dome.GetColorAttr().Set(Gf.Vec3f(*look["dome_color"]))
    key.GetIntensityAttr().Set(look["key_intensity"])
    key.GetColorAttr().Set(Gf.Vec3f(*look["key_color"]))
    # a distant light shines along its -z axis: tilt it down from the zenith by (90 - elevation) toward the azimuth
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), look["key_azimuth_deg"]) * Gf.Rotation(
        Gf.Vec3d(0, 1, 0), 90.0 - look["key_elevation_deg"]
    )
    q = rot.GetQuat()
    xform = UsdGeom.Xformable(key.GetPrim())
    orient = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
    op = orient[0] if orient else xform.AddOrientOp()
    value = op.Get()
    quat_type = type(value) if value is not None else Gf.Quatd
    op.Set(quat_type(q.GetReal(), *q.GetImaginary()))
    for i in range(env.num_envs):
        root = f"/World/envs/env_{i}"
        gray = rng.uniform(0.15, 0.95)
        ceiling = tuple(min(1.0, gray * rng.uniform(0.85, 1.15)) for _ in range(3))
        table = tuple(rng.uniform(0.1, 0.9) for _ in range(3))
        height = rng.uniform(*CEILING_HEIGHT_RANGE)
        _set_color(stage, f"{root}/Ceiling", ceiling)
        _set_height(stage, f"{root}/Ceiling", height)
        if stage.GetPrimAtPath(f"{root}/Table"):  # moving bases have no table
            _set_color(stage, f"{root}/Table", table)
        look["envs"].append({"ceiling_color": ceiling, "ceiling_height": height, "table_color": table})
    return look
