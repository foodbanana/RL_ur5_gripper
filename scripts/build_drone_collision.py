"""Pre-decompose the Iris body collider into convex hulls and write ``iris_collision.usda`` (Newton only).

Why: the Pegasus ``iris.usd`` marks the body collider as ``convexDecomposition``. Newton runs CoACD on it at load
time; the pieces differ between runs and MuJoCo fails on a near-zero-volume piece ("mesh volume is too small").
Writing the decomposition once gives Newton a deterministic collision shape. PhysX keeps its own decomposition
(see ``assets/data/drones/iris/README.md``).

The layer disables collision on the original body mesh (it stays visible) and adds invisible convex
hull pieces with the same local transform. ``iris_0p75_newton.usda`` sublayers it. Pieces below
``--min_volume`` are dropped.

The layer also binds a physics material to the pieces (Newton only, like the layer): friction 1.0 (Newton's default,
unchanged) and torsional friction ``--torsional`` [m]. MuJoCo makes one contact point per finger-hull pair, so the
two fingers alone could not stop the dropped drone from pivoting about the line between them; with torsional
friction (and ``condim`` 4 on the drone colliders, ``assets/drones.py``) it holds like on PhysX.
``--material_only`` rewrites only that material in the existing layer (no new decomposition).

Usage (from the repository root; no simulator needed):

    uv run python scripts/build_drone_collision.py
    uv run python scripts/build_drone_collision.py --material_only --torsional 0.02
"""

import argparse
from pathlib import Path

import coacd
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

REPO_ROOT = Path(__file__).resolve().parents[1]
IRIS_DIR = REPO_ROOT / "src" / "drone_grasp_rl" / "assets" / "data" / "drones" / "iris"
SOURCE_USD = IRIS_DIR / "iris.usd"
OUT_USD = IRIS_DIR / "iris_collision.usda"
BODY_MESH = "/vehicle/body/body"
DRONE_SCALE = 0.75  # applied on /vehicle by iris_0p75.usda; used here only for the report

parser = argparse.ArgumentParser(description="Pre-decompose the Iris body collider.")
parser.add_argument("--threshold", type=float, default=0.03, help="CoACD concavity threshold.")
parser.add_argument("--max_hulls", type=int, default=32, help="Maximum number of convex hulls.")
parser.add_argument("--max_hull_vertices", type=int, default=64, help="Maximum vertices per hull.")
parser.add_argument("--min_volume", type=float, default=1e-7, help="Drop pieces below this volume [m^3, unscaled].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--torsional", type=float, default=0.02, help="Torsional friction of the pieces [m] (Newton).")
parser.add_argument("--material_only", action="store_true", help="Only (re)write the contact material.")
args = parser.parse_args()


def waist_width(mesh: trimesh.Trimesh) -> tuple[float, float] | None:
    """y extent of the cross-section at body x = 0 (where the fingers close)."""
    section = mesh.section(plane_origin=[0.0, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    if section is None:
        return None
    y = section.vertices[:, 1]
    return float(y.min()), float(y.max())


MATERIAL = "/vehicle/body/newton_contact_material"


def bind_contact_material(out: Usd.Stage) -> None:
    """Physics material on every collision piece: friction 1.0, torsional friction ``--torsional``."""
    from pxr import UsdShade

    material = UsdShade.Material.Define(out, MATERIAL)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(1.0)
    api.CreateDynamicFrictionAttr().Set(1.0)
    material.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(args.torsional)
    for prim in out.GetPrimAtPath("/vehicle/body").GetChildren():
        if prim.GetName().startswith("collision_"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
    print(f"[collision] contact material: friction 1.0, torsional friction {args.torsional} m")


def main():
    if args.material_only:
        out = Usd.Stage.Open(str(OUT_USD))
        bind_contact_material(out)
        out.GetRootLayer().Save()
        return
    stage = Usd.Stage.Open(str(SOURCE_USD))
    usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(BODY_MESH))
    points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
    assert np.all(counts == 3), "expected a triangle mesh"
    faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    local = np.array(UsdGeom.Xformable(usd_mesh.GetPrim()).GetLocalTransformation(), dtype=np.float64)

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(points, faces),
        threshold=args.threshold,
        max_convex_hull=args.max_hulls,
        max_ch_vertex=args.max_hull_vertices,
        seed=args.seed,
    )
    hulls = [trimesh.Trimesh(np.asarray(v), np.asarray(f)) for v, f in parts]
    kept = [h for h in hulls if abs(h.volume) >= args.min_volume]

    # Report in the body frame at the drone scale, against the original mesh.
    def body_frame(m: trimesh.Trimesh) -> trimesh.Trimesh:
        m = m.copy()
        m.apply_transform(local.T)  # USD stores row-vector matrices
        m.apply_scale(DRONE_SCALE)
        return m

    orig = body_frame(trimesh.Trimesh(points, faces))
    union = body_frame(trimesh.util.concatenate(kept))
    w_orig, w_hull = waist_width(orig), waist_width(union)
    print(
        f"[collision] CoACD pieces: {len(hulls)}, kept {len(kept)} "
        f"(dropped {len(hulls) - len(kept)} below {args.min_volume} m^3)"
    )
    if w_orig and w_hull:
        print(
            f"[collision] waist width at x=0 (0.75x): mesh {1000 * (w_orig[1] - w_orig[0]):.1f} mm "
            f"[{1000 * w_orig[0]:.1f}, {1000 * w_orig[1]:.1f}], hulls {1000 * (w_hull[1] - w_hull[0]):.1f} mm "
            f"[{1000 * w_hull[0]:.1f}, {1000 * w_hull[1]:.1f}]"
        )

    layer = Sdf.Layer.CreateNew(str(OUT_USD))
    out = Usd.Stage.Open(layer)
    out.SetMetadata(
        "comment",
        f"Generated by scripts/build_drone_collision.py from iris.usd {BODY_MESH} with CoACD "
        f"(threshold={args.threshold}, max_hulls={args.max_hulls}, max_hull_vertices={args.max_hull_vertices}, "
        f"min_volume={args.min_volume}, seed={args.seed}). Do not edit by hand.",
    )
    body_over = out.OverridePrim("/vehicle/body/body")
    body_over.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    out.OverridePrim("/vehicle")
    out.OverridePrim("/vehicle/body")
    for i, h in enumerate(kept):
        prim = UsdGeom.Mesh.Define(out, f"/vehicle/body/collision_{i:02d}")
        prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in h.vertices]))
        prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(h.faces)))
        prim.CreateFaceVertexIndicesAttr(Vt.IntArray(h.faces.flatten().tolist()))
        prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
        prim.AddTransformOp().Set(Gf.Matrix4d(local.tolist()))
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    bind_contact_material(out)
    layer.Save()
    print(f"[collision] wrote {OUT_USD.relative_to(REPO_ROOT)} with {len(kept)} convex hull pieces")


if __name__ == "__main__":
    main()
