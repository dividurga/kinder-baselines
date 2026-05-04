"""Preprocess YCB meshes into VHACD collision meshes and AABB JSON files.

Usage:
    1. Download YCB objects and place their .obj files in ycb_raw/:
           ycb_raw/002_master_chef_can/textured.obj
           ycb_raw/004_sugar_box/textured.obj
           ycb_raw/025_mug/textured.obj
           ...
    2. Run: python prepare_assets.py

Outputs (into assets/):
    assets/visual/<name>.obj    — copy of original mesh for rendering
    assets/vhacd/<name>_vhacd.obj  — VHACD decomposition for fine collision
    assets/aabb/<name>_aabb.json   — AABB half-extents for coarse collision
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
import trimesh

HERE = Path(__file__).parent
YCB_RAW = HERE / "ycb_raw"
ASSETS = HERE / "assets"
VISUAL_DIR = ASSETS / "visual"
VHACD_DIR = ASSETS / "vhacd"
AABB_DIR = ASSETS / "aabb"
URDF_DIR = ASSETS / "urdf"

for d in [VISUAL_DIR, VHACD_DIR, AABB_DIR, URDF_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def find_obj_file(name: str) -> Path | None:
    """Search for the .obj file for a given YCB object name."""
    # Try common YCB directory structures.
    candidates = [
        YCB_RAW / name / "textured.obj",
        YCB_RAW / name / "textured_simple.obj",
        YCB_RAW / name / f"{name}.obj",
        YCB_RAW / f"{name}.obj",
    ]
    for c in candidates:
        if c.exists():
            return c
    # YCB google_16k / google_64k nested structure: search all .obj under the object dir.
    obj_dir = YCB_RAW / name
    if obj_dir.exists():
        matches = sorted(obj_dir.rglob("*.obj"))
        if matches:
            return matches[0]
    # Last resort: recursive search from YCB_RAW root.
    matches = list(YCB_RAW.rglob(f"{name}*.obj"))
    return matches[0] if matches else None


def compute_aabb_half_extents(obj_path: Path) -> list[float]:
    """Load mesh and return AABB half-extents [x, y, z]."""
    mesh = trimesh.load(str(obj_path), force="mesh")
    bounds = mesh.bounds  # shape (2, 3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    half_extents = ((bounds[1] - bounds[0]) / 2.0).tolist()
    return half_extents


def run_vhacd(obj_path: Path, out_path: Path) -> bool:
    """Run PyBullet VHACD on obj_path and save result to out_path."""
    client = p.connect(p.DIRECT)
    log_file = str(out_path.with_suffix(".log"))
    try:
        p.vhacd(
            str(obj_path),
            str(out_path),
            log_file,
            concavity=0.001,
            alpha=0.05,
            beta=0.05,
            minVolumePerCH=0.0001,
            resolution=200_000,
            maxNumVerticesPerCH=64,
            planeDownsampling=4,
            convexhullDownsampling=4,
            pca=0,
            mode=0,
            convexhullApproximation=1,
            physicsClientId=client,
        )
    finally:
        p.disconnect(client)

    return out_path.exists()


def write_urdf(name: str, half_extents: list[float]) -> None:
    """Write box and vhacd URDF files for this object.

    Paths in the URDF are relative to the urdf/ directory so PyBullet
    resolves them correctly regardless of working directory.
    """
    visual_rel = f"../visual/{name}.obj"
    vhacd_rel = f"../vhacd/{name}_vhacd.obj"
    sx, sy, sz = [h * 2 for h in half_extents]  # URDF box uses full extents

    box_urdf = f"""\
<?xml version="1.0"?>
<robot name="{name}_box">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="{visual_rel}"/></geometry>
    </visual>
    <collision>
      <geometry><box size="{sx:.6f} {sy:.6f} {sz:.6f}"/></geometry>
    </collision>
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
</robot>
"""
    vhacd_urdf = f"""\
<?xml version="1.0"?>
<robot name="{name}_vhacd">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="{visual_rel}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{vhacd_rel}"/></geometry>
    </collision>
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
</robot>
"""
    box_out = URDF_DIR / f"{name}_box.urdf"
    vhacd_out = URDF_DIR / f"{name}_vhacd.urdf"
    box_out.write_text(box_urdf)
    vhacd_out.write_text(vhacd_urdf)
    print(f"  urdf    → {box_out.name}, {vhacd_out.name}")


def process_object(name: str) -> bool:
    """Process one YCB object. Returns True on success."""
    print(f"\n[{name}]")

    obj_path = find_obj_file(name)
    if obj_path is None:
        print(f"  WARNING: no .obj found in {YCB_RAW}/{name}/. Skipping.")
        return False
    print(f"  source: {obj_path}")

    # Copy visual mesh.
    visual_out = VISUAL_DIR / f"{name}.obj"
    shutil.copy2(obj_path, visual_out)
    # Copy MTL and textures if present.
    for ext in [".mtl", ".png", ".jpg"]:
        src = obj_path.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, VISUAL_DIR / f"{name}{ext}")
    print(f"  visual  → {visual_out}")

    # AABB.
    half_extents = compute_aabb_half_extents(obj_path)
    aabb_out = AABB_DIR / f"{name}_aabb.json"
    with open(aabb_out, "w") as f:
        json.dump({"half_extents": half_extents, "source": str(obj_path)}, f, indent=2)
    print(f"  aabb    → {aabb_out}  half_extents={[f'{v:.4f}' for v in half_extents]}")

    # VHACD.
    vhacd_out = VHACD_DIR / f"{name}_vhacd.obj"
    ok = run_vhacd(obj_path, vhacd_out)
    if ok:
        print(f"  vhacd   → {vhacd_out}")
    else:
        print(f"  ERROR: VHACD failed for {name}")
        return False

    # URDF files (box + vhacd) — needed so PyBullet unpacks VHACD into
    # separate convex hulls rather than treating it as a single mesh.
    write_urdf(name, half_extents)

    return True


DEFAULT_OBJECTS = [
    "002_master_chef_can",
    "011_banana",
    "006_mustard_bottle",
    "025_mug",
    "021_bleach_cleanser",
    "077_rubiks_cube",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=DEFAULT_OBJECTS,
        help="YCB object names to process (default: all five)",
    )
    args = parser.parse_args()

    successes = []
    failures = []
    for name in args.objects:
        if process_object(name):
            successes.append(name)
        else:
            failures.append(name)

    print(f"\n{'='*50}")
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        print(f"Failed: {failures}")
    if successes:
        print(f"Succeeded: {successes}")
        print(f"\nAssets written to: {ASSETS}")
