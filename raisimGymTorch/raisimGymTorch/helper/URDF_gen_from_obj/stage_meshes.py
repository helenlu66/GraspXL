#!/usr/bin/env python3
"""Stage a directory of raw object meshes into temp/<category>/ for urdf_gen.py.

Usage:
    python stage_meshes.py /path/to/mesh/dir --dest-category grab

Copies every top-level .obj/.ply file in the given directory into
temp/<dest-category>/<name>.obj (converting .ply -> .obj via trimesh where
needed), ready for `python urdf_gen.py`. See the "Adding Custom Objects
(e.g. GRAB)" section in the top-level README for the full pipeline.
"""
import argparse
import shutil
import sys
from pathlib import Path

import trimesh

TASK_DIR = Path(__file__).resolve().parent


def stage(src_dir: Path, dest_category: str) -> None:
    dest_dir = TASK_DIR / "temp" / dest_category
    dest_dir.mkdir(parents=True, exist_ok=True)

    meshes = sorted(list(src_dir.glob("*.obj")) + list(src_dir.glob("*.ply")))
    if not meshes:
        sys.exit(f"No .obj/.ply files found directly under {src_dir}")

    for mesh_path in meshes:
        dest = dest_dir / f"{mesh_path.stem}.obj"
        if mesh_path.suffix == ".obj":
            shutil.copy2(mesh_path, dest)
        else:
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            mesh.export(dest)
        print(f"staged {mesh_path.name} -> {dest.relative_to(TASK_DIR)}")

    print(f"staged {len(meshes)} objects into {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src_dir", type=Path,
        help="Directory of object meshes (.obj/.ply, one file per object)",
    )
    parser.add_argument(
        "--dest-category", default="grab",
        help="Subfolder under temp/ (and later rsc/) to stage into (default: grab)",
    )
    args = parser.parse_args()
    stage(args.src_dir.expanduser().resolve(), args.dest_category)
