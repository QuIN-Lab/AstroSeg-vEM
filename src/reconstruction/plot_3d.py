"""
Load and visualize a 3D mesh from NPZ/PLY with optional axis swap,
stretch scaling, camera controls, and screenshot export via PyVista.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-12-03
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def parse_scale_factors(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("stretch_factors expects 'z,y,x'")
    return np.array([float(p) for p in parts], dtype=np.float32)

def parse_camera_position(value: str):
    if not value:
        return None
    parts = [p.strip() for p in value.replace("|", ";").split(";") if p.strip()]
    if len(parts) != 3:
        raise ValueError("camera_position expects three triplets: pos;focal;up")
    vectors = []
    for triplet in parts:
        nums = [n.strip() for n in triplet.split(",") if n.strip()]
        if len(nums) != 3:
            raise ValueError(f"Invalid triplet '{triplet}', expected 3 values")
        vectors.append(tuple(float(n) for n in nums))
    return tuple(vectors)

def parse_vector3(value: str, name: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} expects three comma-separated values")
    return np.array([float(p) for p in parts], dtype=np.float32)

def faces_to_pyvista(faces):
    faces = faces.astype(np.int64, copy=False)
    return np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces))

def load_mesh_npz(path: Path):
    data = np.load(path)
    verts = np.array(data["verts"], dtype=np.float32)
    faces = np.array(data["faces"], dtype=np.int32)
    meta = {}
    for key in ("scale", "offset"):
        if key in data:
            meta[key] = np.array(data[key], dtype=np.float32)
    print(f"Loaded NPZ mesh: verts={len(verts):,}, faces={len(faces):,}")
    return verts, faces, meta

def load_mesh_ply(path: Path):
    offset = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("end_header"):
                break
            if line.startswith("comment offset_"):
                parts = line.strip().split()
                if len(parts) == 3:
                    axis = parts[1].split("_")[-1]
                    value = float(parts[2])
                    if offset is None:
                        offset = np.zeros(3, dtype=np.float32)
                    if axis == "z":
                        offset[0] = value
                    elif axis == "y":
                        offset[1] = value
                    elif axis == "x":
                        offset[2] = value
    mesh = pv.read(path)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    verts = np.array(mesh.points, dtype=np.float32)
    meta = {"offset": offset} if offset is not None else {}
    print(f"Loaded PLY mesh: verts={len(verts):,}, faces={len(faces):,}")
    return verts, faces, meta

def load_mesh(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_mesh_npz(path)
    if suffix == ".ply":
        return load_mesh_ply(path)
    raise ValueError(f"Unsupported mesh format '{suffix}'")

def compute_auto_camera(verts, direction, up, distance_factor):
    mins, maxs = verts.min(axis=0), verts.max(axis=0)
    center = (mins + maxs) / 2.0
    diag = np.linalg.norm(maxs - mins)
    if diag == 0:
        diag = 1.0
    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0:
        raise ValueError("camera_direction vector must be non-zero")
    direction = direction / dir_norm
    position = center + direction * (diag * distance_factor)
    return (tuple(position.tolist()), tuple(center.tolist()), tuple(up.tolist()))

def visualize_mesh(verts, faces, color, background, off_screen, screenshot, camera_position):
    mesh = pv.PolyData(verts, faces_to_pyvista(faces))
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(background)
    plotter.add_mesh(mesh, color=color)
    plotter.add_axes()
    plotter.show_bounds(xtitle="X", ytitle="Y", ztitle="Z", color="white", ticks="both", location="outer")
    if camera_position is not None:
        plotter.camera_position = camera_position
    if screenshot:
        plotter.show(screenshot=str(screenshot))
        print(f"Saved screenshot → {screenshot}")
    else:
        plotter.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--stretch_factors", type=str, default="1,1,1")
    parser.add_argument("--color", type=str, default="white")
    parser.add_argument("--background", type=str, default="black")
    parser.add_argument("--off_screen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--swap_xz", action="store_true")
    parser.add_argument("--camera_position", type=str, default=None)
    parser.add_argument("--auto_camera", action="store_true")
    parser.add_argument("--camera_direction", type=str, default="1,1,1")
    parser.add_argument("--camera_up", type=str, default="0,0,1")
    parser.add_argument("--camera_distance_factor", type=float, default=2.0)
    args = parser.parse_args()

    stretch = parse_scale_factors(args.stretch_factors)
    verts, faces, meta = load_mesh(args.mesh)

    if args.swap_xz:
        verts = verts[:, [2, 1, 0]]
        print("Swapped x and z axes for visualization.")

    if not np.allclose(stretch, 1):
        verts = verts * stretch[None, :]
        print(f"Applied extra stretch = {stretch.tolist()}")

    if args.camera_position:
        cam_pos = parse_camera_position(args.camera_position)
        print(f"Using manual camera_position = {cam_pos}")
    elif args.auto_camera:
        cam_pos = compute_auto_camera(
            verts,
            parse_vector3(args.camera_direction, "camera_direction"),
            parse_vector3(args.camera_up, "camera_up"),
            args.camera_distance_factor,
        )
        print(f"Using auto camera = {cam_pos}")
    else:
        cam_pos = None
        print("Using default PyVista camera")

    visualize_mesh(
        verts,
        faces,
        args.color,
        args.background,
        args.off_screen,
        args.screenshot,
        cam_pos,
    )

if __name__ == "__main__":
    main()
