"""
RETINA sections → 2D per-section downsample → optional Z subsample
→ marching cubes (global) → stretch (z,y,x) → PyVista render
Optionally export stacked 3D TIFF volume.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-12-03
"""

import argparse
import json
from pathlib import Path

import mcubes
import numpy as np
import pyvista as pv
import tifffile
from PIL import Image
from skimage import transform
from tqdm import tqdm


# Utility
def parse_scale_factors(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("stretch_factors expects 'z,y,x'")
    return np.array([float(p) for p in parts], dtype=np.float32)


def compute_screen_axis_angles(plotter, verts):
    if len(verts) == 0:
        return {}

    renderer = plotter.renderer
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = np.maximum(maxs - mins, 1e-6)
    axis_len = float(np.max(extents) * 0.2)

    origin = center.astype(np.float64)
    axis_points = {
        "x": origin + np.array([axis_len, 0.0, 0.0], dtype=np.float64),
        "y": origin + np.array([0.0, axis_len, 0.0], dtype=np.float64),
        "z": origin + np.array([0.0, 0.0, axis_len], dtype=np.float64),
    }

    def world_to_display(point):
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        x, y, z = renderer.GetDisplayPoint()
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    origin_display = world_to_display(origin)
    payload = {
        "origin_world": origin.tolist(),
        "origin_display": origin_display.tolist(),
        "axis_length_world": axis_len,
        "angles_deg": {},
    }
    for axis_name, point in axis_points.items():
        point_display = world_to_display(point)
        delta = point_display[:2] - origin_display[:2]
        angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))
        payload["angles_deg"][axis_name] = {
            "angle_deg": angle_deg,
            "endpoint_world": point.tolist(),
            "endpoint_display": point_display.tolist(),
            "delta_display_xy": delta.tolist(),
        }
    return payload


def compute_integer_axis_config(verts, preferred_max_labels=5):
    mins = np.floor(verts.min(axis=0)).astype(int)
    maxs = np.ceil(verts.max(axis=0)).astype(int)

    def choose_count(lo, hi, max_labels, min_labels=2):
        diff = int(hi - lo)
        if diff <= 0:
            return 2
        for n in range(int(max_labels), int(min_labels) - 1, -1):
            if diff % (n - 1) == 0:
                return n
        return int(min_labels)

    def expand_range_for_exact_count(lo, hi, n_labels):
        lo = int(lo)
        hi = int(hi)
        if n_labels <= 1:
            return lo, hi
        diff = max(1, hi - lo)
        step = int(np.ceil(diff / float(n_labels - 1)))
        hi = lo + step * (n_labels - 1)
        return lo, hi

    n_xlabels = 2
    n_ylabels = 5
    n_zlabels = 5

    y_lo, y_hi = expand_range_for_exact_count(mins[1], maxs[1], n_ylabels)
    z_lo, z_hi = expand_range_for_exact_count(mins[2], maxs[2], n_zlabels)

    return {
        "axes_ranges": [
            int(mins[0]), int(maxs[0]),
            int(y_lo), int(y_hi),
            int(z_lo), int(z_hi),
        ],
        "n_xlabels": int(n_xlabels),
        "n_ylabels": int(n_ylabels),
        "n_zlabels": int(n_zlabels),
    }


# Volume Loading
def load_and_downsample_volume_streaming(
        input_dir: Path,
        pattern="*.tif",
        dtype="float32",
        downsample_xy=1,
        downsample_z=1,
        skip_first_slices=0,
):
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise ValueError(f"No slices found in {input_dir}")

    skip_first_slices = int(skip_first_slices)
    if skip_first_slices < 0:
        raise ValueError("skip_first_slices must be >= 0")
    if skip_first_slices >= len(files):
        raise ValueError(
            f"skip_first_slices={skip_first_slices} leaves no slices from {len(files)} input files"
        )

    source_files = files[skip_first_slices:]
    first = tifffile.imread(source_files[0]).astype(dtype)
    if first.ndim != 2:
        raise ValueError(f"Expected 2D slices, got shape={first.shape} for {source_files[0]}")

    raw_Z = len(source_files)
    raw_H, raw_W = first.shape
    raw_shape = (raw_Z, raw_H, raw_W)

    target_H = max(1, raw_H // downsample_xy)
    target_W = max(1, raw_W // downsample_xy)
    selected_files = source_files[::max(1, downsample_z)]
    target_Z = len(selected_files)

    out_dtype = np.float32 if downsample_xy > 1 else np.dtype(dtype)
    volume_ds = np.empty((target_Z, target_H, target_W), dtype=out_dtype)

    print(f"Raw volume shape = {raw_shape}, input dtype={np.dtype(dtype)}")
    if skip_first_slices:
        skipped_names = [f.name for f in files[:skip_first_slices]]
        print(f"Skipped first {skip_first_slices} sorted slice(s): {skipped_names}")
    print(f"2D downsample each kept slice: {raw_H}×{raw_W} → {target_H}×{target_W}")
    if downsample_z > 1:
        print(f"Z subsample: keeping every {downsample_z} slice ({raw_Z} → {target_Z})")

    for out_i, f in enumerate(tqdm(selected_files, desc="Loading + downsampling slices")):
        if out_i == 0:
            img = first
        else:
            img = tifffile.imread(f).astype(dtype)

        if img.shape != (raw_H, raw_W):
            raise ValueError(f"Inconsistent slice shape for {f}: {img.shape} != {(raw_H, raw_W)}")

        if downsample_xy == 1:
            if img.dtype != out_dtype:
                img = img.astype(out_dtype, copy=False)
            volume_ds[out_i] = img
        else:
            volume_ds[out_i] = transform.resize(
                img, (target_H, target_W),
                order=1, mode="reflect", preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32, copy=False)

    print(f"Loaded downsampled volume shape = {volume_ds.shape}, dtype={volume_ds.dtype}")
    return volume_ds, files, raw_shape


# Meshing
def marching_cubes_pymcubes(volume, level=0.5):
    print(f"Running PyMCubes marching cubes (level={level})...")
    verts, faces = mcubes.marching_cubes(volume, level)
    faces = faces.astype(np.int32)
    print(f"verts={len(verts):,}, faces={len(faces):,}")
    return verts, faces


# Save Utilities
def save_mesh_npz(path, verts, faces, scale, offset):
    np.savez(path, verts=verts, faces=faces, scale=scale, offset=offset)
    print(f"Saved NPZ mesh → {path}")


def save_mesh_ply(path, verts, faces, offset):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"comment offset_z {offset[0]:.2f}\n")
        f.write(f"comment offset_y {offset[1]:.2f}\n")
        f.write(f"comment offset_x {offset[2]:.2f}\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        f.writelines(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
        f.writelines(f"3 {tri[0]} {tri[1]} {tri[2]}\n" for tri in faces)
    print(f"Saved PLY mesh → {path}")


def save_volume_tif(path, volume):
    volume = volume.astype(np.float32)
    tifffile.imwrite(path, volume)
    print(f"Saved stacked 3D TIFF → {path} (shape={volume.shape})")


def save_params_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved parameters → {path}")


def crop_png_background(path: Path, background_rgb=(0, 0, 0), padding: int = 24):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb)
        bg = np.array(background_rgb, dtype=np.uint8)
        mask = np.any(arr != bg, axis=2)
        if not np.any(mask):
            return {
                "original_size": list(rgb.size),
                "cropped_size": list(rgb.size),
                "crop_box": [0, 0, rgb.size[0], rgb.size[1]],
            }

        ys, xs = np.where(mask)
        left = max(0, int(xs.min()) - int(padding))
        right = min(arr.shape[1], int(xs.max()) + 1 + int(padding))
        top = max(0, int(ys.min()) - int(padding))
        bottom = min(arr.shape[0], int(ys.max()) + 1 + int(padding))
        cropped = rgb.crop((left, top, right, bottom))
        cropped.save(path)
        return {
            "original_size": list(rgb.size),
            "cropped_size": list(cropped.size),
            "crop_box": [left, top, right, bottom],
        }


def pad_png_to_canvas(path: Path, canvas_size, background_rgb=(0, 0, 0)):
    target_w, target_h = int(canvas_size[0]), int(canvas_size[1])
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        src_w, src_h = rgb.size
        if src_w > target_w or src_h > target_h:
            raise ValueError(
                f"Cropped image {rgb.size} exceeds target canvas {(target_w, target_h)} for {path}"
            )
        canvas = Image.new("RGB", (target_w, target_h), tuple(int(v) for v in background_rgb))
        left = (target_w - src_w) // 2
        top = (target_h - src_h) // 2
        canvas.paste(rgb, (left, top))
        canvas.save(path)
        return {
            "source_size": [src_w, src_h],
            "canvas_size": [target_w, target_h],
            "paste_xy": [left, top],
        }


def smooth_mesh(verts, faces, iterations=10, relaxation=0.01):
    if iterations <= 0:
        return verts, faces
    mesh = pv.PolyData(verts, np.hstack((np.full((len(faces), 1), 3), faces)))
    smoothed = mesh.smooth(n_iter=int(iterations), relaxation_factor=float(relaxation))
    new_verts = smoothed.points
    if smoothed.faces.size % 4 != 0:
        raise ValueError("Smoothed mesh faces are not triangles.")
    new_faces = smoothed.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return new_verts, new_faces


def keep_largest_mesh_component(verts, faces):
    if len(faces) == 0:
        return verts, faces
    # Build adjacency on vertices via faces
    adj = [set() for _ in range(len(verts))]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    visited = np.zeros((len(verts),), dtype=bool)
    comp_ids = np.full((len(verts),), -1, dtype=np.int32)
    comp_sizes = []
    cid = 0
    for v in range(len(verts)):
        if visited[v]:
            continue
        # isolated vertex (no faces)
        if not adj[v]:
            visited[v] = True
            comp_ids[v] = cid
            comp_sizes.append(1)
            cid += 1
            continue
        # BFS
        stack = [v]
        visited[v] = True
        comp_ids[v] = cid
        size = 1
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    comp_ids[w] = cid
                    size += 1
                    stack.append(w)
        comp_sizes.append(size)
        cid += 1
    largest = int(np.argmax(comp_sizes))
    keep_vert_mask = comp_ids == largest
    if not keep_vert_mask.any():
        return verts, faces
    # remap vertices
    old_to_new = -np.ones((len(verts),), dtype=np.int32)
    new_idx = np.flatnonzero(keep_vert_mask)
    old_to_new[new_idx] = np.arange(len(new_idx), dtype=np.int32)
    # keep faces fully inside
    face_mask = keep_vert_mask[faces].all(axis=1)
    new_faces = old_to_new[faces[face_mask]]
    new_verts = verts[new_idx]
    return new_verts, new_faces


def render_publication_figure_from_mesh(
        verts,
        faces,
        out_png: Path,
        mesh_smooth_iters: int = 30,
        mesh_smooth_relax: float = 0.05,
        window_size=(2200, 2400),
        camera_zoom: float = 0.85,
        axes_line_width: float = 6.0,
        bounds_line_width: float = 6.0,
        bounds_font_size: int = 18,
        bounds_text_screen_size: float = 18.0,
        bounds_title_offset=(35.0, 10.0),
        bounds_label_offset: float = 48.0,
        axes_widget_viewport=(0.12, 0.12, 0.26, 0.26),
        crop_background: bool = False,
        crop_padding: int = 24,
        keep_canvas_size: bool = False,
):
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Empty mesh; cannot render.")

    cell_faces = np.hstack((np.full((len(faces), 1), 3, dtype=np.int32), faces.astype(np.int32)))
    mesh = pv.PolyData(verts, cell_faces)
    if mesh_smooth_iters > 0:
        mesh = mesh.smooth(n_iter=int(mesh_smooth_iters), relaxation_factor=float(mesh_smooth_relax))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("black")
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(
        mesh,
        color="#5a6f83",
        smooth_shading=True,
        specular=0.25,
        specular_power=30,
        ambient=0.08,
        diffuse=0.95,
    )
    p.add_light(pv.Light(position=(1.2, 1.0, 1.5), intensity=1.3, light_type="scene light"))
    p.add_light(pv.Light(position=(-1.4, -0.2, 0.8), intensity=0.8, light_type="scene light"))
    p.add_light(pv.Light(position=(0.2, -1.1, -1.4), intensity=0.45, light_type="scene light"))
    p.add_axes(
        line_width=float(axes_line_width),
        color="white",
        x_color="white",
        y_color="white",
        z_color="white",
        labels_off=False,
        xlabel="Z",
        ylabel="Y",
        zlabel="X",
        viewport=tuple(float(v) for v in axes_widget_viewport),
    )
    axis_cfg = compute_integer_axis_config(verts)
    bounds_actor = p.show_bounds(
        axes_ranges=axis_cfg["axes_ranges"],
        xtitle="Z",
        ytitle="Y",
        ztitle="X",
        color="white",
        ticks="outside",
        location="outer",
        font_size=int(bounds_font_size),
        grid=False,
        n_xlabels=axis_cfg["n_xlabels"],
        n_ylabels=axis_cfg["n_ylabels"],
        n_zlabels=axis_cfg["n_zlabels"],
        fmt="%.0f",
    )
    bounds_actor.SetScreenSize(float(bounds_text_screen_size))
    bounds_actor.SetTitleOffset(tuple(float(v) for v in bounds_title_offset))
    bounds_actor.SetLabelOffset(float(bounds_label_offset))
    for axes_prop in (
            bounds_actor.GetXAxesLinesProperty(),
            bounds_actor.GetYAxesLinesProperty(),
            bounds_actor.GetZAxesLinesProperty(),
    ):
        axes_prop.SetLineWidth(float(bounds_line_width))
    p.view_isometric()
    p.reset_camera()
    if camera_zoom > 0:
        p.camera.zoom(float(camera_zoom))
    p.render()
    render_meta = {
        "camera": {
            "position": list(map(float, p.camera.position)),
            "focal_point": list(map(float, p.camera.focal_point)),
            "view_up": list(map(float, p.camera.up)),
            "clipping_range": list(map(float, p.camera.clipping_range)),
        },
        "bounds_axes_config": axis_cfg,
        "screen_axes": compute_screen_axis_angles(p, verts),
    }
    p.screenshot(str(out_png))
    p.close()
    if crop_background:
        render_meta["crop_background"] = crop_png_background(
            out_png,
            background_rgb=(0, 0, 0),
            padding=int(crop_padding),
        )
        if keep_canvas_size:
            render_meta["padded_canvas"] = pad_png_to_canvas(
                out_png,
                canvas_size=window_size,
                background_rgb=(0, 0, 0),
            )
    print(f"Saved PyVista figure -> {out_png}")
    return render_meta


def main():
    parser = argparse.ArgumentParser(description="RETINA 3D reconstruction with PyMCubes + global origin alignment.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.tif")
    parser.add_argument("--downsample_xy", type=int, default=1)
    parser.add_argument("--downsample_z", type=int, default=1)
    parser.add_argument("--skip_first_slices", type=int, default=0,
                        help="Exclude this many leading sorted input slices before reconstruction.")
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--stretch_factors", type=str, default="1,1,1")
    parser.add_argument("--dtype", type=str, default="uint8", choices=["float32", "uint8"])
    parser.add_argument("--export_npz", type=Path, default=None)
    parser.add_argument("--export_ply", type=Path, default=None)
    parser.add_argument("--export_tif", type=Path, default=None)
    parser.add_argument("--mesh_smooth_iters", type=int, default=0,
                        help="Apply light Laplacian smoothing to the mesh after reconstruction (0 = off).")
    parser.add_argument("--mesh_smooth_relax", type=float, default=0.01,
                        help="Relaxation factor for mesh smoothing (smaller = gentler).")
    parser.add_argument("--keep_largest_mesh_component", action="store_true",
                        help="Keep only the largest connected mesh component (by vertex count).")
    parser.add_argument("--render_png", type=Path, default=None,
                        help="Output PNG path for publication-style PyVista rendering.")
    parser.add_argument("--render_width", type=int, default=2200)
    parser.add_argument("--render_height", type=int, default=2400)
    parser.add_argument("--camera_zoom", type=float, default=0.85)
    parser.add_argument("--axes_line_width", type=float, default=6.0,
                        help="Line width for the rendered coordinate axes.")
    parser.add_argument("--bounds_line_width", type=float, default=6.0,
                        help="Line width for the rendered mesh bounding axes.")
    parser.add_argument("--bounds_font_size", type=int, default=18,
                        help="Font size for the rendered mesh bounding axes labels.")
    parser.add_argument("--bounds_text_screen_size", type=float, default=18.0,
                        help="Screen-space text size for the rendered mesh bounding axes.")
    parser.add_argument("--bounds_title_offset", type=str, default="35,10",
                        help="Title offset for bounds axes as horizontal,vertical.")
    parser.add_argument("--bounds_label_offset", type=float, default=48.0,
                        help="Label offset for bounds axes tick labels.")
    parser.add_argument("--axes_widget_viewport", type=str, default="0.12,0.12,0.26,0.26",
                        help="Viewport for the corner axes widget as x0,y0,x1,y1.")
    parser.add_argument("--crop_render_background", action="store_true",
                        help="Crop outer black margins from the rendered PNG.")
    parser.add_argument("--crop_padding", type=int, default=24,
                        help="Padding in pixels to keep after cropping the rendered PNG.")
    parser.add_argument("--keep_render_canvas_size", action="store_true",
                        help="After cropping, pad the render back onto the original fixed-size canvas.")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--save_params_json", action="store_true",
                        help="Save run parameters and derived values to JSON.")
    parser.add_argument("--params_json_path", type=Path, default=None,
                        help="Custom JSON path (requires --save_params_json).")
    args = parser.parse_args()

    stretch = parse_scale_factors(args.stretch_factors)
    axes_widget_viewport = [float(x.strip()) for x in args.axes_widget_viewport.split(",") if x.strip()]
    if len(axes_widget_viewport) != 4:
        raise ValueError("--axes_widget_viewport must be x0,y0,x1,y1")
    bounds_title_offset = [float(x.strip()) for x in args.bounds_title_offset.split(",") if x.strip()]
    if len(bounds_title_offset) != 2:
        raise ValueError("--bounds_title_offset must be horizontal,vertical")

    # Load + Downsample
    volume_ds, input_files, raw_shape = load_and_downsample_volume_streaming(
        args.input_dir,
        args.pattern,
        args.dtype,
        args.downsample_xy,
        args.downsample_z,
        args.skip_first_slices,
    )
    ds_shape = tuple(volume_ds.shape)

    if args.export_tif:
        save_volume_tif(args.export_tif, volume_ds)

    # Marching Cubes
    scale = np.array([
        raw_shape[0] / volume_ds.shape[0],
        raw_shape[1] / volume_ds.shape[1],
        raw_shape[2] / volume_ds.shape[2],
    ], dtype=np.float32)

    verts, faces = marching_cubes_pymcubes(volume_ds, args.level)

    # Compute Offset
    z, y, x = np.nonzero(volume_ds)
    offset = np.array([z.min(), y.min(), x.min()], dtype=np.float32)
    verts[:, 0] += offset[0]
    verts[:, 1] += offset[1]
    verts[:, 2] += offset[2]
    print(f"Applied global offset = (z={offset[0]}, y={offset[1]}, x={offset[2]})")

    # Scale + Stretch
    verts = verts * scale[None, :]
    verts = verts * stretch[None, :]
    print(f"Applied stretch factors = {stretch.tolist()}")

    if args.mesh_smooth_iters > 0:
        print(f"Smoothing mesh: iters={args.mesh_smooth_iters}, relax={args.mesh_smooth_relax}")
        verts, faces = smooth_mesh(verts, faces, args.mesh_smooth_iters, args.mesh_smooth_relax)
    if args.keep_largest_mesh_component:
        print("Keeping largest mesh component...")
        verts, faces = keep_largest_mesh_component(verts, faces)

    # Save Outputs
    if args.export_npz:
        save_mesh_npz(args.export_npz, verts, faces, scale * stretch, offset)
    if args.export_ply:
        save_mesh_ply(args.export_ply, verts, faces, offset)

    render_meta = None
    if not args.no_render:
        render_png = args.render_png
        if render_png is None:
            render_png = args.input_dir / "reconstruction_render.png"
        render_meta = render_publication_figure_from_mesh(
            verts,
            faces,
            out_png=render_png,
            mesh_smooth_iters=args.mesh_smooth_iters,
            mesh_smooth_relax=args.mesh_smooth_relax,
            window_size=(int(args.render_width), int(args.render_height)),
            camera_zoom=float(args.camera_zoom),
            axes_line_width=float(args.axes_line_width),
            bounds_line_width=float(args.bounds_line_width),
            bounds_font_size=int(args.bounds_font_size),
            bounds_text_screen_size=float(args.bounds_text_screen_size),
            bounds_title_offset=tuple(bounds_title_offset),
            bounds_label_offset=float(args.bounds_label_offset),
            axes_widget_viewport=tuple(axes_widget_viewport),
            crop_background=bool(args.crop_render_background),
            crop_padding=int(args.crop_padding),
            keep_canvas_size=bool(args.keep_render_canvas_size),
        )

    if args.save_params_json:
        params_path = args.params_json_path
        if params_path is None:
            params_path = args.input_dir / "reconstruction_params.json"
        payload = {
            "input_dir": str(args.input_dir),
            "pattern": args.pattern,
            "downsample_xy": int(args.downsample_xy),
            "downsample_z": int(args.downsample_z),
            "skip_first_slices": int(args.skip_first_slices),
            "level": float(args.level),
            "stretch_factors": args.stretch_factors,
            "dtype": args.dtype,
            "export_npz": str(args.export_npz) if args.export_npz else None,
            "export_ply": str(args.export_ply) if args.export_ply else None,
            "export_tif": str(args.export_tif) if args.export_tif else None,
            "mesh_smooth_iters": int(args.mesh_smooth_iters),
            "mesh_smooth_relax": float(args.mesh_smooth_relax),
            "render_png": str(args.render_png) if args.render_png else None,
            "render_window_size": [int(args.render_width), int(args.render_height)],
            "camera_zoom": float(args.camera_zoom),
            "axes_line_width": float(args.axes_line_width),
            "bounds_line_width": float(args.bounds_line_width),
            "bounds_font_size": int(args.bounds_font_size),
            "bounds_text_screen_size": float(args.bounds_text_screen_size),
            "bounds_title_offset": list(bounds_title_offset),
            "bounds_label_offset": float(args.bounds_label_offset),
            "axes_widget_viewport": list(axes_widget_viewport),
            "crop_render_background": bool(args.crop_render_background),
            "crop_padding": int(args.crop_padding),
            "keep_render_canvas_size": bool(args.keep_render_canvas_size),
            "no_render": bool(args.no_render),
            "raw_shape": raw_shape,
            "downsampled_shape": ds_shape,
            "scale": scale.tolist(),
            "stretch": stretch.tolist(),
            "offset": offset.tolist(),
            "num_verts": len(verts),
            "num_faces": len(faces),
            "render_metadata": render_meta,
        }
        save_params_json(params_path, payload)


if __name__ == "__main__":
    main()
