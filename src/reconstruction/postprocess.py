"""
SegFormer post-processing pipeline for 2D predictions.
Performs soma-guided filtering, 3D top-k component selection,
optional 2D hole filling, and DS/upsampled export.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2026-02-12
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from glob import glob
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage import transform
from tqdm import tqdm


def compute_metrics_csv(pred_dir, gt_dir, csv_out, num_workers=1, no_weighted=False):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    csv_out = Path(csv_out)
    metrics_script = Path(__file__).resolve().parents[1] / "segmentation" / "compute_metrics.py"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth dir not found: {gt_dir}")

    cmd = [
        sys.executable,
        str(metrics_script),
        "--pred_dir", str(pred_dir),
        "--gt_dir", str(gt_dir),
        "--csv_out", str(csv_out),
        "--num_workers", str(num_workers),
    ]
    if no_weighted:
        cmd.append("--no_weighted")

    print(f"[Metrics] {pred_dir.name} -> {csv_out}")
    subprocess.run(cmd, check=True)


def resolve_image_path(pred_path, image_dir, pred_suffix):
    base = os.path.splitext(os.path.basename(pred_path))[0]
    if pred_suffix and base.endswith(pred_suffix):
        base = base[:-len(pred_suffix)]
    ext = os.path.splitext(pred_path)[1]
    cand = os.path.join(image_dir, base + ext)
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(image_dir, os.path.basename(pred_path))
    if os.path.exists(cand2):
        return cand2
    matches = sorted(glob(os.path.join(image_dir, base + ".*")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No matching image for {pred_path} under {image_dir}")


def read_pred_with_image_mask(pred_path, image_dir=None, pred_suffix="_pred_2d"):
    pred = tifffile.imread(pred_path)
    if not image_dir:
        return pred
    img_path = resolve_image_path(pred_path, image_dir, pred_suffix)
    img = tifffile.imread(img_path)
    if img.shape != pred.shape:
        raise ValueError(f"Image shape {img.shape} != pred shape {pred.shape} for {pred_path}")
    pred = pred.copy()
    pred[img == 0] = 0
    return pred


def downsample_binary_mask(mask, downsample_xy):
    if not downsample_xy or int(downsample_xy) <= 1:
        return mask
    ds = int(downsample_xy)
    h, w = mask.shape
    th, tw = max(1, h // ds), max(1, w // ds)
    return transform.resize(mask.astype(np.uint8), (th, tw), order=0,
                            preserve_range=True, anti_aliasing=False).astype(bool)


def upsample_binary_mask(mask, out_shape):
    if mask.shape == out_shape:
        return mask
    return transform.resize(mask.astype(np.uint8), out_shape, order=0,
                            preserve_range=True, anti_aliasing=False).astype(bool)


# Top-K Feature Collection
def collect_topk_dataframe(pred_paths, fg_label=1, top_k=3, image_dir=None, pred_suffix="_pred_2d",
                           downsample_xy=1, return_op_time=False):
    rows = []
    op_time_sec = 0.0

    for z_idx, path in enumerate(tqdm(pred_paths, desc="Collect top-k 2D components")):
        pred = read_pred_with_image_mask(path, image_dir=image_dir, pred_suffix=pred_suffix)
        t_op = time.perf_counter()
        fg = (pred == fg_label)
        fg = downsample_binary_mask(fg, downsample_xy)
        labels, num = ndimage.label(fg, structure=np.ones((3, 3), dtype=np.uint8))
        if num > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            comp_ids = np.where(counts > 0)[0]
            comp_ids = comp_ids[np.argsort(counts[comp_ids])[::-1]]
        else:
            counts = np.array([0], dtype=np.int64)
            comp_ids = np.array([], dtype=np.int64)
        for rank in range(1, int(top_k) + 1):
            if rank <= len(comp_ids):
                cid = int(comp_ids[rank - 1])
                area = int(counts[cid])
                cy, cx = ndimage.center_of_mass(fg, labels=labels, index=cid)
                rows.append(
                    {"slice_index": z_idx, "slice_file": os.path.basename(path), "rank": rank, "area_pixels": area,
                     "centroid_y": float(cy), "centroid_x": float(cx)})
            else:
                rows.append({"slice_index": z_idx, "slice_file": os.path.basename(path), "rank": rank, "area_pixels": 0,
                             "centroid_y": np.nan, "centroid_x": np.nan})
        op_time_sec += time.perf_counter() - t_op

    df = pd.DataFrame(rows)
    if return_op_time:
        return df, op_time_sec
    return df


def run_hdbscan(df, out_json, max_rank=3, min_area=1, z_scale=14.0, y_scale=1.0, x_scale=1.0, min_cluster_size=15,
                min_samples=5, weight_mode="sqrt_replication", max_replication=12, best_cluster_mode="score",
                return_op_time=False):
    t_op = time.perf_counter()
    df = df.copy()
    df = df[(df["rank"] <= int(max_rank)) & (df["area_pixels"] >= int(min_area))]
    df = df.dropna(subset=["centroid_y", "centroid_x"])
    if len(df) == 0: raise ValueError("No valid points")
    z, y, x, area = df["slice_index"].to_numpy(np.float32), df["centroid_y"].to_numpy(np.float32), df[
        "centroid_x"].to_numpy(np.float32), df["area_pixels"].to_numpy(np.float32)
    feats = np.stack([z * z_scale, y * y_scale, x * x_scale], axis=1)
    if weight_mode == "none":
        reps = np.ones_like(area, dtype=np.int32)
    elif weight_mode == "sqrt_replication":
        reps = np.maximum(1, np.rint(np.sqrt(area / max(np.min(area), 1.0))).astype(np.int32))
    else:
        reps = np.maximum(1, np.rint(np.log2(area + 1.0)).astype(np.int32))
    reps = np.clip(reps, 1, int(max_replication))
    rep_feats, rep_orig_idx = np.repeat(feats, reps, axis=0), np.repeat(np.arange(len(df), dtype=np.int32), reps,
                                                                        axis=0)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                metric="euclidean")
    rep_labels = clusterer.fit_predict(rep_feats)
    labels = np.full((len(df),), -1, dtype=np.int32)
    for i in range(len(df)):
        li = rep_labels[rep_orig_idx == i]
        if li.size > 0:
            non_noise = li[li >= 0]
            if non_noise.size > 0:
                uniq, cnt = np.unique(non_noise, return_counts=True)
                labels[i] = int(uniq[np.argmax(cnt)])
    df["cluster_label"] = labels
    cluster_ids = sorted([int(c) for c in np.unique(labels) if c >= 0])
    if not cluster_ids:
        payload = {"status": "no_cluster", "num_points": len(df), "num_clusters": 0}
        op_time_sec = time.perf_counter() - t_op
        with open(out_json, "w") as f: json.dump(payload, f, indent=2)
        if return_op_time:
            return payload, op_time_sec
        return payload
    clusters = []
    for cid in cluster_ids:
        sub = df[df["cluster_label"] == cid]
        zvals, avals, yvals, xvals = sub["slice_index"].to_numpy(), sub["area_pixels"].to_numpy(np.float64), sub[
            "centroid_y"].to_numpy(np.float64), sub["centroid_x"].to_numpy(np.float64)
        z_min, z_max = int(zvals.min()), int(zvals.max())
        z_span = int(z_max - z_min + 1)
        clusters.append(
            {"cluster_id": cid, "num_points": len(sub), "z_min": z_min, "z_max": z_max, "z_span": z_span,
             "area_sum": float(avals.sum()), "score": float(avals.sum() * z_span),
             "center_y_weighted": float(np.average(yvals, weights=avals)),
             "center_x_weighted": float(np.average(xvals, weights=avals))})
    if best_cluster_mode == "area_sum":
        clusters = sorted(clusters, key=lambda c: c["area_sum"], reverse=True)
    elif best_cluster_mode == "area_over_dist":
        ref_y = float(np.average(df["centroid_y"], weights=df["area_pixels"]))
        ref_x = float(np.average(df["centroid_x"], weights=df["area_pixels"]))
        clusters = sorted(
            clusters,
            key=lambda c: c["area_sum"] / (np.hypot(c["center_y_weighted"] - ref_y,
                                                   c["center_x_weighted"] - ref_x) + 1.0),
            reverse=True,
        )
    else:
        clusters = sorted(clusters, key=lambda c: c["score"], reverse=True)
    best = clusters[0]
    best_points = df[df["cluster_label"] == best["cluster_id"]].copy()
    z_tracks = []
    for z_idx, sub in best_points.groupby("slice_index"):
        avals, yvals, xvals = sub["area_pixels"].to_numpy(np.float64), sub["centroid_y"].to_numpy(np.float64), sub[
            "centroid_x"].to_numpy(np.float64)
        z_tracks.append({"slice_index": int(z_idx), "center_y_weighted": float(np.average(yvals, weights=avals)),
                         "center_x_weighted": float(np.average(xvals, weights=avals)), "num_points": len(sub),
                         "area_sum": float(avals.sum())})
    payload = {"status": "ok", "num_points": len(df), "num_clusters": len(cluster_ids),
               "params": {"max_rank": int(max_rank), "min_area": int(min_area), "z_scale": float(z_scale),
                          "y_scale": float(y_scale), "x_scale": float(x_scale),
                          "min_cluster_size": int(min_cluster_size), "min_samples": int(min_samples),
                          "weight_mode": weight_mode, "max_replication": int(max_replication)}, "best_cluster": best,
               "all_clusters": clusters, "best_cluster_z_track": sorted(z_tracks, key=lambda r: r["slice_index"])}
    op_time_sec = time.perf_counter() - t_op
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    if return_op_time:
        return payload, op_time_sec
    return payload


def build_center_lookup(soma_json):
    with open(soma_json, "r") as f: data = json.load(f)
    if data.get("status") != "ok": raise ValueError("Invalid soma json")
    best = data["best_cluster"]
    global_center = (float(best["center_y_weighted"]), float(best["center_x_weighted"]))
    lookup = {int(row["slice_index"]): (float(row["center_y_weighted"]), float(row["center_x_weighted"])) for row in
              data.get("best_cluster_z_track", [])}
    return global_center, lookup


# Component Filtering
def filter_cc_by_center(pred_paths, out_dir, global_center, center_lookup, fg_label=1, rel_area_ratio=0.5,
                        rel_dist_ratio=0.1, image_dir=None, pred_suffix="_pred_2d",
                        single_target_protect=False, merge_radius=0, merge_out_dir=None,
                        cc_select_mode="distance", downsample_xy=1, out_dir_ds=None, return_op_time=False):
    os.makedirs(out_dir, exist_ok=True)
    if out_dir_ds:
        os.makedirs(out_dir_ds, exist_ok=True)
    # merge_out_dir is ignored to avoid writing intermediate dilation outputs
    op_time_sec = 0.0
    for z_idx, p in enumerate(tqdm(pred_paths, desc="Filtering slices")):
        m = read_pred_with_image_mask(p, image_dir=image_dir, pred_suffix=pred_suffix)
        t_op = time.perf_counter()
        fg = (m == fg_label)
        fg_ds = downsample_binary_mask(fg, downsample_xy)
        h, w = fg_ds.shape
        diag = np.hypot(h, w)
        if merge_radius and int(merge_radius) > 0:
            r = int(merge_radius)
            if downsample_xy and int(downsample_xy) > 1:
                r = max(1, int(round(r / int(downsample_xy))))
            y, x = np.ogrid[-r:r+1, -r:r+1]
            structure = (x * x + y * y) <= r * r
            fg_for_cc = ndimage.binary_dilation(fg_ds, structure=structure)
        else:
            fg_for_cc = fg_ds
        # no intermediate merge output
        labels, num = ndimage.label(fg_for_cc, structure=np.ones((3, 3), dtype=np.uint8))
        out = np.zeros_like(m)
        out_ds = None
        if out_dir_ds:
            out_ds = np.zeros_like(fg_ds, dtype=m.dtype)
        if num > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            comp_ids = np.where(counts > 0)[0].astype(np.int32)
            if comp_ids.size > 0:
                ref_y, ref_x = center_lookup.get(z_idx, global_center)
                centers = ndimage.center_of_mass(fg_for_cc, labels=labels, index=comp_ids)
                centers = np.asarray(centers, dtype=np.float32)
                cy = centers[:, 0]
                cx = centers[:, 1]
                areas = counts[comp_ids].astype(np.float32)

                dist_to_ref = np.hypot(cy - ref_y, cx - ref_x)
                if cc_select_mode == "area_over_dist":
                    scores = areas / (dist_to_ref + 1.0)
                    soma_idx = int(np.argmax(scores))
                else:
                    soma_idx = int(np.argmin(dist_to_ref))
                soma_area = float(areas[soma_idx])
                s_cy = float(cy[soma_idx])
                s_cx = float(cx[soma_idx])

                dist_limit = rel_dist_ratio * diag
                dist_to_soma = np.hypot(cy - s_cy, cx - s_cx)
                area_thresh = soma_area * rel_area_ratio
                keep_mask = ~((areas >= area_thresh) & (dist_to_soma > dist_limit))
                if single_target_protect and not np.any(keep_mask):
                    scores = areas / (dist_to_soma + 1.0)
                    keep_mask[np.argmax(scores)] = True
                kept_ids = comp_ids[keep_mask]
                if kept_ids.size > 0:
                    keep_mask_full = np.isin(labels, kept_ids)
                    keep_mask_up = upsample_binary_mask(keep_mask_full, fg.shape)
                    # apply keep mask to original fg (not dilated)
                    out[keep_mask_up & fg] = fg_label
                    if out_ds is not None:
                        out_ds[keep_mask_full & fg_ds] = fg_label
        op_time_sec += time.perf_counter() - t_op
        tifffile.imwrite(os.path.join(out_dir, os.path.basename(p)), out.astype(m.dtype))
        if out_dir_ds and out_ds is not None:
            tifffile.imwrite(os.path.join(out_dir_ds, os.path.basename(p)), out_ds.astype(m.dtype))
    if return_op_time:
        return op_time_sec


def compute_topk_labels_memmap(volume, topk, level, work_dir, return_op_time=False):
    work_dir.mkdir(parents=True, exist_ok=True)
    mask_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mask.dat", dir=work_dir).name
    mask_fg = np.memmap(mask_path, dtype=bool, mode="w+", shape=volume.shape)
    for z in range(volume.shape[0]): mask_fg[z] = volume[z] >= level
    labels_path = tempfile.NamedTemporaryFile(delete=False, suffix=".labels.dat", dir=work_dir).name
    labels_mem = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=volume.shape)
    t_op = time.perf_counter()
    ndimage.label(mask_fg, structure=np.ones((3, 3, 3), dtype=bool), output=labels_mem)
    counts = np.bincount(labels_mem.ravel())
    counts[0] = 0
    keep_labels = set(np.argsort(counts)[::-1][:topk])
    keep_labels.discard(0)
    op_time_sec = time.perf_counter() - t_op
    if return_op_time:
        return labels_mem, keep_labels, mask_path, labels_path, op_time_sec
    return labels_mem, keep_labels, mask_path, labels_path


def build_centers_from_labels(labels_volume, keep_labels, ds_xy, ds_z, orig_z):
    centers_by_z = {}
    if labels_volume is None: return centers_by_z
    for z_ds in range(labels_volume.shape[0]):
        slice_data = labels_volume[z_ds]
        present = [int(l) for l in np.unique(slice_data) if int(l) in keep_labels]
        if not present: continue
        z_start, z_end = int(z_ds * ds_z), int(min(orig_z - 1, (z_ds + 1) * ds_z - 1))
        for lab in present:
            cy, cx = ndimage.center_of_mass(slice_data, labels=slice_data, index=lab)
            for z0 in range(z_start, z_end + 1):
                centers_by_z.setdefault(z0, []).append((int(round(cy * ds_xy)), int(round(cx * ds_xy))))
    return centers_by_z




# Slice Reconstruction
def strip_ds_suffix(stem, ds_tag):
    if ds_tag and stem.endswith(ds_tag):
        return stem[:-len(ds_tag)]
    return stem

def apply_topk_centers_to_slices(pred_paths, out_dir, centers_by_z, fg_label=1, name_suffix="", ds_tag="",
                                 image_dir=None, pred_suffix="_pred_2d", return_op_time=False):
    os.makedirs(out_dir, exist_ok=True)
    op_time_sec = 0.0
    for z_idx, p in enumerate(tqdm(pred_paths, desc="Applying top-k centers")):
        img = read_pred_with_image_mask(p, image_dir=image_dir, pred_suffix=pred_suffix)
        t_op = time.perf_counter()
        labels, num = ndimage.label(img == fg_label, structure=np.ones((3, 3), dtype=np.uint8))
        out = np.zeros_like(img)
        centers = centers_by_z.get(z_idx)
        if num > 0 and centers:
            # Precompute centers for all components once per slice
            comp_ids = np.unique(labels)
            comp_ids = comp_ids[comp_ids > 0]
            comp_centers = {}
            if comp_ids.size > 0:
                centers_all = ndimage.center_of_mass(img == fg_label, labels=labels, index=comp_ids)
                for cid, (cy, cx) in zip(comp_ids, centers_all):
                    comp_centers[int(cid)] = (float(cy), float(cx))
            kept = set()
            h, w = img.shape
            for (y, x) in centers:
                yy, xx = int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))
                cid = int(labels[yy, xx])
                if cid > 0:
                    kept.add(cid)
                else:
                    if comp_centers:
                        dists = [(c, (cy - yy) ** 2 + (cx - xx) ** 2)
                                 for c, (cy, cx) in comp_centers.items()]
                        kept.add(min(dists, key=lambda t: t[1])[0])
            if kept: out[np.isin(labels, list(kept))] = fg_label
        base = os.path.basename(p)
        stem, ext = os.path.splitext(base)
        stem = strip_ds_suffix(stem, ds_tag)
        out_name = f"{stem}{name_suffix}{ext}"
        op_time_sec += time.perf_counter() - t_op
        tifffile.imwrite(os.path.join(out_dir, out_name), out.astype(img.dtype))
    if return_op_time:
        return op_time_sec


def make_disk_structure(radius):
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def scale_single_slice_params_for_downsample(min_area, support_radius, downsample_xy):
    ds = max(1, int(downsample_xy))
    if ds <= 1:
        return int(min_area), int(support_radius)
    scaled_min_area = max(1, int(round(float(min_area) / float(ds * ds))))
    scaled_support_radius = max(1, int(round(float(support_radius) / float(ds))))
    return scaled_min_area, scaled_support_radius


def suppress_single_slice_artifacts_in_memory(volume, fg_label=1,
                                              area_ratio=2.0, support_radius=9, min_area=1000,
                                              debug_print=False, slice_names=None, return_op_time=False):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    masks = np.asarray(volume == fg_label, dtype=bool)
    cleaned_masks = masks.copy()
    areas = np.asarray([int(mask.sum()) for mask in masks], dtype=np.float64)
    structure = make_disk_structure(support_radius)
    cc_structure = np.ones((3, 3), dtype=np.uint8)
    op_time_sec = 0.0
    if slice_names is None:
        slice_names = [str(i) for i in range(len(masks))]
    elif len(slice_names) != len(masks):
        raise ValueError("slice_names length must match number of z-slices")

    if debug_print:
        print("single_slice_artifact_suppression debug:")
        print(
            "slice_index\tslice_file\tarea_prev\tarea_current\tarea_next\t"
            "neighbor_median_area\tarea_ratio\ttriggered\tunsupported_area\t"
            "removed_area\tremoved_cc_count"
        )

    for z in tqdm(range(len(masks)), desc="Suppressing single-slice artifacts"):
        t_op = time.perf_counter()
        current = masks[z]
        cleaned = current.copy()
        neighbor_ref = 0.0
        area_ratio_value = None
        removed_area = 0
        removed_cc_count = 0
        unsupported_area = 0
        triggered = False

        if 0 < z < len(masks) - 1:
            neighbor_ref = float(np.median([areas[z - 1], areas[z + 1]]))
            if neighbor_ref > 0:
                area_ratio_value = float(areas[z] / neighbor_ref)
                triggered = area_ratio_value >= float(area_ratio)

            if triggered:
                support = ndimage.binary_dilation(masks[z - 1] | masks[z + 1], structure=structure)
                unsupported = current & ~support
                unsupported_area = int(np.count_nonzero(unsupported))
                labels, num = ndimage.label(unsupported, structure=cc_structure)
                if num > 0:
                    counts = np.bincount(labels.ravel())
                    counts[0] = 0
                    remove_ids = np.where(counts >= int(min_area))[0]
                    remove_ids = remove_ids[remove_ids > 0]
                    if remove_ids.size > 0:
                        remove_mask = np.isin(labels, remove_ids)
                        removed_area = int(np.count_nonzero(remove_mask))
                        removed_cc_count = int(remove_ids.size)
                        cleaned[remove_mask] = False

        op_time_sec += time.perf_counter() - t_op
        cleaned_masks[z] = cleaned
        if debug_print:
            area_prev = int(areas[z - 1]) if z > 0 else None
            area_next = int(areas[z + 1]) if z < len(masks) - 1 else None
            print(
                f"{int(z)}\t{slice_names[z]}\t{area_prev}\t"
                f"{int(areas[z])}\t{area_next}\t"
                f"{None if not (0 < z < len(masks) - 1) else round(neighbor_ref, 3)}\t"
                f"{None if area_ratio_value is None else round(area_ratio_value, 3)}\t"
                f"{bool(triggered)}\t{int(unsupported_area)}\t"
                f"{int(removed_area)}\t{int(removed_cc_count)}"
            )

    if return_op_time:
        return cleaned_masks, op_time_sec
    return cleaned_masks


def fill_holes_2d(input_dir, output_dir, pattern, dtype, image_dir=None, pred_suffix="_pred_2d", return_op_time=False):
    os.makedirs(output_dir, exist_ok=True)
    op_time_sec = 0.0
    for path in tqdm(sorted(glob(os.path.join(input_dir, pattern))), desc="Filling holes"):
        arr = tifffile.imread(path)
        t_op = time.perf_counter()
        filled = binary_fill_holes(arr > 0).astype(dtype)
        op_time_sec += time.perf_counter() - t_op
        if image_dir:
            img_path = resolve_image_path(path, image_dir, pred_suffix)
            img = tifffile.imread(img_path)
            if img.shape != filled.shape:
                raise ValueError(f"Image shape {img.shape} != pred shape {filled.shape} for {path}")
            t_op = time.perf_counter()
            filled = filled.copy()
            filled[img == 0] = 0
            op_time_sec += time.perf_counter() - t_op
        tifffile.imwrite(os.path.join(output_dir, os.path.basename(path)), filled)
    if return_op_time:
        return op_time_sec


def downsample_mask_dir(input_dir, output_dir, pattern, dtype, downsample_xy, return_op_time=False):
    os.makedirs(output_dir, exist_ok=True)
    op_time_sec = 0.0
    ds = int(downsample_xy)
    for path in tqdm(sorted(glob(os.path.join(input_dir, pattern))), desc="Downsampling 2D masks"):
        img = tifffile.imread(path) > 0
        h, w = img.shape
        th, tw = max(1, h // ds), max(1, w // ds)
        t_op = time.perf_counter()
        ds_mask = transform.resize(img.astype(np.uint8), (th, tw), order=0,
                                   preserve_range=True, anti_aliasing=False).astype(dtype)
        op_time_sec += time.perf_counter() - t_op
        tifffile.imwrite(os.path.join(output_dir, os.path.basename(path)), ds_mask)
    if return_op_time:
        return op_time_sec


def upsample_mask_dir(input_dir, output_dir, pattern, dtype, out_shape, return_op_time=False):
    os.makedirs(output_dir, exist_ok=True)
    op_time_sec = 0.0
    oh, ow = out_shape
    for path in tqdm(sorted(glob(os.path.join(input_dir, pattern))), desc="Upsampling 2D masks"):
        img = tifffile.imread(path) > 0
        t_op = time.perf_counter()
        up_mask = transform.resize(img.astype(np.uint8), (oh, ow), order=0,
                                   preserve_range=True, anti_aliasing=False).astype(dtype)
        op_time_sec += time.perf_counter() - t_op
        tifffile.imwrite(os.path.join(output_dir, os.path.basename(path)), up_mask)
    if return_op_time:
        return op_time_sec


# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pattern", default="*_pred_2d.tif")
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--pred_suffix", type=str, default="_pred_2d")
    p.add_argument("--z_scale", type=float, default=50.0)
    p.add_argument("--min_cluster_size", type=int, default=25)
    p.add_argument("--rel_area_ratio", type=float, default=0.3)
    p.add_argument("--rel_dist_ratio", type=float, default=0.1)
    p.add_argument("--single_target_protect", action="store_true")
    p.add_argument("--merge_cc_radius", type=int, default=0)
    p.add_argument("--merge_cc_out_dir", type=str, default=None)
    p.add_argument("--cc_select_mode", type=str, default="distance",
                   choices=["distance", "area_over_dist"])
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--topk_3d", type=int, default=1)
    p.add_argument("--downsample_xy", type=int, default=1)
    p.add_argument("--best_cluster_mode", type=str, default="score",
                   choices=["score", "area_sum", "area_over_dist"])
    p.add_argument("--topk_map_centers", action="store_true")
    p.add_argument("--no_topk_map_centers", action="store_true")
    p.add_argument("--fill_2d", action="store_true")
    p.add_argument("--single_slice_artifact_suppression", action="store_true",
                   help="As part of volumetric structural consolidation, remove large unsupported single-slice artifacts from the downsampled 3D volume before 3D top-k.")
    p.add_argument("--single_slice_area_ratio", type=float, default=1.5,
                   help="Trigger artifact suppression when slice foreground area exceeds this multiple of adjacent-slice median area.")
    p.add_argument("--single_slice_support_radius", type=int, default=12,
                   help="Pixel radius for dilating previous/next slice support before identifying unsupported regions.")
    p.add_argument("--single_slice_min_area", type=int, default=2000,
                   help="Minimum unsupported connected-component area to remove.")
    p.add_argument("--single_slice_debug", action="store_true",
                   help="Print per-slice single-slice artifact suppression diagnostics.")
    p.add_argument("--compute_metrics", action="store_true",
                   help="Compute metrics.csv for generated output directories after postprocessing.")
    p.add_argument("--gt_dir", type=str, default=None,
                   help="Ground-truth mask directory used when --compute_metrics is enabled.")
    p.add_argument("--metrics_num_workers", type=int, default=2,
                   help="Worker count for metrics computation when --compute_metrics is enabled.")
    p.add_argument("--metrics_no_weighted", action="store_true",
                   help="Skip weighted Dice when computing metrics.")
    p.add_argument("--save_params_json", action="store_true")
    p.add_argument("--params_json_path", type=str, default=None)
    args = p.parse_args()
    run_start = time.perf_counter()
    timing = {}
    pred_paths = sorted(glob(os.path.join(args.pred_dir, args.pattern)))
    if not pred_paths:
        raise ValueError(f"No prediction files found in {args.pred_dir} with pattern {args.pattern}")
    out_root = Path(args.out_dir)
    pred_name = Path(args.pred_dir).name
    ds_xy = int(args.downsample_xy)
    ds_tag = f"_dsxy{ds_xy}" if ds_xy > 1 else ""
    up_tag = f"{ds_tag}_upsampled" if ds_xy > 1 else ""
    s_dir = out_root / f"{pred_name}_soma_filtered{up_tag}"
    s_dir_ds = out_root / f"{pred_name}_soma_filtered{ds_tag}" if ds_xy > 1 else None

    s_json = s_dir / "soma_hdbscan.json"
    p3d_dir = out_root / f"{pred_name}_post3d{up_tag}"
    p3d_dir_ds = out_root / f"{pred_name}_post3d{ds_tag}" if ds_xy > 1 else None
    f2_dir = out_root / f"{pred_name}_fill_2d{up_tag}"
    f2_dir_ds = out_root / f"{pred_name}_fill_2d{ds_tag}" if ds_xy > 1 else None
    s_dir.mkdir(parents=True, exist_ok=True)
    topk_df, collect_op_sec = collect_topk_dataframe(
        pred_paths,
        top_k=args.top_k,
        image_dir=args.image_dir,
        pred_suffix=args.pred_suffix,
        downsample_xy=args.downsample_xy,
        return_op_time=True,
    )
    _, hdbscan_op_sec = run_hdbscan(
        topk_df,
        str(s_json),
        z_scale=args.z_scale,
        min_cluster_size=args.min_cluster_size,
        best_cluster_mode=args.best_cluster_mode,
        return_op_time=True,
    )
    timing["collect_topk_op_sec"] = collect_op_sec
    timing["hdbscan_op_sec"] = hdbscan_op_sec
    timing["collect_topk_hdbscan_sec"] = collect_op_sec + hdbscan_op_sec
    g_c, lk = build_center_lookup(str(s_json))
    merge_out_dir = args.merge_cc_out_dir
    if merge_out_dir:
        merge_out_dir = str(merge_out_dir)
    timing["soma_filter_sec"] = filter_cc_by_center(
        pred_paths,
        str(s_dir),
        g_c,
        lk,
        rel_area_ratio=args.rel_area_ratio,
        rel_dist_ratio=args.rel_dist_ratio,
        image_dir=args.image_dir,
        pred_suffix=args.pred_suffix,
        single_target_protect=args.single_target_protect,
        merge_radius=args.merge_cc_radius,
        merge_out_dir=merge_out_dir,
        cc_select_mode=args.cc_select_mode,
        downsample_xy=args.downsample_xy,
        out_dir_ds=str(s_dir_ds) if s_dir_ds else None,
        return_op_time=True,
    )
    topk_map_centers = True
    if args.no_topk_map_centers:
        topk_map_centers = False
    elif args.topk_map_centers:
        topk_map_centers = True
    files_s = sorted(s_dir.glob(args.pattern))
    sample = tifffile.imread(files_s[0])
    h, w = sample.shape
    th, tw = h // args.downsample_xy, w // args.downsample_xy
    tmp_vol = tempfile.NamedTemporaryFile(delete=False, suffix=".vol.dat")
    tmp_vol.close()
    vol = np.memmap(tmp_vol.name, dtype='uint8', mode='w+', shape=(len(files_s), th, tw))
    volume_prepare_op_sec = 0.0
    for i, f in enumerate(files_s):
        pred = read_pred_with_image_mask(f, image_dir=args.image_dir, pred_suffix=args.pred_suffix)
        t_op = time.perf_counter()
        pred_ds = transform.resize(pred, (th, tw), order=0, preserve_range=True, anti_aliasing=False)
        volume_prepare_op_sec += time.perf_counter() - t_op
        vol[i] = pred_ds
    timing["volume_prepare_op_sec"] = volume_prepare_op_sec
    if args.single_slice_artifact_suppression:
        slice_names = [x.name for x in files_s]
        scaled_min_area, scaled_support_radius = scale_single_slice_params_for_downsample(
            args.single_slice_min_area,
            args.single_slice_support_radius,
            args.downsample_xy,
        )
        if args.single_slice_debug:
            print(
                "single_slice_artifact_suppression params:"
                f" original_min_area={int(args.single_slice_min_area)}"
                f" scaled_min_area={int(scaled_min_area)}"
                f" original_support_radius={int(args.single_slice_support_radius)}"
                f" scaled_support_radius={int(scaled_support_radius)}"
                f" downsample_xy={int(args.downsample_xy)}"
            )
        cleaned_vol, suppression_op_sec = suppress_single_slice_artifacts_in_memory(
            vol,
            fg_label=1,
            area_ratio=args.single_slice_area_ratio,
            support_radius=scaled_support_radius,
            min_area=scaled_min_area,
            debug_print=args.single_slice_debug,
            slice_names=slice_names,
            return_op_time=True,
        )
        vol[:] = cleaned_vol.astype(vol.dtype, copy=False)
        timing["single_slice_artifact_suppression_sec"] = suppression_op_sec
    l_m = None
    m_p = None
    l_p = None
    try:
        l_m, k_l, m_p, l_p, topk_3d_label_op_sec = compute_topk_labels_memmap(
            vol, args.topk_3d, 0.5, out_root, return_op_time=True
        )
        timing["topk_3d_label_sec"] = topk_3d_label_op_sec
        if topk_map_centers:
            post3d_write_op_sec = apply_topk_centers_to_slices(
                [str(x) for x in files_s],
                str(p3d_dir),
                build_centers_from_labels(l_m, k_l, args.downsample_xy, 1, len(files_s)),
                name_suffix="",
                ds_tag=ds_tag,
                image_dir=args.image_dir,
                pred_suffix=args.pred_suffix,
                return_op_time=True,
            )
            if p3d_dir_ds:
                post3d_write_op_sec += downsample_mask_dir(
                    str(p3d_dir), str(p3d_dir_ds), args.pattern, 'uint8', args.downsample_xy, return_op_time=True
                )
            timing["post3d_write_sec"] = post3d_write_op_sec
            if args.fill_2d:
                if p3d_dir_ds and f2_dir_ds:
                    timing["fill_2d_sec"] = (
                        fill_holes_2d(
                            str(p3d_dir_ds), str(f2_dir_ds), args.pattern, 'uint8',
                            image_dir=None, pred_suffix=args.pred_suffix, return_op_time=True
                        )
                        + upsample_mask_dir(
                            str(f2_dir_ds), str(f2_dir), args.pattern, 'uint8', (h, w), return_op_time=True
                        )
                    )
                else:
                    timing["fill_2d_sec"] = fill_holes_2d(
                        str(p3d_dir), str(f2_dir), args.pattern, 'uint8',
                        image_dir=args.image_dir, pred_suffix=args.pred_suffix, return_op_time=True
                    )
        else:
            post3d_write_op_sec = 0.0
            os.makedirs(p3d_dir, exist_ok=True)
            if p3d_dir_ds:
                os.makedirs(p3d_dir_ds, exist_ok=True)
            keep_labels = np.array(sorted(list(k_l)), dtype=np.int32)
            for z in range(l_m.shape[0]):
                t_op = time.perf_counter()
                mask = np.isin(l_m[z], keep_labels)
                out_ds = (mask.astype(np.uint8) * 1)
                out = out_ds
                if args.downsample_xy and int(args.downsample_xy) > 1:
                    out = transform.resize(out, (h, w), order=0, preserve_range=True,
                                           anti_aliasing=False).astype(np.uint8)
                if p3d_dir_ds:
                    base = os.path.basename(files_s[z])
                    out_ds_path = os.path.join(p3d_dir_ds, base)
                base = os.path.basename(files_s[z])
                stem, ext = os.path.splitext(base)
                stem = strip_ds_suffix(stem, ds_tag)
                out_name = f"{stem}{ext}"
                post3d_write_op_sec += time.perf_counter() - t_op
                if p3d_dir_ds:
                    tifffile.imwrite(out_ds_path, out_ds)
                tifffile.imwrite(os.path.join(p3d_dir, out_name), out)
            timing["post3d_write_sec"] = post3d_write_op_sec
            if args.fill_2d:
                if p3d_dir_ds and f2_dir_ds:
                    timing["fill_2d_sec"] = (
                        fill_holes_2d(
                            str(p3d_dir_ds), str(f2_dir_ds), args.pattern, 'uint8',
                            image_dir=None, pred_suffix=args.pred_suffix, return_op_time=True
                        )
                        + upsample_mask_dir(
                            str(f2_dir_ds), str(f2_dir), args.pattern, 'uint8', (h, w), return_op_time=True
                        )
                    )
                else:
                    timing["fill_2d_sec"] = fill_holes_2d(
                        str(p3d_dir), str(f2_dir), args.pattern, 'uint8',
                        image_dir=args.image_dir, pred_suffix=args.pred_suffix, return_op_time=True
                    )

    finally:
        try:
            vol.flush()
        except Exception as e:
            print(f"Warning: failed to flush memmap volume: {e}")
        try:
            del vol
        except Exception as e:
            print(f"Warning: failed to release memmap volume: {e}")
        for p in [tmp_vol.name, m_p, l_p]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception as e:
                    print(f"Warning: failed to remove temp file {p}: {e}")

    run_elapsed = time.perf_counter() - run_start
    timing["total_sec"] = run_elapsed
    final_output_dir = f2_dir if args.fill_2d else p3d_dir
    metric_outputs = {}

    if args.compute_metrics:
        if not args.gt_dir:
            raise ValueError("--gt_dir is required when --compute_metrics is enabled")

        metrics_targets = [("soma_filtered", s_dir)]
        metrics_targets.append(("post3d", p3d_dir))
        if args.fill_2d:
            metrics_targets.append(("fill_2d", f2_dir))

        for label, out_dir in metrics_targets:
            csv_path = Path(out_dir) / "metrics.csv"
            compute_metrics_csv(
                pred_dir=out_dir,
                gt_dir=args.gt_dir,
                csv_out=csv_path,
                num_workers=args.metrics_num_workers,
                no_weighted=args.metrics_no_weighted,
            )
            metric_outputs[label] = str(csv_path)

    if args.save_params_json:
        params_path = Path(args.params_json_path) if args.params_json_path else Path(final_output_dir) / "postprocess_run.json"
        params = {}
        for k, v in vars(args).items():
            params[k] = str(v) if isinstance(v, Path) else v
        params["num_input_slices"] = len(pred_paths)
        params["total_time_sec"] = run_elapsed
        params["timing_sec"] = timing
        outputs = {
            "soma_json": str(s_json),
            "soma_filtered": str(s_dir),
            "post3d": str(p3d_dir),
        }
        if args.fill_2d:
            outputs["fill_2d"] = str(f2_dir)
        if metric_outputs:
            outputs["metrics_csv"] = metric_outputs
        params["outputs"] = outputs
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with params_path.open("w") as f:
            json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
