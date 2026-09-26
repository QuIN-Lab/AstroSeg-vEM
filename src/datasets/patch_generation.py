"""
Generate 2D TIFF training/reconstruction patches from slice images with optional
2.5D context, per-patch z-score, foreground-aware sampling, and metadata export.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2026-01-03
"""

import argparse
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

import numpy as np
import tifffile as tiff
from tqdm import tqdm


def extract_patch(arr, y, x, size):
    return np.asarray(arr[y:y + size, x:x + size], dtype=np.uint8)


def zscore(arr, stats_min=None):
    arr = arr.astype(np.float32)

    if stats_min is not None:
        mask = arr > stats_min
        if not np.any(mask):
            return None
        mean = float(arr[mask].mean())
        std = float(arr[mask].std())
    else:
        mean = float(arr.mean())
        std = float(arr.std())

    if std < 1e-6:
        return None

    return (arr - mean) / std


def to_uint8_view(arr, clip_min=-3.0, clip_max=3.0):
    arr_clipped = np.clip(arr, clip_min, clip_max)
    arr_norm = (arr_clipped - clip_min) / (clip_max - clip_min)
    return np.round(arr_norm * 255).astype(np.uint8)


def process_one_slice(
    i,
    img_paths,
    mask_paths,
    out_dir,
    patch_size,
    patches_per_image,
    per_image_zscore=False,
    zscore_stats_min=None,
    save_uint8_view=False,
    fg_sample_prob=0.3,
):
    base_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    slice_out_dir = os.path.join(out_dir, base_name)
    os.makedirs(slice_out_dir, exist_ok=True)

    with tiff.TiffFile(img_paths[i]) as tif:
        img_curr = tif.asarray(out="memmap")

    mask = None
    has_fg = False
    ys, xs = [], []

    if mask_paths is not None:
        with tiff.TiffFile(mask_paths[i]) as tif:
            mask = tif.asarray(out="memmap")
        mask = (mask == 1).astype(np.uint8)
        H, W = img_curr.shape
        ys, xs = np.nonzero(mask)
        has_fg = len(ys) > 0
    else:
        H, W = img_curr.shape

    k = 0
    attempts = 0
    max_attempts = patches_per_image * 8
    fg_fractions = []
    all_bg_count = 0

    while k < patches_per_image and attempts < max_attempts:
        attempts += 1
        fg_sampled = False
        if has_fg and np.random.rand() < fg_sample_prob:
            fg_idx = np.random.randint(len(ys))
            cy, cx = ys[fg_idx], xs[fg_idx]
            fg_sampled = True
        else:
            cy, cx = np.random.randint(H), np.random.randint(W)
        center_y, center_x = cy, cx

        y = int(np.clip(center_y - patch_size // 2, 0, H - patch_size))
        x = int(np.clip(center_x - patch_size // 2, 0, W - patch_size))

        p_curr = extract_patch(img_curr, y, x, patch_size)
        p_mask = extract_patch(mask, y, x, patch_size) if mask is not None else None

        bg_ratio = np.count_nonzero(p_curr < 10) / p_curr.size
        if bg_ratio > 0.1: continue

        # Simple foreground filter: drop all-zero or all-one masks most of the time.
        if p_mask is not None:
            uniq = np.unique(p_mask)
            if uniq.size == 1 and (uniq[0] == 0 or uniq[0] == 1):
                # keep 5% of extremes to retain some easy examples
                if np.random.rand() > 0.05:
                    continue

        if per_image_zscore:
            z_curr = zscore(p_curr, stats_min=zscore_stats_min)
            if z_curr is None: continue
            p_curr = z_curr
            if save_uint8_view: p_curr_view = to_uint8_view(p_curr)

        patch_dir = os.path.join(slice_out_dir, f"patch_{k:02d}")
        os.makedirs(patch_dir, exist_ok=True)

        tiff.imwrite(os.path.join(patch_dir, "image.tif"), p_curr, dtype=np.float32 if per_image_zscore else np.uint8, compression="zlib", tile=(256, 256), contiguous=True)
        if p_mask is not None:
            tiff.imwrite(os.path.join(patch_dir, "mask.tif"), p_mask, dtype=np.uint8, compression="zlib", tile=(256, 256), contiguous=True)
        if per_image_zscore and save_uint8_view:
            tiff.imwrite(os.path.join(patch_dir, "image_view.tif"), p_curr_view, dtype=np.uint8, compression="zlib", tile=(256, 256), contiguous=True)

        fg_fraction = float(p_mask.mean()) if p_mask is not None else None
        mask_bg_ratio = (1.0 - fg_fraction) if fg_fraction is not None else None
        patch_meta = {
            "slice_index": i,
            "slice_name": base_name,
            "source_path": img_paths[i],
            "mask_path": mask_paths[i] if mask_paths is not None else None,
            "context_mode": "2d",
            "patch_size": patch_size,
            "bbox": {"x": x, "y": y, "width": patch_size, "height": patch_size},
            "center": {"x": int(center_x), "y": int(center_y)},
            "fg_sample_prob": fg_sample_prob if has_fg else 0.0,
            "sampled_from_fg": fg_sampled,
            "mask_bg_ratio": mask_bg_ratio,
            "has_foreground": bool(p_mask is not None and np.any(p_mask)),
            "fg_fraction": fg_fraction,
        }

        with open(os.path.join(patch_dir, "patch_info.json"), "w") as f:
            json.dump(patch_meta, f, indent=2)
        k += 1
        if fg_fraction is not None:
            fg_fractions.append(fg_fraction)
            if fg_fraction == 0.0:
                all_bg_count += 1

    del img_curr, mask
    gc.collect()
    if fg_fractions:
        fg_arr = np.array(fg_fractions, dtype=np.float32)
        return {
            "count": int(fg_arr.size),
            "sum": float(fg_arr.sum()),
            "sum_sq": float((fg_arr ** 2).sum()),
            "min": float(fg_arr.min()),
            "max": float(fg_arr.max()),
            "all_bg": int(all_bg_count),
        }
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
        "all_bg": 0,
    }

def generate_patches_tif(
        image_dir, mask_dir, out_dir,
        patch_size=512, patches_per_image=10,
        per_image_zscore=False, zscore_stats_min=None, save_uint8_view=False,
        fg_sample_prob=0.3
):
    print("\n" + "="*50)
    print("      Astrocyte Patch Generation Settings      ")
    print("="*50)
    print(f"{'Image Directory':<20}: {image_dir}")
    print(f"{'Mask Directory':<20}: {mask_dir if mask_dir else 'None (Reconstruction Mode)'}")
    print(f"{'Output Directory':<20}: {out_dir}")
    print("-" * 50)
    print(f"{'Patch Size':<20}: {patch_size} x {patch_size}")
    print(f"{'Patches / Image':<20}: {patches_per_image}")
    print("-" * 50)
    print(f"{'Per-Image Z-Score':<20}: {'ENABLED' if per_image_zscore else 'DISABLED'}")
    if per_image_zscore:
        stat_msg = f"Values <= {zscore_stats_min} ignored" if zscore_stats_min is not None else "All values used"
        print(f"{'Z-Score Stats':<20}: {stat_msg}")
        print(f"{'Save uint8 View':<20}: {'YES' if save_uint8_view else 'NO'}")
    print("="*50 + "\n")
    # =================================

    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(image_dir, "*.tif*")))

    if mask_dir and mask_dir.lower() != "none":
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif*")))
        assert len(img_paths) == len(mask_paths), "Mismatch between image/mask counts!"
        print("Mode: Training Generation (Masks provided)")
    else:
        mask_paths = None
        print("Mode: Reconstruction Generation (No Masks, Random Sampling)")

    num_slices = len(img_paths)
    print(f"Found {num_slices} slices — generating {patches_per_image} patches per slice.")

    meta = {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "out_dir": out_dir,
        "patch_size": patch_size,
        "patches_per_image": patches_per_image,
        "per_image_zscore": per_image_zscore,
        "zscore_stats_min": zscore_stats_min,
        "save_uint8_view": save_uint8_view,
        "fg_sample_prob": fg_sample_prob,
        "tiff_tiled": True,
        "compression": "zlib",
        "num_slices": num_slices
    }

    process_func = partial(
        process_one_slice,
        img_paths=img_paths,
        mask_paths=mask_paths,
        out_dir=out_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        per_image_zscore=per_image_zscore,
        zscore_stats_min=zscore_stats_min,
        save_uint8_view=save_uint8_view,
        fg_sample_prob=fg_sample_prob
    )

    stats = {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
        "all_bg": 0,
    }

    with ProcessPoolExecutor(max_workers=2) as executor:
        for result in tqdm(
            executor.map(process_func, range(num_slices)),
            total=num_slices,
            desc="Processing slices (parallel)"
        ):
            if not result:
                continue
            stats["count"] += result["count"]
            stats["sum"] += result["sum"]
            stats["sum_sq"] += result["sum_sq"]
            stats["all_bg"] += result["all_bg"]
            if result["min"] is not None:
                stats["min"] = result["min"] if stats["min"] is None else min(stats["min"], result["min"])
            if result["max"] is not None:
                stats["max"] = result["max"] if stats["max"] is None else max(stats["max"], result["max"])

    meta_path = os.path.join(out_dir, "patch_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    if stats["count"] > 0:
        mean = stats["sum"] / stats["count"]
        var = max(0.0, (stats["sum_sq"] / stats["count"]) - (mean ** 2))
        std = float(np.sqrt(var))
        all_bg_ratio = stats["all_bg"] / stats["count"]
        print("\nFG Fraction Summary")
        print(f"  mean={mean:.4f}, std={std:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
              f"n={stats['count']}, all_bg_ratio={all_bg_ratio:.4f}")
    else:
        print("\nFG Fraction Summary: n=0 (no masks)")

    print("\nPatch generation complete.")
    print(f"Patches saved under: {out_dir}")
    print(f"Metadata saved to:   {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2D tiled TIFF patches (Unified Train/Recon)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to cropped images directory")
    parser.add_argument("--mask_dir", type=str, default=None, help="Path to cropped masks directory (Optional)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for patches")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size (default: 512)")
    parser.add_argument("--patches_per_image", type=int, default=10, help="Number of patches per image")
    parser.add_argument("--per_image_zscore", action="store_true", help="Apply per-patch z-score")
    parser.add_argument("--zscore_stats_min", type=float, default=None,
                        help="Exclude values <= this threshold when calculating mean/std (e.g., set 0 to ignore background)")

    parser.add_argument("--save_uint8_view", action="store_true", help="Save extra uint8 view")
    parser.add_argument("--fg_sample_prob", type=float, default=0.3,
                        help="Probability of sampling a foreground-centered patch when foreground exists.")

    args = parser.parse_args()

    generate_patches_tif(
        args.image_dir,
        args.mask_dir,
        args.out_dir,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        per_image_zscore=args.per_image_zscore,
        zscore_stats_min=args.zscore_stats_min,
        save_uint8_view=args.save_uint8_view,
        fg_sample_prob=args.fg_sample_prob,
    )
