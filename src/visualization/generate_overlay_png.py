"""
Astrocyte Segmentation Visualization Script (single slice)

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-11-20
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm


# Resize Helper
def resize_to_small(img, max_size=1024, is_mask=False):
    H, W = img.shape[:2]
    scale = max_size / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    if is_mask:
        img_resized = resize(img, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False)
        return (img_resized > 0.5).astype(np.uint8)   # ensure binary 0/1
    else:
        return resize(img, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)

# Overlay Helper
def make_overlay(img, mask, color=(1.0, 0.2, 0.2), alpha=0.5):
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    rgb_img = gray2rgb(img_norm)
    overlay = rgb_img.copy()
    overlay[mask == 1] = color
    blended = (1 - alpha) * rgb_img + alpha * overlay
    return blended

def _process_one(img_path, mask_dir, out_dir, max_size=1024, color=(1.0, 0.2, 0.2), alpha=0.5):
    # Load original image
    img = imread(img_path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]  # grayscale

    # Load mask
    fname = os.path.basename(img_path).split(".")[0]
    mask_name = f"{fname}_pred_2d.tif"
    # mask_name = f"mask_{fname[6:]}.tif"
    mask_path = os.path.join(mask_dir, mask_name)
    if not os.path.exists(mask_path):
        return (fname, False, "No mask found, skipping")

    mask = imread(mask_path).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)  # ensure binary

    # Resize with safe handling
    img_small = resize_to_small(img, max_size=max_size, is_mask=False)
    mask_small = resize_to_small(mask, max_size=max_size, is_mask=True)

    # Create overlay
    overlay = make_overlay(img_small, mask_small, color=color, alpha=alpha)

    # Save overlay only
    out_path = os.path.join(out_dir, f"{fname}_overlay.png")
    imsave(out_path, (overlay * 255).astype(np.uint8))
    return (fname, True, None)


def generate_overlay_png(image_dir, mask_dir, out_dir, num_workers=1):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(
        glob(os.path.join(image_dir, "*.tif")) +
        glob(os.path.join(image_dir, "*.tiff"))
    )
    if not image_paths:
        raise ValueError(f"No .tif/.tiff images found in: {image_dir}")

    if num_workers <= 1:
        for img_path in tqdm(image_paths, desc="Making overlays"):
            fname, ok, msg = _process_one(img_path, mask_dir, out_dir)
            if not ok and msg:
                print(f"{msg} for {fname}")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_one, img_path, mask_dir, out_dir)
                for img_path in image_paths
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Making overlays"):
                fname, ok, msg = fut.result()
                if not ok and msg:
                    print(f"{msg} for {fname}")

    print(f"\nOverlays saved to {out_dir} (masks should now always be visible)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="overlay visualizations to compare the predicted segmentation mask with the original image。"
    )

    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root image directory")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Root mask directory")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for visualization")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for overlay generation.")
    args = parser.parse_args()

    generate_overlay_png(args.image_dir, args.mask_dir, args.out_dir, num_workers=args.num_workers)
