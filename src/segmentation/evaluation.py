"""
SegFormer Evaluation Script (2D)

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-04-02
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from glob import glob

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import torch
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from segmentation.models.native_unet import NativeUNet

PATCH_COORD_CACHE = {}
WEIGHT_MAP_CACHE = {}


# Utils
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


def build_patch_coords(H, W, patch_size, overlap):
    cache_key = (int(H), int(W), int(patch_size), float(overlap))
    if cache_key in PATCH_COORD_CACHE:
        return PATCH_COORD_CACHE[cache_key]

    stride = int(patch_size * (1 - overlap))
    nH = (H - patch_size + stride - 1) // stride + 1
    nW = (W - patch_size + stride - 1) // stride + 1

    coords = []
    for i in range(nH):
        for j in range(nW):
            y0 = i * stride
            x0 = j * stride
            y1 = min(y0 + patch_size, H)
            x1 = min(x0 + patch_size, W)
            y0 = max(y1 - patch_size, 0)
            x0 = max(x1 - patch_size, 0)
            coords.append((y0, y1, x0, x1))

    PATCH_COORD_CACHE[cache_key] = coords
    return coords


def build_weight_map(H, W, patch_size, overlap):
    cache_key = (int(H), int(W), int(patch_size), float(overlap))
    cached = WEIGHT_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    coords = build_patch_coords(H, W, patch_size, overlap)
    weight_map = np.zeros((H, W), dtype=np.float32)
    for y0, y1, x0, x1 in coords:
        weight_map[y0:y1, x0:x1] += 1.0

    weight_map = np.maximum(weight_map, 1e-8)
    WEIGHT_MAP_CACHE[cache_key] = weight_map
    return weight_map

# Metrics Helpers
try:
    # Try importing local metrics library
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from segmentation.metrics import compute_binary_metrics
except ImportError:
    print("Warning: utils.metrics not found. Using internal metric functions.")
    from sklearn.metrics import jaccard_score


    def dice_coefficient(y_true, y_pred):
        inter = np.sum(y_true * y_pred)
        return (2. * inter) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)


    def compute_binary_metrics(gt_mask, pred_mask, compute_weighted=True):
        y_true = (gt_mask == 1).astype(np.uint8).ravel()
        y_pred = (pred_mask == 1).astype(np.uint8).ravel()
        dice = dice_coefficient(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, average="binary")
        return {"dice": dice, "iou": iou, "weighted_dice": 0.0, "weighted_dice_loss": 0.0}



def save_results_csv(results, output_path):
    import csv
    if not results: return
    with open(output_path, 'w', newline='') as f:
        first_metrics = results[0][1]
        if not first_metrics: return
        columns = ['filename'] + list(first_metrics.keys())
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for fname, metrics in results:
            row = {'filename': fname}
            row.update(metrics)
            writer.writerow(row)
    print(f"Metrics saved to: {output_path}")


# Model Builder

def build_model(weights_path, arch_type="unet", encoder_name="mit_b2", device="cuda", input_size=224):
    print(f"Building {arch_type.upper()} with Encoder: {encoder_name}...")

    if arch_type == "unet":
        # MiT + U-Net Decoder
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=2,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type="scse"
        )
    elif arch_type == "native-unet":
        model = NativeUNet(in_channels=3, num_classes=2)
    elif arch_type == "original":
        # MiT + MLP Decoder
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
    elif arch_type == "swin-unet":
        swin_unet_repo = os.environ.get("SWIN_UNET_REPO", "./Swin-Unet")
        if not os.path.isdir(swin_unet_repo):
            raise FileNotFoundError(f"Swin-Unet repo not found: {swin_unet_repo}")
        if swin_unet_repo not in sys.path:
            sys.path.insert(0, swin_unet_repo)

        from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

        model = SwinTransformerSys(
            img_size=input_size,
            patch_size=4,
            in_chans=3,
            num_classes=2,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first",
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    if arch_type == "swin-unet":
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


# Inference
def inference_sliding_window(model,
                             img_curr,
                             out_path,
                             patch_size=224,
                             batch_size=32,
                             precision="float32",
                             device="cuda",
                             gt_mask=None,
                             overlap=0.5,
                             mean=None,
                             std=None,
                             skip_norm=False,
                             per_image_zscore=False,
                             zscore_stats_min=None):
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]

    H, W = img_curr.shape

    # Resize GT if necessary
    if gt_mask is not None and gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Optional per-image z-score (ignoring values <= zscore_stats_min)
    if per_image_zscore:
        z = zscore(img_curr, stats_min=zscore_stats_min)
        if z is None:
            # fallback to zeros if std too small
            img_curr = np.zeros_like(img_curr, dtype=np.float32)
        else:
            img_curr = z

    if skip_norm or per_image_zscore:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    coords = build_patch_coords(H, W, patch_size, overlap)

    # Keep the legacy aggregation path: accumulate both class probabilities
    # and recover the mask with argmax after overlap averaging.
    prob_map = np.zeros((2, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    batch_size = max(1, int(batch_size))
    use_fp16 = precision == "float16" and str(device).startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        batch_tensors = []
        batch_coords = []

        def flush_batch():
            if not batch_tensors:
                return

            patch_batch = torch.stack(batch_tensors, dim=0).to(device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_fp16 else nullcontext()
            )
            with autocast_ctx:
                output = model(patch_batch)
            probs_batch = torch.softmax(output, dim=1).cpu().numpy()

            for probs, (yy0, yy1, xx0, xx1) in zip(probs_batch, batch_coords):
                prob_map[:, yy0:yy1, xx0:xx1] += probs
                weight_map[yy0:yy1, xx0:xx1] += 1.0

            batch_tensors.clear()
            batch_coords.clear()

        for y0, y1, x0, x1 in coords:
                p_curr = img_curr[y0:y1, x0:x1].astype(np.float32)

                if (not skip_norm and not per_image_zscore) and p_curr.max() > 1:
                    p_curr /= 255.0

                # Replicate 2D slice to 3 channels for models trained on 3-channel input
                patch_stack = np.stack([p_curr, p_curr, p_curr], axis=-1)
                patch_tensor = transform(patch_stack)
                batch_tensors.append(patch_tensor)
                batch_coords.append((y0, y1, x0, x1))
                if len(batch_tensors) >= batch_size:
                    flush_batch()

        flush_batch()

    # Average
    weight_map = np.maximum(weight_map, 1e-8)
    prob_map /= weight_map[None, :, :]

    # Legacy binary decision path from the original commit.
    pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)

    # Save raw class indices (0..num_classes-1)
    tifffile.imwrite(out_path, pred_mask, compression=None)

    if gt_mask is None:
        return {}

    return compute_binary_metrics(gt_mask, pred_mask, compute_weighted=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SegFormer Evaluation (2D Only)")
    parser.add_argument("--model_arch", type=str, default="unet",
                        choices=["unet", "native-unet", "original", "swin-unet"],
                        help="Architecture type used in training: 'unet' (MiT+U-Net), 'native-unet', 'original' (MiT+MLP), or 'swin-unet'")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Root dir containing 'images' and 'masks' folders")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--encoder_name", type=str, default="mit_b2", help="Must match training (mit_b2, mit_b5 etc.)")
    parser.add_argument("--patch_size", type=int, default=224, help="Inference patch size (usually same as training)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sliding-window patches per forward pass during inference.")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"],
                        help="Inference precision. float16 enables CUDA autocast.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mean", nargs="+", type=float, default=[0.5, 0.5, 0.5],
                        help="Normalization Mean (space separated)")
    parser.add_argument("--std", nargs="+", type=float, default=[0.5, 0.5, 0.5],
                        help="Normalization Std (space separated)")
    parser.add_argument("--skip_norm", action="store_true",
                        help="Skip Normalize(mean,std); assume inputs already z-scored")
    parser.add_argument("--per_image_zscore", action="store_true",
                        help="Apply per-image z-score on inputs before inference.")
    parser.add_argument("--zscore_stats_min", type=float, default=None,
                        help="Exclude values <= this when computing z-score mean/std (ignores background).")

    parser.add_argument("--csv_out", type=str, default=None, help="Save metrics to CSV")

    args = parser.parse_args()

    # Setup Dirs
    image_dir = os.path.join(args.data_dir, "images")
    mask_dir = os.path.join(args.data_dir, "masks")
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")

    print("=" * 40)
    print("SegFormer Eval Configuration:")
    print(f"Model: {args.model_path}")
    print(f"Patch Size: {args.patch_size}")
    print(f"Inference Batch Size: {args.batch_size}")
    print(f"Inference Precision: {args.precision}")
    if args.per_image_zscore:
        print(f"Per-image Z-Score: ENABLED (stats_min={args.zscore_stats_min})")
    elif args.skip_norm:
        print("Normalization: SKIPPED (pre-zscore inputs)")
    else:
        print(f"Normalization Mean: {args.mean}")
        print(f"Normalization Std:  {args.std}")
    print("=" * 40)

    # Load Image List
    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")) + glob(os.path.join(image_dir, "*.tiff")))
    print(f"Found {len(image_paths)} images.")

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    # Load Model
    model = build_model(
        weights_path=args.model_path,
        arch_type=args.model_arch,
        encoder_name=args.encoder_name,
        device=args.device,
        input_size=args.patch_size,
    )

    # Inference Loop
    results = []
    slice_times = []
    run_start = time.perf_counter()

    for idx, img_path in enumerate(tqdm(image_paths, desc="Evaluating (2d)")):
        fname = os.path.basename(img_path)
        start_time = time.perf_counter()

        # Load Current Image
        img_curr = tifffile.imread(img_path)

        # Load GT Mask if exists
        gt_mask = None
        if mask_dir is not None:
            # Align mask naming with RETINA pattern: mask<fname[5:]> primary, fallback to replace
            candidate_masks = [
                os.path.join(mask_dir, f"mask{fname[5:]}"),
                os.path.join(mask_dir, fname.replace("image", "mask"))
            ]
            for gt_path in candidate_masks:
                if os.path.exists(gt_path):
                    gt_mask = tifffile.imread(gt_path)
                    break

        # Output Path
        base, ext = os.path.splitext(fname)
        out_path = os.path.join(args.out_dir, f"{base}_pred_2d{ext}")

        # Run Inference with Custom Mean/Std
        metrics = inference_sliding_window(
            model,
            img_curr,
            out_path,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            precision=args.precision,
            device=args.device,
            gt_mask=gt_mask,
            overlap=args.overlap,
            mean=args.mean,
            std=args.std,
            skip_norm=args.skip_norm,
            per_image_zscore=args.per_image_zscore,
            zscore_stats_min=args.zscore_stats_min
        )

        results.append((fname, metrics))
        elapsed = time.perf_counter() - start_time
        slice_times.append((fname, elapsed))
        print(f"{fname}: {elapsed:.3f}s")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    if results and results[0][1]:
        metrics_keys = ["dice", "iou", "weighted_dice", "weighted_dice_loss"]
        for k in metrics_keys:
            vals = [m[k] for _, m in results if k in m]
            if vals:
                print(f"Mean {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nPredictions saved to: {args.out_dir}")
    if args.csv_out:
        save_results_csv(results, args.csv_out)
    if slice_times:
        times_only = [t for _, t in slice_times]
        print(f"\nPer-slice time: mean={np.mean(times_only):.3f}s ± {np.std(times_only):.3f}s")

    run_elapsed = time.perf_counter() - run_start
    run_meta = {
        "args": vars(args),
        "num_images": len(image_paths),
        "total_time_sec": run_elapsed,
        "per_slice_time_sec": [{ "file": f, "time_sec": t } for f, t in slice_times],
    }
    meta_path = os.path.join(args.out_dir, "eval_run.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata saved to: {meta_path}")
