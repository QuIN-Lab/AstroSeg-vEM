"""
Compute metrics from existing predictions.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-11-17
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import tifffile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.metrics import compute_binary_metrics, per_class_metrics


def prediction_to_image_stem(pred_filename):
    base_name = os.path.basename(pred_filename)
    stem, _ = os.path.splitext(base_name)
    stem = stem.removesuffix("_pred_2d")
    return stem


def find_matching_gt(pred_filename, gt_dir):
    image_stem = prediction_to_image_stem(pred_filename)
    candidate_names = [
        image_stem.replace("image", "mask", 1) + ".tif",
        image_stem.replace("image", "mask", 1) + ".tiff",
        image_stem.replace("image_", "mask_", 1) + ".tif",
        image_stem.replace("image_", "mask_", 1) + ".tiff",
    ]
    for candidate_name in candidate_names:
        gt_path = os.path.join(gt_dir, candidate_name)
        if os.path.exists(gt_path):
            return gt_path

    return None


def _process_one(pred_path, gt_dir, num_classes, compute_weighted):
    fname = os.path.basename(pred_path)
    gt_path = find_matching_gt(pred_path, gt_dir)
    if gt_path is None:
        return fname, None, "missing_gt"
    try:
        pred_mask = tifffile.imread(pred_path)
        gt_mask = tifffile.imread(gt_path)
        if num_classes == 2:
            metrics = compute_binary_metrics(gt_mask, pred_mask,
                                             compute_weighted=compute_weighted)
        else:
            metrics = per_class_metrics(gt_mask, pred_mask, num_classes)
        return fname, metrics, None
    except Exception as e:
        return fname, None, f"error: {e}"


def compute_metrics_for_predictions(pred_dir, gt_dir, num_classes=2, compute_weighted=True, num_workers=1):
    # Find all prediction files
    pred_files = sorted(glob(os.path.join(pred_dir, "*.tif")) +
                        glob(os.path.join(pred_dir, "*.tiff")))

    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return []

    print(f"Found {len(pred_files)} prediction files")

    results = []
    missing_gt = []

    if num_workers <= 1:
        for pred_path in tqdm(pred_files, desc="Computing metrics"):
            fname, metrics, err = _process_one(pred_path, gt_dir, num_classes, compute_weighted)
            if err == "missing_gt":
                missing_gt.append(fname)
                continue
            if err:
                print(f"Error processing {fname}: {err}")
                continue
            results.append((fname, metrics))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(_process_one, pred_path, gt_dir, num_classes, compute_weighted): pred_path
                for pred_path in pred_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing metrics"):
                fname, metrics, err = fut.result()
                if err == "missing_gt":
                    missing_gt.append(fname)
                    continue
                if err:
                    print(f"Error processing {fname}: {err}")
                    continue
                results.append((fname, metrics))

    # Report missing ground truths
    if missing_gt:
        print(f"\nWarning: {len(missing_gt)} predictions without matching ground truth:")
        for fname in missing_gt[:5]:  # Show first 5
            print(f"{fname}")
        if len(missing_gt) > 5:
            print(f"... and {len(missing_gt) - 5} more")

    return results


def print_summary(results):
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 60)
    print("Metrics Summary")
    print("=" * 60)

    # Check if we have binary classification metrics
    if results[0][1] and "dice" in results[0][1]:
        all_dice = [m["dice"] for _, m in results if "dice" in m]
        all_iou = [m["iou"] for _, m in results if "iou" in m]
        all_precision = [m["precision"] for _, m in results if "precision" in m]
        all_recall = [m["recall"] for _, m in results if "recall" in m]
        all_wdice = [m["weighted_dice"] for _, m in results if "weighted_dice" in m]
        all_wdice_loss = [m["weighted_dice_loss"] for _, m in results if "weighted_dice_loss" in m]

        if all_dice:
            print(f"Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        if all_iou:
            print(f"Mean IoU: {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
        if all_precision:
            print(f"Mean Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
        if all_recall:
            print(f"Mean Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
        if all_wdice:
            print(f"Mean Weighted Dice (full-mask): {np.mean(all_wdice):.4f} ± {np.std(all_wdice):.4f}")
        if all_wdice_loss:
            print(f"Mean Weighted Dice Loss (full-mask): {np.mean(all_wdice_loss):.4f} ± {np.std(all_wdice_loss):.4f}")
    else:
        # Multi-class metrics
        all_mean_dice = [m["mean_dice"] for _, m in results if "mean_dice" in m]
        all_mean_iou = [m["mean_iou"] for _, m in results if "mean_iou" in m]

        if all_mean_dice:
            print(f"Mean Dice: {np.mean(all_mean_dice):.4f} ± {np.std(all_mean_dice):.4f}")
        if all_mean_iou:
            print(f"Mean IoU: {np.mean(all_mean_iou):.4f} ± {np.std(all_mean_iou):.4f}")

    print(f"\nTotal predictions evaluated: {len(results)}")
    print("=" * 60)


def save_results_csv(results, output_path):
    import csv

    if not results:
        print("No results to save")
        return
    results = sorted(results, key=lambda x: x[0])

    with open(output_path, 'w', newline='') as f:
        # Determine columns from first result
        first_metrics = results[0][1]
        columns = ['filename'] + list(first_metrics.keys())

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for fname, metrics in results:
            row = {'filename': fname}
            row.update(metrics)
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics from existing prediction masks"
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction masks")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth masks")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (default: 2 for binary)")
    parser.add_argument("--no_weighted", action="store_true",
                        help="Skip weighted Dice computation (full-mask mode by default)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker threads (default: 1)")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Save results to CSV file (optional)")
    args = parser.parse_args()

    # Compute metrics
    results = compute_metrics_for_predictions(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        num_classes=args.num_classes,
        compute_weighted=not args.no_weighted,
        num_workers=args.num_workers
    )

    # Print summary
    print_summary(results)

    # Save to CSV if requested
    if args.csv_out:
        save_results_csv(results, args.csv_out)
