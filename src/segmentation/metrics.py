"""
Metrics computation for segmentation evaluation.
Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2025-11-15
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import jaccard_score


def dice_coefficient(y_true, y_pred):
    inter = np.sum(y_true * y_pred)
    return (2. * inter) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def precision_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8).ravel()
    y_pred = (y_pred > 0).astype(np.uint8).ravel()

    tp = np.sum((y_true == 1) & (y_pred == 1), dtype=np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1), dtype=np.float64)
    return float(tp / (tp + fp + 1e-8))


def recall_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8).ravel()
    y_pred = (y_pred > 0).astype(np.uint8).ravel()

    tp = np.sum((y_true == 1) & (y_pred == 1), dtype=np.float64)
    fn = np.sum((y_true == 1) & (y_pred == 0), dtype=np.float64)
    return float(tp / (tp + fn + 1e-8))


def distance_weighted_dice_tiled(pred, gt, alpha=0.10, tile=512):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    H, W = gt.shape
    num = 0.0
    den = 0.0

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)

            pred_blk = pred[y:y1, x:x1].astype(np.float32)
            gt_blk = gt[y:y1, x:x1].astype(np.float32)

            dist = distance_transform_edt(gt_blk == 0)
            w = np.exp(-alpha * dist)

            num += np.sum(w * pred_blk * gt_blk)
            den += np.sum(w * pred_blk) + np.sum(w * gt_blk)

    return (2.0 * num) / (den + 1e-8)


def distance_weighted_dice_full(pred, gt, alpha=0.10):
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    dist = distance_transform_edt(gt == 0)
    w = np.exp(-alpha * dist).astype(np.float32)

    num = np.sum(w * pred * gt, dtype=np.float64)
    den = np.sum(w * pred, dtype=np.float64) + np.sum(w * gt, dtype=np.float64)
    return float((2.0 * num) / (den + 1e-8))


def per_class_metrics(gt_mask, pred_mask, num_classes):
    metrics = {}
    dices, ious = [], []

    for c in range(1, num_classes):
        y_true = (gt_mask == c).astype(np.uint8).ravel()
        y_pred = (pred_mask == c).astype(np.uint8).ravel()

        dice = dice_coefficient(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, average='binary')

        metrics[f"class_{c}_dice"] = dice
        metrics[f"class_{c}_iou"] = iou
        dices.append(dice)
        ious.append(iou)

    metrics["mean_dice"] = np.mean(dices) if dices else 0.0
    metrics["mean_iou"] = np.mean(ious) if ious else 0.0
    return metrics


def compute_binary_metrics(gt_mask, pred_mask, compute_weighted=True):
    y_true = (gt_mask == 1).astype(np.uint8).ravel()
    y_pred = (pred_mask == 1).astype(np.uint8).ravel()

    dice = dice_coefficient(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0.0)
    loss = dice_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    metrics = {
        "dice": dice,
        "iou": iou,
        "loss": loss,
        "precision": precision,
        "recall": recall,
    }

    if compute_weighted:
        w_dice = distance_weighted_dice_full(pred_mask == 1, gt_mask == 1)
        metrics["weighted_dice"] = w_dice
        metrics["weighted_dice_loss"] = 1.0 - w_dice

    return metrics
