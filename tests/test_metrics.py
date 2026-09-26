import numpy as np
import pytest

from segmentation.metrics import (
    compute_binary_metrics,
    dice_coefficient,
    dice_loss,
    distance_weighted_dice_full,
    distance_weighted_dice_tiled,
    per_class_metrics,
    precision_score,
    recall_score,
)


def test_dice_perfect_match_is_one():
    y = np.array([[0, 1], [1, 0]])
    assert dice_coefficient(y, y) == pytest.approx(1.0)


def test_dice_no_overlap_is_zero():
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 0], [0, 1]])
    assert dice_coefficient(a, b) == pytest.approx(0.0, abs=1e-6)


def test_dice_known_partial_overlap():

    a = np.array([1, 1, 1, 0, 0])
    b = np.array([0, 1, 1, 1, 0])
    assert dice_coefficient(a, b) == pytest.approx(4 / 6, rel=1e-4)


def test_dice_loss_is_one_minus_dice():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    assert dice_loss(a, b) == pytest.approx(1 - dice_coefficient(a, b))


def test_precision_recall_hand_computed():

    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1])
    assert precision_score(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-4)
    assert recall_score(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-4)


def test_precision_recall_perfect():
    y = np.array([1, 0, 1, 0, 1])
    assert precision_score(y, y) == pytest.approx(1.0)
    assert recall_score(y, y) == pytest.approx(1.0)


def test_weighted_dice_identical_masks_near_one():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[10:20, 10:20] = 1
    w = distance_weighted_dice_full(gt, gt, alpha=0.1)
    assert w == pytest.approx(1.0, rel=1e-3)


def test_weighted_dice_penalizes_nearby_more_than_far(blob_mask):


    shape = (64, 64)
    gt = blob_mask(shape, center=(32, 32), radius=5)

    pred_near = gt.copy()
    pred_near |= blob_mask(shape, center=(32, 40), radius=3)

    pred_far = gt.copy()
    pred_far |= blob_mask(shape, center=(5, 5), radius=3)

    w_near = distance_weighted_dice_full(pred_near, gt, alpha=0.1)
    w_far = distance_weighted_dice_full(pred_far, gt, alpha=0.1)
    assert w_far > w_near


def test_weighted_dice_tiled_matches_full_on_single_tile():
    rng = np.random.default_rng(1)
    gt = (rng.random((40, 40)) > 0.7).astype(np.uint8)
    pred = (rng.random((40, 40)) > 0.7).astype(np.uint8)
    full = distance_weighted_dice_full(pred, gt, alpha=0.1)
    tiled = distance_weighted_dice_tiled(pred, gt, alpha=0.1, tile=512)
    assert tiled == pytest.approx(full, rel=1e-4)


def test_weighted_dice_tiled_close_to_full_with_small_tiles():


    rng = np.random.default_rng(2)
    gt = (rng.random((50, 37)) > 0.6).astype(np.uint8)
    pred = (rng.random((50, 37)) > 0.6).astype(np.uint8)
    full = distance_weighted_dice_full(pred, gt, alpha=0.1)
    tiled = distance_weighted_dice_tiled(pred, gt, alpha=0.1, tile=16)
    assert tiled == pytest.approx(full, abs=0.01)


def test_per_class_metrics_multiclass():
    gt = np.array([[0, 1, 2], [1, 1, 2]])
    pred = np.array([[0, 1, 2], [1, 2, 2]])
    m = per_class_metrics(gt, pred, num_classes=3)
    assert set(m.keys()) == {"class_1_dice", "class_1_iou", "class_2_dice", "class_2_iou", "mean_dice", "mean_iou"}


    assert m["class_1_dice"] == pytest.approx(0.8, rel=1e-4)


def test_compute_binary_metrics_keys_with_and_without_weighted():
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:8, 4:8] = 1
    pred = gt.copy()

    full = compute_binary_metrics(gt, pred, compute_weighted=True)
    assert {"dice", "iou", "loss", "precision", "recall", "weighted_dice", "weighted_dice_loss"} <= set(full.keys())
    assert full["dice"] == pytest.approx(1.0)
    assert full["loss"] == pytest.approx(0.0, abs=1e-6)

    no_weighted = compute_binary_metrics(gt, pred, compute_weighted=False)
    assert "weighted_dice" not in no_weighted
