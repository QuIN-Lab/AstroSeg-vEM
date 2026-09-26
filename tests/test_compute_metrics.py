import csv

import numpy as np
import pytest

from segmentation.compute_metrics import (
    _process_one,
    compute_metrics_for_predictions,
    find_matching_gt,
    prediction_to_image_stem,
    print_summary,
    save_results_csv,
)


def test_prediction_to_image_stem_strips_suffix():
    assert prediction_to_image_stem("image_001_pred_2d.tif") == "image_001"

    assert prediction_to_image_stem("image_001.tif") == "image_001"


def test_find_matching_gt_replaces_image_with_mask(tmp_path):
    (tmp_path / "mask_001.tif").write_bytes(b"x")
    found = find_matching_gt("image_001_pred_2d.tif", str(tmp_path))
    assert found == str(tmp_path / "mask_001.tif")


def test_find_matching_gt_returns_none_when_missing(tmp_path):
    assert find_matching_gt("image_999_pred_2d.tif", str(tmp_path)) is None


def test_process_one_binary_metrics(tmp_path, tif_writer):
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 1
    pred = gt.copy()
    tif_writer(tmp_path / "mask_001.tif", gt)
    pred_path = tif_writer(tmp_path / "image_001_pred_2d.tif", pred)

    _fname, metrics, err = _process_one(str(pred_path), str(tmp_path), num_classes=2, compute_weighted=True)
    assert err is None
    assert metrics["dice"] == pytest.approx(1.0)
    assert "weighted_dice" in metrics


def test_process_one_missing_gt(tmp_path, tif_writer):
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred_path = tif_writer(tmp_path / "image_002_pred_2d.tif", pred)
    _fname, metrics, err = _process_one(str(pred_path), str(tmp_path), num_classes=2, compute_weighted=True)
    assert err == "missing_gt"
    assert metrics is None


def test_compute_metrics_for_predictions_end_to_end(tmp_path, tif_writer):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    for i in range(3):
        gt = np.zeros((16, 16), dtype=np.uint8)
        gt[2:10, 2:10] = 1
        tif_writer(gt_dir / f"mask_{i:03d}.tif", gt)
        tif_writer(pred_dir / f"image_{i:03d}_pred_2d.tif", gt)


    tif_writer(pred_dir / "image_999_pred_2d.tif", np.zeros((16, 16), dtype=np.uint8))

    results = compute_metrics_for_predictions(str(pred_dir), str(gt_dir), num_workers=1)
    assert len(results) == 3
    for fname, metrics in results:
        assert metrics["dice"] == pytest.approx(1.0)


    results_threaded = compute_metrics_for_predictions(str(pred_dir), str(gt_dir), num_workers=2)
    assert {r[0] for r in results_threaded} == {r[0] for r in results}

    print_summary(results)


def test_save_results_csv(tmp_path):
    results = [("b.tif", {"dice": 0.5, "iou": 0.4}), ("a.tif", {"dice": 0.9, "iou": 0.8})]
    out_csv = tmp_path / "out.csv"
    save_results_csv(results, str(out_csv))

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert [r["filename"] for r in rows] == ["a.tif", "b.tif"]
    assert rows[0]["dice"] == "0.9"
