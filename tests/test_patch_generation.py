import json

import numpy as np
import pytest

from datasets.patch_generation import (
    extract_patch,
    generate_patches_tif,
    process_one_slice,
    to_uint8_view,
    zscore,
)


def test_extract_patch_shape_and_dtype():
    arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
    patch = extract_patch(arr, y=2, x=3, size=4)
    assert patch.shape == (4, 4)
    assert patch.dtype == np.uint8


def test_zscore_normalizes_to_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=50, scale=10, size=(32, 32)).astype(np.float32)
    z = zscore(arr)
    assert z.mean() == pytest.approx(0.0, abs=1e-4)
    assert z.std() == pytest.approx(1.0, rel=1e-4)


def test_zscore_returns_none_for_constant_array():
    arr = np.full((8, 8), 5.0, dtype=np.float32)
    assert zscore(arr) is None


def test_zscore_respects_stats_min_mask():
    arr = np.array([[0, 0], [100, 200]], dtype=np.float32)
    z_all = zscore(arr, stats_min=None)
    z_masked = zscore(arr, stats_min=0)
    assert z_all is not None and z_masked is not None
    assert not np.allclose(z_all, z_masked)


def test_zscore_stats_min_excludes_everything_returns_none():
    arr = np.zeros((4, 4), dtype=np.float32)
    assert zscore(arr, stats_min=0) is None


def test_to_uint8_view_clips_to_full_range():
    arr = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float32)
    view = to_uint8_view(arr, clip_min=-3.0, clip_max=3.0)
    assert view.dtype == np.uint8
    assert view[0] == view[1] == 0
    assert view[3] == view[4] == 255
    assert view[2] == 128


def test_process_one_slice_respects_bg_and_fg_sample_prob(tmp_path, tif_writer, blob_mask):
    rng = np.random.default_rng(42)
    img = rng.integers(50, 255, size=(64, 64), dtype=np.uint8)
    mask = blob_mask((64, 64), center=(20, 20), radius=6)

    img_path = tif_writer(tmp_path / "image_000.tif", img)
    mask_path = tif_writer(tmp_path / "mask_000.tif", mask)
    out_dir = tmp_path / "out"

    np.random.seed(0)
    stats = process_one_slice(
        i=0,
        img_paths=[str(img_path)],
        mask_paths=[str(mask_path)],
        out_dir=str(out_dir),
        patch_size=16,
        patches_per_image=6,
        fg_sample_prob=0.3,
    )

    patch_dirs = sorted((out_dir / "image_000").glob("patch_*"))
    assert len(patch_dirs) == stats["count"] <= 6
    assert stats["count"] > 0

    for pd in patch_dirs:
        assert (pd / "image.tif").exists()
        assert (pd / "mask.tif").exists()
        meta = json.loads((pd / "patch_info.json").read_text())
        assert meta["patch_size"] == 16
        assert meta["bbox"]["width"] == 16


def test_process_one_slice_no_mask_reconstruction_mode(tmp_path, tif_writer):
    rng = np.random.default_rng(1)
    img = rng.integers(50, 255, size=(32, 32), dtype=np.uint8)
    img_path = tif_writer(tmp_path / "image_000.tif", img)
    out_dir = tmp_path / "out"

    stats = process_one_slice(
        i=0,
        img_paths=[str(img_path)],
        mask_paths=None,
        out_dir=str(out_dir),
        patch_size=8,
        patches_per_image=3,
    )
    assert stats["count"] <= 3
    patch_dirs = sorted((out_dir / "image_000").glob("patch_*"))
    for pd in patch_dirs:
        assert (pd / "image.tif").exists()
        assert not (pd / "mask.tif").exists()


def test_generate_patches_tif_end_to_end(tmp_path, tif_writer, blob_mask):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    rng = np.random.default_rng(7)

    for i in range(2):
        img = rng.integers(50, 255, size=(32, 32), dtype=np.uint8)
        mask = blob_mask((32, 32), center=(16, 16), radius=5)
        tif_writer(image_dir / f"image_{i:03d}.tif", img)
        tif_writer(mask_dir / f"mask_{i:03d}.tif", mask)

    out_dir = tmp_path / "patches"
    generate_patches_tif(
        str(image_dir), str(mask_dir), str(out_dir),
        patch_size=8, patches_per_image=2,
    )

    meta_path = out_dir / "patch_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["num_slices"] == 2

    slice_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(slice_dirs) == 2
