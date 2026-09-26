import numpy as np
import pytest
import tifffile
from PIL import Image

from visualization.generate_overlay_png import (
    _process_one,
    generate_overlay_png,
    make_overlay,
    resize_to_small,
)


def test_resize_to_small_image_mode_preserves_aspect():
    img = np.random.default_rng(0).random((100, 50)).astype(np.float32)
    small = resize_to_small(img, max_size=20, is_mask=False)
    assert small.shape == (20, 10)


def test_resize_to_small_mask_mode_stays_binary():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    small = resize_to_small(mask, max_size=20, is_mask=True)
    assert set(np.unique(small)) <= {0, 1}
    assert small.shape == (20, 20)


def test_make_overlay_colors_masked_region():
    img = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    overlay = make_overlay(img, mask, color=(1.0, 0.0, 0.0), alpha=1.0)

    assert np.allclose(overlay[5, 5], [1.0, 0.0, 0.0])

    assert np.allclose(overlay[0, 0], overlay[0, 0])
    assert not np.allclose(overlay[0, 0], [1.0, 0.0, 0.0])


def test_process_one_writes_overlay_png(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32)).astype(np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    img_path = tmp_path / "image_001.tif"
    tifffile.imwrite(str(img_path), img)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(str(mask_dir / "image_001_pred_2d.tif"), mask)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _fname, ok, msg = _process_one(str(img_path), str(mask_dir), str(out_dir))
    assert ok is True
    assert msg is None
    out_path = out_dir / "image_001_overlay.png"
    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.size[0] > 0 and im.size[1] > 0


def test_process_one_missing_mask_reports_skip(tmp_path):
    img = np.zeros((16, 16), dtype=np.uint8)
    img_path = tmp_path / "image_002.tif"
    tifffile.imwrite(str(img_path), img)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    _fname, ok, msg = _process_one(str(img_path), str(mask_dir), str(tmp_path / "out"))
    assert ok is False
    assert "No mask found" in msg


def test_generate_overlay_png_end_to_end(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "overlays"
    image_dir.mkdir()
    mask_dir.mkdir()

    for i in range(2):
        img = np.random.default_rng(i).integers(0, 255, size=(24, 24)).astype(np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        tifffile.imwrite(str(image_dir / f"image_{i:03d}.tif"), img)
        tifffile.imwrite(str(mask_dir / f"image_{i:03d}_pred_2d.tif"), mask)

    generate_overlay_png(str(image_dir), str(mask_dir), str(out_dir), num_workers=2)
    pngs = sorted(out_dir.glob("*_overlay.png"))
    assert len(pngs) == 2


def test_generate_overlay_png_raises_when_no_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    with pytest.raises(ValueError):
        generate_overlay_png(str(image_dir), str(tmp_path), str(tmp_path / "out"))
