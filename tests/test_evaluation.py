import numpy as np
import pytest
import segmentation_models_pytorch as smp
import tifffile
import torch

from segmentation.evaluation import (
    build_model,
    build_patch_coords,
    build_weight_map,
    inference_sliding_window,
    zscore,
)
from segmentation.models.native_unet import NativeUNet


def test_zscore_normalizes():
    rng = np.random.default_rng(0)
    arr = rng.normal(20, 5, size=(16, 16)).astype(np.float32)
    z = zscore(arr)
    assert z.mean() == pytest.approx(0.0, abs=1e-4)
    assert z.std() == pytest.approx(1.0, rel=1e-4)


def test_zscore_constant_array_returns_none():
    assert zscore(np.ones((4, 4), dtype=np.float32)) is None


def test_build_patch_coords_covers_full_image():
    H, W, patch = 50, 37, 20
    coords = build_patch_coords(H, W, patch, overlap=0.5)
    covered = np.zeros((H, W), dtype=bool)
    for y0, y1, x0, x1 in coords:
        assert 0 <= y0 < y1 <= H
        assert 0 <= x0 < x1 <= W
        assert (y1 - y0, x1 - x0) == (patch, patch)
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_build_weight_map_counts_overlaps_and_caches():
    w1 = build_weight_map(32, 32, patch_size=16, overlap=0.5)
    assert w1.shape == (32, 32)
    assert (w1 >= 1.0).all()
    w2 = build_weight_map(32, 32, patch_size=16, overlap=0.5)
    assert w1 is w2


def test_inference_sliding_window_shapes_and_metrics(tmp_path):
    torch.manual_seed(0)
    model = NativeUNet(in_channels=3, num_classes=2)
    model.eval()

    H, W = 48, 48
    img = np.random.default_rng(1).integers(0, 255, size=(H, W)).astype(np.uint8)
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[10:30, 10:30] = 1

    out_path = tmp_path / "pred.tif"
    metrics = inference_sliding_window(
        model, img, str(out_path),
        patch_size=16, batch_size=4, device="cpu",
        gt_mask=gt, overlap=0.5,
    )

    assert out_path.exists()
    pred = tifffile.imread(str(out_path))
    assert pred.shape == (H, W)
    assert set(np.unique(pred)) <= {0, 1}
    assert "dice" in metrics and "weighted_dice" in metrics


def test_inference_sliding_window_without_gt_returns_empty_metrics(tmp_path):
    model = NativeUNet(in_channels=3, num_classes=2)
    model.eval()
    img = np.zeros((32, 32), dtype=np.uint8)
    out_path = tmp_path / "pred.tif"
    metrics = inference_sliding_window(model, img, str(out_path), patch_size=16, device="cpu")
    assert metrics == {}
    assert out_path.exists()


def test_build_model_native_unet_requires_existing_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_model(str(tmp_path / "missing.pth"), arch_type="native-unet", device="cpu")


def test_build_model_unknown_arch_raises(tmp_path):
    ckpt = tmp_path / "dummy.pth"
    torch.save({}, ckpt)
    with pytest.raises(ValueError):
        build_model(str(ckpt), arch_type="not-a-real-arch", device="cpu")


def test_build_model_native_unet_roundtrip(tmp_path):
    src = NativeUNet(in_channels=3, num_classes=2)
    ckpt = tmp_path / "native.pth"
    torch.save(src.state_dict(), ckpt)

    loaded = build_model(str(ckpt), arch_type="native-unet", device="cpu")
    assert isinstance(loaded, NativeUNet)
    assert loaded.training is False
    for p1, p2 in zip(src.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.timeout(120)
def test_build_model_unet_mit_b2_matches_paper_architecture(tmp_path):
    ref = smp.Unet(
        encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=2,
        decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type="scse",
    )
    ckpt = tmp_path / "unet_mit_b2.pth"
    torch.save(ref.state_dict(), ckpt)

    model = build_model(str(ckpt), arch_type="unet", encoder_name="mit_b2", device="cpu")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 224, 224)
