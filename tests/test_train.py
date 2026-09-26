import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from segmentation.models.native_unet import NativeUNet
from segmentation.train import (
    DiceLoss,
    PatchDataset,
    _sanitize_tag,
    build_model,
    create_run_checkpoint_dir,
    get_criterion,
    get_train_transforms,
    set_seed,
)


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_sanitize_tag_replaces_unsafe_characters():
    assert _sanitize_tag("unet/mit_b2 v1:2") == "unet-mit_b2v1-2"


def test_get_train_transforms_normalizes_and_converts_to_tensor():
    transform = get_train_transforms(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.int64)
    out = transform(image=img, mask=mask)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 16, 16)

    assert out["image"].abs().max().item() < 1e-5


class TestDiceLoss:
    def test_perfect_prediction_near_zero_loss(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1] = 100.0
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        for mode in ("per_sample", "batch"):
            loss = DiceLoss(mode=mode)(logits, targets)
            assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_worst_prediction_near_one_loss(self):


        logits = torch.zeros(1, 2, 32, 32)
        logits[:, 0] = 100.0
        targets = torch.ones(1, 32, 32, dtype=torch.long)
        loss = DiceLoss(mode="per_sample")(logits, targets)
        assert loss.item() == pytest.approx(1.0, abs=1e-3)

    def test_rejects_non_binary_logits(self):
        logits = torch.zeros(1, 3, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        with pytest.raises(ValueError):
            DiceLoss()(logits, targets)


def test_get_criterion_returns_ce_and_dice():
    ce, dice = get_criterion(device="cpu", dice_mode="batch")
    assert isinstance(ce, torch.nn.CrossEntropyLoss)
    assert isinstance(dice, DiceLoss)
    assert dice.mode == "batch"


def test_create_run_checkpoint_dir_naming(tmp_path):
    cfg = OmegaConf.create({
        "experiment_name": "unit_test exp",
        "model": {"arch_type": "unet", "encoder_name": "mit_b2", "patch_size": 224},
        "training": {"checkpoint_dir": str(tmp_path), "batch_size": 4, "lr": 0.00006},
    })
    run_dir = create_run_checkpoint_dir(cfg)
    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    name = run_dir.name
    assert "unit_testexp" in name
    assert "unet" in name
    assert "mit_b2" in name
    assert "ps224" in name
    assert "bs4" in name
    assert "lr6e-05" in name


class FakePatchTree:

    def __init__(self, root, tif_writer, num_slices=3, patches_per_slice=2, size=16):
        self.root = root
        for s in range(num_slices):
            slice_dir = root / f"image_{s:03d}"
            for p in range(patches_per_slice):
                patch_dir = slice_dir / f"patch_{p:02d}"
                patch_dir.mkdir(parents=True)
                img = np.full((size, size), 128, dtype=np.uint8)
                mask = np.zeros((size, size), dtype=np.uint8)
                mask[: size // 2, : size // 2] = 1
                tif_writer(patch_dir / "image.tif", img)
                tif_writer(patch_dir / "mask.tif", mask)


def test_patch_dataset_loads_all_samples(tmp_path, tif_writer):
    FakePatchTree(tmp_path, tif_writer, num_slices=2, patches_per_slice=3)
    ds = PatchDataset(str(tmp_path), target_size=16)
    assert len(ds) == 6
    img, mask = ds[0]
    assert img.shape == (16, 16, 3)
    assert mask.shape == (16, 16)


def test_patch_dataset_slice_dirs_filtering(tmp_path, tif_writer):
    FakePatchTree(tmp_path, tif_writer, num_slices=3, patches_per_slice=2)
    only_first_two = {"image_000", "image_001"}
    ds = PatchDataset(str(tmp_path), target_size=16, slice_dirs=only_first_two)
    assert len(ds) == 4


def test_patch_dataset_falls_back_to_curr_tif(tmp_path, tif_writer):
    patch_dir = tmp_path / "image_000" / "patch_00"
    patch_dir.mkdir(parents=True)
    tif_writer(patch_dir / "curr.tif", np.full((8, 8), 100, dtype=np.uint8))
    tif_writer(patch_dir / "mask.tif", np.zeros((8, 8), dtype=np.uint8))
    ds = PatchDataset(str(tmp_path), target_size=8)
    assert len(ds) == 1


def test_patch_dataset_raises_when_empty(tmp_path):
    with pytest.raises(ValueError):
        PatchDataset(str(tmp_path))


def test_build_model_native_unet_roundtrip(tmp_path):
    model = build_model(arch_type="native-unet", device="cpu")
    assert isinstance(model, NativeUNet)

    ckpt = tmp_path / "native.pth"
    torch.save(model.state_dict(), ckpt)

    loaded = build_model(arch_type="native-unet", pretrained_path=str(ckpt), device="cpu")
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)


def test_build_model_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model(arch_type="not-a-real-arch", device="cpu")


def test_training_step_smoke_test():
    torch.manual_seed(0)
    model = NativeUNet(in_channels=3, num_classes=2)
    criterion, aux_loss = get_criterion(device="cpu", dice_mode="per_sample")
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)

    imgs = torch.rand(2, 3, 32, 32)
    masks = torch.randint(0, 2, (2, 32, 32))

    before = [p.clone() for p in model.parameters()]

    logits = model(imgs)
    ce_loss = criterion(logits, masks)
    d_loss = aux_loss(logits, masks)
    loss = 0.5 * ce_loss + 0.5 * d_loss
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
