"""
SegFormer training script for 2D EM segmentation.
Supports MiT encoder with U-Net/SegFormer decoder variants,
ordered slice split, and reproducible training setup.

Author: Bowen Deng
Lab: Quantum Innovation Lab, University of Waterloo
Date: 2026-02-12
"""

import json
import os
import random
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segmentation.models.native_unet import NativeUNet


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Data Augmentation
def get_train_transforms(mean, std, pre_zscore=False):
    transforms_list = [
        ToTensorV2()
    ]
    if not pre_zscore:
        transforms_list.insert(0, A.Normalize(mean=mean, std=std, max_pixel_value=1.0))
    return A.Compose(transforms_list)


def get_val_transforms(mean, std, pre_zscore=False):
    transforms_list = [
        ToTensorV2()
    ]
    if not pre_zscore:
        transforms_list.insert(0, A.Normalize(mean=mean, std=std, max_pixel_value=1.0))
    return A.Compose(transforms_list)


# Dataset

class PatchDataset(Dataset):

    def __init__(
            self,
            patch_roots,
            transform=None,
            target_size=224,
            pre_zscore=False,
            slice_dirs=None,
    ):
        self.transform = transform
        self.target_size = target_size
        self.pre_zscore = pre_zscore
        self.samples = []

        if isinstance(patch_roots, str):
            patch_roots = [patch_roots]
        patch_roots = list(patch_roots)

        for patch_root in patch_roots:
            if os.path.isdir(patch_root):

                all_slices = sorted(glob(os.path.join(patch_root, "*")))
                root_total = 0
                root_valid = 0

                if slice_dirs is not None:
                    if isinstance(slice_dirs, dict):
                        allowed = slice_dirs.get(patch_root, set())
                    else:
                        allowed = slice_dirs
                    all_slices = [
                        s for s in all_slices
                        if os.path.basename(s) in allowed
                    ]

                for slice_dir in all_slices:
                    for patch_dir in sorted(glob(os.path.join(slice_dir, "patch_*"))):
                        root_total += 1

                        img = os.path.join(patch_dir, "image.tif")
                        if not os.path.exists(img):
                            img = os.path.join(patch_dir, "curr.tif")
                        mask = os.path.join(patch_dir, "mask.tif")
                        if os.path.exists(img) and os.path.exists(mask):
                            self.samples.append((img, mask))
                            root_valid += 1

                if root_total > 0:
                    print(f"PatchDataset {os.path.basename(patch_root)} -> "
                          f"valid {root_valid} / total {root_total} "
                          f"(dropped {root_total - root_valid})")

        if len(self.samples) == 0:
            raise ValueError(f"No patches found in given patch_roots: {patch_roots}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path, mask_path = sample
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = img.astype(np.float32)
        if not self.pre_zscore and img.max() > 1:
            img /= 255.0
        img = np.stack([img, img, img], axis=-1)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = mask.astype(np.int64)

        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask


# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, mode="per_sample"):
        super().__init__()
        self.smooth = smooth
        self.mode = mode

    def forward(self, logits, targets):
        if logits.shape[1] != 2:
            raise ValueError("DiceLoss expects 2-class logits.")
        probs = torch.softmax(logits, dim=1)[:, 1, ...]
        targets = (targets > 0).float()

        if self.mode == "batch":
            intersection = (probs * targets).sum()
            union = probs.sum() + targets.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice
        else:
            dims = tuple(range(1, probs.ndim))
            intersection = (probs * targets).sum(dim=dims)
            union = probs.sum(dim=dims) + targets.sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()


def get_criterion(device="cuda", dice_mode="per_sample"):
    weights = torch.ones(2, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    aux_loss = DiceLoss(mode=dice_mode).to(device)
    return criterion, aux_loss


def _sanitize_tag(value):
    return str(value).replace("/", "-").replace(" ", "").replace(":", "-")


def create_run_checkpoint_dir(cfg: DictConfig):
    checkpoint_root = Path(to_absolute_path(cfg.training.checkpoint_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "_".join([
        _sanitize_tag(cfg.experiment_name),
        _sanitize_tag(cfg.model.arch_type),
        _sanitize_tag(cfg.model.encoder_name),
        f"ps{cfg.model.patch_size}",
        f"bs{cfg.training.batch_size}",
        f"lr{cfg.training.lr:g}",
        timestamp,
    ])
    run_dir = checkpoint_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Model Builder
def build_model(arch_type="unet", encoder_name="mit_b2",
                pretrained_path=None, device="cuda", input_size=224):
    if pretrained_path is not None:
        init_weights = None
    else:
        init_weights = "imagenet"

    if arch_type == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=init_weights,
            in_channels=3,
            classes=2,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type="scse"
        )
    elif arch_type == "native-unet":
        model = NativeUNet(in_channels=3, num_classes=2)
    elif arch_type == "original":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=init_weights,
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
        raise ValueError(f"Unknown architecture type: {arch_type}")

    effective_pretrained_path = pretrained_path
    if arch_type == "swin-unet" and effective_pretrained_path is None:
        default_swin_ckpt = os.environ.get(
            "SWIN_UNET_CKPT",
            os.path.join(os.environ.get("SWIN_UNET_REPO", "./Swin-Unet"), "pretrained_ckpt", "swin_tiny_patch4_window7_224.pth"),
        )
        if os.path.isfile(default_swin_ckpt):
            effective_pretrained_path = default_swin_ckpt
            print(f"Using default Swin-Unet pretrained checkpoint: {effective_pretrained_path}")

    if effective_pretrained_path is not None and arch_type == "swin-unet":
        state_dict = torch.load(effective_pretrained_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    elif effective_pretrained_path is not None and arch_type == "native-unet":
        state_dict = torch.load(effective_pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif effective_pretrained_path is not None:
        state_dict = torch.load(effective_pretrained_path, map_location="cpu")
        encoder_keys = set(model.encoder.state_dict().keys())
        sd_keys = set(state_dict.keys())

        # Support encoder-only checkpoints (plain or prefixed with "encoder.")
        if sd_keys.issubset(encoder_keys):
            model.encoder.load_state_dict(state_dict, strict=True)
        elif any(k.startswith("encoder.") for k in sd_keys):
            stripped = {k.replace("encoder.", "", 1): v for k, v in state_dict.items()
                        if k.startswith("encoder.")}
            model.encoder.load_state_dict(stripped, strict=True)
        else:
            # Fallback: assume full-model checkpoint
            model.load_state_dict(state_dict, strict=False)

    return model.to(device)


# Training
def train(cfg: DictConfig):
    set_seed(cfg.training.seed)

    run_checkpoint_dir = create_run_checkpoint_dir(cfg)
    cfg.training.checkpoint_dir = str(run_checkpoint_dir)
    print(f"Checkpoint run directory: {cfg.training.checkpoint_dir}")

    with (run_checkpoint_dir / "run_args.json").open("w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    with (run_checkpoint_dir / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    mean = list(cfg.data.mean)
    std = list(cfg.data.std)

    train_transform = get_train_transforms(mean, std, pre_zscore=bool(cfg.data.pre_zscore))
    val_transform = get_val_transforms(mean, std, pre_zscore=bool(cfg.data.pre_zscore))

    train_slices = {}
    val_slices = {}

    train_patch_roots = list(cfg.data.train_patch_roots)

    for root in train_patch_roots:
        slices = sorted(glob(os.path.join(root, "image_*")))
        slice_names = [os.path.basename(s) for s in slices]
        if len(slice_names) == 0:
            raise ValueError(f"No slice directories found under: {root}")

        split_idx = int(0.8 * len(slice_names))
        train_slices[root] = set(slice_names[:split_idx])
        val_slices[root] = set(slice_names[split_idx:])

        print(f"Split {os.path.basename(root)} -> "
              f"Train: {split_idx} | Val: {len(slice_names) - split_idx}")

    total_train = sum(len(v) for v in train_slices.values())
    total_val = sum(len(v) for v in val_slices.values())
    print(f"Split TOTAL -> Train slices: {total_train} | Val slices: {total_val}")

    train_ds = PatchDataset(
        train_patch_roots,
        train_transform,
        target_size=cfg.model.patch_size,
        pre_zscore=bool(cfg.data.pre_zscore),
        slice_dirs=train_slices,
    )

    val_ds = PatchDataset(
        train_patch_roots,
        val_transform,
        target_size=cfg.model.patch_size,
        pre_zscore=bool(cfg.data.pre_zscore),
        slice_dirs=val_slices,
    )

    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")

    g = torch.Generator()
    g.manual_seed(cfg.training.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    model = build_model(
        arch_type=cfg.model.arch_type,
        encoder_name=cfg.model.encoder_name,
        pretrained_path=cfg.model.pretrained_path,
        device=cfg.training.device,
        input_size=cfg.model.patch_size,
    )

    dice_mode = cfg.training.dice_mode
    criterion, train_aux_loss = get_criterion(cfg.training.device, dice_mode=dice_mode)

    metric_dice_calculator = DiceLoss(mode=dice_mode).to(cfg.training.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    use_amp = cfg.training.device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    project_name = f"segformer-{cfg.model.arch_type}-finetune"
    run = wandb.init(
        project=project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline" if cfg.training.offline else "online",
    )
    run_url = None
    if run is not None and not cfg.training.offline:
        try:
            run_url = run.get_url()
        except Exception as e:
            print(f"Warning: failed to get wandb run URL: {e}")
            run_url = None

    best_val_score = 0.0
    patience_counter = 0

    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss, total_metric_dice = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(cfg.training.device), masks.to(cfg.training.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(imgs)
                ce_loss = criterion(logits, masks)
                aux_loss = train_aux_loss(logits, masks)
                loss = 0.5 * ce_loss + 0.5 * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                real_dice_loss = metric_dice_calculator(logits.float(), masks)

            total_loss += loss.item()
            total_metric_dice += (1 - real_dice_loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        avg_train_dice = total_metric_dice / len(train_loader)

        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(cfg.training.device), masks.to(cfg.training.device)
                logits = model(imgs)

                ce_loss = criterion(logits, masks)
                aux_loss = train_aux_loss(logits, masks)
                loss = 0.5 * ce_loss + 0.5 * aux_loss

                real_d_loss = metric_dice_calculator(logits, masks)

                val_loss += loss.item()
                val_dice += (1 - real_d_loss.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch [{epoch + 1}/{cfg.training.num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss, "train_dice": avg_train_dice,
            "val_loss": avg_val_loss, "val_dice": avg_val_dice,
            "lr": optimizer.param_groups[0]['lr']
        })

        if avg_val_dice > best_val_score:
            best_val_score = avg_val_dice
            patience_counter = 0

            save_name = (
                f"segformer_{cfg.model.arch_type}_{cfg.model.encoder_name}_"
                f"epoch{epoch + 1:02d}_"
                f"dice{avg_val_dice:.4f}_"
                f"2d.pth"
            )
            save_path = os.path.join(cfg.training.checkpoint_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print("Early stopping triggered.")
                break

    wandb.finish()

    if run is not None:
        print(f"Finished training: {cfg.experiment_name}")
        print(f"W&B run id: {run.id}")
        print(f"W&B run path: {'/'.join(run.path)}")
        print(f"W&B local dir: {run.dir}")
        if run_url:
            print(f"W&B url: {run_url}")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train_unet_mit_b2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    main()
