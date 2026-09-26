# AstroSeg-vEM

This repository contains the official implementation for the paper: **Lightweight Deep Learning Framework for Astrocyte-Centric 3D Reconstruction across Normal and Stress-associated Ultrastructural States**.

## Environment Setup

This project is recommended for Python 3.11+.

```bash
pip install -r requirements.txt
```

*Note: Please ensure you install the appropriate version of PyTorch for your GPU drivers (CUDA is used by default).*

## Core Workflow

### 1. Patch Generation
Partition large raw images into training patches.
```bash
bash ./scripts/run_patch_generation.sh
```

### 2. Training
Train the U-Net model with the MiT-B2 backbone.
```bash
bash ./scripts/run_train.sh
```

Training parameters (batch size, learning rate, number of epochs, Dice loss mode, etc.) are configured in `configs/train_unet_mit_b2.yaml`. Individual values can also be overridden at runtime, for example:
```bash
bash ./scripts/run_train.sh train_unet_mit_b2 training.dice_mode=per_sample
```

### 3. Evaluation
Run inference on the configured evaluation dataset(s) to generate 2D prediction masks.

The current `scripts/run_eval_unet_mit_b2_single.sh` script is configured to run `dataset_1_mutant` by default. Update the `datasets` array in that script if you want to evaluate additional datasets.
```bash
bash ./scripts/run_eval_unet_mit_b2_single.sh
```

### 4. Post-processing
This step refines the 2D prediction masks, internally downsampling them by **10x (XY)** for efficient volumetric processing and filtering of artifacts.

With the current `scripts/run_postprocess_single.sh` defaults, the script writes both:
- downsampled outputs such as `*_dsxy10`
- upsampled outputs such as `*_dsxy10_upsampled`

Pay attention to which variant you use in downstream steps.

```bash
bash ./scripts/run_postprocess_single.sh
```

English parameter guide for the post-processing script:
[`src/reconstruction/postprocess_parameters.md`](src/reconstruction/postprocess_parameters.md)

### 5. 3D Reconstruction
Builds the final 3D mesh based on the post-processed results. The current `scripts/run_3d_reconstruction.sh` example uses the **10x downsampled** output from step 4 (for example, `*_fill_2d_dsxy10`). Since that input is downsampled 10x in XY and may have a different Z resolution, the reconstruction applies a **stretch factor** (e.g., `20, 10, 10` for Z, Y, X) to restore the correct physical aspect ratio in the final mesh and volume.

> **Tip**: Adjust `stretch_factors` whenever the downsampling ratio changes. The Y and X components should typically match your `downsample_xy` value to maintain proportionality with the raw data's spatial scale.
```bash
bash ./scripts/run_3d_reconstruction.sh
```

### 6. Visualization
Visualize 3D reconstruction results or generate segmentation overlays.
```bash
bash ./scripts/run_3d_plot_mesh.sh
bash ./scripts/run_overlays_single.sh
```

## Post-processing Parameter Guide

See [`src/reconstruction/postprocess_parameters.md`](src/reconstruction/postprocess_parameters.md) for a full parameter reference including guidance on reducing false positives and maximising true positives.

## Testing

Unit tests for every module, run on small synthetic data (no real vEM data, no GPU, under a minute total):
```bash
pip install -r requirements-dev.txt
pytest
```
See [`tests/README.md`](tests/README.md) for what is and isn't covered.

## Notes

**Coordinate Systems**: Pay close attention to the `z_scale` and `downsample_xy` parameters in `scripts`. These ensure that volumetric components are correctly tracked across slices and that the final 3D output represents the true morphology.
