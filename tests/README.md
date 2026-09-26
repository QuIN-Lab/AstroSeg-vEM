# Tests

Unit tests for every module under `src/`, run against small, synthetic,
in-memory/temp-dir data (tens of pixels, a handful of slices) rather than
real vEM datasets or full training/inference runs. The whole suite finishes
in well under a minute on a laptop CPU, with no GPU and no network access.

## Running

```bash
pip install -r requirements-dev.txt
pytest
```

## What each file covers

| File | Covers |
|---|---|
| `test_patch_generation.py` | Foreground-aware patch sampling (30/70 mixture, 10% background discard, 5% extreme-patch retention) |
| `test_native_unet.py` | Fallback U-Net architecture forward pass |
| `test_train.py` | `PatchDataset`, `DiceLoss` (both modes), seeding, checkpoint-dir naming, `build_model`, and a one-step optimizer smoke test |
| `test_evaluation.py` | Sliding-window inference (patch coords, weight map, aggregation), `build_model` including the exact MiT-B2 + U-Net + scse architecture the paper uses |
| `test_metrics.py` | Dice, precision/recall, weighted Dice (incl. its near/far false-positive weighting property) |
| `test_compute_metrics.py` | Prediction/GT filename matching, batch metrics computation, CSV export |
| `test_postprocess.py` | All three post-processing stages: soma-guided component refinement (top-k components, HDBSCAN, joint size+distance removal), volumetric consolidation (single-slice artifact suppression, 26-connectivity top-k), in-plane hole filling |
| `test_reconstruct_3d.py` | Volume loading/downsampling, marching cubes, mesh I/O, PNG crop/pad helpers |
| `test_plot_3d.py` | Mesh/camera parsing utilities, NPZ/PLY mesh loading |
| `test_generate_overlay_png.py` | Overlay image generation |

## What's intentionally NOT covered

- **The actual `train()` / `main()` training loop.** It's wired to Hydra
  config resolution, wandb logging, and a real multi-epoch DataLoader loop
  (`num_workers=8`) -- exactly the "full-scale run" this suite is meant to
  avoid. Every piece it's built from (dataset, loss, model builder, seeding,
  checkpoint naming) is tested directly instead, plus one single forward/
  backward/optimizer-step smoke test that checks they fit together.
- **PyVista/VTK rendering and screenshot export** (`visualize_mesh` in
  `plot_3d.py`, the render step in `reconstruct_3d.py`). Even off-screen,
  these need a working GL/OSMesa context and produce an image meant for a
  human to look at -- verify those by eye after a real run, not in CI.
- **The `swin-unet` architecture path** in `build_model`. It requires an
  external `Swin-Unet` checkout (`SWIN_UNET_REPO` env var) that isn't part
  of this repository.
- **GPU-specific code** (`torch.autocast("cuda", ...)`, AMP `GradScaler`,
  `cudnn.benchmark`). All tests run with `device="cpu"`.
