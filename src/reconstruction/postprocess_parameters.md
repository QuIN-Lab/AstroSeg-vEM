# `postprocess.py` Parameter Guide

This document explains the command-line parameters used by [`src/reconstruction/postprocess.py`](postprocess.py).

The script performs a multi-stage post-processing pipeline for 2D prediction masks:

1. Collect the top connected components in each slice.
2. Use HDBSCAN to find the main 3D soma trajectory.
3. Filter 2D connected components based on that trajectory.
4. Optionally suppress unsupported single-slice artifacts in the downsampled 3D volume.
5. Keep the largest 3D connected components.
6. Optionally fill 2D holes.
7. Optionally compute evaluation metrics and save a parameter log.

## Required Parameters

### `--pred_dir`
Directory containing the predicted 2D mask files to post-process.

### `--out_dir`
Root output directory. The script creates subfolders such as `*_soma_filtered`, `*_post3d`, and optionally `*_fill_2d` under this location.

## Input and File Matching

### `--pattern`
Filename pattern used to find prediction files in `pred_dir`.
Default: `*_pred_2d.tif`

### `--image_dir`
Optional directory containing the original images. When provided, regions where the raw image equals zero are masked out during processing so empty image background does not remain in the prediction.

### `--pred_suffix`
Suffix removed from prediction filenames when matching them back to raw images in `image_dir`.
Default: `_pred_2d`

## 3D Soma Tracking and Clustering

### `--z_scale`
Scale factor applied to the slice index before HDBSCAN clustering. This controls how strongly z-distance contributes relative to x/y distance when building the 3D soma trajectory.
Default: `50.0`

Higher values make separation across slices more important. Lower values make the clustering rely more on in-plane centroid proximity.

### `--min_cluster_size`
Minimum HDBSCAN cluster size used when identifying the main soma trajectory from top connected components across slices.
Default: `25`

Increase this if you want a more conservative, more stable 3D cluster. Decrease it if the target spans fewer slices.

### `--top_k`
Number of largest 2D connected components collected from each slice before HDBSCAN.
Default: `3`

If your slices contain many large distractor components, increasing this can help preserve the true target candidate set for 3D clustering.

### `--best_cluster_mode`
Rule used to select the final HDBSCAN cluster after clustering is complete.
Choices: `score`, `area_sum`, `area_over_dist`
Default: `score`

Meaning:
- `score`: prefers clusters with both large area and long z-span.
- `area_sum`: prefers the cluster with the largest summed 2D area.
- `area_over_dist`: prefers a large cluster that also stays close to the global centroid reference.

## 2D Connected-Component Filtering

### `--rel_area_ratio`
Area threshold, relative to the selected soma-like component in each slice, for deciding whether another component is large enough to be considered a distractor.
Default: `0.3`

A component is removed when it is both:
- large enough: `component_area >= soma_area * rel_area_ratio`
- far enough from the soma reference: controlled by `--rel_dist_ratio`

### `--rel_dist_ratio`
Distance threshold, expressed as a fraction of the downsampled slice diagonal, for deciding whether a component is too far from the soma reference.
Default: `0.1`

Larger values keep more distant components. Smaller values filter more aggressively.

### `--single_target_protect`
Safety switch that ensures at least one component is kept in a slice if the normal filtering logic would remove everything.

This is useful when your dataset should contain exactly one main target and you want to avoid empty output slices caused by over-filtering.

### `--merge_cc_radius`
Optional radius used to dilate the downsampled binary mask before connected-component analysis.
Default: `0`

This can merge nearby fragments into a single connected component before filtering. If `downsample_xy > 1`, the radius is internally scaled to the downsampled grid.

### `--merge_cc_out_dir`
Legacy/placeholder output directory argument for merged connected components.
Default: `None`

In the current implementation, this value is accepted but intermediate merged outputs are not written.

### `--cc_select_mode`
Rule used to choose the soma-like reference component in each slice before filtering other components.
Choices: `distance`, `area_over_dist`
Default: `distance`

Meaning:
- `distance`: choose the component closest to the per-slice or global soma center.
- `area_over_dist`: choose the component with the best `area / (distance + 1)` score.

## 3D Volume Resolution and Component Selection

### `--downsample_xy`
XY downsampling factor applied before 3D volume operations.
Default: `1`

This affects:
- top-k component collection
- slice filtering in downsampled space
- 3D connected-component extraction
- single-slice artifact suppression scaling

If greater than `1`, the script writes both downsampled and upsampled variants for some outputs.

In the current implementation, this includes paired output directories such as:
- `*_soma_filtered_dsxyN` and `*_soma_filtered_dsxyN_upsampled`
- `*_post3d_dsxyN` and `*_post3d_dsxyN_upsampled`
- `*_fill_2d_dsxyN` and `*_fill_2d_dsxyN_upsampled` when `--fill_2d` is enabled

### `--topk_3d`
Number of largest 3D connected components kept after volume-level labeling.
Default: `1`

Use `1` when you expect a single dominant object. Increase it if multiple disconnected targets should be preserved.

### `--topk_map_centers`
Force the script to convert kept 3D components into per-slice center points and reselect matching 2D components in the original slices.

This is the default behavior unless `--no_topk_map_centers` is specified.

### `--no_topk_map_centers`
Disable center mapping and directly write masks from the kept 3D downsampled labels instead.

Use this if you want the final output to follow the downsampled 3D labels more literally instead of projecting centers back onto original 2D connected components.

## Optional 2D Hole Filling

### `--fill_2d`
After 3D filtering, fill holes inside each 2D mask.

When `downsample_xy > 1`, hole filling is applied to the downsampled post-3D masks first and then upsampled back to the original size.

## Single-Slice Artifact Suppression

These parameters control optional removal of large unsupported objects that appear in only one slice of the downsampled 3D volume.

### `--single_slice_artifact_suppression`
Enable single-slice artifact suppression before 3D top-k labeling.

The algorithm compares each interior slice to its neighboring slices. If a slice is abnormally large and contains regions unsupported by adjacent slices, those unsupported large connected components can be removed.

### `--single_slice_area_ratio`
Trigger suppression when the current slice foreground area is at least this multiple of the median area of the previous and next slice.
Default: `1.5`

Higher values make triggering rarer. Lower values make suppression more aggressive.

### `--single_slice_support_radius`
Radius used to dilate the union of the previous and next slice before deciding which regions in the current slice are unsupported.
Default: `12`

This value is internally scaled when `downsample_xy > 1`.

### `--single_slice_min_area`
Minimum area of an unsupported connected component that can be removed.
Default: `2000`

This value is also internally scaled when `downsample_xy > 1`.

### `--single_slice_debug`
Print per-slice diagnostics for artifact suppression, including area ratios, unsupported area, and the number of removed connected components.

## Metrics and Evaluation

### `--compute_metrics`
Compute `metrics.csv` files for generated outputs after post-processing.

The script evaluates:
- soma-filtered output
- post-3D output
- fill-2D output, if `--fill_2d` is enabled

### `--gt_dir`
Ground-truth mask directory used when `--compute_metrics` is enabled.

Required if `--compute_metrics` is set.

### `--metrics_num_workers`
Number of worker processes used by metrics computation.
Default: `2`

### `--metrics_no_weighted`
Skip weighted Dice during metric computation.

## Logging and Reproducibility

### `--save_params_json`
Save a JSON record of the run configuration, timing statistics, and output paths.

### `--params_json_path`
Optional explicit path for the saved JSON parameter log.
Default behavior: save `postprocess_run.json` under the final output directory.

## Typical Example

```bash
python src/reconstruction/postprocess.py \
  --pred_dir /path/to/predictions \
  --out_dir /path/to/output \
  --image_dir /path/to/images \
  --top_k 3 \
  --topk_3d 1 \
  --downsample_xy 10 \
  --z_scale 2.0 \
  --rel_area_ratio 0.5 \
  --rel_dist_ratio 0.5 \
  --fill_2d \
  --single_target_protect \
  --single_slice_artifact_suppression \
  --single_slice_area_ratio 1.5 \
  --single_slice_support_radius 12 \
  --single_slice_min_area 2000 \
  --save_params_json
```

## Output Folders

Depending on the options, the script typically creates:

- `*_soma_filtered`: masks after per-slice filtering around the soma trajectory when `downsample_xy == 1`
- `*_post3d`: masks after 3D connected-component selection when `downsample_xy == 1`
- `*_fill_2d`: masks after optional 2D hole filling when `downsample_xy == 1`
- `*_soma_filtered_dsxyN` and `*_soma_filtered_dsxyN_upsampled`: downsampled and upsampled soma-filtered outputs when `downsample_xy > 1`
- `*_post3d_dsxyN` and `*_post3d_dsxyN_upsampled`: downsampled and upsampled post-3D outputs when `downsample_xy > 1`
- `*_fill_2d_dsxyN` and `*_fill_2d_dsxyN_upsampled`: downsampled and upsampled fill-2D outputs when `downsample_xy > 1` and `--fill_2d` is enabled
- `soma_hdbscan.json`: clustering summary for the tracked soma-like structure
- `postprocess_run.json`: saved parameter and timing log when enabled
