#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
ROOT_DEFAULT="${ROOT_DEFAULT:-${RESULTS_ROOT}/eval_unet_mit_b2/exp_1_2}"
OUTPUT_ROOT_DEFAULT="${OUTPUT_ROOT_DEFAULT:-${ROOT_DEFAULT}/Astro_Reconstruction_3d}"
ROOT="${1:-$ROOT_DEFAULT}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
SUFFIX="reconstruction"
DS=1
SKIP_FIRST_SLICES="${SKIP_FIRST_SLICES:-0}"
STRETCH_DEFAULT="${STRETCH_DEFAULT:-1,1,1}"
RENDER_WIDTH="${RENDER_WIDTH:-2200}"
RENDER_HEIGHT="${RENDER_HEIGHT:-2400}"
CAMERA_ZOOM="${CAMERA_ZOOM:-0.85}"
CROP_RENDER_BACKGROUND="${CROP_RENDER_BACKGROUND:-0}"
CROP_PADDING="${CROP_PADDING:-24}"
KEEP_RENDER_CANVAS_SIZE="${KEEP_RENDER_CANVAS_SIZE:-0}"
AXES_WIDGET_VIEWPORT="${AXES_WIDGET_VIEWPORT:-0.12,0.12,0.26,0.26}"
BOUNDS_TITLE_OFFSET="${BOUNDS_TITLE_OFFSET:-35,10}"
BOUNDS_LABEL_OFFSET="${BOUNDS_LABEL_OFFSET:-52}"
RECON_PY="${PROJECT_ROOT}/src/reconstruction/reconstruct_3d.py"

datasets=(
  "dataset_4_mutant_fill_2d_dsxy10"
)

for name in "${datasets[@]}"; do
  dataset_key="$(basename "$name")"

  if [[ "$dataset_key" == *"dataset_1_mutant"* || "$dataset_key" == *"dataset_2_ctrl"* ]]; then
    STRETCH="14,10,10"
  elif [[ "$dataset_key" == *"dataset_3_ctrl"* || "$dataset_key" == *"dataset_4_mutant"* ]]; then
    STRETCH="20,10,10"
  else
    STRETCH="$STRETCH_DEFAULT"
    echo "Unknown dataset pattern in ${dataset_key}, fallback STRETCH=${STRETCH}"
  fi

  if [[ "$name" = /* ]]; then
    input_dir="$name"
  else
    input_dir="${ROOT}/${name}"
  fi

  output_dir="${OUTPUT_ROOT}/${dataset_key}_${SUFFIX}"
  mkdir -p "${output_dir}"

  npz="${output_dir}/pred_mask_global_xy_${DS}.npz"
  ply="${output_dir}/pred_mask_global_xy_${DS}.ply"
  tif="${output_dir}/pred_mask_downsampled_xy_${DS}.tif"
  render_png="${output_dir}/reconstruction_render.png"
  params_json="${output_dir}/reconstruction_params.json"

  echo "[Run] ${dataset_key}"
  echo "Input dir: ${input_dir}"
  echo "Using STRETCH=${STRETCH}"
  echo "Skipping first sorted slice(s): ${SKIP_FIRST_SLICES}"
  cmd=(
    python "${RECON_PY}"
    --input_dir "${input_dir}"
    --pattern "*_pred_2d.tif"
    --dtype "uint8"
    --downsample_xy "${DS}"
    --skip_first_slices "${SKIP_FIRST_SLICES}"
    --bounds_line_width 10
    --bounds_font_size 24
    --bounds_text_screen_size 24
    --render_width "${RENDER_WIDTH}"
    --render_height "${RENDER_HEIGHT}"
    --camera_zoom "${CAMERA_ZOOM}"
    --axes_widget_viewport "${AXES_WIDGET_VIEWPORT}"
    --bounds_title_offset "${BOUNDS_TITLE_OFFSET}"
    --bounds_label_offset "${BOUNDS_LABEL_OFFSET}"
    --crop_padding "${CROP_PADDING}"
    --export_npz "${npz}"
    --export_ply "${ply}"
    --export_tif "${tif}"
    --stretch_factors "${STRETCH}"
    --render_png "${render_png}"
    --save_params_json
    --params_json_path "${params_json}"
  )

  if [[ "${CROP_RENDER_BACKGROUND}" == "1" ]]; then
    cmd+=(--crop_render_background)
  fi

  if [[ "${KEEP_RENDER_CANVAS_SIZE}" == "1" ]]; then
    cmd+=(--keep_render_canvas_size)
  fi

  "${cmd[@]}"
done
