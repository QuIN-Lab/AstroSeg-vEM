#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/visualization/generate_overlay_png.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/eval_unet_mit_b2/exp_3}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}}"
OUT_BASE="${OUT_BASE:-${RESULTS_ROOT}/overlays_postprocess_all}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/cropped_datasets}"
NUM_WORKERS="${NUM_WORKERS:-8}"

datasets=(
  "dataset_4_mutant"
)

for DATASET in "${datasets[@]}"; do
  IMG_DIR="${DATA_ROOT}/${DATASET}/images"
  if [[ ! -d "$IMG_DIR" ]]; then
    echo "Skipping ${DATASET}: missing images dir ${IMG_DIR}"
    continue
  fi

  pred_dir_candidates=(
    "${DATASET}_soma_filtered_dsxy10"
    "${DATASET}_soma_filtered_dsxy10_upsampled"
    "${DATASET}_post3d_dsxy10"
    "${DATASET}_post3d_dsxy10_upsampled"
    "${DATASET}_fill_2d_dsxy10"
    "${DATASET}_fill_2d_dsxy10_upsampled"
  )

  for pred_name in "${pred_dir_candidates[@]}"; do
    pred_dir="${EVAL_ROOT}/${pred_name}"
    if [[ ! -d "$pred_dir" ]]; then
      echo "Skipping: missing mask dir ${pred_dir}"
      continue
    fi

    base="$(basename "$pred_dir")"
    out_dir="${OUT_BASE}/${base}"

    echo "Generating overlays for ${base} -> ${out_dir}"
    python "$SCRIPT" \
      --image_dir "$IMG_DIR" \
      --mask_dir "$pred_dir" \
      --out_dir "$out_dir" \
      --num_workers "$NUM_WORKERS"
  done
done
