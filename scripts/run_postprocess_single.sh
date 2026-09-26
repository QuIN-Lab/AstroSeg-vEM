#!/usr/bin/env bash
set -eo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
BASE_PRED_ROOT="${BASE_PRED_ROOT:-${RESULTS_ROOT}/eval_unet_mit_b2}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/eval_unet_mit_b2}"
IMAGE_ROOT="${IMAGE_ROOT:-${PROJECT_ROOT}/data/cropped_datasets}"
MERGE_CC_RADIUS="${MERGE_CC_RADIUS:-0}"
SINGLE_SLICE_ARTIFACT_SUPPRESSION="${SINGLE_SLICE_ARTIFACT_SUPPRESSION:-1}"
SINGLE_SLICE_DEBUG="${SINGLE_SLICE_DEBUG:-1}"
SINGLE_SLICE_AREA_RATIO="${SINGLE_SLICE_AREA_RATIO:-1.5}"
SINGLE_SLICE_SUPPORT_RADIUS="${SINGLE_SLICE_SUPPORT_RADIUS:-12}"
SINGLE_SLICE_MIN_AREA="${SINGLE_SLICE_MIN_AREA:-2000}"

datasets=(
  "dataset_4_mutant"
)
for DATASET in "${datasets[@]}"; do
  if [[ "$DATASET" == "dataset_1_mutant" || "$DATASET" == "dataset_2_ctrl" ]]; then
    Z_SCALE=1.4
  elif [[ "$DATASET" == "dataset_3_ctrl" || "$DATASET" == "dataset_4_mutant" ]]; then
    Z_SCALE=2.0
  else
    echo "Unknown dataset: $DATASET"
    echo "Expected one of: dataset_1_mutant, dataset_2_ctrl, dataset_3_ctrl, dataset_4_mutant"
    continue
  fi

  PRED_DIR="${BASE_PRED_ROOT}/${DATASET}"
  IMAGE_DIR="${IMAGE_ROOT}/${DATASET}/images"

  if [[ ! -d "$PRED_DIR" ]]; then
    echo "Skipping ${DATASET}: prediction dir not found (${PRED_DIR})"
    continue
  fi
  if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Skipping ${DATASET}: image dir not found (${IMAGE_DIR})"
    continue
  fi

  echo "----------------------------------------"
  echo "[Run] ${DATASET}"
  echo "Using z_scale = ${Z_SCALE}"
  echo "Single-slice artifact suppression = ${SINGLE_SLICE_ARTIFACT_SUPPRESSION}"
  echo "----------------------------------------"

  single_slice_args=()
  if [[ "${SINGLE_SLICE_ARTIFACT_SUPPRESSION}" == "1" ]]; then
    single_slice_args+=(
      --single_slice_artifact_suppression
      --single_slice_area_ratio "$SINGLE_SLICE_AREA_RATIO"
      --single_slice_support_radius "$SINGLE_SLICE_SUPPORT_RADIUS"
      --single_slice_min_area "$SINGLE_SLICE_MIN_AREA"
    )
    if [[ "${SINGLE_SLICE_DEBUG}" == "1" ]]; then
      single_slice_args+=(--single_slice_debug)
    fi
  fi

  "$PYTHON_BIN" "${PROJECT_ROOT}/src/reconstruction/postprocess.py" \
    --pred_dir "$PRED_DIR" \
    --out_dir "$OUT_DIR" \
    --image_dir "$IMAGE_DIR" \
    --pattern "*_pred_2d.tif" \
    --top_k 3 \
    --topk_3d 1 \
    --downsample_xy 10 \
    --z_scale "$Z_SCALE" \
    --min_cluster_size 10 \
    --rel_area_ratio 0.5 \
    --rel_dist_ratio 0.5 \
    --merge_cc_radius "$MERGE_CC_RADIUS" \
    --cc_select_mode area_over_dist \
    --best_cluster_mode area_sum \
    --no_topk_map_centers \
    --save_params_json \
    --params_json_path "$OUT_DIR/${DATASET}_fill_2d_dsxy10_upsampled/postprocess_run.json" \
    --fill_2d \
    --single_target_protect \
    "${single_slice_args[@]}" \
    "$@"
done
