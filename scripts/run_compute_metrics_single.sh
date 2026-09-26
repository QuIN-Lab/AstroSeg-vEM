#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/segmentation/compute_metrics.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/eval_unet_mit_b2}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}/exp_3}"
GT_ROOT="${GT_ROOT:-${PROJECT_ROOT}/data/cropped_datasets}"
NUM_WORKERS="${NUM_WORKERS:-2}"

datasets=(
  "dataset_4_mutant"
)

for DATASET in "${datasets[@]}"; do
  GT_DIR="${GT_ROOT}/${DATASET}/masks"
  if [[ ! -d "$GT_DIR" ]]; then
    echo "Skipping ${DATASET}: missing gt dir ${GT_DIR}"
    continue
  fi

  pred_dirs=(
#    "${EVAL_ROOT}/${DATASET}"
    "${EVAL_ROOT}/${DATASET}_soma_filtered_dsxy10_upsampled"
    "${EVAL_ROOT}/${DATASET}_post3d_dsxy10_upsampled"
    "${EVAL_ROOT}/${DATASET}_fill_2d_dsxy10_upsampled"
  )

  for PRED_DIR in "${pred_dirs[@]}"; do
    if [[ ! -d "$PRED_DIR" ]]; then
      echo "Skipping ${DATASET}: missing pred dir ${PRED_DIR}"
      continue
    fi

    OUT_CSV="${PRED_DIR}/metrics.csv"
    echo "[Metrics|full] ${DATASET} -> $(basename "$PRED_DIR") (dice/iou/precision/recall + weighted_dice_full)"
    python "$SCRIPT" \
      --pred_dir "$PRED_DIR" \
      --gt_dir "$GT_DIR" \
      --csv_out "$OUT_CSV" \
      --num_workers "$NUM_WORKERS"
  done
done
