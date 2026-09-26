#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
MODEL_PATH="${MODEL_PATH:-${RESULTS_ROOT}/eval_unet_mit_b2/exp_1_2/exp_1_checkpoints/segformer_unet_mit_b2_epoch30_dice0.9562_2d.pth}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/cropped_datasets}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}/eval_unet_mit_b2}"
DEVICE="${DEVICE:-cuda}"
PATCH_SIZE="${PATCH_SIZE:-224}"
OVERLAP="${OVERLAP:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRECISION="${PRECISION:-float32}"

datasets=(
  "dataset_1_mutant"
)

echo "Using model checkpoint: ${MODEL_PATH}"

for DATASET in "${datasets[@]}"; do
  ds="${DATA_ROOT}/${DATASET}"
  if [[ ! -d "$ds" ]]; then
    echo "Skipping ${DATASET}: dataset directory not found (${ds})"
    continue
  fi

  out_dir="${EVAL_ROOT}/${DATASET}"

  echo "Evaluating ${DATASET} -> ${out_dir}"
  python "${PROJECT_ROOT}/src/segmentation/evaluation.py" \
    --model_arch unet \
    --encoder_name mit_b2 \
    --model_path "$MODEL_PATH" \
    --data_dir "$ds" \
    --out_dir "$out_dir" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --overlap "$OVERLAP" \
    --device "$DEVICE" \
    --csv_out "$out_dir/metrics.csv"
done
