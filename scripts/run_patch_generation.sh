#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="${BASE_DIR:-${PROJECT_ROOT}/data/cropped_datasets}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/data/patches}"
PATCH_GEN="${PATCH_GEN:-${PROJECT_ROOT}/src/datasets/patch_generation.py}"

PATCH_SIZE="${PATCH_SIZE:-224}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-400}"
FG_SAMPLE_PROB="${FG_SAMPLE_PROB:-0.3}"

datasets=(
  "dataset_4_mutant"
)

for ds in "${datasets[@]}"; do
  img_dir="${BASE_DIR}/${ds}/images"
  mask_dir="${BASE_DIR}/${ds}/masks"
  out_dir="${OUT_BASE}/${ds}_patches_${PATCH_SIZE}"

  if [[ ! -d "$img_dir" ]]; then
    echo "Skipping ${ds}: missing images dir ${img_dir}"
    continue
  fi

  mask_arg=(--mask_dir none)
  if [[ -d "$mask_dir" ]]; then
    mask_arg=(--mask_dir "$mask_dir")
  fi

  echo "Generating patches for ${ds} -> ${out_dir}"
  python "$PATCH_GEN" \
    --image_dir "$img_dir" \
    "${mask_arg[@]}" \
    --out_dir "$out_dir" \
    --patch_size "$PATCH_SIZE" \
    --patches_per_image "$PATCHES_PER_IMAGE" \
    --fg_sample_prob "$FG_SAMPLE_PROB" \
    "$@"
done
