#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-train_unet_mit_b2}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_NAME="$1"
  shift
fi

python "${PROJECT_ROOT}/src/segmentation/train.py" --config-name "${CONFIG_NAME}" "$@"
