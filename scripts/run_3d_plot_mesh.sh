#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLOT_PY="${PROJECT_ROOT}/src/reconstruction/plot_3d.py"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
MESH_DEFAULT="${RESULTS_ROOT}/3d_reconstruction/dataset_4_mutant_fill_2d_dsxy10_reconstruction/pred_mask_global_xy_1.ply"
MESH_PATH="${1:-$MESH_DEFAULT}"

python "${PLOT_PY}" --mesh "${MESH_PATH}"
