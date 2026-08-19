#!/bin/bash
# Quick launch script for current single-GPU ScanRefer two-stage priority runs.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
BASE_MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
if ! [[ "${BASE_MASTER_PORT}" =~ ^[0-9]+$ ]]; then
  BASE_MASTER_PORT=$((29500 + RANDOM % 1000))
fi

run_step() {
  local title=$1
  local port=$2
  shift 2
  echo "${title}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" MASTER_PORT="${port}" "$@"
  echo ""
}

echo "=========================================="
echo "ScanRefer two-stage single-GPU priority experiments"
echo "=========================================="
echo "WARNING: legacy S2S/ACD/DHC launcher. New SACR/RAPF/QA-HNL runs live under scripts/new_method_v2/."
echo "GPU: ${GPU_ID}"
echo ""

run_step \
  "[1/3] S2S-only fixed reference" \
  "${BASE_MASTER_PORT}" \
  ./scripts/scanrefer/two-stage/block1_s2s_only_scanrefer.sh

run_step \
  "[2/3] S2S + ACD with rank loss down-weighted to 0.25" \
  "$((BASE_MASTER_PORT + 1))" \
  env EXTRA_ARGS='--acd_rank_weight 0.25 --log_dir ./logs/scanrefer_spacy/two-stage/block2_s2s_acd_rank025_scanrefer' \
  ./scripts/scanrefer/two-stage/block2_s2s_acd_scanrefer.sh

run_step \
  "[3/3] S2S + ACD + DHC fixed mainline" \
  "$((BASE_MASTER_PORT + 2))" \
  ./scripts/scanrefer/two-stage/block5_s2s_acd_dhc_scanrefer.sh

echo "=========================================="
echo "Priority runs complete."
echo "Check logs under ./logs/scanrefer_spacy/two-stage/"
echo "=========================================="
