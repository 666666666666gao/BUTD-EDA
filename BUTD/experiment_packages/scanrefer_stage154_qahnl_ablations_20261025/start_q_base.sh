#!/usr/bin/env bash
# Start the Stage154 base stage (65 epochs, official detector init) for one
# QAHNL internal ablation row, using the same compile/preflight/smoke/execute
# path as the S0-S3 / R0-R3 rows.
#
#   Q1: w/o Hungarian matching      (qahnl_disable_hungarian_pos_rescue)
#   Q2: IoU-only positives          (+ qahnl_disable_top_iou_pos)
#   Q3: Softplus -> Hinge           (qahnl_loss_form=hinge; needs a runtime
#                                    that declares this flag)
#
# Usage: start_q_base.sh <Q1|Q2|Q3> [runtime]
set -Eeuo pipefail

ROW="$1"
RUNTIME="${2:-/home/gb/butd_stage154_runtime_20260907_v2}"
PY=/root/miniconda3/envs/bdetr/bin/python
PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/gb/butd_stage154_qahnl_ablations_20261025
REF=/home/gb/butd_stage154_ablations_20260907
DATA_MANIFEST="${REF}/control/frozen_s0_base_single_server_no_weights_20260923/required_annotations.sha256"
PLAN="${ROOT}/plans/${ROW}_base_v1.json"
CONTROL="${ROOT}/control/${ROW}_base"

export PATH="$(dirname "${PY}"):${PATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${RUNTIME}:${RUNTIME}/pointnet2"

mkdir -p "${ROOT}/plans" "${CONTROL}"
status() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "${CONTROL}/status.log"; }

status COMPILE
"${PY}" "${PKG}/compile_stage.py" \
  --history "${REF}/historical_base_configs.json" "${REF}/historical_stage_configs.json" \
  --runtime "${RUNTIME}" --row "${ROW}" --stage base --parent-epoch 0 \
  --log-dir "${ROOT}/${ROW}/base" --data-root /root/autodl-tmp/DATA_ROOT \
  --output "${PLAN}"

status PREFLIGHT
"${PY}" "${PKG}/execute_stage.py" "${PLAN}" \
  --data-manifest "${DATA_MANIFEST}" --preflight-only > "${CONTROL}/preflight.log" 2>&1

status BASE_SMOKE_RUNNING
cd "${RUNTIME}"
torchrun --nproc_per_node=1 --master_port=33580 \
  "${PKG}/smoke_base.py" "${PLAN}" "${CONTROL}/smoke" > "${CONTROL}/smoke.log" 2>&1
grep -q BASE_TRAINING_SMOKE_PASS "${CONTROL}/smoke/smoke_receipt.json"
# Smoke weights are diagnostic only and never used as a parent.
find "${CONTROL}/smoke" -name '*.pth' -delete

status FORMAL_BASE_RUNNING
"${PY}" "${PKG}/execute_stage.py" "${PLAN}" \
  --data-manifest "${DATA_MANIFEST}" --master-port=33581 > "${CONTROL}/formal.log" 2>&1
status BASE_COMPLETE_DOWNSTREAM_REQUIRED
