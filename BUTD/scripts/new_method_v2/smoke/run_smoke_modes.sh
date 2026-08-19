#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29570}
PP_CHECKPOINT=${PP_CHECKPOINT:-${DATA_ROOT}/gf_detector_l6o256.pth}
SCANREFER_STAGE=${SCANREFER_STAGE:-two_stage}
DATASET=${DATASET:-scanrefer_spacy}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DIAG_ARGS=(--verbose_diagnostics --eval_report_diagnostic_scores)

if [ "${SCANREFER_STAGE}" != "two_stage" ]; then
  echo "five-mode smoke is defined only for ScanRefer two_stage" >&2
  exit 2
fi

SMOKE_LOG_ROOT=${SMOKE_LOG_ROOT:-./logs/new_method_v2/smoke/scanrefer/two_stage}

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

if command -v torchrun >/dev/null 2>&1; then
  DIST_BIN=(torchrun --nproc_per_node 1)
else
  DIST_BIN=(python -m torch.distributed.run --nproc_per_node 1)
fi

run_mode() {
  local mode=$1
  local port=$2
  shift 2
  TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${DIST_BIN[@]}" --master_port "${port}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${DATA_ROOT}" \
    --val_freq 1 --batch_size 2 --save_freq 1 --print_freq 1 \
    --max_epoch 1 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset "${DATASET}" --test_dataset "${DATASET}" \
    --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "${SMOKE_LOG_ROOT}/${mode}" \
    --lr_decay_epochs 1 \
    --pp_checkpoint "${PP_CHECKPOINT}" \
    --butd --self_attend --augment_det \
    "${DIAG_ARGS[@]}" \
    "$@"
  python scripts/new_method_v2/smoke/check_smoke_log.py \
    --mode "${mode}" "${SMOKE_LOG_ROOT}/${mode}"
}

run_mode baseline "$((MASTER_PORT_BASE + 0))"
run_mode quality_only "$((MASTER_PORT_BASE + 1))" --use_quality_head
run_mode sacr_only "$((MASTER_PORT_BASE + 2))" --use_structured_slots --use_sacr --eval_use_structured_scores
run_mode rapf_quality "$((MASTER_PORT_BASE + 3))" --use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality --eval_use_fused_scores
run_mode full "$((MASTER_PORT_BASE + 4))" --use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality --use_qahnl --eval_use_fused_scores
