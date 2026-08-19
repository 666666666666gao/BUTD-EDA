#!/bin/bash
# Block 3 - Ablation 3: Reduced candidate pools (M=16, K=8)
# Uses the Block 2 v2 ACD configuration with smaller target/anchor pools.
# Set RESUME_EPOCH=20 to resume from ckpt_epoch_20.pth.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT=/root/autodl-tmp/DATA_ROOT
LOG_ROOT=/root/autodl-tmp/logs
EXPERIMENT_NAME=block3_acd_reduced_pool_sr3d_v2_eascale01_alpha03
RESUME_RUN_ID=${RESUME_RUN_ID:-1775219541}
RESUME_EPOCH=${RESUME_EPOCH:-}

if [[ -n "${RESUME_EPOCH}" ]]; then
  LOG_DIR_NAME=${LOG_DIR_NAME:-${EXPERIMENT_NAME}_resume_ep${RESUME_EPOCH}}
  CKPT_PATH=${CKPT_PATH:-${LOG_ROOT}/${EXPERIMENT_NAME}/sr3d_spacy/${RESUME_RUN_ID}/ckpt_epoch_${RESUME_EPOCH}.pth}
else
  LOG_DIR_NAME=${LOG_DIR_NAME:-${EXPERIMENT_NAME}}
  CKPT_PATH=${CKPT_PATH:-}
fi

mkdir -p "${LOG_ROOT}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=10
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

if command -v torchrun >/dev/null 2>&1; then
  DIST_LAUNCH=(torchrun --nproc_per_node 1 --master_port "$RANDOM")
else
  DIST_LAUNCH=(python -m torch.distributed.run --nproc_per_node 1 --master_port "$RANDOM")
fi

if [[ -n "${CKPT_PATH}" && ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

COMMON_ARGS=(
    train_dist_mod.py
    --num_decoder_layers 6
    --use_color
    --weight_decay 0.0005
    --data_root "${DATA_ROOT}"
    --val_freq 5
    --batch_size 24
    --save_freq 5
    --print_freq 1000
    --lr_backbone=1e-3
    --lr=1e-4
    --dataset sr3d_spacy
    --test_dataset sr3d_spacy
    --detect_intermediate
    --joint_det
    --use_soft_token_loss
    --use_contrastive_align
    --log_dir "${LOG_ROOT}/${LOG_DIR_NAME}"
    --lr_decay_epochs 25 27
    --pp_checkpoint "${DATA_ROOT}/gf_detector_l6o256.pth"
    --butd_gt
    --self_attend
    --num_workers 8
    --use_amp
    --use_structured_slots
    --use_late_acd
    --slot_pooling attention
    --max_rel_anchor_pairs 3
    --acd_top_m_targets 16
    --acd_top_k_anchors 8
    --acd_use_confidence_fusion
    --acd_global_residual_alpha 0.3
    --acd_initial_alpha 0.1
    --acd_warmup_steps 5000
    --structured_debug
    --acd_ea_scale 0.1
    --max_epoch 300
)

if [[ -n "${CKPT_PATH}" ]]; then
  COMMON_ARGS+=(--checkpoint_path "${CKPT_PATH}")
fi

CUDA_VISIBLE_DEVICES=0 "${DIST_LAUNCH[@]}" "${COMMON_ARGS[@]}"
