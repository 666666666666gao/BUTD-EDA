#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
TB_ROOT=${TB_ROOT:-/root/tf-logs}
MASTER_PORT=${MASTER_PORT:-$RANDOM}

mkdir -p "${TB_ROOT}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

if command -v torchrun >/dev/null 2>&1; then
  DIST_LAUNCH=(torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}")
else
  DIST_LAUNCH=(python -m torch.distributed.run --nproc_per_node 1 --master_port "${MASTER_PORT}")
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${DIST_LAUNCH[@]}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${DATA_ROOT}" \
    --batch_size 24 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset scanrefer_spacy --test_dataset scanrefer_spacy \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend \
    --print_freq 1000 \
    --val_freq 5 \
    --save_freq 5 \
    --max_epoch 300 \
    --lr_decay_epochs 65 \
    --tensorboard_root "${TB_ROOT}" \
    --log_dir "./logs/scanrefer_spacy/onestage/block2_s2s_acd_scanrefer" \
    --use_structured_slots \
    --use_late_acd \
    --slot_pooling attention \
    --max_rel_anchor_pairs 3 \
    --acd_top_m_targets 28 \
    --acd_top_k_anchors 10 \
    --acd_global_residual_alpha 0.25 \
    --acd_initial_alpha 0.05 \
    --acd_warmup_steps 10000 \
    --acd_ea_scale 0.8
