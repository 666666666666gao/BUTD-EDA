#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
MASTER_PORT=${MASTER_PORT:-$RANDOM}
PP_CHECKPOINT=${PP_CHECKPOINT:-${DATA_ROOT}/gf_detector_l6o256.pth}


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
    --val_freq 5 --batch_size 24 --save_freq 5 --print_freq 1000 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "./logs/sr3d/block6_full_method_sr3d" \
    --lr_decay_epochs 25 27 \
    --pp_checkpoint "${PP_CHECKPOINT}" \
    --butd_gt --self_attend \
    --use_structured_slots \
    --use_late_acd \
    --use_dhc \
    --slot_pooling attention \
    --max_rel_anchor_pairs 2 \
    --acd_top_m_targets 24 \
    --acd_top_k_anchors 8 \
    --acd_use_confidence_fusion \
    --acd_global_residual_alpha 0.35 \
    --acd_ea_scale 0.5 \
    --dhc_consistency_weight 0.2 \
    --dhc_ent_hardneg_weight 0.2 \
    --dhc_attr_hardneg_weight 0.05 \
    --dhc_rel_hardneg_weight 0.25
