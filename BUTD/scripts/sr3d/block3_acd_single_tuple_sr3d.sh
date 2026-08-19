#!/bin/bash
# Block 3 - Ablation 2: Single tuple only
# Tests need for multi-tuple reasoning

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT=/root/autodl-tmp/DATA_ROOT
LOG_ROOT=/root/autodl-tmp/logs

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

CUDA_VISIBLE_DEVICES=0 "${DIST_LAUNCH[@]}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${DATA_ROOT} \
    --val_freq 5 --batch_size 40 --save_freq 5 --print_freq 100 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ${LOG_ROOT}/block3_acd_single_tuple_sr3d \
    --lr_decay_epochs 25 27 \
    --pp_checkpoint ${DATA_ROOT}/gf_detector_l6o256.pth \
    --butd_gt --self_attend \
    --num_workers 8 \
    --use_amp \
    --use_structured_slots \
    --use_late_acd \
    --slot_pooling attention \
    --max_rel_anchor_pairs 1 \
    --acd_top_m_targets 32 \
    --acd_top_k_anchors 16 \
    --acd_global_residual_alpha 0.5 \
    --acd_ea_scale 1.0 \
    --max_epoch 30
