#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
DATA_ROOT="${DATA_ROOT%/}/"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT}gf_detector_l6o256.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-33331}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${MODE}" in
    smoke)
        MAX_EPOCH="${MAX_EPOCH:-1}"
        BATCH_SIZE="${BATCH_SIZE:-2}"
        NUM_WORKERS="${NUM_WORKERS:-0}"
        VAL_FREQ="${VAL_FREQ:-1}"
        SAVE_FREQ="${SAVE_FREQ:-1}"
        PRINT_FREQ="${PRINT_FREQ:-1}"
        EXTRA_ARGS=(--debug)
        ;;
    tune)
        MAX_EPOCH="${MAX_EPOCH:-30}"
        BATCH_SIZE="${BATCH_SIZE:-24}"
        NUM_WORKERS="${NUM_WORKERS:-8}"
        VAL_FREQ="${VAL_FREQ:-1}"
        SAVE_FREQ="${SAVE_FREQ:-1}"
        PRINT_FREQ="${PRINT_FREQ:-100}"
        EXTRA_ARGS=()
        ;;
    long)
        MAX_EPOCH="${MAX_EPOCH:-120}"
        BATCH_SIZE="${BATCH_SIZE:-24}"
        NUM_WORKERS="${NUM_WORKERS:-8}"
        VAL_FREQ="${VAL_FREQ:-3}"
        SAVE_FREQ="${SAVE_FREQ:-3}"
        PRINT_FREQ="${PRINT_FREQ:-500}"
        EXTRA_ARGS=()
        ;;
    *)
        echo "Usage: $0 [smoke|tune|long]" >&2
        exit 2
        ;;
esac

LOG_ROOT="${LOG_ROOT:-./logs/eda_sacr_rapf_qahnl/${MODE}}"

# Default to the retained decomposed-data augmentation policy from tuning.
AUG_ARGS=()
if [[ "${ENABLE_AUGMENT_DET:-0}" == "1" ]]; then
    AUG_ARGS+=(--augment_det)
fi
if [[ "${DISABLE_BOX_JITTER:-1}" == "1" ]]; then
    AUG_ARGS+=(--disable_box_jitter)
fi
if [[ "${SPACY_RELATION_FREE_YAW_ONLY_AUG:-1}" == "1" ]]; then
    AUG_ARGS+=(--spacy_relation_free_yaw_only_aug)
fi

export CUDA_VISIBLE_DEVICES
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

python -m torch.distributed.launch \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    train_dist_mod.py \
    --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${DATA_ROOT}" \
    --max_epoch "${MAX_EPOCH}" \
    --val_freq "${VAL_FREQ}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_freq "${SAVE_FREQ}" \
    --print_freq "${PRINT_FREQ}" \
    --lr_backbone 2e-3 \
    --lr 2e-4 \
    --lr_decay_epochs 50 75 \
    --dataset scanrefer_spacy \
    --test_dataset scanrefer_spacy \
    --detect_intermediate \
    --joint_det \
    --use_soft_token_loss \
    --use_contrastive_align \
    --log_dir "${LOG_ROOT}" \
    --pp_checkpoint "${PP_CHECKPOINT}" \
    --butd \
    --self_attend \
    "${AUG_ARGS[@]}" \
    --use_structured_slots \
    --use_sacr \
    --use_rapf \
    --use_reliability_gate \
    --use_quality_head \
    --rapf_use_quality \
    --use_qahnl \
    --eval_use_fused_scores \
    --rapf_struct_residual_clip "${RAPF_STRUCT_RESIDUAL_CLIP:-0.25}" \
    --rapf_quality_weight "${RAPF_QUALITY_WEIGHT:-0.75}" \
    --rapf_gate_loss_weight "${RAPF_GATE_LOSS_WEIGHT:-0.1}" \
    --rapf_initial_gate_bias "${RAPF_INITIAL_GATE_BIAS:--2.5}" \
    --rapf_generic_gate_cap "${RAPF_GENERIC_GATE_CAP:-0.1}" \
    --qahnl_score_source "${QAHNL_SCORE_SOURCE:-fused}" \
    --qahnl_loss_weight "${QAHNL_LOSS_WEIGHT:-0.2}" \
    "${EXTRA_ARGS[@]}"
