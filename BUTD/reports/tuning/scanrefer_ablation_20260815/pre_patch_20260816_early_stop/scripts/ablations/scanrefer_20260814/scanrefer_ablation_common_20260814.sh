#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
cd "${REPO_ROOT}"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"
# Prevent each DataLoader worker from spawning a full BLAS/OpenMP thread pool.
# This only controls CPU scheduling; it does not change model/data semantics.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

: "${ABLATION_ID:?set ABLATION_ID}"
: "${ABLATION_FLAGS:=}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}"
OFFICIAL_INIT="${PP_CHECKPOINT:-${DATA_ROOT}/gf_detector_l6o256.pth}"
LOG_ROOT="${ABLATION_LOG_ROOT:-${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init}"
BATCH_SIZE="${NMV2_BATCH_SIZE:-24}"
MAX_EPOCH="${NMV2_MAX_EPOCH:-100}"
VAL_FREQ="${NMV2_VAL_FREQ:-5}"
PRINT_FREQ="${NMV2_PRINT_FREQ:-100}"
LR_DECAY_EPOCHS="${NMV2_LR_DECAY_EPOCHS:-65}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

read -r -a ABLATION_FLAGS_ARR <<< "${ABLATION_FLAGS}"

CMD=(
  torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}"
  --val_freq "${VAL_FREQ}" --batch_size "${BATCH_SIZE}"
  --save_freq 1000 --print_freq "${PRINT_FREQ}"
  --max_epoch "${MAX_EPOCH}"
  --lr_backbone=1e-3 --lr=1e-4
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --joint_det --use_soft_token_loss --use_contrastive_align
  --log_dir "${LOG_ROOT}/${ABLATION_ID}"
  --lr_decay_epochs "${LR_DECAY_EPOCHS}"
  --pp_checkpoint "${OFFICIAL_INIT}"
  --butd --self_attend --augment_det
  --rng_seed 0
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.25_top1
  --best_checkpoint_min_delta 0
  --verbose_diagnostics --eval_report_diagnostic_scores
  "${ABLATION_FLAGS_ARR[@]}"
  "$@"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
