#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
PP_CHECKPOINT=${PP_CHECKPOINT:-${DATA_ROOT}/gf_detector_l6o256.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
DIAG=${DIAG:-0}

NMV2_BATCH_SIZE=${NMV2_BATCH_SIZE:-24}
NMV2_MAX_EPOCH=${NMV2_MAX_EPOCH:-300}
NMV2_VAL_FREQ=${NMV2_VAL_FREQ:-5}
NMV2_SAVE_FREQ=${NMV2_SAVE_FREQ:-5}
NMV2_PRINT_FREQ=${NMV2_PRINT_FREQ:-1000}
NMV2_LR_DECAY_EPOCHS=${NMV2_LR_DECAY_EPOCHS:-"25 27"}
NMV2_LOG_ROOT=${NMV2_LOG_ROOT:-./logs/new_method_v2}
# Active NR3D mainline uses class-supervised detector boxes.
# Earlier GT/intermediate-box results are legacy-only and not comparable.

DRY_RUN=0
USER_ARGS=()
for arg in "$@"; do
  if [ "${arg}" = "--dry-run" ]; then
    DRY_RUN=1
  else
    USER_ARGS+=("${arg}")
  fi
done

DIAG_ARGS=()
if [ "${DIAG}" = "1" ]; then
  DIAG_ARGS=(--verbose_diagnostics --eval_report_diagnostic_scores)
fi

EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
fi
read -r -a LR_DECAY_EPOCHS_ARR <<< "${NMV2_LR_DECAY_EPOCHS}"

if command -v torchrun >/dev/null 2>&1; then
  DIST_LAUNCH=(torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}")
else
  DIST_LAUNCH=(python -m torch.distributed.run --nproc_per_node 1 --master_port "${MASTER_PORT}")
fi

CMD=(
  "${DIST_LAUNCH[@]}"
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}"
  --val_freq "${NMV2_VAL_FREQ}" --batch_size "${NMV2_BATCH_SIZE}"
  --save_freq "${NMV2_SAVE_FREQ}" --print_freq "${NMV2_PRINT_FREQ}"
  --max_epoch "${NMV2_MAX_EPOCH}"
  --lr_backbone=1e-3 --lr=1e-4
  --dataset nr3d_spacy --test_dataset nr3d_spacy
  --joint_det
  --use_soft_token_loss --use_contrastive_align
  --log_dir "${NMV2_LOG_ROOT}/nr3d/01_baseline"
  --lr_decay_epochs "${LR_DECAY_EPOCHS_ARR[@]}"
  --pp_checkpoint "${PP_CHECKPOINT}"
  --butd_cls --self_attend
  "${DIAG_ARGS[@]}"
  "${USER_ARGS[@]}"
  "${EXTRA_ARGS_ARR[@]}"
)

if [ "${DRY_RUN}" = "1" ]; then
  printf 'TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
