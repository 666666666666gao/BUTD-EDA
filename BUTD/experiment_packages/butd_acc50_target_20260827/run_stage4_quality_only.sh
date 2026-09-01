#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${OLD_PACKAGE}/common.sh"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
cd "${REPO_ROOT}"

ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SOURCE_SELECTION_JSON="${ROOT}/stage3_source_selection.json"
JOB_ROOT="${ROOT}/stage4_quality_head_top5"

if [ -n "${SOURCE_CHECKPOINT_OVERRIDE:-}" ]; then
  SOURCE_CHECKPOINT="$(realpath "${SOURCE_CHECKPOINT_OVERRIDE}")"
else
  [ -s "${SOURCE_SELECTION_JSON}" ] || {
    echo "ERROR: Stage-3 source selection is missing: ${SOURCE_SELECTION_JSON}" >&2
    exit 170
  }
  SOURCE_CHECKPOINT="$("${PYTHON}" -c 'import json,os,sys; print(os.path.realpath(json.load(open(sys.argv[1]))["selected"]["checkpoint"]))' "${SOURCE_SELECTION_JSON}")"
fi
[ -f "${SOURCE_CHECKPOINT}" ] || {
  echo "ERROR: Stage-4 source checkpoint missing: ${SOURCE_CHECKPOINT}" >&2
  exit 171
}
SOURCE_EPOCH="$("${PYTHON}" -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SOURCE_CHECKPOINT}")"
NEW_EPOCHS="${STAGE4_NEW_EPOCHS:-4}"
MAX_EPOCH="$((SOURCE_EPOCH + NEW_EPOCHS))"

require_files
base_command
CMD+=(
  --log_dir "${JOB_ROOT}"
  --max_epoch "${MAX_EPOCH}" --val_freq 1 --save_freq 1000
  --lr=1e-5 --lr_backbone=1e-4 --text_encoder_lr=1e-6
  --lr_decay_epochs 2 3
  --checkpoint_path "${SOURCE_CHECKPOINT}" --reduce_lr
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.25_top1
  --best_checkpoint_min_delta 0
  --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241
  --best_checkpoint_constraint_epsilon 0
  --use_structured_slots --use_sacr
  --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality
  --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25
  --rapf_gate_loss_weight 0.0
  --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --qahnl_pos_iou_thresh 0.50
  --qahnl_neg_iou_thresh 0.25
  --qahnl_disable_top_iou_pos
  --qahnl_disable_hungarian_pos_rescue
  --qahnl_loss_weight 0.50
  --quality_loss_weight 0.25
  --quality_iou_threshold 0.50
  --quality_topk_rerank_weight 2.0
  --quality_topk_rerank_source base
  --quality_topk_rerank_k 5
  --quality_topk_rerank_margin 0.10
  --quality_topk_rerank_min_iou_gap 0.05
  --quality_head_train_only
  --quality_head_lr 5e-4
  --eval_use_fused_scores
  --verbose_diagnostics
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'source_epoch=%q new_epochs=%q max_epoch=%q ' \
    "${SOURCE_EPOCH}" "${NEW_EPOCHS}" "${MAX_EPOCH}"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${JOB_ROOT}" ] || {
  echo "ERROR: refusing to overwrite existing Stage-4 root ${JOB_ROOT}" >&2
  exit 172
}
mkdir -p "${JOB_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

RUN_DIR="$(latest_run_dir "${JOB_ROOT}")"
CHECKPOINT="${RUN_DIR}/ckpt_best_primary.pth"
[ -f "${CHECKPOINT}" ] || {
  echo "ERROR: Stage-4 best checkpoint missing: ${CHECKPOINT}" >&2
  exit 173
}
{
  echo "source_checkpoint=${SOURCE_CHECKPOINT}"
  echo "source_epoch=${SOURCE_EPOCH}"
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint=${CHECKPOINT}"
  sha256sum "${CHECKPOINT}"
  cat "${RUN_DIR}/best_primary.json"
} > "${JOB_ROOT}/summary.txt"
cat "${JOB_ROOT}/summary.txt"

"${PACKAGE_ROOT}/run_stage4_quality_grid.sh" "${CHECKPOINT}"
