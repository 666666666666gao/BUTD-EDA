#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
source "${OLD_PACKAGE}/common.sh"
cd "${REPO_ROOT}"

STAGE1_ROOT="${STAGE1_ROOT:-/root/autodl-tmp/logs/butd_acc50_target_20260827/stage1_qahnl_iou50_universal_only}"
JOB_ROOT="${STAGE2_JOB_ROOT:-/root/autodl-tmp/logs/butd_acc50_target_20260827/stage2_qahnl_iou50_full_finetune}"
mapfile -t stage1_runs < <(find "${STAGE1_ROOT}/scanrefer_spacy" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ "${#stage1_runs[@]}" -eq 1 ] || {
  echo "ERROR: expected one Stage-1 run, found ${#stage1_runs[@]}" >&2
  exit 110
}
SOURCE_RUN="${stage1_runs[0]}"
SOURCE_CHECKPOINT="${SOURCE_RUN}/ckpt_best_primary.pth"
[ -f "${SOURCE_CHECKPOINT}" ] || {
  echo "ERROR: Stage-1 best checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
  exit 111
}
SOURCE_EPOCH="$("${PYTHON}" -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SOURCE_CHECKPOINT}")"
MAX_EPOCH="$((SOURCE_EPOCH + 6))"

require_files
base_command
CMD+=(
  --log_dir "${JOB_ROOT}"
  --max_epoch "${MAX_EPOCH}" --val_freq 1 --save_freq 1000
  --lr=1e-5 --lr_backbone=1e-4 --text_encoder_lr=1e-6
  --lr_decay_epochs 3 5
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
  --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --qahnl_pos_iou_thresh 0.50
  --qahnl_neg_iou_thresh 0.25
  --qahnl_disable_top_iou_pos
  --qahnl_disable_hungarian_pos_rescue
  --qahnl_loss_weight 0.40
  --quality_loss_weight 1.0
  --quality_iou_threshold 0.50
  --quality_topk_rerank_weight 0.20
  --quality_topk_rerank_source fused
  --quality_topk_rerank_k 5
  --quality_topk_rerank_margin 0.05
  --quality_topk_rerank_min_iou_gap 0.02
  --eval_use_fused_scores
  --verbose_diagnostics
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'source_epoch=%q max_epoch=%q ' "${SOURCE_EPOCH}" "${MAX_EPOCH}"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${JOB_ROOT}" ] || {
  echo "ERROR: refusing to overwrite existing run root ${JOB_ROOT}" >&2
  exit 112
}
mkdir -p "${JOB_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
RUN_DIR="$(latest_run_dir "${JOB_ROOT}")"
EVAL_LOG="${RUN_DIR}/eval_epoch_last.log"
{
  echo "source_run=${SOURCE_RUN}"
  echo "source_checkpoint=${SOURCE_CHECKPOINT}"
  echo "source_epoch=${SOURCE_EPOCH}"
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint=${RUN_DIR}/ckpt_best_primary.pth"
  sha256sum "${RUN_DIR}/ckpt_best_primary.pth"
  grep -E 'last__bbs_(unique_|multiple_)?acc0\.(25|50)_top1:' "${EVAL_LOG}" || true
  grep -E 'last__bbs_acc0\.(25|50)_top1:' "${EVAL_LOG}" || true
  cat "${RUN_DIR}/best_primary.json"
} > "${JOB_ROOT}/summary.txt"
cat "${JOB_ROOT}/summary.txt"

ACC50_PACKAGE="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
"${ACC50_PACKAGE}/verify_stage2_reload.sh" "${RUN_DIR}"
