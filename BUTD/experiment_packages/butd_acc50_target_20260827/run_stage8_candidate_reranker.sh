#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${REPO_ROOT}"

SOURCE="${SOURCE_CHECKPOINT_OVERRIDE:-${REPO_ROOT}/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156/ckpt_best_primary.pth}"
JOB_ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage8_candidate_reranker"
[ -f "${SOURCE}" ] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 181; }
SOURCE_EPOCH="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SOURCE}")"
MAX_EPOCH="$((SOURCE_EPOCH + ${STAGE8_NEW_EPOCHS:-3}))"

require_files
base_command
CMD+=(
  --log_dir "${JOB_ROOT}"
  --max_epoch "${MAX_EPOCH}" --val_freq 1 --save_freq 1000
  --checkpoint_path "${SOURCE}"
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.25_top1
  --best_checkpoint_min_delta 0
  --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241
  --best_checkpoint_constraint_epsilon 0
  --use_structured_slots --use_sacr
  --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality
  --rapf_quality_weight 0.25 --rapf_struct_residual_clip 0.25
  --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --use_detector_policy_adapter --detector_policy_adapter_train_only
  --detector_policy_adapter_hidden_dim 64
  --detector_policy_adapter_k 5
  --detector_policy_adapter_delta_scale 0.50
  --detector_policy_adapter_lr 0.001
  --detector_policy_adapter_loss_weight 1.0
  --detector_policy_adapter_margin 0.05
  --detector_policy_adapter_min_iou_gap 0.02
  --eval_use_detector_policy_adapter_scores
  --verbose_diagnostics
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf 'source_epoch=%q max_epoch=%q ' "${SOURCE_EPOCH}" "${MAX_EPOCH}"
  printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${JOB_ROOT}" ] || { echo "refusing to overwrite ${JOB_ROOT}" >&2; exit 182; }
mkdir -p "${JOB_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${PACKAGE_ROOT}/stage8_train.log"
