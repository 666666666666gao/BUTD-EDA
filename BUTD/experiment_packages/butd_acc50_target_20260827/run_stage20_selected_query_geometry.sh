#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage14c_rescue_gate_stable/scanrefer_spacy/1787858656/ckpt_best_primary.pth"
OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage20_selected_query_geometry"
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SRC}")"

base_command
CMD+=(--log_dir "${OUT}" --max_epoch "$((E+1))" --val_freq 1 --save_freq 1000
 --checkpoint_path "${SRC}" --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.25_top1 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241 --best_checkpoint_constraint_epsilon 0
 --disable_train_augmentation
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head
 --rapf_use_quality --rapf_quality_weight 0.25 --rapf_struct_residual_clip 0.25
 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --use_detector_policy_adapter
 --detector_policy_adapter_train_only --detector_policy_geometry_train_only
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0 --detector_policy_adapter_lr 0.0002
 --detector_policy_adapter_loss_weight 0.0
 --detector_policy_geometry_loss_weight 1.0
 --detector_policy_adapter_margin 0.1 --detector_policy_adapter_min_iou_gap 0.02
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 227; }
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${P}/stage20_train.log"
