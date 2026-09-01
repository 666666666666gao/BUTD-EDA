#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage97_nearthreshold_clean_target_only"
EXPECTED_SRC_SHA256="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_UTILS_SHA256="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSSES_SHA256="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
[ "$(sha256sum "${SRC}" | awk '{print $1}')" = "${EXPECTED_SRC_SHA256}" ] || {
  echo "source checkpoint SHA256 mismatch" >&2; exit 228
}
[ "$(sha256sum main_utils.py | awk '{print $1}')" = "${EXPECTED_MAIN_UTILS_SHA256}" ] || {
  echo "main_utils.py SHA256 mismatch" >&2; exit 229
}
[ "$(sha256sum models/losses.py | awk '{print $1}')" = "${EXPECTED_LOSSES_SHA256}" ] || {
  echo "models/losses.py SHA256 mismatch" >&2; exit 230
}
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SRC}")"

base_command
CMD+=(--log_dir "${OUT}" --max_epoch "$((E+1))" --val_freq 1 --save_freq 1000
 --checkpoint_path "${SRC}" --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.50_top1 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241 --best_checkpoint_constraint_epsilon 0
 --disable_train_augmentation --disable_box_jitter
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head
 --rapf_use_quality --rapf_quality_weight 0.25 --rapf_struct_residual_clip 0.25
 --rapf_gate_loss_weight 0.0 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --qahnl_loss_weight 0.0
 --quality_loss_weight 0.0
 --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0
 --detector_policy_adapter_loss_weight 0.0
 --detector_policy_geometry_loss_weight 0.0
 --detector_policy_rank2_rescue_loss_weight 0.0
 --detector_policy_adapter_margin 0.1 --detector_policy_adapter_min_iou_gap 0.02
 --last_box_head_train_only --last_box_head_lr 0.000005
 --last_box_head_freeze_bn_stats --last_box_standard_loss_scale 0.0
 --last_box_target_loss_weight 1.0
 --last_box_target_score_source detector_policy_adapter
 --last_box_target_iou_min 0.40 --last_box_target_iou_max 0.50
 --last_box_target_l1_weight 1.0 --last_box_target_giou_weight 1.0
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 227; }
mkdir -p "${OUT}"
{
  printf 'stage=97\nsource=%s\nsource_epoch=%s\n' "${SRC}" "${E}"
  printf 'source_sha256=%s\nmain_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_SRC_SHA256}" "${EXPECTED_MAIN_UTILS_SHA256}" "${EXPECTED_LOSSES_SHA256}"
  printf 'master_port=%s\n' "${MASTER_PORT}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
  printf 'started_at='; date --iso-8601=seconds
} > "${OUT}/launch_manifest.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${P}/stage97_train.log"
