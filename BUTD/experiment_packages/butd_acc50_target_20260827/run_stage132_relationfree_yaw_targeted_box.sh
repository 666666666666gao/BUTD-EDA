#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage132_relationfree_yaw_targeted_box"
EXPECTED_SRC_SHA256="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_UTILS_SHA256="afcb88f2a9a268bf270ee71b0901a18876e732a08749a79b796b6262b9f30ba8"
EXPECTED_TRAIN_SHA256="4dd8bb64e1b33c66f755c624c3294a40f9615c726bbdfc89674bdcc935c88069"
EXPECTED_BDETR_SHA256="8840ade8f08f78ff89332dec376a935064443c839a792231eb29b06d05558ff5"
EXPECTED_POLICY_SHA256="7d615107551720000f924ba903303d886409bd2ebfd48e363fb484fa902b029e"
EXPECTED_LOSSES_SHA256="acb1eed32c1f16a3696b3d1b9dab13abf94dc5aecf7681be8e670fa26286a60a"
EXPECTED_EVALUATOR_SHA256="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_DATASET_SHA256="6a7483d719d09433f2b7763f73246b0daf43948741ec3394068471906d96a24a"
[ "$(sha256sum "${SRC}" | awk '{print $1}')" = "${EXPECTED_SRC_SHA256}" ] || {
  echo "source checkpoint SHA256 mismatch" >&2; exit 228
}
[ "$(sha256sum main_utils.py | awk '{print $1}')" = "${EXPECTED_MAIN_UTILS_SHA256}" ] || {
  echo "main_utils.py SHA256 mismatch" >&2; exit 229
}
[ "$(sha256sum models/losses.py | awk '{print $1}')" = "${EXPECTED_LOSSES_SHA256}" ] || {
  echo "models/losses.py SHA256 mismatch" >&2; exit 230
}
for spec in \
  "train_dist_mod.py:${EXPECTED_TRAIN_SHA256}" \
  "models/bdetr.py:${EXPECTED_BDETR_SHA256}" \
  "models/detector_policy_sources.py:${EXPECTED_POLICY_SHA256}" \
  "src/grounding_evaluator.py:${EXPECTED_EVALUATOR_SHA256}" \
  "src/joint_det_dataset.py:${EXPECTED_DATASET_SHA256}"; do
  path="${spec%%:*}"
  expected="${spec#*:}"
  [ "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}" ] || {
    echo "${path} SHA256 mismatch" >&2; exit 231
  }
done
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SRC}")"

base_command
CMD+=(--log_dir "${OUT}" --max_epoch "$((E+1))" --val_freq 1 --save_freq 1000
 --checkpoint_path "${SRC}" --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.50_top1 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241 --best_checkpoint_constraint_epsilon 0
 --disable_box_jitter
 --spacy_relation_free_yaw_only_aug
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
 --last_box_head_train_only --last_box_head_lr 0.00001
 --last_box_target_loss_weight 1.0
 --last_box_target_score_source detector_policy_adapter
 --last_box_target_iou_min 0.25 --last_box_target_iou_max 0.50
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
  printf 'stage=132\ncontrolled_change=spacy_relation_free_yaw_only_aug\nsource=%s\nsource_epoch=%s\n' "${SRC}" "${E}"
  printf 'source_sha256=%s\nmain_utils_sha256=%s\nlosses_sha256=%s\ndataset_sha256=%s\n' \
    "${EXPECTED_SRC_SHA256}" "${EXPECTED_MAIN_UTILS_SHA256}" \
    "${EXPECTED_LOSSES_SHA256}" "${EXPECTED_DATASET_SHA256}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
  printf 'started_at='; date --iso-8601=seconds
} > "${OUT}/launch_manifest.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${P}/stage132_train.log"
