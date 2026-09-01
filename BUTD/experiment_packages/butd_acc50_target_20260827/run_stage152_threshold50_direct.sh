#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
OUT="${ROOT}/stage152a_threshold50_direct_rerank_from_stage150e13"
STATUS="${P}/stage152_threshold50_direct_status.txt"
STATE="${P}/state/stage152_threshold50_direct_pre_20260831"
MAIN_CANDIDATE="${P}/main_utils_stage152_candidate.py"
LOSS_CANDIDATE="${P}/losses_stage152_candidate.py"
TEST_NEW="${P}/test_stage152_threshold50_direct.py"

EXPECTED_SRC_SHA="8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
EXPECTED_MAIN_OLD_SHA="c5fcfd1f716cc87e2de3c39f3d26578566f48237a5f19996d559d1a939e67c47"
EXPECTED_LOSS_OLD_SHA="622de46f497883532e74f34e43f788482c55a170c614623e9b49f6edfe0df603"
EXPECTED_MAIN_NEW_SHA="960bb83e7fe9524d81bf48e0e23975558380371e2e73bb0af6eaeb5e925a79f2"
EXPECTED_LOSS_NEW_SHA="86a076c2d8f265b9ddcc809a9c55f4548e83d5ba50c827291e5aaf1b4d325d62"
EXPECTED_TRAIN_SHA="44852f403849266e5d706b39c86e99f04f3bc682652bf8eb06944e11557a25e0"
EXPECTED_BDETR_SHA="325444d2e7474b40764634f410d0f68827945493c3b1b7a73301a97c9da71879"
EXPECTED_POLICY_SHA="396303854f5acacba410093a3953ad7f1c56288a29f7422fa671e8aee3684166"
EXPECTED_TEST_NEW_SHA="d3f71d482497df44eaf24ae990eeb76ff709b50a8c1b89a658bff165321cec7f"

check_sha() {
  local path="$1" expected="$2" actual
  test -s "${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  test "${actual}" = "${expected}" || {
    printf 'SHA mismatch %s expected=%s actual=%s\n' \
      "${path}" "${expected}" "${actual}" >&2
    exit 228
  }
}

fail_status() {
  local rc=$?
  printf 'stage152_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

check_sha "${SRC}" "${EXPECTED_SRC_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_OLD_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_OLD_SHA}"
check_sha "${MAIN_CANDIDATE}" "${EXPECTED_MAIN_NEW_SHA}"
check_sha "${LOSS_CANDIDATE}" "${EXPECTED_LOSS_NEW_SHA}"
check_sha train_dist_mod.py "${EXPECTED_TRAIN_SHA}"
check_sha models/bdetr.py "${EXPECTED_BDETR_SHA}"
check_sha models/detector_policy_sources.py "${EXPECTED_POLICY_SHA}"
check_sha "${TEST_NEW}" "${EXPECTED_TEST_NEW_SHA}"
test ! -e "${OUT}"
test ! -e "${STATE}"

assert_gpu_idle
assert_storage
mkdir -p "${STATE}"
cp -p main_utils.py "${STATE}/main_utils.pre_stage152.py"
cp -p models/losses.py "${STATE}/losses.pre_stage152.py"
cp -p "${MAIN_CANDIDATE}" "${STATE}/main_utils.stage152_candidate.py"
cp -p "${LOSS_CANDIDATE}" "${STATE}/losses.stage152_candidate.py"

rollback_preflight() {
  local rc=$?
  install -m 0644 "${STATE}/main_utils.pre_stage152.py" main_utils.py
  install -m 0644 "${STATE}/losses.pre_stage152.py" models/losses.py
  printf 'stage152_preflight_failed_and_rolled_back rc=%s at=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" > "${STATUS}"
  exit "${rc}"
}
trap rollback_preflight ERR INT TERM
install -m 0644 "${MAIN_CANDIDATE}" main_utils.py
install -m 0644 "${LOSS_CANDIDATE}" models/losses.py
check_sha main_utils.py "${EXPECTED_MAIN_NEW_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_NEW_SHA}"

"${PYTHON}" -m py_compile \
  main_utils.py models/losses.py train_dist_mod.py "${TEST_NEW}"
PYTHONPATH="${R}" "${PYTHON}" "${TEST_NEW}"
PYTHONPATH="${R}" "${PYTHON}" -m pytest -q \
  tests/test_detector_policy_rescue_gate.py \
  tests/test_best_checkpoint_policy.py \
  tests/test_best_checkpoint_final_reload.py

E="$(${PYTHON} -c \
  'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' \
  "${SRC}")"
test "${E}" = 13

MODEL_ARGS=(
  --disable_train_augmentation --disable_box_jitter
  --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
  --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.0
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source adapter_hit50
  --qahnl_tiered_quality --qahnl_tier2_iou_thresh 0.50
  --qahnl_pos_iou_thresh 0.25 --qahnl_neg_iou_thresh 0.10
  --qahnl_tiered_margin21 0.20 --qahnl_tiered_margin10 0.10
  --qahnl_tiered_temperature 1.0 --qahnl_tier2_relation_weight 3.0
  --qahnl_threshold50_direct_weight 3.0
  --qahnl_loss_weight 1.0 --quality_loss_weight 0.0
  --use_detector_policy_adapter
  --detector_policy_adapter_hidden_dim 64
  --detector_policy_adapter_k 5
  --detector_policy_adapter_delta_scale 4.0
  --detector_policy_adapter_loss_weight 0.0
  --detector_policy_geometry_loss_weight 0.0
  --detector_policy_rank2_rescue_loss_weight 0.0
  --detector_policy_alignment_rescue_loss_weight 0.0
  --detector_policy_tier_pair_loss_weight 0.0
  --detector_policy_adapter_margin 0.1
  --detector_policy_adapter_min_iou_gap 0.02
  --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
  --verbose_diagnostics
)

base_command
TRAIN_CMD=("${CMD[@]}" --log_dir "${OUT}" --max_epoch 20
  --val_freq 1 --save_freq 1000 --checkpoint_path "${SRC}"
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.50_top1
  --best_checkpoint_min_delta 0
  --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0
  --best_checkpoint_constraint_epsilon 0
  --early_stopping
  --early_stopping_metric last__bbs_acc0.50_top1
  --early_stopping_min_epoch 14
  --early_stopping_patience 2
  --early_stopping_min_delta 0.0003
  "${MODEL_ARGS[@]}"
  --detector_policy_adapter_train_only
  --detector_policy_rerank_head_train_only
  --detector_policy_adapter_lr 0.00002)

# Code and default-zero loss parity are now proven.  Keep the Stage152 code live even
# if the subsequent training job fails, so the failure remains auditable.
trap fail_status ERR
trap 'exit 130' INT TERM
chmod 0444 "${STATE}"/*
mkdir -p "${OUT}"
{
  printf 'stage=152a_threshold50_direct_rerank_from_stage150e13\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s\nsource_epoch=%s\nsource_sha256=%s\n' \
    "${SRC}" "${E}" "${EXPECTED_SRC_SHA}"
  printf 'diagnosis=full_query_oracle050_7421_vs_top1_3979_and_top5_oracle_5402\n'
  printf 'trainable_scope=detector_policy_adapter.rerank_head_only\n'
  printf 'frozen_scope=query_mlp,scalar_mlp,geometry_and_all_non_adapter\n'
  printf 'tier2_relation_weight=3.0\nthreshold50_direct_weight=3.0\nlearning_rate=0.00002\n'
  printf 'validation_threshold_search=none\n'
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_MAIN_NEW_SHA}" "${EXPECTED_LOSS_NEW_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage152_training %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TRAIN_CMD[@]}" \
  2>&1 | tee "${P}/stage152_threshold50_direct.log"

check_sha main_utils.py "${EXPECTED_MAIN_NEW_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_NEW_SHA}"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
trap - ERR
printf 'stage152_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
