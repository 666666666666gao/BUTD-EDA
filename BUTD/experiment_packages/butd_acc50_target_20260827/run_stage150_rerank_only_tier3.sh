#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
BASELINE_SMOKE="${ROOT}/stage148_prechange_source_smoke96/eval_results.json"
SMOKE_OUT="${ROOT}/stage150a_rerank_only_tier3_smoke96"
OUT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c"
STATUS="${P}/stage150b_rerank_only_tier3_status.txt"
STATE="${P}/state/stage150_rerank_only_tier3_pre_20260831"
MAIN_CANDIDATE="${P}/main_utils.stage150_candidate.py"
LOSS_CANDIDATE="${P}/losses.stage150_candidate.py"
TEST_NEW="${P}/test_stage150_rerank_only_tier_weight.py"
TEST_OLD="${P}/test_qahnl_tiered_ordinal.py"

EXPECTED_SRC_SHA="a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
EXPECTED_MAIN_OLD_SHA="665aed267508aa4a77f8ad014071d3a912642e8631a691fa9320c3f21aa14dd6"
EXPECTED_LOSS_OLD_SHA="a80d4a8934536f1b11f488aa7a38bce2ecf6bfd3ad9b732f81913e2ad78763b4"
EXPECTED_MAIN_NEW_SHA="c5fcfd1f716cc87e2de3c39f3d26578566f48237a5f19996d559d1a939e67c47"
EXPECTED_LOSS_NEW_SHA="622de46f497883532e74f34e43f788482c55a170c614623e9b49f6edfe0df603"
EXPECTED_TRAIN_SHA="44852f403849266e5d706b39c86e99f04f3bc682652bf8eb06944e11557a25e0"
EXPECTED_BDETR_SHA="325444d2e7474b40764634f410d0f68827945493c3b1b7a73301a97c9da71879"
EXPECTED_POLICY_SHA="396303854f5acacba410093a3953ad7f1c56288a29f7422fa671e8aee3684166"
EXPECTED_TEST_NEW_SHA="57b71e4a4edd4b26125d690e4e73ee9d161d76f72821cf6df60a61fd7ee24a46"
EXPECTED_TEST_OLD_SHA="14e041ccdd5f3addd62e18ab15476595db500093349054d2763ee3342c81f13e"

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
  printf 'stage150_failed rc=%s at=%s line=%s\n' \
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
check_sha "${TEST_OLD}" "${EXPECTED_TEST_OLD_SHA}"
test -s "${BASELINE_SMOKE}"
test ! -e "${SMOKE_OUT}"
test ! -e "${OUT}"
test ! -e "${STATE}"

assert_gpu_idle
assert_storage
mkdir -p "${STATE}"
cp -p main_utils.py "${STATE}/main_utils.pre_stage150.py"
cp -p models/losses.py "${STATE}/losses.pre_stage150.py"
cp -p "${MAIN_CANDIDATE}" "${STATE}/main_utils.stage150_candidate.py"
cp -p "${LOSS_CANDIDATE}" "${STATE}/losses.stage150_candidate.py"

rollback_preflight() {
  local rc=$?
  install -m 0644 "${STATE}/main_utils.pre_stage150.py" main_utils.py
  install -m 0644 "${STATE}/losses.pre_stage150.py" models/losses.py
  printf 'stage150_preflight_failed_and_rolled_back rc=%s at=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" > "${STATUS}"
  exit "${rc}"
}
trap rollback_preflight ERR INT TERM
install -m 0644 "${MAIN_CANDIDATE}" main_utils.py
install -m 0644 "${LOSS_CANDIDATE}" models/losses.py
check_sha main_utils.py "${EXPECTED_MAIN_NEW_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_NEW_SHA}"

"${PYTHON}" -m py_compile \
  main_utils.py models/losses.py train_dist_mod.py "${TEST_NEW}" "${TEST_OLD}"
PYTHONPATH="${R}" "${PYTHON}" "${TEST_NEW}"
PYTHONPATH="${R}" "${PYTHON}" "${TEST_OLD}"
PYTHONPATH="${R}" "${PYTHON}" -m pytest -q \
  tests/test_detector_policy_rescue_gate.py \
  tests/test_best_checkpoint_policy.py \
  tests/test_best_checkpoint_final_reload.py

E="$(${PYTHON} -c \
  'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' \
  "${SRC}")"
test "${E}" = 12

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
SMOKE_CMD=("${CMD[@]}" --eval --checkpoint_path "${SRC}"
  --eval_max_samples 96 --log_dir "${SMOKE_OUT}"
  --eval_results_json_path "${SMOKE_OUT}/eval_results.json"
  "${MODEL_ARGS[@]}")

base_command
TRAIN_CMD=("${CMD[@]}" --log_dir "${OUT}" --max_epoch 18
  --val_freq 1 --save_freq 1000 --checkpoint_path "${SRC}"
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.50_top1
  --best_checkpoint_min_delta 0
  --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0
  --best_checkpoint_constraint_epsilon 0
  --early_stopping
  --early_stopping_metric last__bbs_acc0.50_top1
  --early_stopping_min_epoch 13
  --early_stopping_patience 2
  --early_stopping_min_delta 0.0003
  "${MODEL_ARGS[@]}"
  --detector_policy_adapter_train_only
  --detector_policy_rerank_head_train_only
  --detector_policy_adapter_lr 0.00005)

mkdir -p "${SMOKE_OUT}"
printf 'stage150a_smoke %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${SMOKE_CMD[@]}" \
  2>&1 | tee "${P}/stage150a_rerank_only_tier3_smoke96.log"

"${PYTHON}" - "${BASELINE_SMOKE}" "${SMOKE_OUT}/eval_results.json" <<'PY'
import json, sys
before = json.load(open(sys.argv[1], encoding='utf-8'))
after = json.load(open(sys.argv[2], encoding='utf-8'))
keys = [key for key in before if key.startswith('last__bbs_acc')]
assert keys
for key in keys:
    assert before[key] == after[key], (key, before[key], after[key])
print('STAGE150_SMOKE96_SOURCE_PARITY_PASS')
PY

# Code and inference parity are now proven.  Keep the Stage150 code live even
# if the subsequent training job fails, so the failure remains auditable.
trap fail_status ERR
trap 'exit 130' INT TERM
chmod 0444 "${STATE}"/*
mkdir -p "${OUT}"
{
  printf 'stage=150b_rerank_only_tier3_from_stage135c\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s\nsource_epoch=%s\nsource_sha256=%s\n' \
    "${SRC}" "${E}" "${EXPECTED_SRC_SHA}"
  printf 'diagnosis=ranking_fix_break_cancellation\n'
  printf 'trainable_scope=detector_policy_adapter.rerank_head_only\n'
  printf 'frozen_scope=query_mlp,scalar_mlp,geometry_and_all_non_adapter\n'
  printf 'tier2_relation_weight=3.0\nlearning_rate=0.00005\n'
  printf 'validation_threshold_search=none\n'
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_MAIN_NEW_SHA}" "${EXPECTED_LOSS_NEW_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage150b_training %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TRAIN_CMD[@]}" \
  2>&1 | tee "${P}/stage150b_rerank_only_tier3.log"

check_sha main_utils.py "${EXPECTED_MAIN_NEW_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_NEW_SHA}"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
trap - ERR
printf 'stage150b_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
