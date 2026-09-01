#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage163a_dependency_status.txt"
TRAIN_DUMP="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
OUTPUT="${ROOT}/stage163a_tier3_residual_blend"

fail_status() {
  local rc=$?
  printf 'stage163a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test -s "${TRAIN_DUMP}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  train_joint_option_ranker.py stage140_train_eval_nested_blend.py \
  stage162_tier3_option_ranker.py stage163_tier3_residual_blend.py \
  test_stage163_tier3_residual_blend.py
"${PYTHON}" -m unittest -v test_stage163_tier3_residual_blend.py
"${PYTHON}" train_joint_option_ranker.py self-test

printf 'stage163a_training_fixed_tier3_residual_blend %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage163_tier3_residual_blend.py train \
  "${TRAIN_DUMP}" "${STAGE31_LOCK}" "${STAGE33_LOCK}" "${STAGE142_LOCK}" \
  "${OUTPUT}" --num-threads 16 \
  2>&1 | tee "${P}/stage163a_tier3_residual_blend_train.log"

printf 'stage163a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
