#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage164a_dependency_status.txt"
TRAIN_DUMP="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
OLD_STAGE142="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
OUTPUT="${ROOT}/stage164a_same_domain_retrained_trio"

fail_status() {
  local rc=$?
  printf 'stage164a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test -s "${TRAIN_DUMP}"
test -s "${OLD_STAGE142}"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  train_joint_option_ranker.py stage140_train_eval_nested_blend.py \
  stage164_same_domain_trio_gate.py test_stage164_same_domain_trio_gate.py
"${PYTHON}" -m unittest -v test_stage164_same_domain_trio_gate.py
"${PYTHON}" train_joint_option_ranker.py self-test

printf 'stage164a_training_ordinal %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" train_joint_option_ranker.py train \
  "${TRAIN_DUMP}" "${OUTPUT}/ordinal" --num-threads 16 \
  2>&1 | tee "${P}/stage164a_ordinal_train.log"

printf 'stage164a_training_binary50 %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" train_joint_option_ranker.py binary50-train \
  "${TRAIN_DUMP}" "${OUTPUT}/binary50" --num-threads 16 \
  2>&1 | tee "${P}/stage164a_binary50_train.log"

printf 'stage164a_training_pointwise %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" train_joint_option_ranker.py pointwise-train \
  "${TRAIN_DUMP}" "${OUTPUT}/pointwise" --num-threads 16 \
  2>&1 | tee "${P}/stage164a_pointwise_train.log"

printf 'stage164a_locking_inner_blend %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" train_joint_option_ranker.py blend-train \
  "${TRAIN_DUMP}" \
  "${OUTPUT}/ordinal/joint_option_ranker.txt" \
  "${OUTPUT}/ordinal/locked_policy.json" \
  "${OUTPUT}/binary50/binary50_option_ranker.txt" \
  "${OUTPUT}/binary50/locked_binary50_policy.json" \
  "${OUTPUT}/inner_blend" \
  2>&1 | tee "${P}/stage164a_inner_blend.log"

printf 'stage164a_locking_nested_blend %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage140_train_eval_nested_blend.py train \
  "${TRAIN_DUMP}" \
  "${OUTPUT}/inner_blend/locked_blend_policy.json" \
  "${OUTPUT}/pointwise/locked_pointwise_policy.json" \
  "${OUTPUT}/locked_same_domain_trio_nested_policy.json" \
  2>&1 | tee "${P}/stage164a_nested_blend.log"

"${PYTHON}" stage164_same_domain_trio_gate.py authorize \
  "${OLD_STAGE142}" \
  "${OUTPUT}/locked_same_domain_trio_nested_policy.json" \
  "${OUTPUT}/locked_same_domain_trio_authorization.json" \
  2>&1 | tee "${P}/stage164a_internal_gate.log"

printf 'stage164a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" \
  "${OUTPUT}/locked_same_domain_trio_nested_policy.json" \
  "${OUTPUT}/locked_same_domain_trio_authorization.json"
trap - ERR
