#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

STAGE153A_OUT="${ROOT}/stage153a_stage150_e13_train_source_dump"
STAGE153A_EXIT="${P}/stage153a_wrapper.exitcode"
STAGE153B_OUT="${ROOT}/stage153b_train_only_rich_source_selector"
STAGE142_TRAIN_DUMP="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
STAGE150_SOURCE_DUMP="${STAGE153A_OUT}/stage150_e13_train_source_features.pt"
STAGE150_COMPACT_DUMP="${STAGE153A_OUT}/stage150_e13_train_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STATUS="${P}/stage153b_dependency_status.txt"

fail_status() {
  local rc=$?
  printf 'stage153b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage153b_waiting_for_stage153a %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
while test ! -e "${STAGE153A_EXIT}"; do
  sleep 60
done

test "$(tr -d '[:space:]' < "${STAGE153A_EXIT}")" = "0"
grep -q '^stage153a_complete ' "${STAGE153A_OUT}/status.txt"
test -s "${STAGE150_SOURCE_DUMP}"
test -s "${STAGE150_COMPACT_DUMP}"
test "$(sha256sum "${R}/src/grounding_evaluator.py" | awk '{print $1}')" = \
  "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
test ! -e "${STAGE153B_OUT}"

cd "${P}"
"${PYTHON}" -m py_compile \
  stage153_train_source_selector.py \
  test_stage153_train_source_selector.py
"${PYTHON}" -m unittest -v test_stage153_train_source_selector.py

printf 'stage153b_training %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage153_train_source_selector.py train \
  "${STAGE142_TRAIN_DUMP}" \
  "${STAGE150_SOURCE_DUMP}" \
  "${STAGE150_COMPACT_DUMP}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${STAGE153B_OUT}" \
  2>&1 | tee "${P}/stage153b_train.log"

printf 'stage153b_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
