#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STAGE153C_STATUS="${P}/stage153c_dependency_status.txt"
STAGE153A_EXIT="${P}/stage153a_wrapper.exitcode"
STATUS="${P}/stage154a_dependency_status.txt"
OUTPUT="${ROOT}/stage154a_train_only_oof_source_selector"
RAW_TRAIN="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
SOURCE_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
COMPACT_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"

fail_status() {
  local rc=$?
  printf 'stage154a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage154a_waiting_for_stage153c %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage153c_(complete_|not_authorized|failed)' \
  "${STAGE153C_STATUS}" 2>/dev/null; do
  sleep 60
done

if grep -q '^stage153c_complete_goal_met ' "${STAGE153C_STATUS}"; then
  printf 'stage154a_skipped_stage153_goal_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test "$(tr -d '[:space:]' < "${STAGE153A_EXIT}")" = "0"
test -s "${RAW_TRAIN}"
test -s "${SOURCE_TRAIN}"
test -s "${COMPACT_TRAIN}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage153_train_source_selector.py \
  stage154_oof_source_selector.py \
  test_stage154_oof_source_selector.py
"${PYTHON}" -m unittest -v test_stage154_oof_source_selector.py

printf 'stage154a_training_oof %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage154_oof_source_selector.py train \
  "${RAW_TRAIN}" \
  "${SOURCE_TRAIN}" \
  "${COMPACT_TRAIN}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${OUTPUT}" \
  2>&1 | tee "${P}/stage154a_oof_train.log"

printf 'stage154a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
