#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
UPSTREAM="${P}/stage156b_dependency_status.txt"
STATUS="${P}/stage157a_dependency_status.txt"
OUTPUT="${ROOT}/stage157a_all_train_refit_selector"
RAW_TRAIN="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
SOURCE_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
COMPACT_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE154_LOCK="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"

fail_status() {
  local rc=$?
  printf 'stage157a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage157a_waiting_for_stage156b %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage156b_(complete_|not_authorized|failed)' \
  "${UPSTREAM}" 2>/dev/null; do
  sleep 60
done
if grep -q '^stage156b_complete_goal_met ' "${UPSTREAM}"; then
  printf 'stage157a_skipped_stage156_goal_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi
if ! grep -q '^stage156b_complete_goal_not_met ' "${UPSTREAM}"; then
  printf 'stage157a_not_authorized_stage156b_unavailable %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test -s "${RAW_TRAIN}"
test -s "${SOURCE_TRAIN}"
test -s "${COMPACT_TRAIN}"
test -s "${STAGE154_LOCK}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile stage157_all_train_refit_selector.py
"${PYTHON}" -m pytest -q test_stage157_all_train_refit_selector.py
printf 'stage157a_refitting_all_train %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage157_all_train_refit_selector.py refit \
  "${RAW_TRAIN}" \
  "${SOURCE_TRAIN}" \
  "${COMPACT_TRAIN}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${STAGE154_LOCK}" \
  "${OUTPUT}" \
  2>&1 | tee "${P}/stage157a_all_train_refit.log"
printf 'stage157a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
