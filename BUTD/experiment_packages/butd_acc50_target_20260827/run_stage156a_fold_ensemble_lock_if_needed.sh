#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
UPSTREAM="${P}/stage155b_dependency_status.txt"
STATUS="${P}/stage156a_dependency_status.txt"
OUTPUT="${ROOT}/stage156a_train_only_five_fold_mean_selector"
RAW_TRAIN="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
SOURCE_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
COMPACT_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE155_LOCK="${ROOT}/stage155a_train_only_fold_routed_oof_selector/locked_fold_routed_oof_selector.json"

fail_status() {
  local rc=$?
  printf 'stage156a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage156a_waiting_for_stage155b %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage155b_(complete_|not_authorized|failed)' \
  "${UPSTREAM}" 2>/dev/null; do
  sleep 60
done
if grep -q '^stage155b_complete_goal_met ' "${UPSTREAM}"; then
  printf 'stage156a_skipped_stage155_goal_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi
if ! grep -q '^stage155b_not_authorized_internal_gate_failed ' "${UPSTREAM}"; then
  printf 'stage156a_not_authorized_stage155b_unavailable %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test -s "${RAW_TRAIN}"
test -s "${SOURCE_TRAIN}"
test -s "${COMPACT_TRAIN}"
test -s "${STAGE155_LOCK}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile stage156_fold_ensemble_selector.py
"${PYTHON}" -m pytest -q test_stage156_fold_ensemble_selector.py
printf 'stage156a_locking_five_fold_mean %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage156_fold_ensemble_selector.py lock \
  "${RAW_TRAIN}" \
  "${SOURCE_TRAIN}" \
  "${COMPACT_TRAIN}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${STAGE155_LOCK}" \
  "${OUTPUT}" \
  2>&1 | tee "${P}/stage156a_fold_ensemble_lock.log"
printf 'stage156a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
