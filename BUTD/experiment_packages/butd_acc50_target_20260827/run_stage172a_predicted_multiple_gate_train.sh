#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage172a_dependency_status.txt"
OUT="${ROOT}/stage172a_predicted_multiple_gate_stage154_stage165"
STAGE171="${ROOT}/stage171a_invariant_fix_break_risk_stage154_stage165/locked_invariant_fix_break_risk.json"

fail_status() {
  local rc=$?
  printf 'stage172a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage171a_complete ' "${P}/stage171a_dependency_status.txt"
grep -q '^stage171b_not_authorized_internal_gate_failed ' \
  "${P}/stage171b_dependency_status.txt"
test ! -e "${OUT}"
printf 'stage172a_testing_predicted_multiple_gate %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage171_invariant_fix_break_risk.py \
  stage172_predicted_multiple_gate.py
"${PYTHON}" -m pytest -q \
  test_stage171_invariant_fix_break_risk.py \
  test_stage172_predicted_multiple_gate.py

printf 'stage172a_evaluating_train_only_predicted_multiple_gate %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
"${PYTHON}" "${P}/stage172_predicted_multiple_gate.py" train \
  "${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt" \
  "${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json" \
  "${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json" \
  "${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json" \
  "${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json" \
  "${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json" \
  "${STAGE171}" "${OUT}" \
  2>&1 | tee "${P}/stage172a_predicted_multiple_gate_train.log"
printf 'stage172a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
