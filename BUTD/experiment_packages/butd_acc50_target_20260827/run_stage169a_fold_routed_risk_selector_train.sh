#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage169a_dependency_status.txt"
OUT="${ROOT}/stage169a_fold_routed_risk_stage154_stage165_selector"

fail_status() {
  local rc=$?
  printf 'stage169a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage168a_complete ' "${P}/stage168a_dependency_status.txt"
grep -q '^stage168b_not_authorized_internal_gate_failed ' \
  "${P}/stage168b_dependency_status.txt"
test ! -e "${OUT}"
printf 'stage169a_testing_fold_routed_risk_selector %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage155_fold_routed_oof_selector.py \
  stage168_risk_capped_meta_selector.py \
  stage169_fold_routed_risk_selector.py
"${PYTHON}" -m pytest -q \
  test_stage155_fold_routed_oof_selector.py \
  test_stage168_risk_capped_meta_selector.py \
  test_stage169_fold_routed_risk_selector.py

printf 'stage169a_training_fold_routed_risk_selector %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
"${PYTHON}" "${P}/stage169_fold_routed_risk_selector.py" train \
  "${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt" \
  "${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json" \
  "${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json" \
  "${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json" \
  "${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json" \
  "${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json" \
  "${OUT}" \
  2>&1 | tee "${P}/stage169a_fold_routed_risk_train.log"
printf 'stage169a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
