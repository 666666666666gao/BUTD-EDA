#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage168a_dependency_status.txt"
OUT="${ROOT}/stage168a_risk_capped_stage154_stage165_selector"

fail_status() {
  local rc=$?
  printf 'stage168a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test ! -e "${OUT}"
printf 'stage168a_training_risk_capped_scene_oof_selector %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
"${PYTHON}" "${P}/stage168_risk_capped_meta_selector.py" train \
  "${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt" \
  "${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt" \
  "${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json" \
  "${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json" \
  "${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json" \
  "${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json" \
  "${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json" \
  "${OUT}" 2>&1 | tee "${P}/stage168a_risk_capped_train.log"
printf 'stage168a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
trap - ERR
