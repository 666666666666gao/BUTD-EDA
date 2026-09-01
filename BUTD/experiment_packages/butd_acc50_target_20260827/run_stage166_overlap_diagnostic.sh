#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage166_dependency_status.txt"
STAGE31="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE154="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"
STAGE165="${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json"
TRAIN_RAW="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
TRAIN_SOURCE="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
TRAIN_COMPACT="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
VAL_RAW="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
VAL_SOURCE="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_source_features.pt"
VAL_COMPACT="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_adapter_features.pt"
TRAIN_REPORT="${ROOT}/stage166_train_internal_overlap.json"
VAL_REPORT="${ROOT}/stage166_validation_overlap.json"

fail_status() {
  local rc=$?
  printf 'stage166_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

for path in "${STAGE31}" "${STAGE33}" "${STAGE142}" "${STAGE154}" \
  "${STAGE165}" "${TRAIN_RAW}" "${TRAIN_SOURCE}" "${TRAIN_COMPACT}" \
  "${VAL_RAW}" "${VAL_SOURCE}" "${VAL_COMPACT}"; do
  test -s "${path}"
done
test ! -e "${TRAIN_REPORT}"
test ! -e "${VAL_REPORT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage166_stage154_stage165_overlap.py \
  test_stage166_stage154_stage165_overlap.py
"${PYTHON}" -m unittest -v test_stage166_stage154_stage165_overlap.py

printf 'stage166_train_internal_diagnostic %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage166_stage154_stage165_overlap.py \
  "${TRAIN_RAW}" "${TRAIN_SOURCE}" "${TRAIN_COMPACT}" \
  "${STAGE31}" "${STAGE33}" "${STAGE142}" "${STAGE154}" "${STAGE165}" \
  "${TRAIN_REPORT}" --scope internal_test \
  2>&1 | tee "${P}/stage166_train_internal.log"

printf 'stage166_validation_diagnostic %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage166_stage154_stage165_overlap.py \
  "${VAL_RAW}" "${VAL_SOURCE}" "${VAL_COMPACT}" \
  "${STAGE31}" "${STAGE33}" "${STAGE142}" "${STAGE154}" "${STAGE165}" \
  "${VAL_REPORT}" --scope all \
  2>&1 | tee "${P}/stage166_validation.log"

printf 'stage166_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${TRAIN_REPORT}" "${VAL_REPORT}"
trap - ERR
