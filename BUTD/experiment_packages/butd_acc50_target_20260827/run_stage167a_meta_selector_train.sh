#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage167a_dependency_status.txt"
STAGE31="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE154="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"
STAGE165="${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json"
TRAIN_RAW="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
TRAIN_SOURCE="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
TRAIN_COMPACT="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
OUTPUT="${ROOT}/stage167a_stage154_stage165_meta_selector"

fail_status() {
  local rc=$?
  printf 'stage167a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage166_complete ' "${P}/stage166_dependency_status.txt"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage167_stage154_stage165_meta_selector.py \
  test_stage167_stage154_stage165_meta_selector.py
"${PYTHON}" -m unittest -v test_stage167_stage154_stage165_meta_selector.py
printf 'stage167a_training_scene_oof_meta_selector %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage167_stage154_stage165_meta_selector.py train \
  "${TRAIN_RAW}" "${TRAIN_SOURCE}" "${TRAIN_COMPACT}" \
  "${STAGE31}" "${STAGE33}" "${STAGE142}" "${STAGE154}" "${STAGE165}" \
  "${OUTPUT}" 2>&1 | tee "${P}/stage167a_meta_selector_train.log"
printf 'stage167a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${OUTPUT}/locked_meta_selector.json"
trap - ERR
