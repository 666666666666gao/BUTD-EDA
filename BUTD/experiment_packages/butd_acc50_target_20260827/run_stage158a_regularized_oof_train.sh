#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage158a_dependency_status.txt"
OUTPUT="${ROOT}/stage158a_regularized_oof_source_selector"
RAW_TRAIN="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
SOURCE_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_source_features.pt"
COMPACT_TRAIN="${ROOT}/stage153a_stage150_e13_train_source_dump/stage150_e13_train_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"

fail_status() {
  local rc=$?
  printf 'stage158a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test -s "${RAW_TRAIN}"
test -s "${SOURCE_TRAIN}"
test -s "${COMPACT_TRAIN}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage153_train_source_selector.py \
  stage154_oof_source_selector.py \
  stage158_regularized_selector.py \
  test_stage158_regularized_selector.py
"${PYTHON}" -m unittest -v test_stage158_regularized_selector.py

printf 'stage158a_training_regularized_oof %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage158_regularized_selector.py train \
  "${RAW_TRAIN}" \
  "${SOURCE_TRAIN}" \
  "${COMPACT_TRAIN}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${OUTPUT}" \
  2>&1 | tee "${P}/stage158a_regularized_oof_train.log"

printf 'stage158a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
