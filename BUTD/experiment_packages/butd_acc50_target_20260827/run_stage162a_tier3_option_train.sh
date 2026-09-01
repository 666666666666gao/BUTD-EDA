#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage162a_dependency_status.txt"
TRAIN_DUMP="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
OUTPUT="${ROOT}/stage162a_stage135c_tier3_option_ranker"

fail_status() {
  local rc=$?
  printf 'stage162a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test -s "${TRAIN_DUMP}"
test ! -e "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  train_joint_option_ranker.py stage162_tier3_option_ranker.py \
  test_stage162_tier3_option_ranker.py
"${PYTHON}" -m unittest -v test_stage162_tier3_option_ranker.py
"${PYTHON}" train_joint_option_ranker.py self-test

printf 'stage162a_training_tier3_option_ranker %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage162_tier3_option_ranker.py train \
  "${TRAIN_DUMP}" "${OUTPUT}" --num-threads 16 \
  2>&1 | tee "${P}/stage162a_tier3_option_train.log"

printf 'stage162a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
