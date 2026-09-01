#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage165a_dependency_status.txt"
TRAIN_DUMP="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump/stage135c_e12_raw_train_geometry_with_ids.pt"
OLD_STAGE142="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
PARENT="${ROOT}/stage164a_same_domain_retrained_trio/locked_same_domain_trio_nested_policy.json"
OUTPUT="${ROOT}/stage165a_capped_same_domain_trio"
POLICY="${OUTPUT}/locked_capped_nested_policy.json"
AUTH="${OUTPUT}/locked_capped_authorization.json"

fail_status() {
  local rc=$?
  printf 'stage165a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage164a_complete ' "${P}/stage164a_dependency_status.txt"
grep -q '^stage164b_not_authorized_internal_gate_failed ' \
  "${P}/stage164b_dependency_status.txt"
test -s "${TRAIN_DUMP}"
test -s "${OLD_STAGE142}"
test -s "${PARENT}"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}"
cd "${P}"
"${PYTHON}" -m py_compile \
  stage165_capped_nested_gate.py test_stage165_capped_nested_gate.py
"${PYTHON}" -m unittest -v test_stage165_capped_nested_gate.py
printf 'stage165a_locking_train_dev_90pct_change_cap %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" stage165_capped_nested_gate.py lock \
  "${TRAIN_DUMP}" "${PARENT}" "${OLD_STAGE142}" "${POLICY}" "${AUTH}" \
  2>&1 | tee "${P}/stage165a_capped_nested_gate.log"
printf 'stage165a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${POLICY}" "${AUTH}"
trap - ERR
