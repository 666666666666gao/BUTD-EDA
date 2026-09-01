#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage164b_dependency_status.txt"
OUTPUT="${ROOT}/stage164a_same_domain_retrained_trio"
AUTH="${OUTPUT}/locked_same_domain_trio_authorization.json"
POLICY="${OUTPUT}/locked_same_domain_trio_nested_policy.json"
RAW_VAL="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
RAW_RESULT="${ROOT}/stage164b_same_domain_trio_raw_validation_result.json"
RESULT="${ROOT}/stage164b_same_domain_trio_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage164b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage164a_complete ' "${P}/stage164a_dependency_status.txt"
test -s "${AUTH}"
AUTHORIZED=$("${PYTHON}" - "${AUTH}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='164_stage135c_same_domain_retrained_trio_nested_blend'
assert x['validation_labels_used_for_selection'] is False
print('1' if x['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage164b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test -s "${RAW_VAL}"
test ! -e "${RAW_RESULT}"
test ! -e "${RESULT}"
printf 'stage164b_locked_same_domain_trio_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage140_train_eval_nested_blend.py" evaluate \
  "${RAW_VAL}" "${POLICY}" "${RAW_RESULT}" \
  2>&1 | tee "${P}/stage164b_same_domain_trio_validation.log"
"${PYTHON}" "${P}/stage164_same_domain_trio_gate.py" wrap-validation \
  "${AUTH}" "${RAW_RESULT}" "${RESULT}" \
  2>&1 | tee -a "${P}/stage164b_same_domain_trio_validation.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print('goal_met' if x['strict_goal_met_offline'] else 'goal_not_met')
PY
)
printf 'stage164b_complete_%s %s\n' "${MET}" \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RAW_RESULT}" "${RESULT}"
trap - ERR
