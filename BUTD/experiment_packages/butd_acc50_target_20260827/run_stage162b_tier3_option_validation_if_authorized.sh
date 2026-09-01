#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage162b_dependency_status.txt"
POLICY_LOCK="${ROOT}/stage162a_stage135c_tier3_option_ranker/locked_tier3_option_policy.json"
RAW_VAL="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
RESULT="${ROOT}/stage162b_tier3_option_ranker_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage162b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage162a_complete ' "${P}/stage162a_dependency_status.txt"
test -s "${POLICY_LOCK}"
AUTHORIZED=$("${PYTHON}" - "${POLICY_LOCK}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='162_stage135c_same_domain_tier3_option_ranker'
assert x['validation_labels_used_for_selection'] is False
print('1' if x['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage162b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test -s "${RAW_VAL}"
test ! -e "${RESULT}"
printf 'stage162b_locked_tier3_option_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage162_tier3_option_ranker.py" evaluate \
  "${RAW_VAL}" "${POLICY_LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage162b_tier3_option_validation.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print('goal_met' if x['strict_goal_met_offline'] else 'goal_not_met')
PY
)
printf 'stage162b_complete_%s %s\n' "${MET}" \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
trap - ERR
