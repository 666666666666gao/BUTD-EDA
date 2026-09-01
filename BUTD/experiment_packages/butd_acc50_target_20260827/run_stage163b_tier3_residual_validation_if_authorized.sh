#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage163b_dependency_status.txt"
POLICY_LOCK="${ROOT}/stage163a_tier3_residual_blend/locked_residual_blend_policy.json"
RAW_VAL="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
RESULT="${ROOT}/stage163b_tier3_residual_blend_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage163b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage163a_complete ' "${P}/stage163a_dependency_status.txt"
test -s "${POLICY_LOCK}"
AUTHORIZED=$("${PYTHON}" - "${POLICY_LOCK}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='163_stage142_plus_tier3_fixed_residual_option_ranker'
assert x['validation_labels_used_for_selection'] is False
print('1' if x['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage163b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test -s "${RAW_VAL}"
test ! -e "${RESULT}"
printf 'stage163b_locked_tier3_residual_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage163_tier3_residual_blend.py" evaluate \
  "${RAW_VAL}" "${STAGE31_LOCK}" "${STAGE33_LOCK}" "${STAGE142_LOCK}" \
  "${POLICY_LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage163b_tier3_residual_validation.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print('goal_met' if x['strict_goal_met_offline'] else 'goal_not_met')
PY
)
printf 'stage163b_complete_%s %s\n' "${MET}" \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
trap - ERR
