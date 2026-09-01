#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage168b_dependency_status.txt"
POLICY="${ROOT}/stage168a_risk_capped_stage154_stage165_selector/locked_risk_capped_selector.json"
RESULT="${ROOT}/stage168b_risk_capped_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage168b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage168a_complete ' "${P}/stage168a_dependency_status.txt"
AUTHORIZED=$("${PYTHON}" - "${POLICY}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x['stage'] == '168_train_only_risk_capped_oof_stage154_stage165_selector'
assert x['validation_labels_used_for_selection'] is False
print('1' if x['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage168b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test ! -e "${RESULT}"
printf 'stage168b_locked_risk_capped_selector_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
"${PYTHON}" "${P}/stage168_risk_capped_meta_selector.py" evaluate \
  "${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt" \
  "${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_source_features.pt" \
  "${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_adapter_features.pt" \
  "${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json" \
  "${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json" \
  "${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json" \
  "${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json" \
  "${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json" \
  "${POLICY}" "${RESULT}" \
  2>&1 | tee "${P}/stage168b_risk_capped_validation.log"
MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print('goal_met' if x['strict_goal_met_offline'] else 'goal_not_met')
PY
)
printf 'stage168b_complete_%s %s\n' "${MET}" \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
trap - ERR
