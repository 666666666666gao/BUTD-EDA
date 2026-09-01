#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STATUS="${P}/stage167b_dependency_status.txt"
STAGE31="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE154="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"
STAGE165="${ROOT}/stage165a_capped_same_domain_trio/locked_capped_nested_policy.json"
VAL_RAW="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
VAL_SOURCE="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_source_features.pt"
VAL_COMPACT="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_adapter_features.pt"
POLICY="${ROOT}/stage167a_stage154_stage165_meta_selector/locked_meta_selector.json"
RESULT="${ROOT}/stage167b_meta_selector_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage167b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

grep -q '^stage167a_complete ' "${P}/stage167a_dependency_status.txt"
AUTHORIZED=$("${PYTHON}" - "${POLICY}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='167_train_only_oof_stage154_stage165_meta_selector'
assert x['validation_labels_used_for_selection'] is False
print('1' if x['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage167b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

test ! -e "${RESULT}"
printf 'stage167b_locked_meta_selector_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage167_stage154_stage165_meta_selector.py" evaluate \
  "${VAL_RAW}" "${VAL_SOURCE}" "${VAL_COMPACT}" \
  "${STAGE31}" "${STAGE33}" "${STAGE142}" "${STAGE154}" "${STAGE165}" \
  "${POLICY}" "${RESULT}" 2>&1 | tee "${P}/stage167b_meta_selector_validation.log"
MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print('goal_met' if x['strict_goal_met_offline'] else 'goal_not_met')
PY
)
printf 'stage167b_complete_%s %s\n' "${MET}" \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
trap - ERR
