#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
UPSTREAM="${P}/stage155a_dependency_status.txt"
STATUS="${P}/stage155b_dependency_status.txt"
POLICY_LOCK="${ROOT}/stage155a_train_only_fold_routed_oof_selector/locked_fold_routed_oof_selector.json"
SOURCE_ROOT="${ROOT}/stage154b_stage150_e13_val_source_dump"
RAW_VAL="${SOURCE_ROOT}/stage135c_e12_raw_val_geometry_with_scene_id.pt"
SOURCE_DUMP="${SOURCE_ROOT}/stage150_e13_val_source_features.pt"
COMPACT_DUMP="${SOURCE_ROOT}/stage150_e13_val_adapter_features.pt"
METRICS_PATH="${SOURCE_ROOT}/eval_results.json"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
RESULT="${ROOT}/stage155b_fold_routed_oof_selector_validation_result.json"

fail_status() {
  local rc=$?
  printf 'stage155b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR INT TERM

printf 'stage155b_waiting_for_stage155a %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage155a_(complete|skipped|failed|not_authorized)' \
  "${UPSTREAM}" 2>/dev/null; do
  sleep 60
done

if grep -q '^stage155a_skipped_stage154_goal_met ' "${UPSTREAM}"; then
  printf 'stage155b_skipped_stage154_goal_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi
if grep -q -E '^stage155a_(failed|not_authorized)' "${UPSTREAM}"; then
  printf 'stage155b_not_authorized_stage155a_unavailable %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi

test -s "${POLICY_LOCK}"
AUTHORIZED=$("${PYTHON}" - "${POLICY_LOCK}" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding='utf-8'))
assert lock['stage'] == '155_train_only_fold_routed_oof_stage142_stage150_selector'
assert lock['validation_labels_used_for_selection'] is False
print('1' if lock['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage155b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi

test -s "${RAW_VAL}"
test -s "${SOURCE_DUMP}"
test -s "${COMPACT_DUMP}"
test -s "${METRICS_PATH}"
test ! -e "${RESULT}"
"${PYTHON}" - "${SOURCE_DUMP}" "${COMPACT_DUMP}" "${METRICS_PATH}" <<'PY'
import json, os, sys, torch, numpy as np
source_path, compact_path, metrics_path = sys.argv[1:]
manifest = torch.load(source_path, map_location='cpu')
assert manifest['format'] == 'source_choice_feature_dump_sharded_v1'
assert manifest['topk'] == 1 and manifest['row_count'] == 9508
base = os.path.dirname(source_path)
assert sum(len(torch.load(os.path.join(base, relative), map_location='cpu')['rows'])
           for relative in manifest['shards']) == 9508
rows = torch.load(compact_path, map_location='cpu')['rows']
assert len(rows) == 9508
hits = [0, 0]
for row in rows:
    selected = int(np.argmax(np.asarray(row['adapter_score_at_candidate'])))
    iou = float(row['adapter_iou_at_candidate'][selected])
    hits[0] += int(iou > 0.25)
    hits[1] += int(iou > 0.50)
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [round(metrics['last__bbs_acc0.25_top1'] * 9508),
            round(metrics['last__bbs_acc0.50_top1'] * 9508)]
assert hits == official == [5234, 3979], (hits, official)
print('STAGE155B_PRIMARY_FEATURE_PARITY_PASS', hits)
PY

printf 'stage155b_locked_fold_routed_selector_evaluating %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage155_fold_routed_oof_selector.py" evaluate \
  "${RAW_VAL}" \
  "${SOURCE_DUMP}" \
  "${COMPACT_DUMP}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${POLICY_LOCK}" \
  "${RESULT}" \
  2>&1 | tee "${P}/stage155b_locked_fold_routed_selector.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding='utf-8'))
print('goal_met' if result['strict_goal_met_offline'] else 'goal_not_met')
PY
)
trap - ERR INT TERM
printf 'stage155b_complete_%s %s\n' \
  "${MET}" "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
