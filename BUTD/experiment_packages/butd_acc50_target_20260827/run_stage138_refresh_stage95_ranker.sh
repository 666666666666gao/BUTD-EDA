#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PY="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage133_stage95_e11_raw_train_geometry_dump/stage95_e11_raw_train_geometry_with_ids.pt"
VAL_DUMP="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
OLD_DIR="${ROOT}/stage29_binary50_ranker"
OLD_MODEL="${OLD_DIR}/binary50_option_ranker.txt"
OLD_LOCK="${OLD_DIR}/locked_binary50_policy.json"
SCRIPT="${P}/train_joint_option_ranker.py"
ANALYZER="${P}/analyze_stage29_thresholds.py"
OUT="${ROOT}/stage138_stage95_binary50_ranker"
OLD_AUDIT="${ROOT}/stage138_old_ranker_on_stage135c_threshold_audit.json"
NEW_RESULT="${ROOT}/stage138_stage95_ranker_on_stage135c_eval.json"
NEW_AUDIT="${ROOT}/stage138_stage95_ranker_on_stage135c_threshold_audit.json"
RECEIPT="${ROOT}/stage138_stage95_ranker_receipt.json"
STATUS="${P}/stage138_status.txt"

EXPECTED_TRAIN_DUMP_SHA="b89f72a67b4b494d2a66e9b34d603e365afeb934f194a5d129c618ba7bf20313"
EXPECTED_VAL_DUMP_SHA="6a837f903f69b0ec15f43bf0544230344352adf14303c6ea0d13a7e842825508"
EXPECTED_OLD_MODEL_SHA="4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e"
EXPECTED_OLD_LOCK_SHA="5acafbca18320a077b16ac6407fd696e6ac68a124c292a26bad455cbba15cdce"
EXPECTED_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"
EXPECTED_ANALYZER_SHA="4e4b1abece08ed02d0fe914341a305f217fe6b9efe2845ccd0ac65de9d340fb3"

fail_status() {
  rc=$?
  printf 'stage138_failed rc=%s at=%s line=%s\n' \
    "$rc" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" > "$STATUS"
  exit "$rc"
}
trap fail_status ERR

check_sha() {
  local path="$1" expected="$2"
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

cd "$R"
test ! -e "$OUT"
test ! -e "$OLD_AUDIT"
test ! -e "$NEW_RESULT"
test ! -e "$NEW_AUDIT"
test ! -e "$RECEIPT"
check_sha "$TRAIN_DUMP" "$EXPECTED_TRAIN_DUMP_SHA"
check_sha "$VAL_DUMP" "$EXPECTED_VAL_DUMP_SHA"
check_sha "$OLD_MODEL" "$EXPECTED_OLD_MODEL_SHA"
check_sha "$OLD_LOCK" "$EXPECTED_OLD_LOCK_SHA"
check_sha "$SCRIPT" "$EXPECTED_SCRIPT_SHA"
check_sha "$ANALYZER" "$EXPECTED_ANALYZER_SHA"
test "$(df --output=avail -k /root/autodl-tmp | tail -1)" -gt 5242880

printf 'stage138_self_test %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
PYTHONPATH="$P:$R" "$PY" "$SCRIPT" self-test \
  2>&1 | tee "$P/stage138_self_test.log"

printf 'stage138_old_ranker_audit %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
PYTHONPATH="$P:$R" "$PY" "$ANALYZER" \
  "$VAL_DUMP" "$OLD_MODEL" "$OLD_LOCK" "$OLD_AUDIT" \
  2>&1 | tee "$P/stage138_old_ranker_threshold_audit.log"

printf 'stage138_training_stage95_ranker %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
PYTHONPATH="$P:$R" "$PY" "$SCRIPT" binary50-train \
  "$TRAIN_DUMP" "$OUT" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "$P/stage138_stage95_binary50_train.log"

NEW_MODEL="${OUT}/binary50_option_ranker.txt"
NEW_LOCK="${OUT}/locked_binary50_policy.json"
test -s "$NEW_MODEL" && test -s "$NEW_LOCK"

printf 'stage138_evaluating_stage95_ranker %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
PYTHONPATH="$P:$R" "$PY" "$SCRIPT" evaluate \
  "$VAL_DUMP" "$NEW_MODEL" "$NEW_LOCK" "$NEW_RESULT" \
  2>&1 | tee "$P/stage138_stage95_ranker_eval.log"
PYTHONPATH="$P:$R" "$PY" "$ANALYZER" \
  "$VAL_DUMP" "$NEW_MODEL" "$NEW_LOCK" "$NEW_AUDIT" \
  2>&1 | tee "$P/stage138_stage95_ranker_threshold_audit.log"

"$PY" - "$TRAIN_DUMP" "$VAL_DUMP" "$OLD_AUDIT" "$NEW_RESULT" \
  "$NEW_AUDIT" "$NEW_MODEL" "$NEW_LOCK" "$RECEIPT" <<'PY'
import hashlib
import json
import sys

(train_dump, val_dump, old_audit_path, new_result_path, new_audit_path,
 new_model, new_lock, receipt_path) = sys.argv[1:]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

old_audit = json.load(open(old_audit_path, encoding='utf-8'))
new_result = json.load(open(new_result_path, encoding='utf-8'))
new_audit = json.load(open(new_audit_path, encoding='utf-8'))
selected = new_result['selected']
receipt = {
    'stage': '138_stage95_ranker_refresh',
    'status': 'complete',
    'train_dump_sha256': digest(train_dump),
    'val_dump_sha256': digest(val_dump),
    'new_model_sha256': digest(new_model),
    'new_lock_sha256': digest(new_lock),
    'old_locked': old_audit['locked'],
    'old_best_threshold': old_audit['best_threshold_acc025_gt_5440'],
    'new_locked': new_audit['locked'],
    'new_best_threshold': new_audit['best_threshold_acc025_gt_5440'],
    'new_frozen_evaluation': selected,
    'new_frozen_hits': {
        'acc025': round(selected['acc025'] * selected['count']),
        'acc050': round(selected['acc050'] * selected['count']),
    },
    'strict_goal_met_offline': bool(
        selected['acc025'] > 0.5391 and selected['acc050'] > 0.4241
    ),
    'diagnostic_only_until_integrated_and_reloaded': True,
}
with open(receipt_path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "$RECEIPT" "$OLD_AUDIT" "$NEW_RESULT" "$NEW_AUDIT"
trap - ERR
printf 'stage138_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
