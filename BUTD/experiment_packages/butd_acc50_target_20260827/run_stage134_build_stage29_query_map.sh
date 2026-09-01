#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PY="/root/miniconda3/envs/bdetr/bin/python"
STAGE133="${ROOT}/stage133_stage95_e11_raw_train_geometry_dump"
DUMP="${STAGE133}/stage95_e11_raw_train_geometry_with_ids.pt"
SOURCE_RECEIPT="${STAGE133}/stage133b_id_repair_receipt.json"
MODEL="${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt"
LOCK="${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json"
BUILDER="${P}/build_stage29_query_map.py"
OUT="${ROOT}/stage134_stage29_query_map"
MAP="${OUT}/stage29_query_action_map.pt"
SUMMARY="${OUT}/stage134_map_summary.json"
STATUS="${P}/stage134_query_map_status.txt"

EXPECTED_MODEL_SHA="4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e"
EXPECTED_LOCK_SHA="5acafbca18320a077b16ac6407fd696e6ac68a124c292a26bad455cbba15cdce"
EXPECTED_BUILDER_SHA="91778a9db2530a987a94faa37d0d5c382075bd22f8d2937457bb36bc11f0c3fe"

check_sha() {
  local path="$1" expected="$2"
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

cd "$R"
test ! -e "$OUT"
test -s "$SOURCE_RECEIPT"
check_sha "$MODEL" "$EXPECTED_MODEL_SHA"
check_sha "$LOCK" "$EXPECTED_LOCK_SHA"
check_sha "$BUILDER" "$EXPECTED_BUILDER_SHA"

"$PY" - "$SOURCE_RECEIPT" "$DUMP" <<'PY'
import hashlib
import json
import sys

receipt_path, dump = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding='utf-8'))
assert receipt['stage'] == '133b_id_repair' and receipt['status'] == 'complete'
assert receipt['rows'] == 36665 and receipt['unique_keys'] == 36665
assert receipt['copied_fields'] == ['scene_id', 'object_id', 'ann_id']
assert receipt['max_gt_abs_delta'] == 0.0
assert receipt['metadata_mismatches'] == 0
h = hashlib.sha256()
with open(dump, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == receipt['corrected_dump_sha256']
print('STAGE133B_SOURCE_RECEIPT_PASS', receipt['corrected_dump_sha256'])
PY

mkdir -p "$OUT"
{
  printf 'stage=134\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_dump=%s\n' "$DUMP"
  printf 'source_dump_sha256='; sha256sum "$DUMP" | awk '{print $1}'
  printf 'stage29_model_sha256=%s\n' "$EXPECTED_MODEL_SHA"
  printf 'stage29_lock_sha256=%s\n' "$EXPECTED_LOCK_SHA"
  printf 'builder_sha256=%s\n' "$EXPECTED_BUILDER_SHA"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage134_mapping %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
PYTHONPATH="$P" "$PY" "$BUILDER" \
  "$DUMP" "$MODEL" "$LOCK" "$MAP" "$SUMMARY" \
  --predict-batch-size 256 2>&1 | tee "$P/stage134_query_map.log"

"$PY" - "$MAP" "$SUMMARY" <<'PY'
import hashlib
import json
import sys
import torch

map_path, summary_path = sys.argv[1:]
payload = torch.load(map_path, map_location='cpu')
summary = json.load(open(summary_path, encoding='utf-8'))
assert payload['format'] == 'stage29_query_action_map_v1'
assert len(payload['entries']) == 36665
assert summary['rows'] == summary['unique_keys'] == 36665
assert summary['source_hashes']['model_sha256'] == (
    '4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e'
)
h = hashlib.sha256()
with open(map_path, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == summary['output_map_sha256']
print('STAGE134_QUERY_MAP_AUDIT_PASS', summary['selected_option'])
PY

chmod 0444 "$MAP" "$SUMMARY" "$OUT/launch_manifest.txt"
printf 'stage134_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
