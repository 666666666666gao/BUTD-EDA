#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STATUS="${P}/stage135c_137c_chain_status.txt"
BACKUP="${P}/state/stage135c_jointmask_pre_20260830"
RUN135="${P}/run_stage135c_stage29_option_last_box_noaug.sh"
RUN136="${P}/run_stage136c_137c_stage135c_dump_stage29_eval.sh"
MAP="${ROOT}/stage134b_stage29_query_map_exampleid/stage29_query_action_map_v2.pt"
MAP_RECEIPT="${ROOT}/stage134b_stage29_query_map_exampleid/stage134b_example_id_map_receipt.json"
TEST="${P}/test_stage135c_query_map_domain.py"
AUDIT="${P}/audit_stage135c_id_domain.py"
DUMP="${ROOT}/stage133_stage95_e11_raw_train_geometry_dump/stage95_e11_raw_train_geometry_with_ids.pt"

EXPECTED_MAIN_SHA="11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4"
EXPECTED_OLD_LOSS_SHA="376c6412adf5f6fcf823c797fcb423114a9ab0a88e6af9e37db889d98f0928d7"
EXPECTED_NEW_LOSS_SHA="a95443a958c0170faac513c46001e8c60f46cfaacd559b620c79eb622ec2c852"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_MAP_SHA="ed6857f8ef2f8053a0bb039c6dd8aa83b7d867c9fe3c92fd5f03cbb8f623927e"
EXPECTED_TEST_SHA="0f6215b8bae1b2ea3cfc3ca70d7276fb21138cfc4d5a0f958e842c5ed16043d0"
EXPECTED_AUDIT_SHA="8c1da98fadb48be5bd7db54fbcd9ba1bf53311bf8ae8a4c2a2895cbff38e7171"

fail_status() {
  rc=$?
  printf 'failed rc=%s at=%s line=%s\n' "$rc" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" > "$STATUS"
  exit "$rc"
}
trap fail_status ERR

check_sha() {
  local path="$1" expected="$2"
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

cd "$R"
test ! -e "$BACKUP"
test ! -e "$ROOT/stage135c_stage29_option_last_box_noaug_jointmask"
test ! -e "$ROOT/stage136c_stage135c_raw_val_geometry_dump"
test ! -e "$ROOT/stage137c_stage29_on_stage135c_locked_val_eval.json"
test -x "$RUN135" && test -x "$RUN136"
check_sha main_utils.py "$EXPECTED_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_OLD_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_DATASET_SHA"
check_sha /tmp/stage135_candidate_losses_v3.py "$EXPECTED_NEW_LOSS_SHA"
check_sha "$MAP" "$EXPECTED_MAP_SHA"
check_sha "$TEST" "$EXPECTED_TEST_SHA"
check_sha "$AUDIT" "$EXPECTED_AUDIT_SHA"
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 500

/root/miniconda3/envs/bdetr/bin/python - "$MAP_RECEIPT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['stage']=='134b_example_id_map' and d['status']=='complete'
assert d['entries_by_example_id']==36665
assert d['example_id_min']==0 and d['example_id_max']==36664
assert d['v2_map_sha256']=='ed6857f8ef2f8053a0bb039c6dd8aa83b7d867c9fe3c92fd5f03cbb8f623927e'
assert d['validation_behavior']=='mapping_disabled_fallback_only'
print('STAGE134B_MAP_RECEIPT_PASS')
PY

mkdir -p "$BACKUP"
cp -p models/losses.py "$BACKUP/losses.pre_stage135c.py"
cp -p src/joint_det_dataset.py "$BACKUP/joint_det_dataset.stage135c.py"
chmod 0444 "$BACKUP/losses.pre_stage135c.py" "$BACKUP/joint_det_dataset.stage135c.py"
install -m 0644 /tmp/stage135_candidate_losses_v3.py models/losses.py
check_sha models/losses.py "$EXPECTED_NEW_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_DATASET_SHA"
cp -p models/losses.py "$BACKUP/losses.stage135c.py"
chmod 0444 "$BACKUP/losses.stage135c.py"
/root/miniconda3/envs/bdetr/bin/python -m py_compile models/losses.py src/joint_det_dataset.py
grep -q '"example_id": np.int64(index)' src/joint_det_dataset.py

PYTHONPATH="$R" /root/miniconda3/envs/bdetr/bin/python "$TEST" "$MAP" \
  > "$BACKUP/stage135c_query_map_domain_test.log"
grep -q 'STAGE135C_QUERY_MAP_DOMAIN_PASS' "$BACKUP/stage135c_query_map_domain_test.log"
/root/miniconda3/envs/bdetr/bin/python "$AUDIT" \
  --annotations /root/autodl-tmp/DATA_ROOT/scanrefer/ScanRefer_filtered_train_spacy_refined.json \
  --scan-ids /root/autodl-tmp/DATA_ROOT/scanrefer/ScanRefer_filtered_train.txt \
  --dump "$DUMP" > "$BACKUP/stage135c_id_domain_audit.log"
grep -q 'STAGE135C_ID_DOMAIN_AUDIT_PASS' "$BACKUP/stage135c_id_domain_audit.log"

DRY_RUN=1 bash "$RUN135" > "$BACKUP/stage135c_dry_run.txt"
grep -q -- '--last_box_target_query_map_mode option' "$BACKUP/stage135c_dry_run.txt"
grep -q 'stage29_query_action_map_v2.pt' "$BACKUP/stage135c_dry_run.txt"
chmod 0444 "$BACKUP"/*.log "$BACKUP/stage135c_dry_run.txt"

printf 'stage135c_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN135"
grep -q '^stage135c_complete ' "$P/stage135c_option_last_box_status.txt"

printf 'stage136c_137c_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN136"
grep -q '^stage137c_complete ' "$P/stage136c_137c_chain_status.txt"

trap - ERR
printf 'stage135c_137c_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
