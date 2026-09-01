#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STATUS="${P}/stage135b_137b_chain_status.txt"
BACKUP="${P}/state/stage135b_exampleid_map_pre_20260830"
RUN135="${P}/run_stage135b_stage29_option_last_box_noaug.sh"
RUN136="${P}/run_stage136b_137b_stage135b_dump_stage29_eval.sh"
MAP="${ROOT}/stage134b_stage29_query_map_exampleid/stage29_query_action_map_v2.pt"
MAP_RECEIPT="${ROOT}/stage134b_stage29_query_map_exampleid/stage134b_example_id_map_receipt.json"

EXPECTED_MAIN_SHA="11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4"
EXPECTED_OLD_LOSS_SHA="fd78ef2cfd661dbdadcc66644d2debad6eaaea46195a645a193fee999bbcddbe"
EXPECTED_NEW_LOSS_SHA="376c6412adf5f6fcf823c797fcb423114a9ab0a88e6af9e37db889d98f0928d7"
EXPECTED_OLD_DATASET_SHA="6a7483d719d09433f2b7763f73246b0daf43948741ec3394068471906d96a24a"
EXPECTED_NEW_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_MAP_SHA="ed6857f8ef2f8053a0bb039c6dd8aa83b7d867c9fe3c92fd5f03cbb8f623927e"

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
test ! -e "$ROOT/stage135b_stage29_option_last_box_noaug_exampleid"
test ! -e "$ROOT/stage136b_stage135b_raw_val_geometry_dump"
test ! -e "$ROOT/stage137b_stage29_on_stage135b_locked_val_eval.json"
test -x "$RUN135" && test -x "$RUN136"
check_sha main_utils.py "$EXPECTED_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_OLD_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_OLD_DATASET_SHA"
check_sha /tmp/stage135_candidate_losses_v2.py "$EXPECTED_NEW_LOSS_SHA"
check_sha /tmp/stage135_candidate_joint_det_dataset.py "$EXPECTED_NEW_DATASET_SHA"
check_sha "$MAP" "$EXPECTED_MAP_SHA"
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
cp -p models/losses.py "$BACKUP/losses.pre_stage135b.py"
cp -p src/joint_det_dataset.py "$BACKUP/joint_det_dataset.pre_stage135b.py"
chmod 0444 "$BACKUP/losses.pre_stage135b.py" "$BACKUP/joint_det_dataset.pre_stage135b.py"
install -m 0644 /tmp/stage135_candidate_losses_v2.py models/losses.py
install -m 0644 /tmp/stage135_candidate_joint_det_dataset.py src/joint_det_dataset.py
check_sha models/losses.py "$EXPECTED_NEW_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_NEW_DATASET_SHA"
cp -p models/losses.py "$BACKUP/losses.stage135b.py"
cp -p src/joint_det_dataset.py "$BACKUP/joint_det_dataset.stage135b.py"
chmod 0444 "$BACKUP/losses.stage135b.py" "$BACKUP/joint_det_dataset.stage135b.py"
/root/miniconda3/envs/bdetr/bin/python -m py_compile models/losses.py src/joint_det_dataset.py
grep -q '"example_id": np.int64(index)' src/joint_det_dataset.py

DRY_RUN=1 bash "$RUN135" > "$BACKUP/stage135b_dry_run.txt"
grep -q -- '--last_box_target_query_map_mode option' "$BACKUP/stage135b_dry_run.txt"
grep -q 'stage29_query_action_map_v2.pt' "$BACKUP/stage135b_dry_run.txt"
chmod 0444 "$BACKUP/stage135b_dry_run.txt"

printf 'stage135b_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN135"
grep -q '^stage135b_complete ' "$P/stage135b_option_last_box_status.txt"

printf 'stage136b_137b_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN136"
grep -q '^stage137b_complete ' "$P/stage136b_137b_chain_status.txt"

trap - ERR
printf 'stage135b_137b_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
