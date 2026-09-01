#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STATUS="${P}/stage133b_137_resume_chain_status.txt"
BACKUP="${P}/state/stage135_stage29_query_map_pre_20260830"
RUN134="${P}/run_stage134_build_stage29_query_map.sh"
RUN135="${P}/run_stage135_stage29_option_last_box_noaug.sh"
RUN136="${P}/run_stage136_137_stage135_dump_stage29_eval.sh"
REPAIR_RECEIPT="${ROOT}/stage133_stage95_e11_raw_train_geometry_dump/stage133b_id_repair_receipt.json"

EXPECTED_OLD_MAIN_SHA="afcb88f2a9a268bf270ee71b0901a18876e732a08749a79b796b6262b9f30ba8"
EXPECTED_OLD_LOSS_SHA="acb1eed32c1f16a3696b3d1b9dab13abf94dc5aecf7681be8e670fa26286a60a"
EXPECTED_NEW_MAIN_SHA="11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4"
EXPECTED_NEW_LOSS_SHA="fd78ef2cfd661dbdadcc66644d2debad6eaaea46195a645a193fee999bbcddbe"
EXPECTED_LIVE_EVAL_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

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
test -s "$REPAIR_RECEIPT"
check_sha main_utils.py "$EXPECTED_OLD_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_OLD_LOSS_SHA"
check_sha src/grounding_evaluator.py "$EXPECTED_LIVE_EVAL_SHA"
check_sha /tmp/stage133_candidate_main_utils.py "$EXPECTED_NEW_MAIN_SHA"
check_sha /tmp/stage133_candidate_losses.py "$EXPECTED_NEW_LOSS_SHA"
test -x "$RUN134" && test -x "$RUN135" && test -x "$RUN136"
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 500

/root/miniconda3/envs/bdetr/bin/python - "$REPAIR_RECEIPT" <<'PY'
import json, sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
assert receipt['stage']=='133b_id_repair' and receipt['status']=='complete'
assert receipt['rows']==receipt['unique_keys']==36665
assert receipt['max_gt_abs_delta']==0.0 and receipt['metadata_mismatches']==0
assert receipt['corrected_dump_sha256']=='b89f72a67b4b494d2a66e9b34d603e365afeb934f194a5d129c618ba7bf20313'
print('STAGE133B_REPAIR_RECEIPT_PASS')
PY

mkdir -p "$BACKUP"
cp -p main_utils.py "$BACKUP/main_utils.pre_stage135.py"
cp -p models/losses.py "$BACKUP/losses.pre_stage135.py"
chmod 0444 "$BACKUP/main_utils.pre_stage135.py" "$BACKUP/losses.pre_stage135.py"
install -m 0644 /tmp/stage133_candidate_main_utils.py main_utils.py
install -m 0644 /tmp/stage133_candidate_losses.py models/losses.py
check_sha main_utils.py "$EXPECTED_NEW_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_NEW_LOSS_SHA"
cp -p main_utils.py "$BACKUP/main_utils.stage135.py"
cp -p models/losses.py "$BACKUP/losses.stage135.py"
chmod 0444 "$BACKUP/main_utils.stage135.py" "$BACKUP/losses.stage135.py"
/root/miniconda3/envs/bdetr/bin/python -m py_compile main_utils.py models/losses.py
/root/miniconda3/envs/bdetr/bin/python train_dist_mod.py --help \
  | grep -q -- '--last_box_target_query_map'

printf 'stage134_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN134"
grep -q '^stage134_complete ' "$P/stage134_query_map_status.txt"

DRY_RUN=1 bash "$RUN135" > "$BACKUP/stage135_dry_run.txt"
grep -q -- '--last_box_target_query_map_mode option' "$BACKUP/stage135_dry_run.txt"
chmod 0444 "$BACKUP/stage135_dry_run.txt"
printf 'stage135_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN135"
grep -q '^stage135_complete ' "$P/stage135_option_last_box_status.txt"

printf 'stage136_137_starting %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$RUN136"
grep -q '^stage137_complete ' "$P/stage136_137_chain_status.txt"

trap - ERR
printf 'stage133b_137_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
