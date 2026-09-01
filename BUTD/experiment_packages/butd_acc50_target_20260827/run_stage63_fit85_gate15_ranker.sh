#!/usr/bin/env bash
set -euo pipefail

R='/home/gb/new butd/butd_detr-main'
P="$R/experiment_packages/butd_acc50_target_20260827"
ROOT='/root/autodl-tmp/logs/butd_acc50_target_20260827'
PY='/root/miniconda3/envs/bdetr/bin/python'
TRAIN_DUMP="$ROOT/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
VAL_DUMP="$ROOT/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
SOURCE="$ROOT/stage29_binary50_ranker"
OUT="$ROOT/stage63_binary50_fit85_gate15"
RESULT="$ROOT/stage64_binary50_fit85_gate15_locked_val_eval.json"
SCRIPT="$P/refit_fit85_calibrate15_binary50.py"

cd "$R"
test -f "$TRAIN_DUMP"
test -f "$VAL_DUMP"
test -f "$SOURCE/binary50_option_ranker.txt"
test -f "$SOURCE/locked_binary50_policy.json"
test ! -e "$OUT"
test ! -e "$RESULT"

"$PY" -m py_compile "$SCRIPT"
"$PY" "$SCRIPT" \
  "$TRAIN_DUMP" \
  "$SOURCE/binary50_option_ranker.txt" \
  "$SOURCE/locked_binary50_policy.json" \
  "$OUT" --num-threads 16

MODEL="$OUT/binary50_option_ranker_fit85.txt"
LOCK="$OUT/locked_binary50_fit85_policy.json"
sha256sum "$MODEL" "$LOCK" > "$OUT/lock_sha256_before_val.txt"
"$PY" "$P/train_joint_option_ranker.py" evaluate \
  "$VAL_DUMP" "$MODEL" "$LOCK" "$RESULT"
sha256sum "$MODEL" "$LOCK" > "$OUT/lock_sha256_after_val.txt"
diff -u "$OUT/lock_sha256_before_val.txt" "$OUT/lock_sha256_after_val.txt"
"$PY" - "$RESULT" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], 'r'))
s = r['selected']
print('STAGE64_FIT85_GATE15_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'strict_goal={}'.format(
          s['acc025'], s['acc050'],
          bool(s['acc025'] > 0.5391 and s['acc050'] > 0.4241)))
PY
