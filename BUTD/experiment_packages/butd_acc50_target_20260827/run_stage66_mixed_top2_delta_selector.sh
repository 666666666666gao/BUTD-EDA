#!/usr/bin/env bash
set -euo pipefail

R='/home/gb/new butd/butd_detr-main'
P="$R/experiment_packages/butd_acc50_target_20260827"
ROOT='/root/autodl-tmp/logs/butd_acc50_target_20260827'
PY='/root/miniconda3/envs/bdetr/bin/python'
CLEAN="$ROOT/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
AUG="$ROOT/stage35_stage16_e8_augmented_train_dump/stage16_e8_augmented_train_geometry.pt"
VAL="$ROOT/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
TOP1_SCRIPT="$P/train_joint_option_ranker.py"
TOP2_SCRIPT="$P/train_joint_option_ranker_top2.py"
TOP1_MODEL="$ROOT/stage29_binary50_ranker/binary50_option_ranker.txt"
TOP2_MODEL="$ROOT/stage69_rebuild_stage49_top2/binary50_option_ranker.txt"
TOP1_LOCK="$ROOT/stage29_binary50_ranker/locked_binary50_policy.json"
TOP2_LOCK="$ROOT/stage69_rebuild_stage49_top2/locked_binary50_policy.json"
SCRIPT="$P/train_top2_delta_rescue_selector_mixed.py"
EVAL_SCRIPT="$P/train_top2_delta_rescue_selector.py"
OUT="$ROOT/stage66_mixed_top2_delta_selector"
RESULT="$ROOT/stage67_mixed_top2_delta_locked_val_eval.json"

cd "$R"
for f in "$CLEAN" "$AUG" "$VAL" "$TOP1_SCRIPT" "$TOP2_SCRIPT" \
  "$TOP1_MODEL" "$TOP2_MODEL" "$TOP1_LOCK" "$TOP2_LOCK"; do
  test -f "$f"
done
test ! -e "$OUT"
test ! -e "$RESULT"
"$PY" -m py_compile "$SCRIPT"
"$PY" "$SCRIPT" "$CLEAN" "$AUG" "$OUT" --num-threads 16 \
  --top1-script "$TOP1_SCRIPT" --top2-script "$TOP2_SCRIPT" \
  --top1-model "$TOP1_MODEL" --top2-model "$TOP2_MODEL" \
  --top1-lock "$TOP1_LOCK" --top2-lock "$TOP2_LOCK"

MODEL="$OUT/top2_delta_selector.txt"
LOCK="$OUT/locked_top2_delta_policy.json"
sha256sum "$MODEL" "$LOCK" > "$OUT/lock_sha256_before_val.txt"
"$PY" "$EVAL_SCRIPT" evaluate "$VAL" "$MODEL" "$LOCK" "$RESULT" \
  --top1-script "$TOP1_SCRIPT" --top2-script "$TOP2_SCRIPT" \
  --top1-model "$TOP1_MODEL" --top2-model "$TOP2_MODEL" \
  --top1-lock "$TOP1_LOCK" --top2-lock "$TOP2_LOCK"
sha256sum "$MODEL" "$LOCK" > "$OUT/lock_sha256_after_val.txt"
diff -u "$OUT/lock_sha256_before_val.txt" "$OUT/lock_sha256_after_val.txt"
"$PY" - "$RESULT" <<'PY'
import json
import sys
r=json.load(open(sys.argv[1]))
s=r['selected']
print('STAGE67_MIXED_TOP2_DELTA acc025={:.10f} acc050={:.10f} '
      'strict_goal={}'.format(
          s['acc025'], s['acc050'],
          bool(s['acc025'] > 0.5391 and s['acc050'] > 0.4241)))
PY
