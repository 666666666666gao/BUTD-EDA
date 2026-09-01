#!/usr/bin/env bash
set -euo pipefail

ROOT='/home/gb/new butd/butd_detr-main'
PKG="$ROOT/experiment_packages/butd_acc50_target_20260827"
PY='/root/miniconda3/envs/bdetr/bin/python'
SCRIPT="$PKG/train_stage38_meta_selector.py"
JOINT="$PKG/train_joint_option_ranker.py"
CLEAN='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt'
VAL='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt'
LOCK29='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage29_binary50_ranker/locked_binary50_policy.json'
LOCK36='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage36_mixed_binary50_ranker/locked_mixed_policy.json'
OUT='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage38_meta_selector'
META="$OUT/locked_stage38_meta.json"
RESULT='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage39_meta_locked_val_eval.json'
STATUS="$PKG/stage38_chain_status.txt"

cd "$ROOT"
for path in "$SCRIPT" "$JOINT" "$CLEAN" "$VAL" "$LOCK29" "$LOCK36"; do
  test -s "$path"
done
test ! -e "$OUT"
test ! -e "$RESULT"

printf 'stage38_selftest %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PY" "$SCRIPT" self-test

printf 'stage38_training %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PY" -u "$SCRIPT" train \
  --joint-script "$JOINT" \
  --stage29-lock "$LOCK29" \
  --stage36-lock "$LOCK36" \
  --dump "$CLEAN" \
  --require-scene \
  --output-dir "$OUT" \
  --seed 17 \
  --num-threads 16 \
  2>&1 | tee "$PKG/stage38_meta_selector_train.log"

test -s "$META"
printf 'stage39_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PY" -u "$SCRIPT" evaluate \
  --joint-script "$JOINT" \
  --stage29-lock "$LOCK29" \
  --stage36-lock "$LOCK36" \
  --dump "$VAL" \
  --meta-lock "$META" \
  --output-json "$RESULT" \
  2>&1 | tee "$PKG/stage39_meta_locked_val_eval.log"

test -s "$RESULT"
"$PY" - "$RESULT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
s = d['selected']
print('STAGE39_META_LOCKED_VAL acc025={:.10f} acc050={:.10f} offline_goal={}'.format(
    s['acc025'], s['acc050'], d['goal_achieved_offline']))
PY
printf 'stage39_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
