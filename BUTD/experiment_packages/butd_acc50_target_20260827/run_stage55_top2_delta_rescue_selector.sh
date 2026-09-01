#!/usr/bin/env bash
set -euo pipefail

ROOT='/home/gb/new butd/butd_detr-main'
PKG="$ROOT/experiment_packages/butd_acc50_target_20260827"
PY='/root/miniconda3/envs/bdetr/bin/python'
BASE='/root/autodl-tmp/logs/butd_acc50_target_20260827'
TRAIN_DUMP="$BASE/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
VAL_DUMP="$BASE/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
OUT="$BASE/stage55_top2_delta_rescue_selector"
RESULT="$BASE/stage56_top2_delta_rescue_locked_val_eval.json"
TOP1_SCRIPT="$PKG/train_joint_option_ranker.py"
TOP2_SCRIPT="$PKG/train_joint_option_ranker_top2.py"
TOP1_MODEL="$BASE/stage29_binary50_ranker/binary50_option_ranker.txt"
TOP2_MODEL="$BASE/stage49_top2_binary50_ranker/binary50_option_ranker.txt"
TOP1_LOCK="$BASE/stage29_binary50_ranker/locked_binary50_policy.json"
TOP2_LOCK="$BASE/stage49_top2_binary50_ranker/locked_binary50_policy.json"
SCRIPT="$PKG/train_top2_delta_rescue_selector.py"

cd "$ROOT"
test ! -e "$OUT"
test ! -e "$RESULT"
"$PY" "$SCRIPT" self-test
"$PY" "$SCRIPT" train "$TRAIN_DUMP" "$OUT" \
  --num-threads 16 \
  --top1-script "$TOP1_SCRIPT" --top2-script "$TOP2_SCRIPT" \
  --top1-model "$TOP1_MODEL" --top2-model "$TOP2_MODEL" \
  --top1-lock "$TOP1_LOCK" --top2-lock "$TOP2_LOCK"
sha256sum "$OUT/top2_delta_selector.txt" \
  "$OUT/locked_top2_delta_policy.json" > "$OUT/lock_sha256_before_val.txt"
"$PY" "$SCRIPT" evaluate "$VAL_DUMP" \
  "$OUT/top2_delta_selector.txt" "$OUT/locked_top2_delta_policy.json" \
  "$RESULT" \
  --top1-script "$TOP1_SCRIPT" --top2-script "$TOP2_SCRIPT" \
  --top1-model "$TOP1_MODEL" --top2-model "$TOP2_MODEL" \
  --top1-lock "$TOP1_LOCK" --top2-lock "$TOP2_LOCK"
sha256sum "$OUT/top2_delta_selector.txt" \
  "$OUT/locked_top2_delta_policy.json" > "$OUT/lock_sha256_after_val.txt"
diff -u "$OUT/lock_sha256_before_val.txt" "$OUT/lock_sha256_after_val.txt"
