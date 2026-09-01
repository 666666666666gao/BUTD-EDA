#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/logs/butd_acc50_target_20260827
PKG='/home/gb/new butd/butd_detr-main/experiment_packages/butd_acc50_target_20260827'
TRAIN_DUMP="$ROOT/stage133_stage95_e11_raw_train_geometry_dump/stage95_e11_raw_train_geometry_with_ids.pt"
VAL_DUMP="$ROOT/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="$ROOT/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="$ROOT/stage33_pointwise_ranker/locked_pointwise_policy.json"
OUTDIR="$ROOT/stage140_train_only_nested_blend"
POLICY="$OUTDIR/locked_train_only_nested_blend_policy.json"
RESULT="$OUTDIR/stage140_on_stage135c_eval.json"
LOG="$OUTDIR/stage140.log"
PY=/root/miniconda3/envs/bdetr/bin/python

test "$(sha256sum "$TRAIN_DUMP" | awk '{print $1}')" = \
  b89f72a67b4b494d2a66e9b34d603e365afeb934f194a5d129c618ba7bf20313
test "$(sha256sum "$VAL_DUMP" | awk '{print $1}')" = \
  6a837f903f69b0ec15f43bf0544230344352adf14303c6ea0d13a7e842825508
test "$(sha256sum "$STAGE31_LOCK" | awk '{print $1}')" = \
  4c1be1199fe2bc62dc3e4679c4ad26af4af193db1f5dda551b8cc559620b83c9
test "$(sha256sum "$STAGE33_LOCK" | awk '{print $1}')" = \
  da1e020bc190d9792a6df57bf83b3d8be41a7e754eb353ab350a40d11588705d
test "$(sha256sum "$PKG/stage140_train_eval_nested_blend.py" | awk '{print $1}')" = \
  0a17816dc5285dee56fdaab333b3818a856367c3cbd3235ef3cbd8833c86d7ff
test "$(sha256sum "$PKG/train_joint_option_ranker.py" | awk '{print $1}')" = \
  67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407
test ! -e "$OUTDIR"
mkdir -p "$OUTDIR"

record_exit() {
  code=$?
  printf '%s\n' "$code" > "$OUTDIR/exit_code.txt"
}
trap record_exit EXIT

{
  printf 'stage=140_train_only_nested_blend\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'selection_scope=scanrefer_train_scene_hash_dev_only\n'
  printf 'validation_labels_used_for_selection=false\n'
  printf 'train_dump_sha256=%s\n' "$(sha256sum "$TRAIN_DUMP" | awk '{print $1}')"
  printf 'val_dump_sha256=%s\n' "$(sha256sum "$VAL_DUMP" | awk '{print $1}')"
  printf 'stage31_lock_sha256=%s\n' "$(sha256sum "$STAGE31_LOCK" | awk '{print $1}')"
  printf 'stage33_lock_sha256=%s\n' "$(sha256sum "$STAGE33_LOCK" | awk '{print $1}')"
  printf 'script_sha256=%s\n' "$(sha256sum "$PKG/stage140_train_eval_nested_blend.py" | awk '{print $1}')"
  printf 'gpu_visible=none\n'
} > "$OUTDIR/launch_manifest.txt"

cd "$PKG"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
"$PY" stage140_train_eval_nested_blend.py train \
  "$TRAIN_DUMP" "$STAGE31_LOCK" "$STAGE33_LOCK" "$POLICY" \
  2>&1 | tee "$LOG"
test -s "$POLICY"
sha256sum "$POLICY" > "$OUTDIR/locked_train_only_nested_blend_policy.json.sha256"

"$PY" stage140_train_eval_nested_blend.py evaluate \
  "$VAL_DUMP" "$POLICY" "$RESULT" \
  2>&1 | tee -a "$LOG"
test -s "$RESULT"
sha256sum "$RESULT" > "$OUTDIR/stage140_on_stage135c_eval.json.sha256"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "$OUTDIR/launch_manifest.txt"
