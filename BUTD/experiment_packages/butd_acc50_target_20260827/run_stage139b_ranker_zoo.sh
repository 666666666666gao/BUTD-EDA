#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/logs/butd_acc50_target_20260827
PKG='/home/gb/new butd/butd_detr-main/experiment_packages/butd_acc50_target_20260827'
DUMP="$ROOT/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
OUTDIR="$ROOT/stage139b_stage135c_ranker_zoo"
RESULT="$OUTDIR/stage139b_ranker_zoo.json"
LOG="$OUTDIR/stage139b.log"
PY=/root/miniconda3/envs/bdetr/bin/python

test "$(sha256sum "$DUMP" | awk '{print $1}')" = \
  6a837f903f69b0ec15f43bf0544230344352adf14303c6ea0d13a7e842825508
test "$(sha256sum "$PKG/analyze_stage139_ranker_zoo.py" | awk '{print $1}')" = \
  2d9d06a9ad96c9033ff9b3409635733f9efc4a68f23783937c63f805f2840bfb
test "$(sha256sum "$PKG/train_joint_option_ranker.py" | awk '{print $1}')" = \
  67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407
test ! -e "$OUTDIR"
mkdir -p "$OUTDIR"

{
  printf 'stage=139b_stage135c_ranker_zoo\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'dump=%s\n' "$DUMP"
  printf 'dump_sha256=%s\n' "$(sha256sum "$DUMP" | awk '{print $1}')"
  printf 'script_sha256=%s\n' "$(sha256sum "$PKG/analyze_stage139_ranker_zoo.py" | awk '{print $1}')"
  printf 'trainer_sha256=%s\n' "$(sha256sum "$PKG/train_joint_option_ranker.py" | awk '{print $1}')"
  printf 'diagnostic_only=true\n'
  printf 'gpu_visible=none\n'
} > "$OUTDIR/launch_manifest.txt"

cd "$PKG"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
set +e
"$PY" analyze_stage139_ranker_zoo.py \
  "$DUMP" "$ROOT" "$RESULT" \
  --top-models 10 --threads 16 2>&1 | tee "$LOG"
code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$code" > "$OUTDIR/exit_code.txt"
if [[ "$code" -ne 0 ]]; then
  exit "$code"
fi
test -s "$RESULT"
sha256sum "$RESULT" > "$OUTDIR/stage139b_ranker_zoo.json.sha256"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "$OUTDIR/launch_manifest.txt"
