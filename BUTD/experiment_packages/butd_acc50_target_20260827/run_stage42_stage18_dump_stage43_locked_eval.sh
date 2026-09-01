#!/usr/bin/env bash
set -euo pipefail

R='/home/gb/new butd/butd_detr-main'
P="$R/experiment_packages/butd_acc50_target_20260827"
source "$R/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

ROOT='/root/autodl-tmp/logs/butd_acc50_target_20260827'
OUT="$ROOT/stage42_stage18_e8_geometry_dump"
CKPT="$ROOT/stage18_geometry_action_head/scanrefer_spacy/1787892530/ckpt_best_primary.pth"
PATCHED="$P/grounding_evaluator_adapter_dump.py"
BACKUP="$P/state/stage42_stage18_e8_geometry_dump_20260828"
DUMP="$OUT/stage18_e8_geometry.pt"
MODEL="$ROOT/stage29_binary50_ranker/binary50_option_ranker.txt"
LOCK="$ROOT/stage29_binary50_ranker/locked_binary50_policy.json"
RESULT="$ROOT/stage43_stage29_on_stage18_locked_val_eval.json"
STATUS="$P/stage42_chain_status.txt"
LIVE='src/grounding_evaluator.py'

for path in "$CKPT" "$PATCHED" "$MODEL" "$LOCK" "$LIVE"; do
  test -s "$path"
done
test ! -e "$OUT"
test ! -e "$RESULT"
assert_gpu_idle
assert_storage

mkdir -p "$BACKUP" "$OUT"
cp -p "$LIVE" "$BACKUP/grounding_evaluator.py"
LIVE_HASH_BEFORE="$(sha256sum "$LIVE" | awk '{print $1}')"
restore() {
  cp -p "$BACKUP/grounding_evaluator.py" "$LIVE"
}
trap restore EXIT INT TERM
cp -p "$PATCHED" "$LIVE"

printf 'stage42_dumping %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
base_command
CMD+=(--eval --checkpoint_path "$CKPT" --log_dir "$OUT"
 --eval_results_json_path "$OUT/eval_results.json"
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
 --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1
 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="$DUMP" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${CMD[@]}" \
  2>&1 | tee "$P/stage42_stage18_geometry_dump.log"

restore
trap - EXIT INT TERM
LIVE_HASH_AFTER="$(sha256sum "$LIVE" | awk '{print $1}')"
test "$LIVE_HASH_BEFORE" = "$LIVE_HASH_AFTER"

"$PYTHON" - "$DUMP" "$CKPT" <<'PY'
import hashlib, sys, torch
dump, checkpoint = sys.argv[1:]
payload = torch.load(dump, map_location='cpu')
rows = payload['rows']
assert len(rows) == 9508, len(rows)
assert all(row.get('adapter_candidate_query') for row in rows)
assert all('adapter_hit50_logit_at_candidate' in row for row in rows)
assert all('adapter_box_at_candidate' in row for row in rows)
assert all('gt_box' in row and 'detected_box' in row for row in rows)
print('STAGE42_STAGE18_DUMP_PASS rows={} checkpoint_epoch={} dump_bytes={}'.format(
    len(rows), torch.load(checkpoint, map_location='cpu')['epoch'],
    __import__('os').path.getsize(dump)))
PY

printf 'stage43_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" "$P/train_joint_option_ranker.py" evaluate \
  "$DUMP" "$MODEL" "$LOCK" "$RESULT" \
  2>&1 | tee "$P/stage43_stage29_on_stage18_locked_val_eval.log"

test -s "$RESULT"
"$PYTHON" - "$RESULT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
s = d['selected']
print('STAGE43_STAGE29_ON_STAGE18 acc025={:.10f} acc050={:.10f} offline_goal={}'.format(
    s['acc025'], s['acc050'], d['goal_achieved_offline']))
PY
printf 'stage43_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
