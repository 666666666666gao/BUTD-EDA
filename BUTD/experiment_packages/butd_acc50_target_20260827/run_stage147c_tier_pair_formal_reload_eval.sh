#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33950}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

TRAIN_OUT="${ROOT}/stage147b_tier_pair_trainonly_k2"
OUT="${ROOT}/stage147c_tier_pair_formal_reload_eval"
LOG="${P}/stage147c_tier_pair_formal_reload_eval.log"
STATUS="${P}/stage147c_tier_pair_formal_reload_status.txt"

mapfile -t CKPTS < <(find "$TRAIN_OUT" -type f -name ckpt_best_primary.pth | sort)
test "${#CKPTS[@]}" = 1
CKPT="${CKPTS[0]}"
test ! -e "$OUT"
test "$(sha256sum models/detector_policy_sources.py | awk '{print $1}')" = \
  "396303854f5acacba410093a3953ad7f1c56288a29f7422fa671e8aee3684166"
test "$(sha256sum models/bdetr.py | awk '{print $1}')" = \
  "325444d2e7474b40764634f410d0f68827945493c3b1b7a73301a97c9da71879"
test "$(sha256sum models/losses.py | awk '{print $1}')" = \
  "4584dfc9cdbaa83adf2b2e294b9b3d95b37df3bf35866e9fda09df8095019305"
test "$(sha256sum main_utils.py | awk '{print $1}')" = \
  "31cb486e03165ee16e0c08b3dde3029d9360132770a1020fa0ea02c451174fba"
test "$(sha256sum train_dist_mod.py | awk '{print $1}')" = \
  "44852f403849266e5d706b39c86e99f04f3bc682652bf8eb06944e11557a25e0"
test "$(sha256sum src/grounding_evaluator.py | awk '{print $1}')" = \
  "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

base_command
CMD+=(--eval --checkpoint_path "$CKPT" --log_dir "$OUT"
 --eval_results_json_path "$OUT/eval_results.json"
 --disable_train_augmentation --disable_box_jitter
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
 --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.0
 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --qahnl_loss_weight 0.0
 --quality_loss_weight 0.0 --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0
 --detector_policy_adapter_loss_weight 0.0
 --detector_policy_geometry_loss_weight 0.0
 --detector_policy_rank2_rescue_loss_weight 0.0
 --detector_policy_alignment_rescue_loss_weight 0.0
 --detector_policy_tier_pair_rescue_head
 --detector_policy_tier_pair_candidate_k 2
 --detector_policy_tier_pair_override_threshold 0.0
 --detector_policy_tier_pair_loss_weight 0.0
 --detector_policy_adapter_margin 0.1
 --detector_policy_adapter_min_iou_gap 0.02
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "$OUT"
CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
{
  printf 'stage=147c_tier_pair_formal_reload_eval\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'checkpoint=%s\ncheckpoint_sha256=%s\n' "$CKPT" "$CKPT_SHA"
  printf 'candidate_policy=fixed_current_score_top2\n'
  printf 'inference_threshold=fixed_zero_no_validation_sweep\n'
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage147c_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "$LOG"

"${PYTHON}" - "$OUT/eval_results.json" "$OUT/formal_receipt.json" \
  "$CKPT" "$CKPT_SHA" <<'PY'
import json
import sys
import torch

metrics_path, receipt_path, checkpoint, checkpoint_sha = sys.argv[1:]
metrics = json.load(open(metrics_path, encoding='utf-8'))
acc25 = metrics['last__bbs_acc0.25_top1']
acc50 = metrics['last__bbs_acc0.50_top1']
rows = 9508
receipt = {
    'stage': '147c_tier_pair_formal_reload_eval',
    'status': 'complete',
    'checkpoint': checkpoint,
    'checkpoint_sha256': checkpoint_sha,
    'checkpoint_epoch': int(torch.load(checkpoint, map_location='cpu')['epoch']),
    'rows': rows,
    'acc025': acc25,
    'acc050': acc50,
    'hits025': round(acc25 * rows),
    'hits050': round(acc50 * rows),
    'strict_goal': {'acc025_gt': 0.5391, 'acc050_gt': 0.4241},
    'strict_goal_met': bool(acc25 > 0.5391 and acc50 > 0.4241),
    'independent_reload': True,
    'gt_used_at_inference': False,
    'threshold_selected_on_validation': False,
}
with open(receipt_path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
chmod 0444 "$OUT/launch_manifest.txt" "$OUT/eval_results.json" \
  "$OUT/formal_receipt.json"
printf 'stage147c_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
