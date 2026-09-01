#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33947}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

TRAIN_OUT="${ROOT}/stage145b_alignment_rescue_trainonly_k16"
OUT="${ROOT}/stage145c_alignment_rescue_formal_reload_eval"
LOG="${P}/stage145c_alignment_formal_reload_eval.log"
STATUS="${P}/stage145c_alignment_formal_reload_status.txt"

mapfile -t CKPTS < <(find "$TRAIN_OUT" -type f -name ckpt_best_primary.pth | sort)
test "${#CKPTS[@]}" = 1
CKPT="${CKPTS[0]}"
test ! -e "$OUT"
test "$(sha256sum models/detector_policy_sources.py | awk '{print $1}')" = \
  "da33ab79e20fbb2e35858a7da9729979609878588df20da6754dc61c10c54ff2"
test "$(sha256sum models/bdetr.py | awk '{print $1}')" = \
  "00083a5498de8a1aa8442b046a849159b3c643db24395ee1d8c25eab2259f233"
test "$(sha256sum models/losses.py | awk '{print $1}')" = \
  "b84d6cad8778877dd2a58c818736376ccc5cb553cfac308f87ee493c31c8e930"
test "$(sha256sum main_utils.py | awk '{print $1}')" = \
  "7de08db9b8826af8a62a32d9b331a704a57ced6da583c89646c31a46706e81b4"
test "$(sha256sum train_dist_mod.py | awk '{print $1}')" = \
  "96686a7987d0bac3dcea39bfcfbb3cad0354a3cf4eb937436cd3cdc67a1ae86d"

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
 --detector_policy_alignment_rescue_head
 --detector_policy_alignment_candidate_k 16
 --detector_policy_alignment_override_threshold 0.0
 --detector_policy_alignment_rescue_loss_weight 0.0
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
  printf 'stage=145c_alignment_formal_reload_eval\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'checkpoint=%s\ncheckpoint_sha256=%s\n' "$CKPT" "$CKPT_SHA"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage145c_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
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
    'stage': '145c_alignment_formal_reload_eval',
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
}
with open(receipt_path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
chmod 0444 "$OUT/launch_manifest.txt" "$OUT/eval_results.json" \
  "$OUT/formal_receipt.json"
printf 'stage145c_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
