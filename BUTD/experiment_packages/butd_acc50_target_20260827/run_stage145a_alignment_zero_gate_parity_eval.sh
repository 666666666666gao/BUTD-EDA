#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33945}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

CKPT="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
OUT="${ROOT}/stage145a_alignment_rescue_zero_gate_parity_eval"
LOG="${P}/stage145a_alignment_zero_gate_parity_eval.log"
STATUS="${P}/stage145a_alignment_zero_gate_parity_status.txt"

test ! -e "$OUT"
test "$(sha256sum "$CKPT" | awk '{print $1}')" = \
  "a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
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
{
  printf 'stage=145a_alignment_zero_gate_parity\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'checkpoint=%s\ncheckpoint_sha256=%s\n' "$CKPT" \
    "$(sha256sum "$CKPT" | awk '{print $1}')"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage145a_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "$LOG"

"${PYTHON}" - "$OUT/eval_results.json" <<'PY'
import json
import math
import sys

path = sys.argv[1]
metrics = json.load(open(path, encoding='utf-8'))
expected25 = 0.548485485906605
expected50 = 0.41607067732435843
actual25 = metrics['last__bbs_acc0.25_top1']
actual50 = metrics['last__bbs_acc0.50_top1']
assert math.isclose(actual25, expected25, rel_tol=0.0, abs_tol=0.0), (
    actual25, expected25
)
assert math.isclose(actual50, expected50, rel_tol=0.0, abs_tol=0.0), (
    actual50, expected50
)
print('STAGE145A_EXACT_PARITY_PASS acc025={:.15f} acc050={:.15f}'.format(
    actual25, actual50
))
PY
chmod 0444 "$OUT/launch_manifest.txt" "$OUT/eval_results.json"
printf 'stage145a_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
