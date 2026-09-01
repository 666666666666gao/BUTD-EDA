#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33936}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

TRAIN_OUT="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask"
mapfile -t CKPTS < <(find "$TRAIN_OUT" -type f -name ckpt_best_primary.pth | sort)
test "${#CKPTS[@]}" = 1
CKPT="${CKPTS[0]}"
OUT="${ROOT}/stage136c_stage135c_raw_val_geometry_dump"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${P}/state/stage136c_stage135c_raw_val_dump_20260830"
DUMP="${OUT}/stage135c_e12_raw_val_geometry.pt"
MODEL="${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt"
LOCK="${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json"
RANKER_SCRIPT="${P}/train_joint_option_ranker.py"
RESULT="${ROOT}/stage137c_stage29_on_stage135c_locked_val_eval.json"
STATUS="${P}/stage136c_137c_chain_status.txt"

EXPECTED_MAIN_SHA="11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4"
EXPECTED_LOSS_SHA="a95443a958c0170faac513c46001e8c60f46cfaacd559b620c79eb622ec2c852"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"
EXPECTED_MODEL_SHA="4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e"
EXPECTED_LOCK_SHA="5acafbca18320a077b16ac6407fd696e6ac68a124c292a26bad455cbba15cdce"
EXPECTED_RANKER_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"

check_sha() {
  local path="$1" expected="$2"
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

test ! -e "$OUT" && test ! -e "$RESULT" && test ! -e "$BACKUP"
check_sha main_utils.py "$EXPECTED_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_DATASET_SHA"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "$EXPECTED_COMMON_SHA"
check_sha "$LIVE" "$EXPECTED_LIVE_SHA"
check_sha "$PATCHED" "$EXPECTED_PATCHED_SHA"
check_sha "$MODEL" "$EXPECTED_MODEL_SHA"
check_sha "$LOCK" "$EXPECTED_LOCK_SHA"
check_sha "$RANKER_SCRIPT" "$EXPECTED_RANKER_SHA"
CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "$CKPT")"
test "$E" = 12

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
mkdir -p "$BACKUP" "$OUT"
cp -p "$LIVE" "$BACKUP/grounding_evaluator.pre_stage136c.py"
chmod 0444 "$BACKUP/grounding_evaluator.pre_stage136c.py"
restore() {
  install -m 0644 "$BACKUP/grounding_evaluator.pre_stage136c.py" "$LIVE"
}
trap restore EXIT INT TERM ERR
install -m 0644 "$PATCHED" "$LIVE"
check_sha "$LIVE" "$EXPECTED_PATCHED_SHA"

{
  printf 'stage=136c_137c\nstarted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_checkpoint=%s\nsource_checkpoint_sha256=%s\n' "$CKPT" "$CKPT_SHA"
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\ndataset_sha256=%s\n' "$EXPECTED_MAIN_SHA" "$EXPECTED_LOSS_SHA" "$EXPECTED_DATASET_SHA"
  printf 'dump_evaluator_sha256=%s\n' "$EXPECTED_PATCHED_SHA"
  printf 'stage29_model_sha256=%s\nstage29_lock_sha256=%s\n' "$EXPECTED_MODEL_SHA" "$EXPECTED_LOCK_SHA"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage136c_dumping %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="$DUMP" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "$P/stage136c_stage135c_raw_val_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "$LIVE" "$EXPECTED_LIVE_SHA"

"${PYTHON}" - "$DUMP" "$OUT/eval_results.json" "$CKPT" <<'PY'
import json
import os
import sys
import torch

dump, metrics_path, checkpoint = sys.argv[1:]
rows = torch.load(dump, map_location='cpu')['rows']
assert len(rows) == 9508, len(rows)
required = ('adapter_candidate_query', 'adapter_hit50_logit_at_candidate',
            'adapter_box_at_candidate', 'gt_box', 'detected_box')
for key in required:
    assert all(key in row for row in rows), key
metrics = json.load(open(metrics_path, encoding='utf-8'))
assert int(torch.load(checkpoint, map_location='cpu')['epoch']) == 12
print('STAGE136C_DUMP_PASS rows={} acc025={:.12f} acc050={:.12f} bytes={}'.format(
    len(rows), metrics['last__bbs_acc0.25_top1'],
    metrics['last__bbs_acc0.50_top1'], os.path.getsize(dump)))
PY

printf 'stage137c_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"${PYTHON}" "$RANKER_SCRIPT" evaluate \
  "$DUMP" "$MODEL" "$LOCK" "$RESULT" \
  2>&1 | tee "$P/stage137c_stage29_on_stage135c_locked_val_eval.log"

"${PYTHON}" - "$OUT" "$DUMP" "$RESULT" "$OUT/eval_results.json" "$CKPT_SHA" <<'PY'
import hashlib
import json
import os
import sys

out, dump, result_path, metrics_path, checkpoint_sha = sys.argv[1:]
result = json.load(open(result_path, encoding='utf-8'))
metrics = json.load(open(metrics_path, encoding='utf-8'))
selected = result['selected']
formal = {
    'acc025': metrics['last__bbs_acc0.25_top1'],
    'acc050': metrics['last__bbs_acc0.50_top1'],
}
receipt = {
    'stage': '136c_137c', 'status': 'complete',
    'source_checkpoint_sha256': checkpoint_sha,
    'formal_adapter_metrics': formal,
    'ranker_baseline_metrics': result['baseline'],
    'stage29_selected_metrics': selected,
    'selected_hits': {
        'acc025': round(selected['acc025'] * 9508),
        'acc050': round(selected['acc050'] * 9508),
    },
    'strict_goal': {'acc025_gt': 0.5391, 'acc050_gt': 0.4241},
    'strict_goal_met_offline': bool(
        selected['acc025'] > 0.5391 and selected['acc050'] > 0.4241
    ),
    'diagnostic_only_until_integrated_and_reloaded': True,
}
for key, path in [('dump_sha256', dump), ('result_sha256', result_path)]:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    receipt[key] = h.hexdigest()
path = os.path.join(out, 'stage136c_137c_receipt.json')
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "$OUT/stage136c_137c_receipt.json" "$OUT/launch_manifest.txt"
printf 'stage137c_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
