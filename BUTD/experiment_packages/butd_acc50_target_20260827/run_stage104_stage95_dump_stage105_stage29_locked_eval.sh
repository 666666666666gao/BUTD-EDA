#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33904}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed_v2.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${P}/state/stage104_stage95_e11_calibrated_geometry_dump_20260830"
DUMP="${OUT}/stage95_e11_geometry.pt"
MODEL="${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt"
LOCK="${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json"
RANKER_SCRIPT="${P}/train_joint_option_ranker.py"
RESULT="${ROOT}/stage105_stage29_on_stage95_calibrated_locked_val_eval.json"
STATUS="${P}/stage104_105_chain_status.txt"
SMOKE_RECEIPT="${ROOT}/stage103_stage95_calibrated_dump_smoke96/stage103_receipt.json"

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="51017942ea99e7e734e59c9c89d895de6a775069e0e00b5426b8f1b0ac476794"
EXPECTED_MODEL_SHA="4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e"
EXPECTED_LOCK_SHA="5acafbca18320a077b16ac6407fd696e6ac68a124c292a26bad455cbba15cdce"
EXPECTED_RANKER_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"
EXPECTED_SMOKE_RECEIPT_SHA="bc99b19029cd45bf012ac4d78965cde6064bd6b40baa1e4c1132b0d801a60bcf"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUT}"
test ! -e "${RESULT}"
test ! -e "${BACKUP}"
check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${MODEL}" "${EXPECTED_MODEL_SHA}"
check_sha "${LOCK}" "${EXPECTED_LOCK_SHA}"
check_sha "${RANKER_SCRIPT}" "${EXPECTED_RANKER_SCRIPT_SHA}"
check_sha "${SMOKE_RECEIPT}" "${EXPECTED_SMOKE_RECEIPT_SHA}"

base_command
CMD+=(--eval --checkpoint_path "${CKPT}" --log_dir "${OUT}"
 --eval_results_json_path "${OUT}/eval_results.json"
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

mkdir -p "${BACKUP}" "${OUT}"
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage104.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage104.py"
restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage104.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=104_105\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'master_port=%s\n' "${MASTER_PORT}"
  printf 'source_checkpoint=%s\n' "${CKPT}"
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'main_utils_sha256=%s\n' "${EXPECTED_MAIN_SHA}"
  printf 'losses_sha256=%s\n' "${EXPECTED_LOSS_SHA}"
  printf 'live_evaluator_before_sha256=%s\n' "${EXPECTED_LIVE_SHA}"
  printf 'stage103_calibrated_dump_receipt=%s\n' "${SMOKE_RECEIPT}"
  printf 'stage103_calibrated_dump_receipt_sha256=%s\n' "${EXPECTED_SMOKE_RECEIPT_SHA}"
  printf 'dump_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_SHA}"
  printf 'stage29_model_sha256=%s\n' "${EXPECTED_MODEL_SHA}"
  printf 'stage29_lock_sha256=%s\n' "${EXPECTED_LOCK_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage104_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage104_stage95_calibrated_geometry_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}/eval_results.json" "${CKPT}" <<'PY'
import json
import os
import sys
import torch

dump, metrics_path, checkpoint = sys.argv[1:]
payload = torch.load(dump, map_location='cpu')
rows = payload['rows']
assert len(rows) == 9508, len(rows)
assert all(row.get('adapter_candidate_query') for row in rows)
assert all('adapter_hit50_logit_at_candidate' in row for row in rows)
assert all('adapter_box_at_candidate' in row for row in rows)
assert all('gt_box' in row and 'detected_box' in row for row in rows)
metrics = json.load(open(metrics_path, encoding='utf-8'))
expected = (0.5477492637778713, 0.41533445519562473)
actual = (metrics['last__bbs_acc0.25_top1'], metrics['last__bbs_acc0.50_top1'])
assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected)), (actual, expected)
assert int(torch.load(checkpoint, map_location='cpu')['epoch']) == 11
print('STAGE104_DUMP_PARITY_PASS rows={} acc025={:.12f} acc050={:.12f} bytes={}'.format(
    len(rows), actual[0], actual[1], os.path.getsize(dump)))
PY

printf 'stage105_evaluating %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${RANKER_SCRIPT}" evaluate \
  "${DUMP}" "${MODEL}" "${LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage105_stage29_on_stage95_calibrated_locked_val_eval.log"
test -s "${RESULT}"

"${PYTHON}" - "${OUT}" "${DUMP}" "${RESULT}" <<'PY'
import hashlib
import json
import os
import sys

out, dump, result_path = sys.argv[1:]
result = json.load(open(result_path, encoding='utf-8'))
baseline = result['baseline']
expected_baseline = (0.5477492637778713, 0.41533445519562473)
actual_baseline = (baseline['acc025'], baseline['acc050'])
assert all(
    abs(actual - expected) < 1e-12
    for actual, expected in zip(actual_baseline, expected_baseline)
), (actual_baseline, expected_baseline)
selected = result['selected']
receipt = {
    'stage': '104_105',
    'status': 'complete',
    'source_checkpoint_sha256': 'f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811',
    'source_metrics': {'acc025': 0.5477492637778713, 'acc050': 0.41533445519562473},
    'ranker_baseline_metrics': baseline,
    'ranker_baseline_matches_evaluator': True,
    'selected_metrics': selected,
    'selected_hits': {
        'acc025': round(selected['acc025'] * 9508),
        'acc050': round(selected['acc050'] * 9508),
    },
    'strict_goal': {'acc025_gt': 0.5391, 'acc050_gt': 0.4241},
    'strict_goal_met_offline': bool(selected['acc025'] > 0.5391 and selected['acc050'] > 0.4241),
    'diagnostic_only_until_integrated_and_reloaded': True,
}
for key, path in [('dump_sha256', dump), ('result_sha256', result_path)]:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    receipt[key] = h.hexdigest()
path = os.path.join(out, 'stage104_105_receipt.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
chmod 0444 "${OUT}/stage104_105_receipt.json" "${OUT}/launch_manifest.txt"
printf 'stage105_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
