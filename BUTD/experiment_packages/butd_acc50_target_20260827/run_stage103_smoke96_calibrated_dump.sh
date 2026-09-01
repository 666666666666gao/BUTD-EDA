#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33903}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${ROOT}/stage103_stage95_calibrated_dump_smoke96"
BASELINE_METRICS="${ROOT}/stage100a_stage95_live_smoke96/eval_results.json"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed_v2.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${P}/state/stage103_calibrated_dump_smoke96_20260830"
DUMP="${OUT}/stage95_e11_calibrated_geometry_smoke96.pt"
MODEL="${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt"
LOCK="${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json"
RANKER_SCRIPT="${P}/train_joint_option_ranker.py"
RESULT="${OUT}/stage29_locked_smoke96_eval.json"
STATUS="${P}/stage103_calibrated_dump_smoke96_status.txt"

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="51017942ea99e7e734e59c9c89d895de6a775069e0e00b5426b8f1b0ac476794"
EXPECTED_BASELINE_METRICS_SHA="e0246b24f9ba8ee0fd111cec18a6d7a1da42f77b988130b75b41990f20f6b1b8"
EXPECTED_MODEL_SHA="4ee630977503cc7bfa303bea6d67a180d30394fea7f47746cae3b2e067431e2e"
EXPECTED_LOCK_SHA="5acafbca18320a077b16ac6407fd696e6ac68a124c292a26bad455cbba15cdce"
EXPECTED_RANKER_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUT}"
test ! -e "${BACKUP}"
check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${BASELINE_METRICS}" "${EXPECTED_BASELINE_METRICS_SHA}"
check_sha "${MODEL}" "${EXPECTED_MODEL_SHA}"
check_sha "${LOCK}" "${EXPECTED_LOCK_SHA}"
check_sha "${RANKER_SCRIPT}" "${EXPECTED_RANKER_SCRIPT_SHA}"

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
 --eval_max_samples 96 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${OUT}"
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage103.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage103.py"
restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage103.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

printf 'stage103_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage103_stage95_calibrated_dump_smoke96.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" "${RANKER_SCRIPT}" evaluate \
  "${DUMP}" "${MODEL}" "${LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage103_stage29_locked_smoke96_eval.log"

"${PYTHON}" - "${BASELINE_METRICS}" "${OUT}/eval_results.json" \
  "${DUMP}" "${RESULT}" "${OUT}" <<'PY'
import hashlib
import json
import os
import sys
import torch

baseline_path, dump_metrics_path, dump_path, result_path, out = sys.argv[1:]
baseline = json.load(open(baseline_path, encoding='utf-8'))
dump_metrics = json.load(open(dump_metrics_path, encoding='utf-8'))
row_count = dump_metrics.pop('detector_topk_compact_rows')
assert row_count == 96, row_count
assert dump_metrics == baseline
rows = torch.load(dump_path, map_location='cpu')['rows']
assert len(rows) == 96
result = json.load(open(result_path, encoding='utf-8'))
expected = (
    baseline['last__bbs_acc0.25_top1'],
    baseline['last__bbs_acc0.50_top1'],
)
actual = (result['baseline']['acc025'], result['baseline']['acc050'])
assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected)), (
    actual, expected
)
receipt = {
    'stage': '103',
    'status': 'calibrated_dump_semantics_pass',
    'samples': len(rows),
    'shared_eval_metrics_exact_match': True,
    'ranker_baseline_matches_evaluator': True,
    'baseline': result['baseline'],
    'selected': result['selected'],
    'dump_evaluator_sha256': '51017942ea99e7e734e59c9c89d895de6a775069e0e00b5426b8f1b0ac476794',
}
for key, path in (
    ('dump_sha256', dump_path),
    ('result_sha256', result_path),
):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    receipt[key] = h.hexdigest()
receipt_path = os.path.join(out, 'stage103_receipt.json')
with open(receipt_path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage103_receipt.json"
printf 'stage103_semantics_pass %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
