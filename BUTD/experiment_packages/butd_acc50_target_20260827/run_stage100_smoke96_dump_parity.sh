#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33900}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

LIVE_OUT="${ROOT}/stage100a_stage95_live_smoke96"
DUMP_OUT="${ROOT}/stage100b_stage95_dump_smoke96"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${P}/state/stage100_dump_parity_smoke96_20260830"
DUMP="${DUMP_OUT}/stage95_e11_geometry_smoke96.pt"
STATUS="${P}/stage100_dump_parity_smoke96_status.txt"

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${LIVE_OUT}"
test ! -e "${DUMP_OUT}"
test ! -e "${BACKUP}"
check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"

base_command
CMD_BASE=("${CMD[@]}"
 --eval --checkpoint_path "${CKPT}"
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
  printf '%q ' "${CMD_BASE[@]}" --log_dir "${LIVE_OUT}" \
    --eval_results_json_path "${LIVE_OUT}/eval_results.json"
  printf '\n'
  printf '%q ' "${CMD_BASE[@]}" --log_dir "${DUMP_OUT}" \
    --eval_results_json_path "${DUMP_OUT}/eval_results.json"
  printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${LIVE_OUT}" "${DUMP_OUT}"
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage100.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage100.py"

restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage100.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR

printf 'stage100_live_smoke %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD_BASE[@]}" \
  --log_dir "${LIVE_OUT}" \
  --eval_results_json_path "${LIVE_OUT}/eval_results.json" \
  2>&1 | tee "${P}/stage100a_stage95_live_smoke96.log"

install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"
printf 'stage100_dump_smoke %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD_BASE[@]}" \
  --log_dir "${DUMP_OUT}" \
  --eval_results_json_path "${DUMP_OUT}/eval_results.json" \
  2>&1 | tee "${P}/stage100b_stage95_dump_smoke96.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - \
  "${LIVE_OUT}/eval_results.json" \
  "${DUMP_OUT}/eval_results.json" \
  "${DUMP}" "${DUMP_OUT}" <<'PY'
import hashlib
import json
import os
import sys
import torch

live_path, dump_metrics_path, dump_path, out = sys.argv[1:]
with open(live_path, encoding='utf-8') as f:
    live = json.load(f)
with open(dump_metrics_path, encoding='utf-8') as f:
    dumped = json.load(f)
dump_row_count = dumped.pop('detector_topk_compact_rows')
assert dump_row_count == 96, dump_row_count
assert live == dumped, {
    key: (live.get(key), dumped.get(key))
    for key in sorted(set(live) | set(dumped))
    if live.get(key) != dumped.get(key)
}
payload = torch.load(dump_path, map_location='cpu')
rows = payload['rows']
assert len(rows) == 96, len(rows)
assert all(row.get('adapter_candidate_query') for row in rows)
for required in (
    'adapter_hit50_logit_at_candidate',
    'adapter_box_at_candidate',
    'gt_box',
    'detected_box',
):
    assert all(required in row for row in rows), required

receipt = {
    'stage': '100',
    'status': 'smoke_parity_pass',
    'samples': len(rows),
    'live_evaluator_sha256': '50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86',
    'dump_evaluator_sha256': 'cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f',
    'acc025': live['last__bbs_acc0.25_top1'],
    'acc050': live['last__bbs_acc0.50_top1'],
    'all_eval_metrics_exact_match': True,
}
for key, path in (
    ('live_metrics_sha256', live_path),
    ('dump_metrics_sha256', dump_metrics_path),
    ('dump_sha256', dump_path),
):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    receipt[key] = h.hexdigest()
receipt_path = os.path.join(out, 'stage100_smoke_parity_receipt.json')
with open(receipt_path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${DUMP_OUT}/stage100_smoke_parity_receipt.json"
printf 'stage100_smoke_parity_pass %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
