#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?usage: verify_stage0_reload.sh CHECKPOINT QUALITY_WEIGHT GRID_SELECTION}"
QUALITY_WEIGHT="${2:?usage: verify_stage0_reload.sh CHECKPOINT QUALITY_WEIGHT GRID_SELECTION}"
GRID_SELECTION="${3:?usage: verify_stage0_reload.sh CHECKPOINT QUALITY_WEIGHT GRID_SELECTION}"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
source "${OLD_PACKAGE}/common.sh"
cd "${REPO_ROOT}"

VERIFY_ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage0_reload_verify"
RESULT_JSON="${VERIFY_ROOT}/eval_results.json"
RECEIPT="${VERIFY_ROOT}/goal_receipt.json"
ORIGINAL_EVALUATOR_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"

[ -f "${CHECKPOINT}" ] || {
  echo "ERROR: Stage-0 checkpoint missing: ${CHECKPOINT}" >&2
  exit 180
}
[ -f "${GRID_SELECTION}" ] || {
  echo "ERROR: Stage-0 grid selection missing: ${GRID_SELECTION}" >&2
  exit 181
}
[ ! -e "${VERIFY_ROOT}" ] || {
  echo "ERROR: refusing to overwrite Stage-0 verification root" >&2
  exit 182
}
[ "$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')" = "${ORIGINAL_EVALUATOR_SHA}" ] || {
  echo "ERROR: evaluator is not original for Stage-0 reload" >&2
  exit 183
}

assert_gpu_idle
mkdir -p "${VERIFY_ROOT}"
base_command
CMD+=(
  --eval --checkpoint_path "${CHECKPOINT}"
  --log_dir "${VERIFY_ROOT}"
  --eval_results_json_path "${RESULT_JSON}"
  --use_structured_slots --use_sacr
  --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality
  --rapf_quality_weight "${QUALITY_WEIGHT}"
  --rapf_struct_residual_clip 0.25
  --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --eval_use_fused_scores
  --verbose_diagnostics
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

"${PYTHON}" - "${RESULT_JSON}" "${CHECKPOINT}" "${RECEIPT}" \
  "${QUALITY_WEIGHT}" "${GRID_SELECTION}" <<'PY'
import hashlib
import json
import os
import sys

result_path, checkpoint, receipt, weight, selection_path = sys.argv[1:]
with open(result_path) as f:
    metrics = json.load(f)
acc025 = float(metrics['last__bbs_acc0.25_top1'])
acc050 = float(metrics['last__bbs_acc0.50_top1'])
sha = hashlib.sha256()
with open(checkpoint, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
payload = {
    'checkpoint': os.path.realpath(checkpoint),
    'checkpoint_sha256': sha.hexdigest(),
    'eval_results_json': result_path,
    'quality_grid_selection': selection_path,
    'rapf_quality_weight': float(weight),
    'overall_acc0.25': acc025,
    'overall_acc0.50': acc050,
    'required_acc0.25_strictly_above': 0.5391,
    'required_acc0.50_strictly_above': 0.4241,
    'pass_acc0.25': acc025 > 0.5391,
    'pass_acc0.50': acc050 > 0.4241,
    'independent_full_reload': True,
}
payload['goal_achieved'] = payload['pass_acc0.25'] and payload['pass_acc0.50']
tmp = receipt + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
