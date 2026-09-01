#!/usr/bin/env bash
set -euo pipefail

TRAIN_RUN="${1:?usage: verify_stage1_reload.sh TRAIN_RUN_DIR}"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
source "${OLD_PACKAGE}/common.sh"
cd "${REPO_ROOT}"

CHECKPOINT="${TRAIN_RUN}/ckpt_best_primary.pth"
VERIFY_ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage1_reload_verify"
RESULT_JSON="${VERIFY_ROOT}/eval_results.json"
RECEIPT="${VERIFY_ROOT}/goal_receipt.json"
ORIGINAL_EVALUATOR_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"

[ -f "${CHECKPOINT}" ] || {
  echo "ERROR: stage1 checkpoint missing: ${CHECKPOINT}" >&2
  exit 100
}
[ ! -e "${VERIFY_ROOT}" ] || {
  echo "ERROR: refusing to overwrite verification root ${VERIFY_ROOT}" >&2
  exit 101
}
current_sha="$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')"
[ "${current_sha}" = "${ORIGINAL_EVALUATOR_SHA}" ] || {
  echo "ERROR: evaluator is not at frozen original SHA: ${current_sha}" >&2
  exit 102
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
  --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25
  --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --eval_use_fused_scores
  --verbose_diagnostics
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

"${PYTHON}" - "${RESULT_JSON}" "${CHECKPOINT}" "${RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys

result_path, checkpoint_path, receipt_path = sys.argv[1:]
with open(result_path) as f:
    metrics = json.load(f)
overall_025 = float(metrics['last__bbs_acc0.25_top1'])
overall_050 = float(metrics['last__bbs_acc0.50_top1'])
sha = hashlib.sha256()
with open(checkpoint_path, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
payload = {
    'checkpoint': checkpoint_path,
    'checkpoint_sha256': sha.hexdigest(),
    'eval_results_json': result_path,
    'overall_acc0.25': overall_025,
    'overall_acc0.50': overall_050,
    'required_acc0.25_strictly_above': 0.5391,
    'required_acc0.50_strictly_above': 0.4241,
    'pass_acc0.25': overall_025 > 0.5391,
    'pass_acc0.50': overall_050 > 0.4241,
}
payload['goal_achieved'] = payload['pass_acc0.25'] and payload['pass_acc0.50']
os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
with open(receipt_path, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
