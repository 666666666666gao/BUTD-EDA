#!/usr/bin/env bash
set -euo pipefail

K="${1:?usage: verify_stage6_detector_topk_reload.sh K}"
case "${K}" in
  2|3|4|5) ;;
  *) echo "ERROR: K must be one of 2,3,4,5" >&2; exit 183 ;;
esac

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
source "${OLD_PACKAGE}/common.sh"
cd "${REPO_ROOT}"

ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
CHECKPOINT="${REPO_ROOT}/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156/ckpt_best_primary.pth"
CANDIDATE="${ROOT}/stage6_detector_quality_top${K}_candidate/candidate_receipt.json"
VERIFY_ROOT="${ROOT}/stage6_detector_quality_top${K}_reload_verify"
RESULT_JSON="${VERIFY_ROOT}/eval_results.json"
RECEIPT="${VERIFY_ROOT}/goal_receipt.json"
SOURCE="detector_quality_top${K}_target_rerank"
EVALUATOR_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"

[ -f "${CHECKPOINT}" ]
[ -f "${CANDIDATE}" ]
"${PYTHON}" - "${CANDIDATE}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not bool(r.get('goal_achieved')):
    raise SystemExit('candidate did not pass both strict thresholds')
PY
[ "$(sha256sum src/grounding_evaluator.py | awk '{print $1}')" = "${EVALUATOR_SHA}" ]
[ ! -e "${VERIFY_ROOT}" ] || {
  echo "ERROR: refusing to overwrite ${VERIFY_ROOT}" >&2
  exit 184
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
  --eval_primary_score_source "${SOURCE}"
  --eval_target_cid_source text
  --verbose_diagnostics
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

"${PYTHON}" - "${RESULT_JSON}" "${CHECKPOINT}" "${RECEIPT}" "${SOURCE}" "${CANDIDATE}" <<'PY'
import hashlib
import json
import os
import sys

result_path, checkpoint_path, receipt_path, source, candidate_path = sys.argv[1:]
metrics = json.load(open(result_path))
acc025 = float(metrics['last__bbs_acc0.25_top1'])
acc050 = float(metrics['last__bbs_acc0.50_top1'])
sha = hashlib.sha256()
with open(checkpoint_path, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
payload = {
    'checkpoint': os.path.realpath(checkpoint_path),
    'checkpoint_sha256': sha.hexdigest(),
    'eval_results_json': os.path.realpath(result_path),
    'candidate_receipt': os.path.realpath(candidate_path),
    'eval_primary_score_source': source,
    'eval_target_cid_source': 'text',
    'overall_acc0.25': acc025,
    'overall_acc0.50': acc050,
    'required_acc0.25_strictly_above': 0.5391,
    'required_acc0.50_strictly_above': 0.4241,
    'pass_acc0.25': acc025 > 0.5391,
    'pass_acc0.50': acc050 > 0.4241,
    'independent_full_reload': True,
    'protocol_note': 'deployable detected boxes/classes plus utterance-derived text target CID; no GT rerank inputs',
}
payload['goal_achieved'] = payload['pass_acc0.25'] and payload['pass_acc0.50']
tmp = receipt_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
