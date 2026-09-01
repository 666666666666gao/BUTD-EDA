#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
ACC50_PACKAGE="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${OLD_PACKAGE}/common.sh"
cd "${REPO_ROOT}"

OUT_ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827/text_policy_diagnostic"
RESULT_JSON="${OUT_ROOT}/eval_results.json"
GRID_SELECTION="${OUT_ROOT}/quality_grid_selection.json"
STAGE0_VERIFY_ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage0_reload_verify"
STAGE0_RECEIPT="${STAGE0_VERIFY_ROOT}/goal_receipt.json"
EVALUATOR="${REPO_ROOT}/src/grounding_evaluator.py"
BACKUP="${EVALUATOR}.pre_acc50_textdiag_20260827"
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
PATCHED_SHA="93337696e376762bad44788d8e027aa62fc1fae0c63c852bc85a48f6b8925945"

restore_evaluator() {
  if [ -f "${BACKUP}" ]; then
    cp -p "${BACKUP}" "${EVALUATOR}"
    local restored_sha
    restored_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
    if [ "${restored_sha}" != "${ORIGINAL_SHA}" ]; then
      echo "ERROR: evaluator restore SHA mismatch: ${restored_sha}" >&2
      return 70
    fi
  fi
}
trap restore_evaluator EXIT INT TERM

current_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
backup_sha="$(sha256sum "${BACKUP}" | awk '{print $1}')"
if [ "${backup_sha}" != "${ORIGINAL_SHA}" ]; then
  echo "ERROR: evaluator backup SHA mismatch: ${backup_sha}" >&2
  exit 72
fi
if [ "${current_sha}" = "${ORIGINAL_SHA}" ]; then
  patch --batch --forward -p0 \
    -i "${ACC50_PACKAGE}/grounding_evaluator_textdiag.patch"
  current_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
fi
if [ "${current_sha}" != "${PATCHED_SHA}" ]; then
  echo "ERROR: expected diagnostic evaluator SHA ${PATCHED_SHA}, got ${current_sha}" >&2
  exit 71
fi

CUDA_VISIBLE_DEVICES='' "${PYTHON}" \
  "${ACC50_PACKAGE}/test_stage0_receipt_protocol.py"

require_files
assert_gpu_idle
mkdir -p "${OUT_ROOT}"
base_command
CMD+=(
  --eval --checkpoint_path "${M3_CHECKPOINT}"
  --log_dir "${OUT_ROOT}"
  --eval_results_json_path "${RESULT_JSON}"
  --use_structured_slots --use_sacr
  --use_rapf --use_reliability_gate --use_quality_head
  --rapf_use_quality --use_qahnl --qahnl_score_source fused
  --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25
  --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5
  --rapf_generic_gate_cap 0.1
  --eval_primary_score_source detector_countboost
  --eval_target_cid_source text
  --eval_report_diagnostic_scores
  --verbose_diagnostics
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

restore_evaluator
trap - EXIT INT TERM
[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]

"${PYTHON}" - "${RESULT_JSON}" "${GRID_SELECTION}" <<'PY'
import json
import os
import sys

result_path, selection_path = sys.argv[1:]
weights = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00,
)
with open(result_path) as f:
    metrics = json.load(f)
rows = []
for weight in weights:
    source = 'rapf_qw_{:03d}'.format(int(round(weight * 100.0)))
    acc025 = float(metrics['diag_{}@0.25'.format(source)])
    acc050 = float(metrics['diag_{}@0.50'.format(source)])
    row = {
        'source': source,
        'rapf_quality_weight': weight,
        'overall_acc0.25': acc025,
        'overall_acc0.50': acc050,
        'pass_acc0.25': acc025 > 0.5391,
        'pass_acc0.50': acc050 > 0.4241,
        'joint_margin': min(acc025 - 0.5391, acc050 - 0.4241),
    }
    row['goal_achieved'] = row['pass_acc0.25'] and row['pass_acc0.50']
    rows.append(row)
feasible = [row for row in rows if row['goal_achieved']]
if feasible:
    selected = max(
        feasible,
        key=lambda row: (
            row['overall_acc0.25'],
            row['overall_acc0.50'],
            -abs(row['rapf_quality_weight'] - 0.25),
        ),
    )
    policy = 'strict_dual_threshold_then_maximize_acc0.25'
else:
    selected = max(
        rows,
        key=lambda row: (
            row['joint_margin'],
            row['overall_acc0.25'],
            row['overall_acc0.50'],
        ),
    )
    policy = 'closest_joint_margin_no_feasible_candidate'
payload = {
    'result_json': result_path,
    'selection_policy': policy,
    'selected': selected,
    'grid': rows,
}
tmp = selection_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, selection_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

SELECTED_FEASIBLE="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["selected"]["goal_achieved"])).lower())' "${GRID_SELECTION}")"
SELECTED_WEIGHT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["rapf_quality_weight"])' "${GRID_SELECTION}")"
if [ "${SELECTED_FEASIBLE}" = "true" ]; then
  "${ACC50_PACKAGE}/verify_stage0_reload.sh" \
    "${M3_CHECKPOINT}" "${SELECTED_WEIGHT}" "${GRID_SELECTION}"
else
  mkdir -p "${STAGE0_VERIFY_ROOT}"
  "${PYTHON}" - "${M3_CHECKPOINT}" "${GRID_SELECTION}" \
    "${STAGE0_RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys

checkpoint, selection_path, receipt_path = sys.argv[1:]
with open(selection_path) as f:
    selected = json.load(f)['selected']
sha = hashlib.sha256()
with open(checkpoint, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
payload = {
    'checkpoint': os.path.realpath(checkpoint),
    'checkpoint_sha256': sha.hexdigest(),
    'quality_grid_selection': selection_path,
    'rapf_quality_weight': selected['rapf_quality_weight'],
    'overall_acc0.25': selected['overall_acc0.25'],
    'overall_acc0.50': selected['overall_acc0.50'],
    'pass_acc0.25': selected['pass_acc0.25'],
    'pass_acc0.50': selected['pass_acc0.50'],
    'goal_achieved': False,
    'independent_full_reload': False,
    'reason': 'accepted Full checkpoint has no feasible preregistered quality weight',
}
tmp = receipt_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
fi

RUN_DIR="$(latest_run_dir "${OUT_ROOT}")"
EVAL_LOG="${RUN_DIR}/eval.log"
if [ ! -f "${EVAL_LOG}" ]; then
  EVAL_LOG="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'eval*.log' | sort | tail -n 1)"
fi
{
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint=${M3_CHECKPOINT}"
  echo "target_cid_source=text"
  cat "${GRID_SELECTION}"
  grep -E 'last__(bbs|diag_detector_).*acc0\\.(25|50)_top1:' "${EVAL_LOG}" || true
} > "${OUT_ROOT}/summary.txt"
cat "${OUT_ROOT}/summary.txt"
