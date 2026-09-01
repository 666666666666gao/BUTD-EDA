#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?usage: run_stage4_quality_grid.sh CHECKPOINT}"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${OLD_PACKAGE}/common.sh"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
cd "${REPO_ROOT}"

ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
OUT_ROOT="${ROOT}/stage4_quality_grid"
RESULT_JSON="${OUT_ROOT}/calibration_eval_results.json"
CALIBRATION_JSON="${OUT_ROOT}/calibration_selection.json"
VERIFY_ROOT="${ROOT}/stage4_reload_verify"
RECEIPT="${VERIFY_ROOT}/goal_receipt.json"
EVALUATOR="${REPO_ROOT}/src/grounding_evaluator.py"
BACKUP="${EVALUATOR}.pre_stage4_qualitygrid_20260827"
PATCH_FILE="${PACKAGE_ROOT}/grounding_evaluator_qualitygrid.patch"
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
PATCHED_SHA="68fce9647e532a4e80412f70ae70555dbb295428e68e04c4e596202a41ac9122"

[ -f "${CHECKPOINT}" ] || {
  echo "ERROR: Stage-4 checkpoint missing: ${CHECKPOINT}" >&2
  exit 174
}
[ ! -e "${OUT_ROOT}" ] || {
  echo "ERROR: refusing to overwrite Stage-4 calibration root ${OUT_ROOT}" >&2
  exit 175
}
[ ! -e "${VERIFY_ROOT}" ] || {
  echo "ERROR: refusing to overwrite Stage-4 verification root ${VERIFY_ROOT}" >&2
  exit 176
}

restore_evaluator() {
  if [ -f "${BACKUP}" ]; then
    cp -p "${BACKUP}" "${EVALUATOR}"
    [ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]
  fi
}
trap restore_evaluator EXIT INT TERM

[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]
cp -p "${EVALUATOR}" "${BACKUP}"
[ "$(sha256sum "${BACKUP}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]
assert_gpu_idle
mkdir -p "${OUT_ROOT}"
patch --batch --forward -p0 -i "${PATCH_FILE}"
[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${PATCHED_SHA}" ]

base_command
CMD+=(
  --eval --checkpoint_path "${CHECKPOINT}"
  --log_dir "${OUT_ROOT}/calibration_eval"
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
  --eval_report_diagnostic_scores
  --verbose_diagnostics
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

restore_evaluator
trap - EXIT INT TERM

"${PYTHON}" - "${RESULT_JSON}" "${CALIBRATION_JSON}" <<'PY'
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
    rows.append({
        'source': source,
        'rapf_quality_weight': weight,
        'overall_acc0.25': acc025,
        'overall_acc0.50': acc050,
        'pass_acc0.25': acc025 > 0.5391,
        'pass_acc0.50': acc050 > 0.4241,
        'joint_margin': min(acc025 - 0.5391, acc050 - 0.4241),
    })
for row in rows:
    row['goal_achieved'] = row['pass_acc0.25'] and row['pass_acc0.50']
feasible = [row for row in rows if row['goal_achieved']]
if feasible:
    selected = max(
        feasible,
        key=lambda row: (
            row['overall_acc0.25'], row['overall_acc0.50'],
            -abs(row['rapf_quality_weight'] - 0.25),
        ),
    )
    policy = 'strict_dual_threshold_then_maximize_acc0.25'
else:
    selected = max(
        rows,
        key=lambda row: (
            row['joint_margin'], row['overall_acc0.25'],
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

SELECTED_FEASIBLE="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["selected"]["goal_achieved"])).lower())' "${CALIBRATION_JSON}")"
SELECTED_WEIGHT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["rapf_quality_weight"])' "${CALIBRATION_JSON}")"

if [ "${SELECTED_FEASIBLE}" = "true" ]; then
  "${PACKAGE_ROOT}/verify_stage4_reload.sh" \
    "${CHECKPOINT}" "${SELECTED_WEIGHT}" "${CALIBRATION_JSON}"
else
  mkdir -p "${VERIFY_ROOT}"
  "${PYTHON}" - "${CHECKPOINT}" "${CALIBRATION_JSON}" "${RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys

checkpoint, calibration_path, receipt_path = sys.argv[1:]
with open(calibration_path) as f:
    selected = json.load(f)['selected']
sha = hashlib.sha256()
with open(checkpoint, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
payload = {
    'checkpoint': os.path.realpath(checkpoint),
    'checkpoint_sha256': sha.hexdigest(),
    'calibration_json': calibration_path,
    'rapf_quality_weight': selected['rapf_quality_weight'],
    'overall_acc0.25': selected['overall_acc0.25'],
    'overall_acc0.50': selected['overall_acc0.50'],
    'pass_acc0.25': selected['pass_acc0.25'],
    'pass_acc0.50': selected['pass_acc0.50'],
    'goal_achieved': False,
    'independent_full_reload': False,
    'reason': 'Stage-4 quality-only training produced no feasible grid weight',
}
tmp = receipt_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
fi
