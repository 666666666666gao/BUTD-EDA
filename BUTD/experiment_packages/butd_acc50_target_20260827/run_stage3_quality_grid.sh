#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${OLD_PACKAGE}/common.sh"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
cd "${REPO_ROOT}"

ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STAGE1_RECEIPT="${ROOT}/stage1_reload_verify/goal_receipt.json"
STAGE2_RECEIPT="${ROOT}/stage2_reload_verify/goal_receipt.json"
SOURCE_SELECTION_JSON="${ROOT}/stage3_source_selection.json"
OUT_ROOT="${ROOT}/stage3_rapf_quality_grid"
RESULT_JSON="${OUT_ROOT}/calibration_eval_results.json"
CALIBRATION_JSON="${OUT_ROOT}/calibration_selection.json"
STAGE3_VERIFY_ROOT="${ROOT}/stage3_reload_verify"
STAGE3_RECEIPT="${STAGE3_VERIFY_ROOT}/goal_receipt.json"
EVALUATOR="${REPO_ROOT}/src/grounding_evaluator.py"
BACKUP="${EVALUATOR}.pre_acc50_qualitygrid_20260827"
PATCH_FILE="${PACKAGE_ROOT}/grounding_evaluator_qualitygrid.patch"
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
PATCHED_SHA="68fce9647e532a4e80412f70ae70555dbb295428e68e04c4e596202a41ac9122"

[ -s "${STAGE1_RECEIPT}" ] || {
  echo "ERROR: Stage-1 receipt missing: ${STAGE1_RECEIPT}" >&2
  exit 149
}
[ -s "${STAGE2_RECEIPT}" ] || {
  echo "ERROR: Stage-2 receipt missing: ${STAGE2_RECEIPT}" >&2
  exit 150
}
SOURCE_CHECKPOINT="$("${PYTHON}" - \
  "${STAGE1_RECEIPT}" "${STAGE2_RECEIPT}" "${SOURCE_SELECTION_JSON}" <<'PY'
import json
import os
import sys

stage1_path, stage2_path, output_path = sys.argv[1:]
candidates = []
for stage, receipt_path in (("stage1", stage1_path), ("stage2", stage2_path)):
    with open(receipt_path) as f:
        receipt = json.load(f)
    acc025 = float(receipt["overall_acc0.25"])
    acc050 = float(receipt["overall_acc0.50"])
    checkpoint = os.path.realpath(receipt["checkpoint"])
    candidates.append({
        "stage": stage,
        "receipt": receipt_path,
        "checkpoint": checkpoint,
        "overall_acc0.25": acc025,
        "overall_acc0.50": acc050,
        "pass_acc0.25": acc025 > 0.5391,
        "pass_acc0.50": acc050 > 0.4241,
        "joint_margin": min(acc025 - 0.5391, acc050 - 0.4241),
    })
selected = max(
    candidates,
    key=lambda row: (
        row["pass_acc0.25"] and row["pass_acc0.50"],
        row["joint_margin"],
        row["overall_acc0.25"],
        row["overall_acc0.50"],
    ),
)
payload = {
    "selection_policy": "dual_threshold_feasibility_then_joint_margin",
    "candidates": candidates,
    "selected": selected,
}
tmp = output_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, output_path)
print(selected["checkpoint"])
PY
)"
[ -f "${SOURCE_CHECKPOINT}" ] || {
  echo "ERROR: selected Stage-3 source checkpoint missing: ${SOURCE_CHECKPOINT}" >&2
  exit 151
}
[ ! -e "${OUT_ROOT}" ] || {
  echo "ERROR: refusing to overwrite Stage-3 root ${OUT_ROOT}" >&2
  exit 152
}
[ ! -e "${STAGE3_VERIFY_ROOT}" ] || {
  echo "ERROR: refusing to overwrite Stage-3 verify root ${STAGE3_VERIFY_ROOT}" >&2
  exit 153
}

restore_evaluator() {
  if [ -f "${BACKUP}" ]; then
    cp -p "${BACKUP}" "${EVALUATOR}"
    local restored_sha
    restored_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
    if [ "${restored_sha}" != "${ORIGINAL_SHA}" ]; then
      echo "ERROR: evaluator restore SHA mismatch: ${restored_sha}" >&2
      return 154
    fi
  fi
}
trap restore_evaluator EXIT INT TERM

current_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
backup_sha="$(sha256sum "${BACKUP}" | awk '{print $1}')"
[ "${current_sha}" = "${ORIGINAL_SHA}" ] || {
  echo "ERROR: evaluator is not original before Stage 3: ${current_sha}" >&2
  exit 155
}
[ "${backup_sha}" = "${ORIGINAL_SHA}" ] || {
  echo "ERROR: Stage-3 evaluator backup SHA mismatch: ${backup_sha}" >&2
  exit 156
}

assert_gpu_idle
mkdir -p "${OUT_ROOT}"
patch --batch --forward -p0 -i "${PATCH_FILE}"
current_sha="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
[ "${current_sha}" = "${PATCHED_SHA}" ] || {
  echo "ERROR: quality-grid evaluator SHA mismatch: ${current_sha}" >&2
  exit 157
}

base_command
CMD+=(
  --eval --checkpoint_path "${SOURCE_CHECKPOINT}"
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
[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]

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
    key025 = 'diag_{}@0.25'.format(source)
    key050 = 'diag_{}@0.50'.format(source)
    if key025 not in metrics or key050 not in metrics:
        raise SystemExit('missing quality-grid metrics for ' + source)
    acc025 = float(metrics[key025])
    acc050 = float(metrics[key050])
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

SELECTED_FEASIBLE="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["selected"]["goal_achieved"])).lower())' "${CALIBRATION_JSON}")"
SELECTED_WEIGHT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["rapf_quality_weight"])' "${CALIBRATION_JSON}")"

if [ "${SELECTED_FEASIBLE}" = "true" ]; then
  "${PACKAGE_ROOT}/verify_stage3_reload.sh" \
    "${SOURCE_CHECKPOINT}" "${SELECTED_WEIGHT}" "${CALIBRATION_JSON}"
else
  mkdir -p "${STAGE3_VERIFY_ROOT}"
  "${PYTHON}" - "${SOURCE_CHECKPOINT}" "${CALIBRATION_JSON}" \
    "${STAGE3_RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys

checkpoint, calibration_path, receipt_path = sys.argv[1:]
with open(calibration_path) as f:
    calibration = json.load(f)
selected = calibration['selected']
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
    'reason': 'no pre-registered RAPF quality weight met both strict thresholds',
}
tmp = receipt_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
fi
