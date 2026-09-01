#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_monotonic_main_ablations_20260825"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
source "${OLD_PACKAGE}/common.sh"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_acc50_target_20260827"
cd "${REPO_ROOT}"

ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
OUT_ROOT="${ROOT}/stage7_detector_topk_compact_dump"
DUMP_PATH="${OUT_ROOT}/detector_topk_compact.pt"
RESULT_JSON="${OUT_ROOT}/eval_results.json"
CHECKPOINT="${REPO_ROOT}/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156/ckpt_best_primary.pth"
EVALUATOR="${REPO_ROOT}/src/grounding_evaluator.py"
PATCHED_EVALUATOR="${PACKAGE_ROOT}/grounding_evaluator_compact.py"
BACKUP="${EVALUATOR}.pre_stage7_compact_20260827"
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
PATCHED_SHA="deef4ca154362fe554ef79c8e6e73b7bbe7e7cae5a76231dc618ca80bfd2ee99"

[ -f "${CHECKPOINT}" ]
[ -f "${PATCHED_EVALUATOR}" ]
[ ! -e "${OUT_ROOT}" ] || {
  echo "ERROR: refusing to overwrite ${OUT_ROOT}" >&2
  exit 186
}
[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]
[ "$(sha256sum "${PATCHED_EVALUATOR}" | awk '{print $1}')" = "${PATCHED_SHA}" ]
assert_gpu_idle

restore_evaluator() {
  if [ -f "${BACKUP}" ]; then
    cp -p "${BACKUP}" "${EVALUATOR}"
    [ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]
  fi
}
trap restore_evaluator EXIT INT TERM
cp -p "${EVALUATOR}" "${BACKUP}"
cp "${PATCHED_EVALUATOR}" "${EVALUATOR}"
[ "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = "${PATCHED_SHA}" ]
mkdir -p "${OUT_ROOT}"

base_command
CMD+=(
  --eval --checkpoint_path "${CHECKPOINT}"
  --log_dir "${OUT_ROOT}"
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
  --eval_target_cid_source text
  --verbose_diagnostics
)
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP_PATH}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

restore_evaluator
trap - EXIT INT TERM
[ -s "${DUMP_PATH}" ]
"${PYTHON}" - "${DUMP_PATH}" <<'PY'
import torch, sys
payload = torch.load(sys.argv[1], map_location='cpu')
rows = payload.get('rows', [])
if len(rows) != 9508:
    raise SystemExit('expected 9508 compact rows, got {}'.format(len(rows)))
print('compact_rows={}'.format(len(rows)))
PY
printf '%s stage7_detector_compact_dump_complete\n' "$(date -Is)" > "${PACKAGE_ROOT}/stage7_watch_status.txt"
