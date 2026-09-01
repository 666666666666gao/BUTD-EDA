#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

CKPT="${ROOT}/stage148b_tiered_qahnl_adapter_trainonly/scanrefer_spacy/1788136855/ckpt_best_primary.pth"
BASELINE_DUMP="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
OUT="${ROOT}/stage149a_stage148_e15_raw_val_dump"
DUMP="${OUT}/stage148_e15_raw_val_geometry.pt"
REPORT="${OUT}/stage149_fix_break_report.json"
STATUS="${P}/stage149a_fix_break_status.txt"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
COMPARE="${P}/compare_stage149_fix_break.py"
STATE="${P}/state/stage149a_fix_break_dump_20260831"

EXPECTED_CKPT_SHA="1b5a881016797f9eacda64e2010746e4dcf4b77db19eb2d5924a5970c4698bff"
EXPECTED_MAIN_SHA="665aed267508aa4a77f8ad014071d3a912642e8631a691fa9320c3f21aa14dd6"
EXPECTED_LOSS_SHA="a80d4a8934536f1b11f488aa7a38bce2ecf6bfd3ad9b732f81913e2ad78763b4"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"
EXPECTED_COMPARE_SHA="5b1212be2be186f05dd4006581a8f309f5c6cbb9b1ebb8e2ee532b96ae8491d3"

fail_status() {
  local rc=$?
  printf 'stage149a_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha src/joint_det_dataset.py "${EXPECTED_DATASET_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${COMPARE}" "${EXPECTED_COMPARE_SHA}"
test -s "${BASELINE_DUMP}"
test ! -e "${OUT}"
test ! -e "${STATE}"

MODEL_ARGS=(
 --disable_train_augmentation --disable_box_jitter
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
 --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.0
 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source adapter_hit50
 --qahnl_tiered_quality --qahnl_tier2_iou_thresh 0.50
 --qahnl_pos_iou_thresh 0.25 --qahnl_neg_iou_thresh 0.10
 --qahnl_tiered_margin21 0.20 --qahnl_tiered_margin10 0.10
 --qahnl_tiered_temperature 1.0 --qahnl_loss_weight 1.0
 --quality_loss_weight 0.0 --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0
 --detector_policy_adapter_loss_weight 0.0
 --detector_policy_geometry_loss_weight 0.0
 --detector_policy_rank2_rescue_loss_weight 0.0
 --detector_policy_alignment_rescue_loss_weight 0.0
 --detector_policy_tier_pair_loss_weight 0.0
 --detector_policy_adapter_margin 0.1
 --detector_policy_adapter_min_iou_gap 0.02
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics
)

base_command
EVAL_CMD=("${CMD[@]}" --eval --checkpoint_path "${CKPT}"
 --log_dir "${OUT}" --eval_results_json_path "${OUT}/eval_results.json"
 "${MODEL_ARGS[@]}")

if [[ "${DRY_RUN:-0}" = 1 ]]; then
  printf '%q ' "${EVAL_CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${OUT}" "${STATE}"
cp -p "${LIVE}" "${STATE}/grounding_evaluator.pre_stage149a.py"
chmod 0444 "${STATE}/grounding_evaluator.pre_stage149a.py"
restore() {
  install -m 0644 "${STATE}/grounding_evaluator.pre_stage149a.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=149a_stage148_e15_fix_break_dump\n'
  printf 'started_at=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "${CKPT}" "${EXPECTED_CKPT_SHA}"
  printf 'baseline_dump=%s\n' "${BASELINE_DUMP}"
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_MAIN_SHA}" "${EXPECTED_LOSS_SHA}"
  printf 'dump_evaluator_sha256=%s\ncomparator_sha256=%s\n' \
    "${EXPECTED_PATCHED_SHA}" "${EXPECTED_COMPARE_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage149a_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage149a_stage148_e15_raw_val_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}/eval_results.json" <<'PY'
import json,os,sys,torch
dump,metrics_path=sys.argv[1:]
rows=torch.load(dump,map_location='cpu')['rows']
assert len(rows)==9508,len(rows)
required=('adapter_candidate_query','adapter_rescue_query',
          'adapter_box_at_candidate','adapter_iou_at_candidate','gt_box')
for key in required: assert all(key in row for row in rows),key
metrics=json.load(open(metrics_path,encoding='utf-8'))
assert round(metrics['last__bbs_acc0.25_top1']*9508)==5220,metrics
assert round(metrics['last__bbs_acc0.50_top1']*9508)==3971,metrics
print('STAGE149_DUMP_FORMAL_METRIC_PARITY_PASS',len(rows),os.path.getsize(dump))
PY

"${PYTHON}" "${COMPARE}" "${BASELINE_DUMP}" "${DUMP}" "${REPORT}" \
  2>&1 | tee "${P}/stage149a_fix_break_compare.log"

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
sha256sum "${DUMP}" "${REPORT}" >> "${OUT}/launch_manifest.txt"
chmod 0444 "${OUT}/launch_manifest.txt" "${REPORT}"
trap - ERR
printf 'stage149a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
