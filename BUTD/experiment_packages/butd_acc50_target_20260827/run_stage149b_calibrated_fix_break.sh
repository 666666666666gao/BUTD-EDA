#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

BASE_CKPT="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
CAND_CKPT="${ROOT}/stage148b_tiered_qahnl_adapter_trainonly/scanrefer_spacy/1788136855/ckpt_best_primary.pth"
BASE_OUT="${ROOT}/stage149b_stage135c_e12_calibrated_dump"
CAND_OUT="${ROOT}/stage149b_stage148_e15_calibrated_dump"
BASE_DUMP="${BASE_OUT}/stage135c_e12_calibrated_geometry.pt"
CAND_DUMP="${CAND_OUT}/stage148_e15_calibrated_geometry.pt"
REPORT="${CAND_OUT}/stage149b_calibrated_fix_break_report.json"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_stage149b_calibrated_dump.py"
COMPARE="${P}/compare_stage149b_calibrated_fix_break.py"
STATE="${P}/state/stage149b_calibrated_fix_break_20260831"
STATUS="${P}/stage149b_calibrated_fix_break_status.txt"

EXPECTED_BASE_CKPT_SHA="a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
EXPECTED_CAND_CKPT_SHA="1b5a881016797f9eacda64e2010746e4dcf4b77db19eb2d5924a5970c4698bff"
EXPECTED_MAIN_SHA="665aed267508aa4a77f8ad014071d3a912642e8631a691fa9320c3f21aa14dd6"
EXPECTED_LOSS_SHA="a80d4a8934536f1b11f488aa7a38bce2ecf6bfd3ad9b732f81913e2ad78763b4"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
EXPECTED_COMPARE_SHA="68a543113ea25f1a1973135f53ff7d8af86dad4755dc4ce09228ec3aa1481c74"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

fail_status() {
  local rc=$?
  printf 'stage149b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

check_sha "${BASE_CKPT}" "${EXPECTED_BASE_CKPT_SHA}"
check_sha "${CAND_CKPT}" "${EXPECTED_CAND_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha src/joint_det_dataset.py "${EXPECTED_DATASET_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${COMPARE}" "${EXPECTED_COMPARE_SHA}"
test ! -e "${BASE_OUT}"
test ! -e "${CAND_OUT}"
test ! -e "${STATE}"

COMMON_MODEL_ARGS=(
  --disable_train_augmentation --disable_box_jitter
  --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
  --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.0
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
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

build_eval_command() {
  local checkpoint="$1" output="$2" mode="$3"
  base_command
  EVAL_CMD=("${CMD[@]}" --eval --checkpoint_path "${checkpoint}"
    --log_dir "${output}" --eval_results_json_path "${output}/eval_results.json"
    "${COMMON_MODEL_ARGS[@]}")
  if [[ "${mode}" = baseline ]]; then
    EVAL_CMD+=(--use_qahnl --qahnl_score_source fused --qahnl_loss_weight 0.0)
  else
    EVAL_CMD+=(
      --use_qahnl --qahnl_score_source adapter_hit50
      --qahnl_tiered_quality --qahnl_tier2_iou_thresh 0.50
      --qahnl_pos_iou_thresh 0.25 --qahnl_neg_iou_thresh 0.10
      --qahnl_tiered_margin21 0.20 --qahnl_tiered_margin10 0.10
      --qahnl_tiered_temperature 1.0 --qahnl_loss_weight 1.0
    )
  fi
}

assert_gpu_idle
assert_storage
mkdir -p "${BASE_OUT}" "${CAND_OUT}" "${STATE}"
cp -p "${LIVE}" "${STATE}/grounding_evaluator.pre_stage149b.py"
chmod 0444 "${STATE}/grounding_evaluator.pre_stage149b.py"
restore() {
  install -m 0644 "${STATE}/grounding_evaluator.pre_stage149b.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=149b_calibrated_fix_break\nstarted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'baseline_checkpoint=%s\nbaseline_checkpoint_sha256=%s\n' "${BASE_CKPT}" "${EXPECTED_BASE_CKPT_SHA}"
  printf 'candidate_checkpoint=%s\ncandidate_checkpoint_sha256=%s\n' "${CAND_CKPT}" "${EXPECTED_CAND_CKPT_SHA}"
  printf 'patched_evaluator_sha256=%s\ncomparator_sha256=%s\n' "${EXPECTED_PATCHED_SHA}" "${EXPECTED_COMPARE_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${CAND_OUT}/launch_manifest.txt"

printf 'stage149b_baseline_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
build_eval_command "${BASE_CKPT}" "${BASE_OUT}" baseline
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${BASE_DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage149b_stage135c_calibrated_dump.log"

printf 'stage149b_candidate_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
build_eval_command "${CAND_CKPT}" "${CAND_OUT}" candidate
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${CAND_DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage149b_stage148_calibrated_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${BASE_DUMP}" "${BASE_OUT}/eval_results.json" 5215 3956 baseline <<'PY'
import json, sys, torch
dump_path, metrics_path, expected25, expected50, label = sys.argv[1:]
expected25, expected50 = int(expected25), int(expected50)
rows = torch.load(dump_path, map_location='cpu')['rows']
assert len(rows) == 9508, len(rows)
hits = [0, 0]
for row in rows:
    queries = [int(value) for value in row['adapter_candidate_query']]
    selected = int(row['adapter_rescue_query'])
    iou = float(row['adapter_iou_at_candidate'][queries.index(selected)])
    hits[0] += iou > 0.25
    hits[1] += iou > 0.50
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [
    round(metrics['last__bbs_acc0.25_top1'] * 9508),
    round(metrics['last__bbs_acc0.50_top1'] * 9508),
]
assert official == [expected25, expected50], (label, official)
assert hits == official, (label, hits, official)
print('STAGE149B_CALIBRATED_PARITY_PASS', label, hits)
PY

"${PYTHON}" - "${CAND_DUMP}" "${CAND_OUT}/eval_results.json" 5220 3971 candidate <<'PY'
import json, sys, torch
dump_path, metrics_path, expected25, expected50, label = sys.argv[1:]
expected25, expected50 = int(expected25), int(expected50)
rows = torch.load(dump_path, map_location='cpu')['rows']
assert len(rows) == 9508, len(rows)
hits = [0, 0]
for row in rows:
    queries = [int(value) for value in row['adapter_candidate_query']]
    selected = int(row['adapter_rescue_query'])
    iou = float(row['adapter_iou_at_candidate'][queries.index(selected)])
    hits[0] += iou > 0.25
    hits[1] += iou > 0.50
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [
    round(metrics['last__bbs_acc0.25_top1'] * 9508),
    round(metrics['last__bbs_acc0.50_top1'] * 9508),
]
assert official == [expected25, expected50], (label, official)
assert hits == official, (label, hits, official)
print('STAGE149B_CALIBRATED_PARITY_PASS', label, hits)
PY

"${PYTHON}" "${COMPARE}" "${BASE_DUMP}" "${CAND_DUMP}" "${REPORT}" \
  2>&1 | tee "${P}/stage149b_calibrated_fix_break_compare.log"

"${PYTHON}" - "${REPORT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
assert report['thresholds']['0.25']['baseline_hit'] == 5215, report
assert report['thresholds']['0.25']['candidate_hit'] == 5220, report
assert report['thresholds']['0.5']['baseline_hit'] == 3956, report
assert report['thresholds']['0.5']['candidate_hit'] == 3971, report
print('STAGE149B_REPORT_FORMAL_PARITY_PASS')
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${CAND_OUT}/launch_manifest.txt"
sha256sum "${BASE_DUMP}" "${CAND_DUMP}" "${REPORT}" >> "${CAND_OUT}/launch_manifest.txt"
chmod 0444 "${CAND_OUT}/launch_manifest.txt" "${REPORT}"
trap - ERR
printf 'stage149b_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
