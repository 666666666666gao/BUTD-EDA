#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

CKPT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
BASELINE_DUMP="${ROOT}/stage149b_stage135c_e12_calibrated_dump/stage135c_e12_calibrated_geometry.pt"
OUT="${ROOT}/stage151b_stage150_e13_full_query_oracle_dump"
DUMP="${OUT}/stage150_e13_full_query_oracle_geometry.pt"
FIXBREAK="${OUT}/stage151b_stage150_fix_break_report.json"
TIER_REPORT="${OUT}/stage151b_full_query_oracle_report.json"
STATUS="${P}/stage151b_full_query_oracle_status.txt"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_stage151b_full_query_oracle.py"
COMPARE="${P}/compare_stage149b_calibrated_fix_break.py"
ANALYZE="${P}/analyze_stage151b_full_query_oracle.py"
STATE="${P}/state/stage151b_stage150_full_query_oracle_20260831"

EXPECTED_CKPT_SHA="8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
EXPECTED_BASELINE_DUMP_SHA="3de197aa7f15ec6e795389eb9a464213bb21d30daedf22e61cdae17fb68ed5ad"
EXPECTED_MAIN_SHA="c5fcfd1f716cc87e2de3c39f3d26578566f48237a5f19996d559d1a939e67c47"
EXPECTED_LOSS_SHA="622de46f497883532e74f34e43f788482c55a170c614623e9b49f6edfe0df603"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="f5f35cbc051ad41b24c907e899c027683c96bb210a46aa1917f8651fb8dfc2ee"
EXPECTED_COMPARE_SHA="68a543113ea25f1a1973135f53ff7d8af86dad4755dc4ce09228ec3aa1481c74"
EXPECTED_ANALYZE_SHA="da627aff87836d83e71a5e40d2193415a4dcf129a2baeb110706169279dcfdd0"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

fail_status() {
  local rc=$?
  printf 'stage151b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha "${BASELINE_DUMP}" "${EXPECTED_BASELINE_DUMP_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha src/joint_det_dataset.py "${EXPECTED_DATASET_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${COMPARE}" "${EXPECTED_COMPARE_SHA}"
check_sha "${ANALYZE}" "${EXPECTED_ANALYZE_SHA}"
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
  --qahnl_tiered_temperature 1.0 --qahnl_tier2_relation_weight 3.0
  --qahnl_loss_weight 1.0 --quality_loss_weight 0.0
  --use_detector_policy_adapter
  --detector_policy_adapter_hidden_dim 64
  --detector_policy_adapter_k 5
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

assert_gpu_idle
assert_storage
mkdir -p "${OUT}" "${STATE}"
cp -p "${LIVE}" "${STATE}/grounding_evaluator.pre_stage151b.py"
chmod 0444 "${STATE}/grounding_evaluator.pre_stage151b.py"
restore() {
  install -m 0644 "${STATE}/grounding_evaluator.pre_stage151b.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=151b_stage150_full_query_oracle\n'
  printf 'started_at=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "${CKPT}" "${EXPECTED_CKPT_SHA}"
  printf 'baseline_dump=%s\n' "${BASELINE_DUMP}"
  printf 'analyzer_sha256=%s\n' "${EXPECTED_ANALYZE_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage151b_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage151b_stage150_full_query_oracle_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}/eval_results.json" <<'PY'
import json, sys, torch
dump_path, metrics_path = sys.argv[1:]
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
assert official == [5234, 3979], official
assert hits == official, (hits, official)
print('STAGE151B_STAGE150_FULL_QUERY_ORACLE_PARITY_PASS', hits)
PY

"${PYTHON}" "${COMPARE}" "${BASELINE_DUMP}" "${DUMP}" "${FIXBREAK}" \
  > "${P}/stage151b_stage150_fix_break_compare.log"
"${PYTHON}" "${ANALYZE}" "${BASELINE_DUMP}" "${DUMP}" "${TIER_REPORT}" \
  > "${P}/stage151b_full_query_oracle_analysis.log"

"${PYTHON}" - "${FIXBREAK}" "${TIER_REPORT}" <<'PY'
import json, sys
fixbreak = json.load(open(sys.argv[1], encoding='utf-8'))
tier = json.load(open(sys.argv[2], encoding='utf-8'))
assert fixbreak['thresholds']['0.25']['baseline_hit'] == 5215
assert fixbreak['thresholds']['0.25']['candidate_hit'] == 5234
assert fixbreak['thresholds']['0.5']['baseline_hit'] == 3956
assert fixbreak['thresholds']['0.5']['candidate_hit'] == 3979
assert tier['hits']['candidate_025'] == 5234
assert tier['hits']['candidate_050'] == 3979
assert tier['oracle_scope'] == 'all_predicted_queries'
assert tier['hits']['candidate_oracle_025'] >= tier['hits']['candidate_025']
assert tier['hits']['candidate_oracle_050'] >= tier['hits']['candidate_050']
print('STAGE151B_FULL_QUERY_REPORT_PARITY_PASS')
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
sha256sum "${DUMP}" "${FIXBREAK}" "${TIER_REPORT}" >> "${OUT}/launch_manifest.txt"
chmod 0444 "${OUT}/launch_manifest.txt" "${FIXBREAK}" "${TIER_REPORT}"
trap - ERR
printf 'stage151b_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
