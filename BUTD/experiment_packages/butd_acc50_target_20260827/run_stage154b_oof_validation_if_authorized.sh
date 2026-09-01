#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

STAGE154A_STATUS="${P}/stage154a_dependency_status.txt"
STATUS="${P}/stage154b_dependency_status.txt"
POLICY_LOCK="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"
RAW_VAL="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
STAGE153C_OUT="${ROOT}/stage153c_stage150_e13_val_source_dump"
STAGE153C_SOURCE="${STAGE153C_OUT}/stage150_e13_val_source_features.pt"
STAGE153C_COMPACT="${STAGE153C_OUT}/stage150_e13_val_adapter_features.pt"
FALLBACK_OUT="${ROOT}/stage154b_stage150_e13_val_source_dump"
FALLBACK_SOURCE="${FALLBACK_OUT}/stage150_e13_val_source_features.pt"
FALLBACK_COMPACT="${FALLBACK_OUT}/stage150_e13_val_adapter_features.pt"
RESULT="${ROOT}/stage154b_oof_selector_validation_result.json"
CKPT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_stage149b_calibrated_dump.py"
PRE_EVAL=""

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

restore_evaluator() {
  if test -n "${PRE_EVAL}" && test -s "${PRE_EVAL}"; then
    install -m 0644 "${PRE_EVAL}" "${LIVE}"
    PRE_EVAL=""
  fi
}

fail_status() {
  local rc=$?
  set +e
  restore_evaluator
  printf 'stage154b_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR INT TERM

printf 'stage154b_waiting_for_stage154a %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage154a_(complete|skipped|failed)' \
  "${STAGE154A_STATUS}" 2>/dev/null; do
  sleep 60
done

if grep -q '^stage154a_skipped_stage153_goal_met ' "${STAGE154A_STATUS}"; then
  printf 'stage154b_skipped_stage153_goal_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi
if grep -q '^stage154a_failed ' "${STAGE154A_STATUS}"; then
  printf 'stage154b_not_authorized_stage154a_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi

test -s "${POLICY_LOCK}"
AUTHORIZED=$("${PYTHON}" - "${POLICY_LOCK}" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding='utf-8'))
assert lock['stage'] == '154_train_only_scene_oof_stage142_stage150_source_selector'
assert lock['validation_labels_used_for_selection'] is False
print('1' if lock['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage154b_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi

test -s "${RAW_VAL}"
test ! -e "${RESULT}"
if grep -q '^stage153c_complete_goal_not_met ' "${P}/stage153c_dependency_status.txt" \
  && test -s "${STAGE153C_SOURCE}" && test -s "${STAGE153C_COMPACT}"; then
  SOURCE_DUMP="${STAGE153C_SOURCE}"
  COMPACT_DUMP="${STAGE153C_COMPACT}"
  METRICS_PATH="${STAGE153C_OUT}/eval_results.json"
  printf 'stage154b_reusing_stage153c_fixed_features %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
else
  check_sha "${CKPT}" "8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
  check_sha main_utils.py "960bb83e7fe9524d81bf48e0e23975558380371e2e73bb0af6eaeb5e925a79f2"
  check_sha models/losses.py "86a076c2d8f265b9ddcc809a9c55f4548e83d5ba50c827291e5aaf1b4d325d62"
  check_sha src/joint_det_dataset.py "5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
  check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
  check_sha "${PATCHED}" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
  test ! -e "${FALLBACK_OUT}"
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
    --eval_dump_source_choice_features_path "${FALLBACK_SOURCE}"
    --eval_dump_source_choice_topk 1 --verbose_diagnostics
  )
  base_command
  EVAL_CMD=("${CMD[@]}" --eval --checkpoint_path "${CKPT}"
    --log_dir "${FALLBACK_OUT}"
    --eval_results_json_path "${FALLBACK_OUT}/eval_results.json"
    "${MODEL_ARGS[@]}")
  assert_gpu_idle
  assert_storage
  mkdir -p "${FALLBACK_OUT}"
  PRE_EVAL="${FALLBACK_OUT}/grounding_evaluator.pre_stage154b.py"
  cp -p "${LIVE}" "${PRE_EVAL}"
  install -m 0644 "${PATCHED}" "${LIVE}"
  check_sha "${LIVE}" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
  printf 'stage154b_validation_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
  NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${FALLBACK_COMPACT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
    2>&1 | tee "${P}/stage154b_validation_dump.log"
  restore_evaluator
  check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
  SOURCE_DUMP="${FALLBACK_SOURCE}"
  COMPACT_DUMP="${FALLBACK_COMPACT}"
  METRICS_PATH="${FALLBACK_OUT}/eval_results.json"
fi

"${PYTHON}" - "${SOURCE_DUMP}" "${COMPACT_DUMP}" "${METRICS_PATH}" <<'PY'
import json, os, sys, torch, numpy as np
source_path, compact_path, metrics_path = sys.argv[1:]
manifest = torch.load(source_path, map_location='cpu')
assert manifest['format'] == 'source_choice_feature_dump_sharded_v1'
assert manifest['topk'] == 1 and manifest['row_count'] == 9508
base = os.path.dirname(source_path)
assert sum(len(torch.load(os.path.join(base, relative), map_location='cpu')['rows'])
           for relative in manifest['shards']) == 9508
rows = torch.load(compact_path, map_location='cpu')['rows']
assert len(rows) == 9508
hits = [0, 0]
for row in rows:
    selected = int(np.argmax(np.asarray(row['adapter_score_at_candidate'])))
    iou = float(row['adapter_iou_at_candidate'][selected])
    hits[0] += int(iou > 0.25)
    hits[1] += int(iou > 0.50)
assert hits == [5234, 3979], hits
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [round(metrics['last__bbs_acc0.25_top1'] * 9508),
            round(metrics['last__bbs_acc0.50_top1'] * 9508)]
assert official == hits, (official, hits)
print('STAGE154B_PRIMARY_FEATURE_PARITY_PASS', hits)
PY

printf 'stage154b_locked_oof_selector_evaluating %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage154_oof_source_selector.py" evaluate \
  "${RAW_VAL}" \
  "${SOURCE_DUMP}" \
  "${COMPACT_DUMP}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${POLICY_LOCK}" \
  "${RESULT}" \
  2>&1 | tee "${P}/stage154b_locked_oof_selector.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding='utf-8'))
print('goal_met' if result['strict_goal_met_offline'] else 'goal_not_met')
PY
)
restore_evaluator
trap - ERR INT TERM
printf 'stage154b_complete_%s %s\n' "${MET}" "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RESULT}"
