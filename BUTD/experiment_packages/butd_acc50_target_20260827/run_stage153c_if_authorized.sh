#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

STAGE153B_STATUS="${P}/stage153b_dependency_status.txt"
POLICY_LOCK="${ROOT}/stage153b_train_only_rich_source_selector/locked_source_selector.json"
OUT="${ROOT}/stage153c_stage150_e13_val_source_dump"
SOURCE_DUMP="${OUT}/stage150_e13_val_source_features.pt"
COMPACT_DUMP="${OUT}/stage150_e13_val_adapter_features.pt"
RESULT="${OUT}/stage153c_locked_selector_validation_result.json"
STATUS="${P}/stage153c_dependency_status.txt"
CKPT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
STAGE142_VAL_DUMP="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_stage149b_calibrated_dump.py"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

fail_status() {
  local rc=$?
  printf 'stage153c_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage153c_waiting_for_stage153b %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q '^stage153b_complete ' "${STAGE153B_STATUS}" 2>/dev/null; do
  if grep -q '^stage153b_failed ' "${STAGE153B_STATUS}" 2>/dev/null; then
    printf 'stage153c_not_authorized_stage153b_failed %s\n' \
      "$(date --iso-8601=seconds)" > "${STATUS}"
    chmod 0444 "${STATUS}"
    trap - ERR
    exit 0
  fi
  sleep 60
done

test -s "${POLICY_LOCK}"
AUTHORIZED=$("${PYTHON}" - "${POLICY_LOCK}" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding='utf-8'))
assert lock['validation_labels_used_for_selection'] is False
print('1' if lock['validation_evaluation_authorized'] is True else '0')
PY
)
if test "${AUTHORIZED}" != "1"; then
  printf 'stage153c_not_authorized_internal_gate_failed %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR
  exit 0
fi

check_sha "${CKPT}" "8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
check_sha main_utils.py "960bb83e7fe9524d81bf48e0e23975558380371e2e73bb0af6eaeb5e925a79f2"
check_sha models/losses.py "86a076c2d8f265b9ddcc809a9c55f4548e83d5ba50c827291e5aaf1b4d325d62"
check_sha src/joint_det_dataset.py "5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
check_sha "${PATCHED}" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
test ! -e "${OUT}"

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
  --eval_dump_source_choice_features_path "${SOURCE_DUMP}"
  --eval_dump_source_choice_topk 1
  --verbose_diagnostics
)

CMD=(
  torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --use_soft_token_loss --use_contrastive_align
  --pp_checkpoint "${OFFICIAL_INIT}"
  --butd --self_attend --rng_seed 0 --print_freq 100
)
EVAL_CMD=("${CMD[@]}" --eval --checkpoint_path "${CKPT}"
  --log_dir "${OUT}" --eval_results_json_path "${OUT}/eval_results.json"
  "${MODEL_ARGS[@]}")

assert_gpu_idle
assert_storage
mkdir -p "${OUT}"
PRE_EVAL="${OUT}/grounding_evaluator.pre_stage153c.py"
cp -p "${LIVE}" "${PRE_EVAL}"
restore_evaluator() {
  install -m 0644 "${PRE_EVAL}" "${LIVE}"
}
trap restore_evaluator EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
printf 'stage153c_validation_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"

NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${COMPACT_DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage153c_validation_dump.log"

restore_evaluator
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

"${PYTHON}" - "${SOURCE_DUMP}" "${COMPACT_DUMP}" "${OUT}/eval_results.json" <<'PY'
import json, os, sys, torch, numpy as np
source_path, compact_path, metrics_path = sys.argv[1:]
manifest = torch.load(source_path, map_location='cpu')
assert manifest['format'] == 'source_choice_feature_dump_sharded_v1'
assert manifest['topk'] == 1
assert manifest['row_count'] == 9508
base = os.path.dirname(source_path)
count = 0
for relative in manifest['shards']:
    count += len(torch.load(os.path.join(base, relative), map_location='cpu')['rows'])
assert count == 9508
rows = torch.load(compact_path, map_location='cpu')['rows']
assert len(rows) == 9508
assert [int(row['example_id']) for row in rows] == list(range(9508))
hits = [0, 0]
for row in rows:
    scores = np.asarray(row['adapter_score_at_candidate'], dtype=np.float32)
    ious = np.asarray(row['adapter_iou_at_candidate'], dtype=np.float32)
    selected = int(np.argmax(scores))
    hits[0] += int(ious[selected] > 0.25)
    hits[1] += int(ious[selected] > 0.50)
assert hits == [5234, 3979], hits
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [
    round(metrics['last__bbs_acc0.25_top1'] * 9508),
    round(metrics['last__bbs_acc0.50_top1'] * 9508),
]
assert official == hits, (official, hits)
print('STAGE153C_DUMP_FORMAL_PARITY_PASS', hits, len(manifest['shards']))
PY

printf 'stage153c_locked_selector_evaluating %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage153_train_source_selector.py" evaluate \
  "${STAGE142_VAL_DUMP}" \
  "${SOURCE_DUMP}" \
  "${COMPACT_DUMP}" \
  "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" \
  "${POLICY_LOCK}" \
  "${RESULT}" \
  2>&1 | tee "${P}/stage153c_locked_selector_evaluate.log"

MET=$("${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding='utf-8'))
print('goal_met' if result['strict_goal_met_offline'] else 'goal_not_met')
PY
)
sha256sum "${SOURCE_DUMP}" "${COMPACT_DUMP}" "${RESULT}" > "${OUT}/artifact_sha256.txt"
chmod 0444 "${RESULT}" "${OUT}/artifact_sha256.txt"
printf 'stage153c_complete_%s %s\n' "${MET}" "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}"
trap - ERR
