#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

STATUS="${P}/stage163d_dependency_status.txt"
UPSTREAM_STATUS="${P}/stage163b_dependency_status.txt"
ALT_SOURCE_CKPT="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
PRIMARY_SOURCE_CKPT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
SELECTOR_LOCK="${ROOT}/stage163a_tier3_residual_blend/locked_residual_blend_policy.json"
LOCKED_RESULT="${ROOT}/stage163b_tier3_residual_blend_validation_result.json"
BUNDLE="${ROOT}/stage163d_final_artifact_bundle"
TEMP_ALT_CKPT="${ROOT}/stage163d_materialized_stage135c_tmp.pth"
ALT_OUT="${ROOT}/stage163d_fresh_bundle_stage135c_raw_val"
ALT_DUMP="${ALT_OUT}/stage135c_fresh_raw_val_geometry.pt"
FRESH_RESULT="${ROOT}/stage163d_fresh_bundle_residual_validation_result.json"
RELOAD_RECEIPT="${ROOT}/stage163d_fresh_bundle_reload_receipt.json"
LIVE="src/grounding_evaluator.py"
RAW_PATCH="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
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

cleanup() {
  restore_evaluator
  if test -e "${TEMP_ALT_CKPT}"; then
    rm -- "${TEMP_ALT_CKPT}"
  fi
}

fail_status() {
  local rc=$?
  set +e
  cleanup
  printf 'stage163d_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR INT TERM

printf 'stage163d_waiting_for_stage163b %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
while ! grep -q -E '^stage163b_(complete_|not_authorized|failed)' \
  "${UPSTREAM_STATUS}" 2>/dev/null; do
  sleep 60
done

if ! grep -q '^stage163b_complete_goal_met ' "${UPSTREAM_STATUS}"; then
  printf 'stage163d_not_authorized_offline_goal_not_met %s\n' \
    "$(date --iso-8601=seconds)" > "${STATUS}"
  chmod 0444 "${STATUS}"
  trap - ERR INT TERM
  exit 0
fi

check_sha "${ALT_SOURCE_CKPT}" "a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
check_sha "${PRIMARY_SOURCE_CKPT}" "8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
check_sha main_utils.py "960bb83e7fe9524d81bf48e0e23975558380371e2e73bb0af6eaeb5e925a79f2"
check_sha models/losses.py "86a076c2d8f265b9ddcc809a9c55f4548e83d5ba50c827291e5aaf1b4d325d62"
check_sha src/joint_det_dataset.py "5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
check_sha "${RAW_PATCH}" "cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"
test -s "${SELECTOR_LOCK}"
test -s "${LOCKED_RESULT}"
test ! -e "${BUNDLE}"
test ! -e "${TEMP_ALT_CKPT}"
test ! -e "${ALT_OUT}"
test ! -e "${FRESH_RESULT}"
test ! -e "${RELOAD_RECEIPT}"

printf 'stage163d_building_bundle %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${P}/stage153_build_final_artifact.py" build \
  "${ALT_SOURCE_CKPT}" "${PRIMARY_SOURCE_CKPT}" \
  "${STAGE31_LOCK}" "${STAGE33_LOCK}" "${STAGE142_LOCK}" \
  "${SELECTOR_LOCK}" "${LOCKED_RESULT}" "${BUNDLE}" \
  --runtime-file "${P}/stage163_tier3_residual_blend.py" \
  --runtime-file "${P}/stage162_tier3_option_ranker.py" \
  --runtime-file "${P}/train_joint_option_ranker.py" \
  --runtime-file "${P}/stage140_train_eval_nested_blend.py" \
  --runtime-file "${P}/stage143_same_checkpoint_complement_gate.py" \
  --runtime-file "${P}/stage153_build_final_artifact.py" \
  --runtime-file "${P}/stage153_materialize_alternate_checkpoint.py" \
  --runtime-file "${P}/stage153_finalize_artifact.py" \
  --runtime-file "${RAW_PATCH}" \
  2>&1 | tee "${P}/stage163d_build_bundle.log"

"${PYTHON}" "${BUNDLE}/runtime/stage153_materialize_alternate_checkpoint.py" \
  "${BUNDLE}" "${TEMP_ALT_CKPT}" \
  2>&1 | tee "${P}/stage163d_materialize_alternate.log"

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

assert_gpu_idle
assert_storage
mkdir -p "${ALT_OUT}"
PRE_EVAL="${ALT_OUT}/grounding_evaluator.pre_stage163d_alt.py"
cp -p "${LIVE}" "${PRE_EVAL}"
install -m 0644 "${BUNDLE}/runtime/grounding_evaluator_adapter_dump_parityfixed.py" "${LIVE}"
check_sha "${LIVE}" "cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"
base_command
ALT_CMD=("${CMD[@]}" --eval --checkpoint_path "${TEMP_ALT_CKPT}"
  --log_dir "${ALT_OUT}" --eval_results_json_path "${ALT_OUT}/eval_results.json"
  "${COMMON_MODEL_ARGS[@]}"
  --use_qahnl --qahnl_score_source fused --qahnl_loss_weight 0.0)
printf 'stage163d_fresh_alternate_inference %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${ALT_DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${ALT_CMD[@]}" \
  2>&1 | tee "${P}/stage163d_fresh_alternate_inference.log"
restore_evaluator
check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

"${PYTHON}" - "${ALT_DUMP}" "${ALT_OUT}/eval_results.json" <<'PY'
import json, sys, torch
dump_path, metrics_path = sys.argv[1:]
rows = torch.load(dump_path, map_location='cpu')['rows']
assert len(rows) == 9508
hits = [0, 0]
for row in rows:
    queries = [int(value) for value in row['adapter_candidate_query']]
    selected = int(row['adapter_rescue_query'])
    iou = float(row['adapter_iou_at_candidate'][queries.index(selected)])
    hits[0] += int(iou > 0.25)
    hits[1] += int(iou > 0.50)
metrics = json.load(open(metrics_path, encoding='utf-8'))
official = [round(metrics['last__bbs_acc0.25_top1'] * 9508),
            round(metrics['last__bbs_acc0.50_top1'] * 9508)]
assert hits == official == [5215, 3956], (hits, official)
print('STAGE163D_ALTERNATE_FRESH_PARITY_PASS', hits)
PY

ALT_TEMP_SHA=$(sha256sum "${TEMP_ALT_CKPT}" | awk '{print $1}')
rm -- "${TEMP_ALT_CKPT}"
test ! -e "${TEMP_ALT_CKPT}"

printf 'stage163d_fresh_locked_residual_ranker %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
"${PYTHON}" "${BUNDLE}/runtime/stage163_tier3_residual_blend.py" evaluate \
  "${ALT_DUMP}" "${BUNDLE}/locks/stage31.json" \
  "${BUNDLE}/locks/stage33.json" "${BUNDLE}/locks/stage142.json" \
  "${BUNDLE}/locks/stage153.json" "${FRESH_RESULT}" \
  2>&1 | tee "${P}/stage163d_fresh_locked_residual_ranker.log"

"${PYTHON}" - "${LOCKED_RESULT}" "${FRESH_RESULT}" "${RELOAD_RECEIPT}" \
  "${ALT_DUMP}" "${ALT_TEMP_SHA}" <<'PY'
import hashlib, json, sys
locked_path, fresh_path, receipt_path, alt_dump, temp_sha = sys.argv[1:]
locked=json.load(open(locked_path,encoding='utf-8'))
fresh=json.load(open(fresh_path,encoding='utf-8'))
assert locked['strict_goal_met_offline'] is True
assert fresh['strict_goal_met_offline'] is True
lsel=locked['metrics']['selected']; fsel=fresh['metrics']['selected']
assert int(fsel['hits025']) == int(lsel['hits025'])
assert int(fsel['hits050']) == int(lsel['hits050'])
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
receipt={
  'stage':'stage163d_fresh_bundle_reload',
  'status':'complete',
  'no_ground_truth_used_for_inference':True,
  'ground_truth_used_only_for_posthoc_metric_computation':True,
  'primary_stage150_hits':None,
  'primary_inference_required_for_deployment':False,
  'alternate_stage135c_hits':[5215,3956],
  'selected_hits':[int(fsel['hits025']),int(fsel['hits050'])],
  'selected_acc':[float(fsel['acc025']),float(fsel['acc050'])],
  'locked_validation_result_sha256':sha(fresh_path),
  'offline_reference_result_sha256':sha(locked_path),
  'alternate_materialized_checkpoint_sha256':temp_sha,
  'fresh_dump_sha256':{'alternate_raw':sha(alt_dump)},
  'evaluator_restored_sha256':'50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86',
  'temporary_alternate_checkpoint_deleted':True,
}
with open(receipt_path,'w',encoding='utf-8') as handle:
    json.dump(receipt,handle,indent=2,sort_keys=True); handle.write('\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

"${PYTHON}" "${BUNDLE}/runtime/stage153_finalize_artifact.py" \
  "${BUNDLE}" "${FRESH_RESULT}" "${RELOAD_RECEIPT}" \
  2>&1 | tee "${P}/stage163d_finalize_bundle.log"
"${PYTHON}" "${BUNDLE}/runtime/stage153_build_final_artifact.py" validate \
  "${BUNDLE}" 2>&1 | tee "${P}/stage163d_final_bundle_validate.log"

cleanup
trap - ERR INT TERM
printf 'stage163d_complete_fresh_bundle_reload %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${STATUS}" "${RELOAD_RECEIPT}" "${FRESH_RESULT}"
