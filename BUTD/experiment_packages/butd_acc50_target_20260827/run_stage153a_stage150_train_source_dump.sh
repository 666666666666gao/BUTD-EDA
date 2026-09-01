#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

CKPT="${ROOT}/stage150b_rerank_only_tier3_from_stage135c/scanrefer_spacy/1788152996/ckpt_best_primary.pth"
OUT="${STAGE153_OUT:-${ROOT}/stage153a_stage150_e13_train_source_dump}"
DUMP="${OUT}/stage150_e13_train_source_features.pt"
COMPACT_DUMP="${OUT}/stage150_e13_train_adapter_features.pt"
STATUS="${OUT}/status.txt"
EVAL_MAX_SAMPLES="${STAGE153_EVAL_MAX_SAMPLES:--1}"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

fail_status() {
  local rc=$?
  if test -d "${OUT}"; then
    printf 'stage153a_failed rc=%s at=%s line=%s\n' \
      "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
      > "${STATUS}"
  fi
  exit "${rc}"
}
trap fail_status ERR

check_sha "${CKPT}" "8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
check_sha main_utils.py "960bb83e7fe9524d81bf48e0e23975558380371e2e73bb0af6eaeb5e925a79f2"
check_sha models/losses.py "86a076c2d8f265b9ddcc809a9c55f4548e83d5ba50c827291e5aaf1b4d325d62"
check_sha src/joint_det_dataset.py "5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
check_sha src/grounding_evaluator.py "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
check_sha "${P}/grounding_evaluator_stage149b_calibrated_dump.py" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
check_sha train_dist_mod.py "44852f403849266e5d706b39c86e99f04f3bc682652bf8eb06944e11557a25e0"
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
  --eval_dump_source_choice_features_path "${DUMP}"
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
EVAL_CMD=("${CMD[@]}" --eval_train --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --checkpoint_path "${CKPT}" --log_dir "${OUT}"
  --eval_results_json_path "${OUT}/eval_results.json" "${MODEL_ARGS[@]}")

assert_gpu_idle
assert_storage
mkdir -p "${OUT}"
LIVE="src/grounding_evaluator.py"
PATCHED="${P}/grounding_evaluator_stage149b_calibrated_dump.py"
PRE_EVAL="${OUT}/grounding_evaluator.pre_stage153a.py"
cp -p "${LIVE}" "${PRE_EVAL}"
restore_evaluator() {
  install -m 0644 "${PRE_EVAL}" "${LIVE}"
}
trap restore_evaluator EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "8ad56df00fd30502259b9f4d49715e2f4f56918ae392ebe07a81a9971ff33d71"
{
  printf 'stage=153a_stage150_train_source_dump\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'checkpoint=%s\ncheckpoint_sha256=%s\n' \
    "${CKPT}" "8888af47d293d5449b9d68e323ff69db882a3115db839c6d65899a87edd9dc27"
  printf 'eval_max_samples=%s\n' "${EVAL_MAX_SAMPLES}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"
printf 'stage153a_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"

NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${COMPACT_DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${OUT}/run.log"

restore_evaluator
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

"${PYTHON}" - "${DUMP}" "${COMPACT_DUMP}" "${EVAL_MAX_SAMPLES}" <<'PY'
import os, sys, torch
dump_path, compact_path, limit_text = sys.argv[1:]
payload = torch.load(dump_path, map_location='cpu')
assert payload['format'] == 'source_choice_feature_dump_sharded_v1'
assert payload['topk'] == 1
expected = 36665 if int(limit_text) < 0 else int(limit_text)
assert payload['row_count'] == expected, (payload['row_count'], expected)
base = os.path.dirname(dump_path)
rows = []
for relative in payload['shards']:
    shard = torch.load(os.path.join(base, relative), map_location='cpu')
    rows.extend(shard['rows'])
assert len(rows) == expected
required = {
    'base_top_query', 'base_top_iou',
    'fused_top_query', 'fused_top_iou',
    'quality_top_query', 'quality_top_iou',
    'target_detector_logit_top_query',
    'target_detector_logit_top_iou',
}
assert required.issubset(rows[0]), sorted(required - set(rows[0]))
compact = torch.load(compact_path, map_location='cpu')['rows']
assert len(compact) == expected
assert [int(row['example_id']) for row in compact] == list(range(expected))
compact_required = {
    'adapter_candidate_query', 'adapter_score_at_candidate',
    'adapter_iou_at_candidate', 'adapter_hit25_logit_at_candidate',
    'adapter_hit50_logit_at_candidate', 'adapter_rescue_gate',
    'adapter_rescue_query', 'adapter_fallback_query',
}
assert compact_required.issubset(compact[0]), sorted(
    compact_required - set(compact[0])
)
print('STAGE153A_DUMP_PARITY_PASS', expected, len(payload['shards']))
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
sha256sum "${DUMP}" "${COMPACT_DUMP}" >> "${OUT}/launch_manifest.txt"
printf 'stage153a_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${OUT}/launch_manifest.txt" "${STATUS}"
trap - ERR
