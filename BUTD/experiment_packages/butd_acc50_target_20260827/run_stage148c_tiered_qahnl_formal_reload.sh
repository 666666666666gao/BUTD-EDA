#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

TRAIN_OUT="${ROOT}/stage148b_tiered_qahnl_adapter_trainonly"
OUT="${ROOT}/stage148c_tiered_qahnl_formal_reload"
STATUS="${P}/stage148c_tiered_qahnl_reload_status.txt"
EXPECTED_MAIN_SHA="665aed267508aa4a77f8ad014071d3a912642e8631a691fa9320c3f21aa14dd6"
EXPECTED_LOSS_SHA="a80d4a8934536f1b11f488aa7a38bce2ecf6bfd3ad9b732f81913e2ad78763b4"

fail_status() {
  local rc=$?
  printf 'stage148c_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR

test "$(sha256sum main_utils.py | awk '{print $1}')" = "${EXPECTED_MAIN_SHA}"
test "$(sha256sum models/losses.py | awk '{print $1}')" = "${EXPECTED_LOSS_SHA}"
mapfile -t CKPTS < <(find "${TRAIN_OUT}" -type f -name ckpt_best_primary.pth | sort)
test "${#CKPTS[@]}" = 1
CKPT="${CKPTS[0]}"
BEST_JSON="$(dirname "${CKPT}")/best_primary.json"
test -s "${BEST_JSON}"
test ! -e "${OUT}"

"${PYTHON}" - "${BEST_JSON}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
m=d['constraint_values']
assert m['overall_025'] > 0.5391, m
assert m['overall_050'] > 0.4241, m
print('STAGE148_TRAIN_CANDIDATE_STRICT_GOAL_PASS',m)
PY

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
 --quality_loss_weight 0.0
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

if [[ "${DRY_RUN:-0}" = 1 ]]; then
  printf '%q ' "${EVAL_CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${OUT}"
CKPT_SHA="$(sha256sum "${CKPT}" | awk '{print $1}')"
{
  printf 'stage=148c_tiered_qahnl_formal_reload\n'
  printf 'started_at=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "${CKPT}" "${CKPT_SHA}"
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_MAIN_SHA}" "${EXPECTED_LOSS_SHA}"
  printf 'validation_threshold_search=none\n'
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage148c_reloading %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${EVAL_CMD[@]}" \
  2>&1 | tee "${P}/stage148c_tiered_qahnl_formal_reload.log"

"${PYTHON}" - \
  "${BEST_JSON}" "${OUT}/eval_results.json" "${OUT}" "${CKPT_SHA}" <<'PY'
import json,os,sys
best_path,result_path,out,checkpoint_sha=sys.argv[1:]
best=json.load(open(best_path,encoding='utf-8'))['constraint_values']
result=json.load(open(result_path,encoding='utf-8'))
acc25=result['last__bbs_acc0.25_top1']
acc50=result['last__bbs_acc0.50_top1']
assert acc25 == best['overall_025'], (acc25,best['overall_025'])
assert acc50 == best['overall_050'], (acc50,best['overall_050'])
assert acc25 > 0.5391 and acc50 > 0.4241, (acc25,acc50)
receipt={
  'stage':'148c_tiered_qahnl_formal_reload',
  'status':'complete',
  'checkpoint_sha256':checkpoint_sha,
  'acc025':acc25,
  'acc050':acc50,
  'strict_goal_met':True,
  'gt_used_at_inference':False,
  'validation_threshold_search':False,
}
path=os.path.join(out,'stage148c_formal_receipt.json')
with open(path,'w',encoding='utf-8') as f:
  json.dump(receipt,f,indent=2,sort_keys=True); f.write('\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

chmod 0444 "${OUT}/launch_manifest.txt" "${OUT}/stage148c_formal_receipt.json"
trap - ERR
printf 'stage148c_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
