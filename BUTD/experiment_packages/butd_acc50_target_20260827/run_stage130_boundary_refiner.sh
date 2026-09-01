#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
SMOKE_OUT="${ROOT}/stage130_boundary_refiner_smoke96"
OUT="${ROOT}/stage130_boundary_refiner_train"
BASELINE_SMOKE="${ROOT}/stage100a_stage95_live_smoke96/eval_results.json"
EXPECTED_SRC_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="afcb88f2a9a268bf270ee71b0901a18876e732a08749a79b796b6262b9f30ba8"
EXPECTED_TRAIN_SHA="4dd8bb64e1b33c66f755c624c3294a40f9615c726bbdfc89674bdcc935c88069"
EXPECTED_BDETR_SHA="8840ade8f08f78ff89332dec376a935064443c839a792231eb29b06d05558ff5"
EXPECTED_POLICY_SHA="7d615107551720000f924ba903303d886409bd2ebfd48e363fb484fa902b029e"
EXPECTED_LOSS_SHA="acb1eed32c1f16a3696b3d1b9dab13abf94dc5aecf7681be8e670fa26286a60a"
EXPECTED_EVAL_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [ "${actual}" = "${expected}" ] || {
    printf 'SHA mismatch %s expected=%s actual=%s\n' \
      "${path}" "${expected}" "${actual}" >&2
    exit 228
  }
}
check_sha "${SRC}" "${EXPECTED_SRC_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha train_dist_mod.py "${EXPECTED_TRAIN_SHA}"
check_sha models/bdetr.py "${EXPECTED_BDETR_SHA}"
check_sha models/detector_policy_sources.py "${EXPECTED_POLICY_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha src/grounding_evaluator.py "${EXPECTED_EVAL_SHA}"

"${PYTHON}" -m py_compile \
  main_utils.py train_dist_mod.py models/bdetr.py \
  models/detector_policy_sources.py models/losses.py
"${PYTHON}" "${P}/test_boundary_refiner_contract.py"
E="$("${PYTHON}" -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SRC}")"

MODEL_ARGS=(
 --disable_box_jitter
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
 --use_quality_head --rapf_use_quality --rapf_quality_weight 0.25
 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.0
 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --qahnl_loss_weight 0.0
 --quality_loss_weight 0.0 --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0
 --detector_policy_adapter_loss_weight 0.0
 --detector_policy_geometry_loss_weight 0.0
 --detector_policy_rank2_rescue_loss_weight 0.0
 --detector_policy_adapter_margin 0.1
 --detector_policy_adapter_min_iou_gap 0.02
 --detector_policy_boundary_refiner
 --detector_policy_boundary_refiner_scale 0.25
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics
)

base_command
SMOKE_CMD=("${CMD[@]}" --eval --checkpoint_path "${SRC}"
 --disable_train_augmentation "${MODEL_ARGS[@]}"
 --eval_max_samples 96 --log_dir "${SMOKE_OUT}"
 --eval_results_json_path "${SMOKE_OUT}/eval_results.json")

base_command
TRAIN_CMD=("${CMD[@]}" --log_dir "${OUT}" --max_epoch "$((E+3))"
 --val_freq 1 --save_freq 1000 --checkpoint_path "${SRC}"
 --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.50_top1
 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241
 --best_checkpoint_constraint_epsilon 0
 "${MODEL_ARGS[@]}"
 --detector_policy_adapter_train_only
 --detector_policy_boundary_refiner_train_only
 --detector_policy_adapter_lr 0.0005
 --detector_policy_boundary_refiner_loss_weight 1.0
 --detector_policy_boundary_refiner_iou_min 0.25
 --detector_policy_boundary_refiner_iou_max 0.55
 --detector_policy_boundary_refiner_stability_weight 0.5)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${SMOKE_CMD[@]}"; printf '\n'
  printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
[ ! -e "${SMOKE_OUT}" ] || { echo "refusing to overwrite ${SMOKE_OUT}" >&2; exit 226; }
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 227; }
mkdir -p "${SMOKE_OUT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${SMOKE_CMD[@]}" 2>&1 \
  | tee "${P}/stage130_smoke96.log"
"${PYTHON}" "${P}/verify_stage130_smoke_parity.py" \
  "${BASELINE_SMOKE}" "${SMOKE_OUT}/eval_results.json" \
  "${SMOKE_OUT}/stage130_smoke_parity_receipt.json"

mkdir -p "${OUT}"
{
  printf 'stage=130\nsource=%s\nsource_epoch=%s\n' "${SRC}" "${E}"
  printf 'source_sha256=%s\nmain_utils_sha256=%s\nlosses_sha256=%s\n' \
    "${EXPECTED_SRC_SHA}" "${EXPECTED_MAIN_SHA}" "${EXPECTED_LOSS_SHA}"
  printf 'architecture=zero_init_query_detector_continuous_boundary_refiner\n'
  printf 'smoke_receipt=%s\n' "${SMOKE_OUT}/stage130_smoke_parity_receipt.json"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
  printf 'started_at='; date --iso-8601=seconds
} > "${OUT}/launch_manifest.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TRAIN_CMD[@]}" 2>&1 \
  | tee "${P}/stage130_train.log"
