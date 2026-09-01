#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

SRC="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage18_geometry_action_head/scanrefer_spacy/1787892530/ckpt_best_primary.pth"
SRC_SHA="c7aa796359314b95575c9af035535d8432ca2f31be4aa6dff52a59e5eb39213f"
PARITY_OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage48_geometry_extension_parity"
PARITY_JSON="${PARITY_OUT}/eval_results.json"
PARITY_RECEIPT="${PARITY_OUT}/parity_receipt.json"
OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage48_geometry_extension_head"
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "${SRC}")"

[ -f "${SRC}" ]
[ "$(sha256sum "${SRC}" | awk '{print $1}')" = "${SRC_SHA}" ]
[ "$(sha256sum src/grounding_evaluator.py | awk '{print $1}')" = "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86" ]

append_method_flags() {
  CMD+=(
    --use_structured_slots --use_sacr --use_rapf
    --use_reliability_gate --use_quality_head
    --rapf_use_quality --rapf_quality_weight 0.25
    --rapf_struct_residual_clip 0.25
    --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5
    --rapf_generic_gate_cap 0.1
    --use_qahnl --qahnl_score_source fused
    --use_detector_policy_adapter
    --detector_policy_adapter_hidden_dim 64
    --detector_policy_adapter_k 5
    --detector_policy_adapter_delta_scale 4.0
    --detector_policy_geometry_extension_head
    --eval_use_detector_policy_adapter_scores
    --eval_target_cid_source text
    --verbose_diagnostics
  )
}

append_training_flags() {
  CMD+=(
    --log_dir "${OUT}" --max_epoch "$((E+3))" --val_freq 1
    --save_freq 1000 --checkpoint_path "${SRC}" --best_checkpoint_only
    --best_checkpoint_metric last__bbs_acc0.25_top1
    --best_checkpoint_min_delta 0
    --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241
    --best_checkpoint_constraint_epsilon 0
    --detector_policy_adapter_train_only
    --detector_policy_geometry_extension_train_only
    --detector_policy_adapter_lr 0.0002
    --detector_policy_adapter_loss_weight 0.0
    --detector_policy_geometry_loss_weight 1.0
    --detector_policy_adapter_margin 0.1
    --detector_policy_adapter_min_iou_gap 0.02
  )
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  base_command
  append_training_flags
  append_method_flags
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage

if [ -f "${PARITY_RECEIPT}" ]; then
  "${PYTHON}" - "${PARITY_RECEIPT}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['parity_pass'] is True, r
PY
else
  [ ! -e "${PARITY_OUT}" ] || {
    echo "refusing incomplete parity directory ${PARITY_OUT}" >&2
    exit 227
  }
  mkdir -p "${PARITY_OUT}"
  base_command
  CMD+=(
    --eval --checkpoint_path "${SRC}" --log_dir "${PARITY_OUT}"
    --eval_results_json_path "${PARITY_JSON}"
  )
  append_method_flags
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
    2>&1 | tee "${P}/stage48_parity_eval.log"
  "${PYTHON}" - "${PARITY_JSON}" "${PARITY_RECEIPT}" \
    "${SRC}" <<'PY'
import hashlib, json, os, sys
result_path, receipt_path, checkpoint_path = sys.argv[1:]
metrics = json.load(open(result_path))
acc025 = float(metrics['last__bbs_acc0.25_top1'])
acc050 = float(metrics['last__bbs_acc0.50_top1'])
expected025 = 0.5472233908287758
expected050 = 0.41196886832141355
h = hashlib.sha256()
with open(checkpoint_path, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
payload = {
    'checkpoint': os.path.realpath(checkpoint_path),
    'checkpoint_sha256': h.hexdigest(),
    'legacy_actions': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'residual_extension_actions': [0.75, 1.0],
    'overall_acc0.25': acc025,
    'overall_acc0.50': acc050,
    'expected_acc0.25': expected025,
    'expected_acc0.50': expected050,
    'eval_target_cid_source': 'text',
    'parity_pass': (
        abs(acc025 - expected025) <= 1e-12
        and abs(acc050 - expected050) <= 1e-12
    ),
}
tmp = receipt_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload['parity_pass']:
    raise SystemExit('geometry extension checkpoint parity failed')
PY
fi

[ ! -e "${OUT}" ] || {
  echo "refusing to overwrite ${OUT}" >&2
  exit 228
}
mkdir -p "${OUT}"
sha256sum \
  main_utils.py train_dist_mod.py models/bdetr.py \
  models/detector_policy_sources.py models/losses.py \
  src/grounding_evaluator.py "${SRC}" \
  > "${OUT}/launch_sha256.txt"

base_command
append_training_flags
append_method_flags
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage48_train.log"
