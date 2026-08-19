#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INPUT_RUN="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/02_full_sacr_rapf_qahnl/scanrefer_spacy/1786908904"
INPUT_CHECKPOINT="${INPUT_RUN}/ckpt_best_primary.pth"
INPUT_RECEIPT="${INPUT_RUN}/best_primary.json"
TARGET="${SCANREFER_TARGET:-0.5391}"
MAX_EPOCH="${SCANREFER_MAX_EPOCH:-80}"
VAL_FREQ="${SCANREFER_VAL_FREQ:-1}"
RUN_ROOT="${REPO_ROOT}/logs/butd_universal_target/main_results_20260819/scanrefer_same_sacr_rapf_qahnl_e65_to_target"
STDOUT_LOG="${STATE_ROOT}/scanrefer_continuation_stdout.log"
ACCEPTANCE_JSON="${STATE_ROOT}/scanrefer_acceptance.json"

assert_checkpoint "${INPUT_CHECKPOINT}" 65
assert_three_module_boundary "${INPUT_CHECKPOINT}" "${STATE_ROOT}/scanrefer_input_boundary.json"
wait_gpu_idle
mkdir -p "${RUN_ROOT}"

CMD=(
  torchrun --nproc_per_node 1 --master_port 30241 train_dist_mod.py
  --num_decoder_layers 6 --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}" --val_freq "${VAL_FREQ}" --batch_size 24
  --save_freq 1000 --print_freq 100 --max_epoch "${MAX_EPOCH}"
  --lr_backbone=1e-3 --lr=1e-4
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --joint_det --use_soft_token_loss --use_contrastive_align
  --log_dir "${RUN_ROOT}" --lr_decay_epochs 55 60
  --pp_checkpoint "${DATA_ROOT}/gf_detector_l6o256.pth"
  --butd --self_attend --augment_det --rng_seed 0
  --best_checkpoint_only --best_checkpoint_metric last__bbs_acc0.25_top1
  --best_checkpoint_min_delta 0 --verbose_diagnostics --eval_report_diagnostic_scores
  --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
  --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused
  --eval_use_fused_scores --rapf_quality_weight 0.75
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
  --checkpoint_path "${INPUT_CHECKPOINT}"
)

echo "[$(timestamp)] Starting exact three-module ScanRefer continuation from epoch 65; target > ${TARGET}; val_freq=${VAL_FREQ}."
setsid env TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" >"${STDOUT_LOG}" 2>&1 &
TRAIN_PID=$!

RUN_DIR=""
for _ in $(seq 1 30); do
  RUN_DIR="$(latest_run_dir "${RUN_ROOT}" 2>/dev/null || true)"
  [ -n "${RUN_DIR}" ] && break
  sleep 5
done
[ -n "${RUN_DIR}" ] || { echo "Failed to discover ScanRefer continuation run directory" >&2; kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true; exit 52; }
echo "${RUN_DIR}" > "${STATE_ROOT}/scanrefer_run_dir.txt"

input_score="$(${PYTHON} - "${INPUT_RECEIPT}" <<'PY'
import json, sys
print(float(json.load(open(sys.argv[1]))['score']))
PY
)"
ln "${INPUT_CHECKPOINT}" "${RUN_DIR}/ckpt_best_primary.pth"
"${PYTHON}" - "${RUN_DIR}" "${input_score}" <<'PY'
import json, sys
run, score = sys.argv[1], float(sys.argv[2])
payload = {
    'checkpoint': run + '/ckpt_best_primary.pth',
    'comparison': 'strict_greater_than',
    'epoch': 65,
    'metric': 'last__bbs_acc0.25_top1',
    'min_delta': 0.0,
    'mode': 'max',
    'score': score,
}
open(run + '/best_primary.json', 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

TARGET_HIT=0
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  if [ -f "${RUN_DIR}/best_primary.json" ]; then
    read -r epoch score < <("${PYTHON}" - "${RUN_DIR}/best_primary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); print(int(d['epoch']), float(d['score']))
PY
)
    echo "[$(timestamp)] ScanRefer strict best: epoch=${epoch}, Acc@0.25=${score}."
    if "${PYTHON}" - "${score}" "${TARGET}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      TARGET_HIT=1
      echo "[$(timestamp)] Target exceeded; stopping continuation after durable checkpoint write."
      kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done

set +e
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e
echo "[$(timestamp)] ScanRefer continuation process ended rc=${TRAIN_RC}, target_hit=${TARGET_HIT}."
wait_gpu_idle

BEST_CHECKPOINT="${RUN_DIR}/ckpt_best_primary.pth"
assert_checkpoint "${BEST_CHECKPOINT}" 65
assert_three_module_boundary "${BEST_CHECKPOINT}" "${STATE_ROOT}/scanrefer_final_boundary.json"

FINAL_JSON="${STATE_ROOT}/scanrefer_final_eval_results.json"
FINAL_LOG_ROOT="${RUN_ROOT}_strict_reload_eval"
torchrun --nproc_per_node 1 --master_port 30242 train_dist_mod.py \
  --eval --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8 \
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy --joint_det \
  --use_soft_token_loss --use_contrastive_align --log_dir "${FINAL_LOG_ROOT}" \
  --lr_decay_epochs 55 60 --pp_checkpoint "${DATA_ROOT}/gf_detector_l6o256.pth" \
  --butd --self_attend --augment_det --rng_seed 0 \
  --verbose_diagnostics --eval_report_diagnostic_scores \
  --eval_results_json_path "${FINAL_JSON}" \
  --use_structured_slots --use_sacr --use_rapf --use_reliability_gate \
  --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused \
  --eval_use_fused_scores --rapf_quality_weight 0.75 \
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1 \
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1 \
  --checkpoint_path "${BEST_CHECKPOINT}" \
  >"${STATE_ROOT}/scanrefer_strict_reload_eval_stdout.log" 2>&1

"${PYTHON}" - "${FINAL_JSON}" "${TARGET}" "${BEST_CHECKPOINT}" "${ACCEPTANCE_JSON}" <<'PY'
import hashlib, json, sys
result_path, target, checkpoint, output = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
d=json.load(open(result_path))
metric=float(d['last__bbs_acc0.25_top1'])
h=hashlib.sha256()
with open(checkpoint,'rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
payload={
    'status':'PASS' if metric > target else 'FAIL',
    'module_boundary':'SACR+RAPF+QAHNL only',
    'metric':'last__bbs_acc0.25_top1',
    'score':metric,
    'target_strictly_greater_than':target,
    'checkpoint':checkpoint,
    'checkpoint_sha256':h.hexdigest(),
    'eval_results_json':result_path,
}
open(output,'w').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 62)
PY

echo "[$(timestamp)] ScanRefer strict acceptance PASS: ${ACCEPTANCE_JSON}"
