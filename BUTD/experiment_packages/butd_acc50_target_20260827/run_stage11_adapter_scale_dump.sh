#!/usr/bin/env bash
set -euo pipefail
R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"
OUT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage11_adapter_scale_dump_textcid"
CKPT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage9_candidate_reranker_wide/scanrefer_spacy/1787837476/ckpt_best_primary.pth"
OLD="${P}/state/stage10_dualthreshold_20260827_2248/models"
PATCHED="${P}/grounding_evaluator_adapter_dump.py"
LIVE_BACKUP="${P}/state/stage11_live_restore"
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 211; }
[ -f "${CKPT}" ] && [ -f "${PATCHED}" ]
assert_gpu_idle; assert_storage
mkdir -p "${LIVE_BACKUP}/models" "${OUT}"
cp -p models/detector_policy_sources.py models/bdetr.py models/losses.py "${LIVE_BACKUP}/models/"
cp -p src/grounding_evaluator.py "${LIVE_BACKUP}/grounding_evaluator.py"
restore() {
  cp -p "${LIVE_BACKUP}/models/"*.py models/
  cp -p "${LIVE_BACKUP}/grounding_evaluator.py" src/grounding_evaluator.py
}
trap restore EXIT INT TERM
cp -p "${OLD}/detector_policy_sources.py" models/detector_policy_sources.py
cp -p "${OLD}/bdetr.py" models/bdetr.py
cp -p "${OLD}/losses.py" models/losses.py
cp -p "${PATCHED}" src/grounding_evaluator.py
base_command
CMD+=(--eval --checkpoint_path "${CKPT}" --log_dir "${OUT}"
 --eval_results_json_path "${OUT}/eval_results.json"
 --use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head
 --rapf_use_quality --rapf_quality_weight 0.25 --rapf_struct_residual_clip 0.25
 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
 --use_qahnl --qahnl_score_source fused --use_detector_policy_adapter
 --detector_policy_adapter_hidden_dim 64 --detector_policy_adapter_k 5
 --detector_policy_adapter_delta_scale 4.0 --eval_use_detector_policy_adapter_scores
 --eval_target_cid_source text --verbose_diagnostics)
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${OUT}/adapter_scale_dump.pt" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
restore; trap - EXIT INT TERM
"${PYTHON}" - "${OUT}/adapter_scale_dump.pt" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location='cpu'); rows=p['rows']
assert len(rows)==9508, len(rows)
assert all('adapter_delta_at_candidate' in r for r in rows)
print('adapter_rows={}'.format(len(rows)))
PY
