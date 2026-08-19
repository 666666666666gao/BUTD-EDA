#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SCAN_ACCEPTANCE="${STATE_ROOT}/scanrefer_acceptance.json"
[ -f "${SCAN_ACCEPTANCE}" ] || { echo "Missing ScanRefer acceptance receipt" >&2; exit 2; }
SCAN_CHECKPOINT="$(${PYTHON} - "${SCAN_ACCEPTANCE}" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d['status']=='PASS'; print(d['checkpoint'])
PY
)"
assert_three_module_boundary "${SCAN_CHECKPOINT}" "${STATE_ROOT}/transfer_scan_input_boundary.json"
wait_gpu_idle

SLIM_INIT="${STATE_ROOT}/scanrefer_best_model_epoch0_for_domain_adaptation.pth"
"${PYTHON}" - "${SCAN_CHECKPOINT}" "${SLIM_INIT}" <<'PY'
import sys, torch
src, dst = sys.argv[1:]
d=torch.load(src,map_location='cpu')
torch.save({'model':d['model'],'epoch':0,'source_checkpoint':src},dst)
print(dst)
PY
assert_checkpoint "${SLIM_INIT}" 0

COMMON_MODULE_ARGS=(
  --use_structured_slots --use_sacr --use_rapf --use_reliability_gate
  --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused
  --eval_use_fused_scores --rapf_quality_weight 0.75
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
)

NR_ROOT="${REPO_ROOT}/logs/butd_universal_target/main_results_20260819/nr3d_same_sacr_rapf_qahnl_e1"
mkdir -p "${NR_ROOT}"
echo "[$(timestamp)] Starting one Nr3D adaptation epoch to mirror the EDA transfer sequence."
torchrun --nproc_per_node 1 --master_port 30243 train_dist_mod.py \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" --val_freq 1 --batch_size 24 --save_freq 1000 \
  --print_freq 200 --max_epoch 1 --lr_backbone=1e-3 --lr=1e-4 \
  --dataset nr3d_spacy --test_dataset nr3d_spacy --joint_det --butd_cls --self_attend \
  --use_soft_token_loss --use_contrastive_align --log_dir "${NR_ROOT}" \
  --lr_decay_epochs 25 27 --pp_checkpoint "${DATA_ROOT}/gf_detector_l6o256.pth" \
  --rng_seed 0 --best_checkpoint_only --best_checkpoint_metric last__bbs_acc \
  --best_checkpoint_min_delta 0 --verbose_diagnostics --eval_report_diagnostic_scores \
  --checkpoint_path "${SLIM_INIT}" --reduce_lr "${COMMON_MODULE_ARGS[@]}" \
  >"${STATE_ROOT}/nr3d_adaptation_stdout.log" 2>&1

NR_RUN="$(latest_run_dir "${NR_ROOT}")"
NR_CHECKPOINT="${NR_RUN}/ckpt_best_primary.pth"
assert_checkpoint "${NR_CHECKPOINT}" 1
assert_three_module_boundary "${NR_CHECKPOINT}" "${STATE_ROOT}/nr3d_final_boundary.json"

SR_ROOT="${REPO_ROOT}/logs/butd_universal_target/main_results_20260819/sr3d_same_sacr_rapf_qahnl_e1"
mkdir -p "${SR_ROOT}"
echo "[$(timestamp)] Starting one Sr3D adaptation epoch from the clean Nr3D checkpoint."
torchrun --nproc_per_node 1 --master_port 30244 train_dist_mod.py \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" --val_freq 1 --batch_size 24 --save_freq 1000 \
  --print_freq 200 --max_epoch 2 --lr_backbone=1e-3 --lr=1e-4 \
  --dataset sr3d_spacy --test_dataset sr3d_spacy --joint_det --butd_cls --self_attend \
  --use_soft_token_loss --use_contrastive_align --log_dir "${SR_ROOT}" \
  --lr_decay_epochs 25 27 --pp_checkpoint "${DATA_ROOT}/gf_detector_l6o256.pth" \
  --rng_seed 0 --best_checkpoint_only --best_checkpoint_metric last__bbs_acc \
  --best_checkpoint_min_delta 0 --verbose_diagnostics --eval_report_diagnostic_scores \
  --checkpoint_path "${NR_CHECKPOINT}" --reduce_lr "${COMMON_MODULE_ARGS[@]}" \
  >"${STATE_ROOT}/sr3d_adaptation_stdout.log" 2>&1

SR_RUN="$(latest_run_dir "${SR_ROOT}")"
SR_CHECKPOINT="${SR_RUN}/ckpt_best_primary.pth"
assert_checkpoint "${SR_CHECKPOINT}" 2
assert_three_module_boundary "${SR_CHECKPOINT}" "${STATE_ROOT}/sr3d_final_boundary.json"

"${PYTHON}" - "${NR_RUN}/best_primary.json" "${NR_CHECKPOINT}" "${SR_RUN}/best_primary.json" "${SR_CHECKPOINT}" "${STATE_ROOT}/domain_transfer_acceptance.json" <<'PY'
import hashlib, json, sys
nr_receipt, nr_ckpt, sr_receipt, sr_ckpt, output = sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
nr=json.load(open(nr_receipt)); sr=json.load(open(sr_receipt))
payload={
    'status':'PASS',
    'module_boundary':'SACR+RAPF+QAHNL only',
    'protocol':'ScanRefer accepted checkpoint -> one Nr3D epoch -> one Sr3D epoch',
    'nr3d':{'score':float(nr['score']),'metric':nr['metric'],'epoch':int(nr['epoch']),'checkpoint':nr_ckpt,'sha256':sha(nr_ckpt)},
    'sr3d':{'score':float(sr['score']),'metric':sr['metric'],'epoch':int(sr['epoch']),'checkpoint':sr_ckpt,'sha256':sha(sr_ckpt)},
}
open(output,'w').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
PY

safe_remove_transient_weight "${SLIM_INIT}"
echo "[$(timestamp)] Domain transfer complete; retained exactly one best checkpoint for Nr3D and one for Sr3D."

