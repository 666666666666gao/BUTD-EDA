#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${STAGE24_OUT:-/root/autodl-tmp/logs/butd_acc50_target_20260827/stage24_stage16_e8_train_geometry_dump}"
MAX_SAMPLES="${STAGE24_MAX_SAMPLES:--1}"
EXPECTED_ROWS="${STAGE24_EXPECTED_ROWS:-36665}"
CKPT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage16_hit50_rank_augmented/scanrefer_spacy/1787883778/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_joint_train_dump.py"

[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 224; }
[ -f "${CKPT}" ] && [ -f "${PATCHED}" ]
assert_gpu_idle
assert_storage
mkdir -p "${OUT}"
cp -p src/grounding_evaluator.py "${OUT}/grounding_evaluator.original.py"
restore() {
  cp -p "${OUT}/grounding_evaluator.original.py" src/grounding_evaluator.py
}
trap restore EXIT INT TERM
cp -p "${PATCHED}" src/grounding_evaluator.py

# Deliberately omit --joint_det: eval_train must contain only the 36,665
# ScanRefer descriptions, not the auxiliary ScanNet detection annotations.
CMD=(
  torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --use_soft_token_loss --use_contrastive_align
  --pp_checkpoint "${OFFICIAL_INIT}"
  --butd --self_attend --rng_seed 0 --print_freq 100
  --eval_train --eval_max_samples "${MAX_SAMPLES}"
  --disable_train_augmentation --disable_box_jitter
  --checkpoint_path "${CKPT}" --log_dir "${OUT}"
  --eval_results_json_path "${OUT}/eval_results.json"
  --use_structured_slots --use_sacr --use_rapf
  --use_reliability_gate --use_quality_head
  --rapf_use_quality --rapf_quality_weight 0.25
  --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1
  --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1
  --use_qahnl --qahnl_score_source fused
  --use_detector_policy_adapter
  --detector_policy_adapter_hidden_dim 64
  --detector_policy_adapter_k 5
  --detector_policy_adapter_delta_scale 4.0
  --eval_use_detector_policy_adapter_scores
  --eval_target_cid_source text --verbose_diagnostics
)

NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${OUT}/stage16_e8_train_geometry.pt" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

restore
trap - EXIT INT TERM
"${PYTHON}" - "${OUT}" "${EXPECTED_ROWS}" <<'PY'
import glob
import json
import os
import sys
import torch

out = sys.argv[1]
expected = int(sys.argv[2])
path = os.path.join(out, 'stage16_e8_train_geometry.pt')
payload = torch.load(path, map_location='cpu')
rows = payload['rows']
assert len(rows) == expected, (len(rows), expected)
assert len({int(row['example_id']) for row in rows}) == expected
assert all(row.get('scene_id') for row in rows)
assert all(row.get('adapter_candidate_query') for row in rows)
assert all('adapter_hit50_logit_at_candidate' in row for row in rows)
assert all('adapter_box_at_candidate' in row for row in rows)
assert all('gt_box' in row and 'detected_box' in row for row in rows)
configs = glob.glob(os.path.join(out, 'scanrefer_spacy', '*', 'config.json'))
assert len(configs) == 1, configs
config = json.load(open(configs[0], 'r'))
assert config['eval_train'] is True
assert config['joint_det'] is False
assert config['disable_train_augmentation'] is True
assert config['disable_box_jitter'] is True
print('STAGE24_TRAIN_DUMP_PASS rows={} scenes={}'.format(
    len(rows), len({row['scene_id'] for row in rows})
))
PY
