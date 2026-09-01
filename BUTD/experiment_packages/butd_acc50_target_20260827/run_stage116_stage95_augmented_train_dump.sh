#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33916}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${STAGE116_OUT:-${ROOT}/stage116_stage95_e11_augmented_train_geometry_dump}"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_calibrated_train_dump.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${STAGE116_BACKUP:-${P}/state/stage116_stage95_augmented_train_dump_20260830}"
DUMP="${OUT}/stage95_e11_augmented_train_geometry.pt"
STATUS="${STAGE116_STATUS:-${P}/stage116_augmented_train_dump_status.txt}"
EXPECTED_ROWS="${STAGE116_EXPECTED_ROWS:-36665}"
MAX_SAMPLES="${STAGE116_MAX_SAMPLES:--1}"

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUT}"
test ! -e "${BACKUP}"
check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"

CMD=(
 torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
 train_dist_mod.py --num_decoder_layers 6
 --use_color --weight_decay 0.0005
 --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
 --dataset scanrefer_spacy --test_dataset scanrefer_spacy
 --use_soft_token_loss --use_contrastive_align
 --pp_checkpoint "${OFFICIAL_INIT}"
 --butd --self_attend --augment_det --rng_seed 17 --print_freq 100
 --eval_train --eval_max_samples "${MAX_SAMPLES}"
 --checkpoint_path "${CKPT}" --log_dir "${OUT}"
 --eval_results_json_path "${OUT}/eval_results.json"
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
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${OUT}"
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage116.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage116.py"
restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage116.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=116\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_checkpoint=%s\n' "${CKPT}"
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'dump_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_SHA}"
  printf 'augmentation_seed=17\n'
  printf 'expected_rows=%s\n' "${EXPECTED_ROWS}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage116_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage116_stage95_augmented_train_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}" "${EXPECTED_ROWS}" <<'PY'
import glob
import hashlib
import json
import os
import sys
import torch

dump, out, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = torch.load(dump, map_location='cpu')['rows']
assert len(rows) == expected, (len(rows), expected)
required = (
    'scene_id', 'adapter_candidate_query', 'adapter_box_at_candidate',
    'adapter_hit50_logit_at_candidate', 'gt_box', 'detected_box',
)
for key in required:
    assert all(key in row for row in rows), key
scene_count = len({row['scene_id'] for row in rows})
assert scene_count >= (1 if expected < 1000 else 100), scene_count
configs = glob.glob(os.path.join(out, 'scanrefer_spacy', '*', 'config.json'))
assert len(configs) == 1, configs
config = json.load(open(configs[0], encoding='utf-8'))
assert config['eval_train'] is True
assert config['joint_det'] is False
assert config['augment_det'] is True
assert config['disable_train_augmentation'] is False
assert config['disable_box_jitter'] is False
assert config['rng_seed'] == 17
h = hashlib.sha256()
with open(dump, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
receipt = {
    'stage': '116',
    'status': 'complete',
    'rows': len(rows),
    'scene_count': scene_count,
    'dump': dump,
    'dump_sha256': h.hexdigest(),
    'source_checkpoint_sha256': 'f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811',
    'dump_evaluator_sha256': '0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6',
    'augmentation_seed': 17,
}
path = os.path.join(out, 'stage116_receipt.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage116_receipt.json" "${OUT}/launch_manifest.txt"
printf 'stage116_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
