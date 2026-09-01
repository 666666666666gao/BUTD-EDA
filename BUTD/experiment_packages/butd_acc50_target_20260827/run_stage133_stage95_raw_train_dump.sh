#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33933}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${STAGE133_OUT:-${ROOT}/stage133_stage95_e11_raw_train_geometry_dump}"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
LIVE="src/grounding_evaluator.py"
BACKUP="${STAGE133_BACKUP:-${P}/state/stage133_stage95_raw_train_dump_20260830}"
DUMP="${OUT}/stage95_e11_raw_train_geometry.pt"
STATUS="${P}/stage133_raw_train_dump_status.txt"
EXPECTED_ROWS="${STAGE133_EXPECTED_ROWS:-36665}"
MAX_SAMPLES="${STAGE133_MAX_SAMPLES:--1}"

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="afcb88f2a9a268bf270ee71b0901a18876e732a08749a79b796b6262b9f30ba8"
EXPECTED_LOSS_SHA="acb1eed32c1f16a3696b3d1b9dab13abf94dc5aecf7681be8e670fa26286a60a"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_SHA="cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f"

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
 --butd --self_attend --rng_seed 0 --print_freq 100
 --eval_train --eval_max_samples "${MAX_SAMPLES}"
 --checkpoint_path "${CKPT}" --log_dir "${OUT}"
 --eval_results_json_path "${OUT}/eval_results.json"
 --disable_train_augmentation --disable_box_jitter
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
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage133.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage133.py"
restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage133.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=133\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_checkpoint=%s\n' "${CKPT}"
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'dump_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_SHA}"
  printf 'expected_rows=%s\n' "${EXPECTED_ROWS}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage133_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage133_stage95_raw_train_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}" "${EXPECTED_ROWS}" <<'PY'
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
    'scene_id', 'object_id', 'ann_id', 'adapter_candidate_query',
    'adapter_box_at_candidate', 'adapter_hit50_logit_at_candidate',
    'gt_box', 'detected_box',
)
for key in required:
    assert all(key in row for row in rows), key
keys = [
    (str(row['scene_id']), str(row['object_id']), str(row['ann_id']))
    for row in rows
]
assert len(set(keys)) == len(keys), (len(set(keys)), len(keys))
scene_count = len({key[0] for key in keys})
assert scene_count >= (1 if expected < 1000 else 100), scene_count
h = hashlib.sha256()
with open(dump, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
receipt = {
    'stage': '133',
    'status': 'complete',
    'rows': len(rows),
    'unique_keys': len(set(keys)),
    'scene_count': scene_count,
    'dump': dump,
    'dump_sha256': h.hexdigest(),
    'source_checkpoint_sha256': 'f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811',
    'dump_evaluator_sha256': 'cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f',
}
path = os.path.join(out, 'stage133_receipt.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage133_receipt.json" "${OUT}/launch_manifest.txt"
printf 'stage133_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
