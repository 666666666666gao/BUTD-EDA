#!/usr/bin/env bash
set -Eeuo pipefail

R='/home/gb/new butd/butd_detr-main'
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT=/root/autodl-tmp/logs/butd_acc50_target_20260827
export MASTER_PORT="${MASTER_PORT:-33941}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump"
CKPT="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
PATCHED="${P}/grounding_evaluator_adapter_dump_parityfixed.py"
LIVE=src/grounding_evaluator.py
BACKUP="${P}/state/stage141_stage135c_raw_train_dump_20260830"
RAW_DUMP="${OUT}/stage135c_e12_raw_train_geometry.pt"
CORRECTED_DUMP="${OUT}/stage135c_e12_raw_train_geometry_with_ids.pt"
ID_SOURCE="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
REPAIR_RECEIPT="${OUT}/stable_id_join_receipt.json"
STATUS="${P}/stage141_raw_train_dump_status.txt"
EXPECTED_ROWS=36665

EXPECTED_CKPT_SHA=a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b
EXPECTED_MAIN_SHA=11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4
EXPECTED_LOSS_SHA=a95443a958c0170faac513c46001e8c60f46cfaacd559b620c79eb622ec2c852
EXPECTED_DATASET_SHA=5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a
EXPECTED_COMMON_SHA=b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc
EXPECTED_LIVE_SHA=50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86
EXPECTED_PATCHED_SHA=cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f
EXPECTED_REPAIR_SHA=bc1433b9ae42cd2e55415f723bd2cbdd32edcbb7a0f0d7d9f54f97dc88e7fb7b
EXPECTED_ID_SOURCE_SHA=6df4cbd9a2177e1470ecf2209d614a12cd66e51d89e6c0cc20745b70a1bf70fb

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
check_sha src/joint_det_dataset.py "${EXPECTED_DATASET_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"
check_sha "${PATCHED}" "${EXPECTED_PATCHED_SHA}"
check_sha "${P}/repair_stage133_stable_ids.py" "${EXPECTED_REPAIR_SHA}"
check_sha "${ID_SOURCE}" "${EXPECTED_ID_SOURCE_SHA}"

CMD=(
 torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
 train_dist_mod.py --num_decoder_layers 6
 --use_color --weight_decay 0.0005
 --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
 --dataset scanrefer_spacy --test_dataset scanrefer_spacy
 --use_soft_token_loss --use_contrastive_align
 --pp_checkpoint "${OFFICIAL_INIT}"
 --butd --self_attend --rng_seed 0 --print_freq 100
 --eval_train --eval_max_samples -1
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

if [[ "${DRY_RUN:-0}" = 1 ]]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${OUT}"
cp -p "${LIVE}" "${BACKUP}/grounding_evaluator.pre_stage141.py"
chmod 0444 "${BACKUP}/grounding_evaluator.pre_stage141.py"
restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.pre_stage141.py" "${LIVE}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED}" "${LIVE}"
check_sha "${LIVE}" "${EXPECTED_PATCHED_SHA}"

{
  printf 'stage=141_stage135c_raw_train_dump\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_checkpoint=%s\n' "${CKPT}"
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'dump_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_SHA}"
  printf 'dataset_sha256=%s\n' "${EXPECTED_DATASET_SHA}"
  printf 'expected_rows=%s\n' "${EXPECTED_ROWS}"
  printf 'augmentation=disabled\nbox_jitter=disabled\n'
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

printf 'stage141_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${RAW_DUMP}" \
CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" \
  2>&1 | tee "${P}/stage141_stage135c_raw_train_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE}" "${EXPECTED_LIVE_SHA}"

"${PYTHON}" - "${RAW_DUMP}" "${OUT}" "${EXPECTED_ROWS}" <<'PY'
import hashlib
import json
import os
import sys
import torch

dump, out, expected_text = sys.argv[1:]
expected = int(expected_text)
payload = torch.load(dump, map_location='cpu')
rows = payload['rows']
assert len(rows) == expected, (len(rows), expected)
required = (
    'example_id', 'adapter_candidate_query', 'adapter_box_at_candidate',
    'adapter_hit50_logit_at_candidate', 'adapter_hit25_logit_at_candidate',
    'adapter_fused_at_candidate', 'adapter_rescue_logit_at_candidate',
    'adapter_score_at_candidate', 'adapter_delta_at_candidate',
    'gt_box', 'detected_box',
)
for key in required:
    assert all(key in row for row in rows), key
ids = [int(row['example_id']) for row in rows]
assert ids == list(range(expected)), (ids[:3], ids[-3:])
for key in ('scene_id', 'object_id', 'ann_id'):
    assert not any(key in row for row in rows), key
h = hashlib.sha256()
with open(dump, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(chunk)
receipt = {
    'stage': '141_raw_dump',
    'status': 'complete',
    'rows': len(rows),
    'example_id_min': min(ids),
    'example_id_max': max(ids),
    'example_id_exact_order': True,
    'raw_dump': os.path.abspath(dump),
    'raw_dump_sha256': h.hexdigest(),
}
path = os.path.join(out, 'stage141_raw_receipt.json')
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

"${PYTHON}" "${P}/repair_stage133_stable_ids.py" \
  "${RAW_DUMP}" "${ID_SOURCE}" "${CORRECTED_DUMP}" "${REPAIR_RECEIPT}"

"${PYTHON}" - \
  "${OUT}" "${CKPT}" "${EXPECTED_CKPT_SHA}" \
  "${CORRECTED_DUMP}" "${REPAIR_RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys
import torch

out, checkpoint, checkpoint_sha, corrected, repair_receipt = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
payload = torch.load(corrected, map_location='cpu')
rows = payload['rows']
keys = [
    (str(row['scene_id']), str(row['object_id']), str(row['ann_id']))
    for row in rows
]
assert len(rows) == 36665
assert len(set(keys)) == 36665
repair = json.load(open(repair_receipt, encoding='utf-8'))
assert repair['status'] == 'complete'
receipt = {
    'stage': '141_stage135c_train_geometry_with_ids',
    'status': 'complete',
    'rows': len(rows),
    'unique_stable_keys': len(set(keys)),
    'source_checkpoint': checkpoint,
    'source_checkpoint_sha256': checkpoint_sha,
    'corrected_dump': os.path.abspath(corrected),
    'corrected_dump_sha256': sha(corrected),
    'stable_id_join_receipt': os.path.abspath(repair_receipt),
    'stable_id_join_receipt_sha256': sha(repair_receipt),
    'validation_labels_used': False,
}
path = os.path.join(out, 'stage141_receipt.json')
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUT}/launch_manifest.txt"
chmod 0444 \
  "${OUT}/launch_manifest.txt" \
  "${OUT}/stage141_raw_receipt.json" \
  "${REPAIR_RECEIPT}" \
  "${OUT}/stage141_receipt.json"
printf 'stage141_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
