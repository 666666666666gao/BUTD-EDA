#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33923}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

OUT="${ROOT}/stage123_stage95_e11_p09_augmented_train_geometry_dump"
CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED_EVAL="${P}/grounding_evaluator_adapter_calibrated_train_dump.py"
LIVE_EVAL="src/grounding_evaluator.py"
PATCHED_DATA="${P}/joint_det_dataset_stage122.py"
LIVE_DATA="src/joint_det_dataset.py"
BACKUP="${P}/state/stage123_stage95_p09_augmented_train_dump_20260830"
DUMP="${OUT}/stage95_e11_p09_augmented_train_geometry.pt"
STATUS="${P}/stage123_p09_augmented_train_dump_status.txt"
EXPECTED_ROWS=36665
CORRUPTION_PROBABILITY=0.9

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_EVAL_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_EVAL_SHA="0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6"
EXPECTED_LIVE_DATA_SHA="6a7483d719d09433f2b7763f73246b0daf43948741ec3394068471906d96a24a"
EXPECTED_PATCHED_DATA_SHA="5578f8161afcfee2acb6e02de858c3b4e5f278ac709fa6ebdf29ac0323c94c4e"

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
check_sha "${LIVE_EVAL}" "${EXPECTED_LIVE_EVAL_SHA}"
check_sha "${PATCHED_EVAL}" "${EXPECTED_PATCHED_EVAL_SHA}"
check_sha "${LIVE_DATA}" "${EXPECTED_LIVE_DATA_SHA}"
check_sha "${PATCHED_DATA}" "${EXPECTED_PATCHED_DATA_SHA}"

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${OUT}"
cp -p "${LIVE_EVAL}" "${BACKUP}/grounding_evaluator.py"
cp -p "${LIVE_DATA}" "${BACKUP}/joint_det_dataset.py"
chmod 0444 "${BACKUP}/grounding_evaluator.py" "${BACKUP}/joint_det_dataset.py"

restore() {
  install -m 0644 "${BACKUP}/grounding_evaluator.py" "${LIVE_EVAL}"
  install -m 0644 "${BACKUP}/joint_det_dataset.py" "${LIVE_DATA}"
}
trap restore EXIT INT TERM ERR
install -m 0644 "${PATCHED_EVAL}" "${LIVE_EVAL}"
install -m 0644 "${PATCHED_DATA}" "${LIVE_DATA}"
check_sha "${LIVE_EVAL}" "${EXPECTED_PATCHED_EVAL_SHA}"
check_sha "${LIVE_DATA}" "${EXPECTED_PATCHED_DATA_SHA}"

{
  printf 'stage=123\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_checkpoint=%s\n' "${CKPT}"
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'dump_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_EVAL_SHA}"
  printf 'patched_dataset_sha256=%s\n' "${EXPECTED_PATCHED_DATA_SHA}"
  printf 'augmentation_seed=17\n'
  printf 'detector_corruption_probability=%s\n' "${CORRUPTION_PROBABILITY}"
  printf 'expected_rows=%s\n' "${EXPECTED_ROWS}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

CMD=(
  torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --use_soft_token_loss --use_contrastive_align
  --pp_checkpoint "${OFFICIAL_INIT}"
  --butd --self_attend --augment_det --rng_seed 17 --print_freq 100
  --eval_train --eval_max_samples -1
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

printf 'stage123_dumping %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
BUTD_AUGMENT_DET_CORRUPT_PROB="${CORRUPTION_PROBABILITY}" \
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "${P}/stage123_stage95_p09_augmented_train_dump.log"

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE_EVAL}" "${EXPECTED_LIVE_EVAL_SHA}"
check_sha "${LIVE_DATA}" "${EXPECTED_LIVE_DATA_SHA}"

"${PYTHON}" - "${DUMP}" "${OUT}" "${EXPECTED_ROWS}" <<'PY'
import glob
import hashlib
import json
import os
import sys
import torch

dump, out, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = torch.load(dump, map_location="cpu")["rows"]
assert len(rows) == expected, (len(rows), expected)
required = (
    "scene_id", "adapter_candidate_query", "adapter_box_at_candidate",
    "adapter_hit50_logit_at_candidate", "gt_box", "detected_box",
)
for key in required:
    assert all(key in row for row in rows), key
scene_count = len({row["scene_id"] for row in rows})
assert scene_count >= 100, scene_count
configs = glob.glob(os.path.join(out, "scanrefer_spacy", "*", "config.json"))
assert len(configs) == 1, configs
config = json.load(open(configs[0], encoding="utf-8"))
assert config["eval_train"] is True
assert config["joint_det"] is False
assert config["augment_det"] is True
assert config["disable_train_augmentation"] is False
assert config["disable_box_jitter"] is False
assert config["rng_seed"] == 17
digest = hashlib.sha256()
with open(dump, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
receipt = {
    "stage": "123",
    "status": "complete",
    "rows": len(rows),
    "scene_count": scene_count,
    "dump": dump,
    "dump_sha256": digest.hexdigest(),
    "source_checkpoint_sha256": "f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811",
    "dump_evaluator_sha256": "0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6",
    "patched_dataset_sha256": "5578f8161afcfee2acb6e02de858c3b4e5f278ac709fa6ebdf29ac0323c94c4e",
    "augmentation_seed": 17,
    "detector_corruption_probability": 0.9,
}
path = os.path.join(out, "stage123_receipt.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage123_receipt.json" "${OUT}/launch_manifest.txt"
printf 'stage123_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
