#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33922}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "${R}"

CKPT="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
PATCHED_EVAL="${P}/grounding_evaluator_adapter_calibrated_train_dump.py"
LIVE_EVAL="src/grounding_evaluator.py"
PATCHED_DATA="${P}/joint_det_dataset_stage122.py"
LIVE_DATA="src/joint_det_dataset.py"
ANALYZER="${P}/stage122_summarize_corruption_smoke.py"
OUT_ROOT="${ROOT}/stage122_corruption_smoke_sweep_n2400"
BACKUP="${P}/state/stage122_corruption_smoke_sweep_20260830"
STATUS="${P}/stage122_corruption_smoke_sweep_status.txt"
MAX_SAMPLES=2400

EXPECTED_CKPT_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="5cb0a1d1bdb805b8fa3c997ec263bba10024d988473fcc82b9109c2fcc4d3ff7"
EXPECTED_LOSS_SHA="1ee7550a835b0e4af179ac74cd2f75853a607c037f4087fbb600b16ab076cf44"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"
EXPECTED_LIVE_EVAL_SHA="50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
EXPECTED_PATCHED_EVAL_SHA="0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6"
EXPECTED_LIVE_DATA_SHA="6a7483d719d09433f2b7763f73246b0daf43948741ec3394068471906d96a24a"
EXPECTED_PATCHED_DATA_SHA="5578f8161afcfee2acb6e02de858c3b4e5f278ac709fa6ebdf29ac0323c94c4e"
EXPECTED_ANALYZER_SHA="4c8b78671346c73240dee3220004b0a281e896b1b9e647f0816c675d7d0feb17"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUT_ROOT}"
test ! -e "${BACKUP}"
check_sha "${CKPT}" "${EXPECTED_CKPT_SHA}"
check_sha main_utils.py "${EXPECTED_MAIN_SHA}"
check_sha models/losses.py "${EXPECTED_LOSS_SHA}"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "${EXPECTED_COMMON_SHA}"
check_sha "${LIVE_EVAL}" "${EXPECTED_LIVE_EVAL_SHA}"
check_sha "${PATCHED_EVAL}" "${EXPECTED_PATCHED_EVAL_SHA}"
check_sha "${LIVE_DATA}" "${EXPECTED_LIVE_DATA_SHA}"
check_sha "${PATCHED_DATA}" "${EXPECTED_PATCHED_DATA_SHA}"
check_sha "${ANALYZER}" "${EXPECTED_ANALYZER_SHA}"

assert_gpu_idle
assert_storage
mkdir -p "${BACKUP}" "${OUT_ROOT}"
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

printf 'stage122_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
REFERENCE="${ROOT}/stage116_stage95_e11_augmented_train_geometry_dump/stage95_e11_augmented_train_geometry.pt"
/root/miniconda3/envs/bdetr/bin/python "${ANALYZER}" \
  "${P}" "${REFERENCE}" "${OUT_ROOT}/p03_reference_n2400.json" 0.3 "${MAX_SAMPLES}"

for probability in 0.5 0.7 0.9; do
  label="${probability/./}"
  OUT="${OUT_ROOT}/p${label}_n${MAX_SAMPLES}"
  DUMP="${OUT}/stage95_e11_augmented_train_geometry.pt"
  mkdir -p "${OUT}"
  {
    printf 'stage=122\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'corruption_probability=%s\n' "${probability}"
    printf 'max_samples=%s\n' "${MAX_SAMPLES}"
    printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
    printf 'patched_dataset_sha256=%s\n' "${EXPECTED_PATCHED_DATA_SHA}"
    printf 'patched_evaluator_sha256=%s\n' "${EXPECTED_PATCHED_EVAL_SHA}"
  } > "${OUT}/launch_manifest.txt"

  CMD=(
    torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}"
    train_dist_mod.py --num_decoder_layers 6
    --use_color --weight_decay 0.0005
    --data_root "${DATA_ROOT}" --batch_size 24 --num_workers 8
    --dataset scanrefer_spacy --test_dataset scanrefer_spacy
    --use_soft_token_loss --use_contrastive_align
    --pp_checkpoint "${OFFICIAL_INIT}"
    --butd --self_attend --augment_det --rng_seed 17 --print_freq 50
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

  BUTD_AUGMENT_DET_CORRUPT_PROB="${probability}" \
  NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="${DUMP}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
    2>&1 | tee "${P}/stage122_corruption_p${label}_n${MAX_SAMPLES}.log"

  /root/miniconda3/envs/bdetr/bin/python - "${DUMP}" "${MAX_SAMPLES}" <<'PY'
import sys
import torch

dump, expected_text = sys.argv[1:]
rows = torch.load(dump, map_location="cpu")["rows"]
assert len(rows) == int(expected_text), (len(rows), expected_text)
PY
  /root/miniconda3/envs/bdetr/bin/python "${ANALYZER}" \
    "${P}" "${DUMP}" "${OUT_ROOT}/p${label}_n${MAX_SAMPLES}.json" \
    "${probability}" "${MAX_SAMPLES}"
done

restore
trap - EXIT INT TERM ERR
check_sha "${LIVE_EVAL}" "${EXPECTED_LIVE_EVAL_SHA}"
check_sha "${LIVE_DATA}" "${EXPECTED_LIVE_DATA_SHA}"

/root/miniconda3/envs/bdetr/bin/python - "${OUT_ROOT}" <<'PY'
import glob
import hashlib
import json
import os
import sys

out = sys.argv[1]
paths = sorted(glob.glob(os.path.join(out, "p*.json")))
assert len(paths) == 4, paths
rows = [json.load(open(path, encoding="utf-8")) for path in paths]
summary = {
    "stage": "122",
    "status": "complete",
    "diagnostic_only": True,
    "sample_count_per_probability": 2400,
    "results": rows,
}
path = os.path.join(out, "stage122_summary.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT_ROOT}/stage122_summary.json"
printf 'stage122_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
