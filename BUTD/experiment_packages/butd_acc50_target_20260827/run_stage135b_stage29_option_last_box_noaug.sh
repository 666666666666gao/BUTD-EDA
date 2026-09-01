#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

SRC="${ROOT}/stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth"
MAP_DIR="${ROOT}/stage134b_stage29_query_map_exampleid"
MAP="${MAP_DIR}/stage29_query_action_map_v2.pt"
MAP_SUMMARY="${MAP_DIR}/stage134b_example_id_map_receipt.json"
OUT="${ROOT}/stage135b_stage29_option_last_box_noaug_exampleid"
STATUS="${P}/stage135b_option_last_box_status.txt"

EXPECTED_SRC_SHA="f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811"
EXPECTED_MAIN_SHA="11c36148dd2f3188cce64ee37f46bba564f6777dfc4d19759af5c345ea6617f4"
EXPECTED_LOSS_SHA="376c6412adf5f6fcf823c797fcb423114a9ab0a88e6af9e37db889d98f0928d7"
EXPECTED_DATASET_SHA="5f2da69539be82e90aba64f2c7ab6ddc6551af6fb15d3df2db77aaacdae67d3a"
EXPECTED_COMMON_SHA="b95e3d433f94010230cf77d5409992487e1dac8eafa1b947186a043ca8dcdbdc"

check_sha() {
  local path="$1" expected="$2"
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_sha "$SRC" "$EXPECTED_SRC_SHA"
check_sha main_utils.py "$EXPECTED_MAIN_SHA"
check_sha models/losses.py "$EXPECTED_LOSS_SHA"
check_sha src/joint_det_dataset.py "$EXPECTED_DATASET_SHA"
check_sha experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh "$EXPECTED_COMMON_SHA"
test -s "$MAP" && test -s "$MAP_SUMMARY"
MAP_SHA="$(${PYTHON} - "$MAP" "$MAP_SUMMARY" <<'PY'
import hashlib
import json
import sys

path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
assert summary['stage'] == '134b_example_id_map'
assert summary['rows'] == summary['entries_by_example_id'] == 36665
h = hashlib.sha256()
with open(path, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
actual = h.hexdigest()
assert actual == summary['v2_map_sha256']
print(actual)
PY
)"
E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "$SRC")"
test "$E" = 11

base_command
CMD+=(--log_dir "$OUT" --max_epoch "$((E+1))" --val_freq 1 --save_freq 1000
 --checkpoint_path "$SRC" --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.50_top1 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0.4241
 --best_checkpoint_constraint_epsilon 0
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
 --last_box_head_train_only --last_box_head_lr 0.00001
 --last_box_standard_loss_scale 1.0
 --last_box_target_loss_weight 1.0
 --last_box_target_score_source detector_policy_adapter
 --last_box_target_query_map "$MAP"
 --last_box_target_query_map_mode option
 --last_box_target_iou_min 0.25 --last_box_target_iou_max 0.50
 --last_box_target_l1_weight 1.0 --last_box_target_giou_weight 1.0
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
test ! -e "$OUT"
mkdir -p "$OUT"
{
  printf 'stage=135b\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s\nsource_epoch=%s\nsource_sha256=%s\n' "$SRC" "$E" "$EXPECTED_SRC_SHA"
  printf 'query_map=%s\nquery_map_sha256=%s\n' "$MAP" "$MAP_SHA"
  printf 'main_utils_sha256=%s\nlosses_sha256=%s\ndataset_sha256=%s\n' "$EXPECTED_MAIN_SHA" "$EXPECTED_LOSS_SHA" "$EXPECTED_DATASET_SHA"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage135b_training %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" \
  2>&1 | tee "$P/stage135b_option_last_box_train.log"
printf 'stage135b_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
