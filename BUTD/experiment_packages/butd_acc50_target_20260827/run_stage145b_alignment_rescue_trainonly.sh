#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33946}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

SRC="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
PARITY="${ROOT}/stage145a_alignment_rescue_zero_gate_parity_eval/eval_results.json"
OUT="${ROOT}/stage145b_alignment_rescue_trainonly_k16"
LOG="${P}/stage145b_alignment_rescue_trainonly.log"
STATUS="${P}/stage145b_alignment_rescue_trainonly_status.txt"

test ! -e "$OUT"
test -s "$PARITY"
test "$(sha256sum "$SRC" | awk '{print $1}')" = \
  "a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
test "$(sha256sum models/detector_policy_sources.py | awk '{print $1}')" = \
  "da33ab79e20fbb2e35858a7da9729979609878588df20da6754dc61c10c54ff2"
test "$(sha256sum models/bdetr.py | awk '{print $1}')" = \
  "00083a5498de8a1aa8442b046a849159b3c643db24395ee1d8c25eab2259f233"
test "$(sha256sum models/losses.py | awk '{print $1}')" = \
  "b84d6cad8778877dd2a58c818736376ccc5cb553cfac308f87ee493c31c8e930"
test "$(sha256sum main_utils.py | awk '{print $1}')" = \
  "7de08db9b8826af8a62a32d9b331a704a57ced6da583c89646c31a46706e81b4"
test "$(sha256sum train_dist_mod.py | awk '{print $1}')" = \
  "96686a7987d0bac3dcea39bfcfbb3cad0354a3cf4eb937436cd3cdc67a1ae86d"

"${PYTHON}" - "$PARITY" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding='utf-8'))
assert d['last__bbs_acc0.25_top1'] == 0.548485485906605
assert d['last__bbs_acc0.50_top1'] == 0.41607067732435843
PY

E="$(${PYTHON} -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu")["epoch"]))' "$SRC")"
test "$E" = 12

base_command
CMD+=(--log_dir "$OUT" --max_epoch 18 --val_freq 1 --save_freq 1000
 --checkpoint_path "$SRC" --best_checkpoint_only
 --best_checkpoint_metric last__bbs_acc0.50_top1
 --best_checkpoint_min_delta 0
 --best_checkpoint_constraint_lower 0 0 0 0 0.5391 0
 --best_checkpoint_constraint_epsilon 0
 --early_stopping
 --early_stopping_metric last__bbs_acc0.50_top1
 --early_stopping_min_epoch 13 --early_stopping_patience 2
 --early_stopping_min_delta 0.0003
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
 --detector_policy_alignment_rescue_head
 --detector_policy_alignment_candidate_k 16
 --detector_policy_alignment_override_threshold 0.0
 --detector_policy_alignment_rescue_loss_weight 1.0
 --detector_policy_adapter_train_only
 --detector_policy_alignment_rescue_train_only
 --detector_policy_adapter_lr 0.001
 --detector_policy_adapter_margin 0.1
 --detector_policy_adapter_min_iou_gap 0.02
 --eval_use_detector_policy_adapter_scores --eval_target_cid_source text
 --verbose_diagnostics)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

assert_gpu_idle
assert_storage
mkdir -p "$OUT"
{
  printf 'stage=145b_alignment_rescue_trainonly\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s\nsource_epoch=%s\nsource_sha256=%s\n' \
    "$SRC" "$E" "$(sha256sum "$SRC" | awk '{print $1}')"
  printf 'candidate_policy=hard_target_class_union_target_confidence_top16\n'
  printf 'validation_protocol=official_scene_disjoint_scanrefer_val_each_epoch\n'
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage145b_training %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "$LOG"
printf 'stage145b_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
