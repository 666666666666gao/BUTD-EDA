#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
export MASTER_PORT="${MASTER_PORT:-33949}"
source "${R}/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

SRC="${ROOT}/stage135c_stage29_option_last_box_noaug_jointmask/scanrefer_spacy/1788089093/ckpt_best_primary.pth"
PARITY="${ROOT}/stage147a_tier_pair_zero_gate_parity_eval/eval_results.json"
OUT="${ROOT}/stage147b_tier_pair_trainonly_k2"
LOG="${P}/stage147b_tier_pair_trainonly.log"
STATUS="${P}/stage147b_tier_pair_trainonly_status.txt"

test ! -e "$OUT"
test -s "$PARITY"
test "$(sha256sum "$SRC" | awk '{print $1}')" = \
  "a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b"
test "$(sha256sum models/detector_policy_sources.py | awk '{print $1}')" = \
  "396303854f5acacba410093a3953ad7f1c56288a29f7422fa671e8aee3684166"
test "$(sha256sum models/bdetr.py | awk '{print $1}')" = \
  "325444d2e7474b40764634f410d0f68827945493c3b1b7a73301a97c9da71879"
test "$(sha256sum models/losses.py | awk '{print $1}')" = \
  "4584dfc9cdbaa83adf2b2e294b9b3d95b37df3bf35866e9fda09df8095019305"
test "$(sha256sum main_utils.py | awk '{print $1}')" = \
  "31cb486e03165ee16e0c08b3dde3029d9360132770a1020fa0ea02c451174fba"
test "$(sha256sum train_dist_mod.py | awk '{print $1}')" = \
  "44852f403849266e5d706b39c86e99f04f3bc682652bf8eb06944e11557a25e0"

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
 --detector_policy_alignment_rescue_loss_weight 0.0
 --detector_policy_tier_pair_rescue_head
 --detector_policy_tier_pair_candidate_k 2
 --detector_policy_tier_pair_override_threshold 0.0
 --detector_policy_tier_pair_loss_weight 1.0
 --detector_policy_adapter_train_only
 --detector_policy_tier_pair_train_only
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
  printf 'stage=147b_tier_pair_trainonly\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s\nsource_epoch=%s\nsource_sha256=%s\n' \
    "$SRC" "$E" "$(sha256sum "$SRC" | awk '{print $1}')"
  printf 'candidate_policy=fixed_current_score_top2\n'
  printf 'tier_policy=tier2_iou_ge_050__tier1_025_to_050__tier0_iou_le_010\n'
  printf 'ambiguous_policy=iou_gt_010_lt_025_ignored\n'
  printf 'inference_threshold=fixed_zero_no_validation_sweep\n'
  printf 'validation_protocol=official_scene_disjoint_scanrefer_val_each_epoch\n'
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "$OUT/launch_manifest.txt"

printf 'stage147b_training %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "$LOG"
printf 'stage147b_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
