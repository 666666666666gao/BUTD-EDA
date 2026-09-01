#!/usr/bin/env bash
set -euo pipefail

R='/home/gb/new butd/butd_detr-main'
P="$R/experiment_packages/butd_acc50_target_20260827"
source "$R/experiment_packages/scanrefer_monotonic_main_ablations_20260825/common.sh"
cd "$R"

ROOT='/root/autodl-tmp/logs/butd_acc50_target_20260827'
CKPT="$ROOT/stage18_geometry_action_head/scanrefer_spacy/1787892530/ckpt_best_primary.pth"
OUT="$ROOT/stage44_stage18_e8_train_geometry_dump"
DUMP="$OUT/stage18_e8_train_geometry.pt"
PATCHED="$P/grounding_evaluator_joint_train_dump.py"
LIVE='src/grounding_evaluator.py'
RANK_OUT="$ROOT/stage45_stage18_binary50_ranker"
VAL_DUMP="$ROOT/stage42_stage18_e8_geometry_dump/stage18_e8_geometry.pt"
RESULT="$ROOT/stage46_stage18_binary50_locked_val_eval.json"
STATUS="$P/stage44_chain_status.txt"

for path in "$CKPT" "$PATCHED" "$LIVE" "$VAL_DUMP"; do
  test -s "$path"
done
test ! -e "$OUT"
test ! -e "$RANK_OUT"
test ! -e "$RESULT"
assert_gpu_idle
assert_storage

mkdir -p "$OUT"
cp -p "$LIVE" "$OUT/grounding_evaluator.original.py"
LIVE_HASH_BEFORE="$(sha256sum "$LIVE" | awk '{print $1}')"
restore() {
  cp -p "$OUT/grounding_evaluator.original.py" "$LIVE"
}
trap restore EXIT INT TERM
cp -p "$PATCHED" "$LIVE"

printf 'stage44_dumping %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
CMD=(
  torchrun --nproc_per_node 1 --master_port 33344
  train_dist_mod.py --num_decoder_layers 6
  --use_color --weight_decay 0.0005
  --data_root "$DATA_ROOT" --batch_size 24 --num_workers 8
  --dataset scanrefer_spacy --test_dataset scanrefer_spacy
  --use_soft_token_loss --use_contrastive_align
  --pp_checkpoint "$OFFICIAL_INIT"
  --butd --self_attend --rng_seed 0 --print_freq 100
  --eval_train --eval_max_samples -1
  --disable_train_augmentation --disable_box_jitter
  --checkpoint_path "$CKPT" --log_dir "$OUT"
  --eval_results_json_path "$OUT/eval_results.json"
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
NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH="$DUMP" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${CMD[@]}" \
  2>&1 | tee "$P/stage44_stage18_train_geometry_dump.log"

restore
trap - EXIT INT TERM
LIVE_HASH_AFTER="$(sha256sum "$LIVE" | awk '{print $1}')"
test "$LIVE_HASH_BEFORE" = "$LIVE_HASH_AFTER"

"$PYTHON" - "$OUT" "$DUMP" "$CKPT" <<'PY'
import glob, json, os, sys, torch
out, dump, checkpoint = sys.argv[1:]
rows = torch.load(dump, map_location='cpu')['rows']
assert len(rows) == 36665, len(rows)
assert len({int(row['example_id']) for row in rows}) == 36665
assert len({row['scene_id'] for row in rows}) == 562
assert all(row.get('adapter_candidate_query') for row in rows)
assert all('adapter_hit50_logit_at_candidate' in row for row in rows)
assert all('adapter_box_at_candidate' in row for row in rows)
assert all('gt_box' in row and 'detected_box' in row for row in rows)
configs = glob.glob(os.path.join(out, 'scanrefer_spacy', '*', 'config.json'))
assert len(configs) == 1, configs
config = json.load(open(configs[0], encoding='utf-8'))
assert config['eval_train'] is True
assert config['joint_det'] is False
assert config['disable_train_augmentation'] is True
assert config['disable_box_jitter'] is True
assert os.path.realpath(config['checkpoint_path']) == os.path.realpath(checkpoint)
print('STAGE44_STAGE18_TRAIN_DUMP_PASS rows={} scenes={} epoch={} bytes={}'.format(
    len(rows), len({row['scene_id'] for row in rows}),
    torch.load(checkpoint, map_location='cpu')['epoch'], os.path.getsize(dump)))
PY

printf 'stage45_training %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" "$P/train_joint_option_ranker.py" self-test
"$PYTHON" -u "$P/train_joint_option_ranker.py" binary50-train \
  "$DUMP" "$RANK_OUT" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "$P/stage45_stage18_binary50_train.log"

MODEL="$RANK_OUT/binary50_option_ranker.txt"
LOCK="$RANK_OUT/locked_binary50_policy.json"
test -s "$MODEL" && test -s "$LOCK"
sha256sum "$LOCK" > "$ROOT/stage45_lock_sha256_before_val.txt"

printf 'stage46_evaluating %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u "$P/train_joint_option_ranker.py" evaluate \
  "$VAL_DUMP" "$MODEL" "$LOCK" "$RESULT" \
  2>&1 | tee "$P/stage46_stage18_binary50_locked_val_eval.log"

test -s "$RESULT"
"$PYTHON" - "$RESULT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
s = d['selected']
print('STAGE46_STAGE18_BINARY50 acc025={:.10f} acc050={:.10f} offline_goal={}'.format(
    s['acc025'], s['acc050'], d['goal_achieved_offline']))
PY
printf 'stage46_complete %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
