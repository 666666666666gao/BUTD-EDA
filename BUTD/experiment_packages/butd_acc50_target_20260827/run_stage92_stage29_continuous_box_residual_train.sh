#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage73_stage16_e8_train_geometry_semantic_dump/stage16_e8_train_geometry_semantic.pt"
OUT="${ROOT}/stage92_stage29_continuous_box_residual"
LOG="${P}/stage92_stage29_continuous_box_residual_train.log"

cd "${R}"
[ -f "${DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 292; }
avail_kb="$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
[ "${avail_kb}" -ge 2097152 ] || {
  echo "insufficient /root/autodl-tmp free space: ${avail_kb} KiB" >&2
  exit 192
}
"${PYTHON}" -u "${P}/train_stage29_continuous_box_residual.py" train \
  "${DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --device cuda --hidden-dim 128 --lr 0.001 \
  --epochs 80 --patience 15 --batch-size 512 \
  2>&1 | tee "${LOG}"
[ -f "${OUT}/box_residual_model.pt" ]
[ -f "${OUT}/locked_box_residual_policy.json" ]
[ -f "${OUT}/internal_evidence.npz" ]
sha256sum \
  "${OUT}/box_residual_model.pt" \
  "${OUT}/locked_box_residual_policy.json" \
  "${OUT}/internal_evidence.npz" > "${OUT}/sha256.txt"
"${PYTHON}" - "${OUT}/locked_box_residual_policy.json" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding='utf-8'))
print('STAGE92_INTERNAL_NET', {
    'acc025': lock['test_net025'], 'acc050': lock['test_net050']
})
print('STAGE92_EXTERNAL_EVAL_WORTHY', lock['external_eval_worthy'])
PY
echo STAGE92_CONTINUOUS_BOX_RESIDUAL_LOCKED
