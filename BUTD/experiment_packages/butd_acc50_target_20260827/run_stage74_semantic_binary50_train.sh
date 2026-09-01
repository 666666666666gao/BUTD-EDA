#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage73_stage16_e8_train_geometry_semantic_dump/stage16_e8_train_geometry_semantic.pt"
OUT="${ROOT}/stage74_semantic_binary50_ranker"

cd "${R}"
[ -f "${TRAIN_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 274; }

"${PYTHON}" "${P}/train_joint_option_ranker_semantic.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker_semantic.py" binary50-train \
  "${TRAIN_DUMP}" "${OUT}" --max-candidates 8 --num-threads 32 \
  2>&1 | tee "${P}/stage74_semantic_binary50_train.log"

[ -f "${OUT}/binary50_option_ranker.txt" ]
[ -f "${OUT}/locked_binary50_policy.json" ]
sha256sum "${OUT}/binary50_option_ranker.txt" \
  "${OUT}/locked_binary50_policy.json" \
  > "${OUT}/sha256.txt"
echo STAGE74_SEMANTIC_POLICY_LOCKED
