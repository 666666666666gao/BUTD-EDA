#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage73_stage16_e8_train_geometry_semantic_dump/stage16_e8_train_geometry_semantic.pt"
OUT="${ROOT}/stage78_semantic_override_gate"

cd "${R}"
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 278; }
"${PYTHON}" "${P}/train_semantic_override_gate.py" train \
  "${DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${P}/train_joint_option_ranker_semantic.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage74_semantic_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage74_semantic_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --num-threads 32 --preserve-acc025-drop 0.0 \
  2>&1 | tee "${P}/stage78_semantic_override_gate_train.log"
[ -f "${OUT}/semantic_override_gate.txt" ]
[ -f "${OUT}/locked_semantic_override_gate.json" ]
sha256sum "${OUT}/semantic_override_gate.txt" \
  "${OUT}/locked_semantic_override_gate.json" > "${OUT}/sha256.txt"
echo STAGE78_SEMANTIC_OVERRIDE_GATE_LOCKED
