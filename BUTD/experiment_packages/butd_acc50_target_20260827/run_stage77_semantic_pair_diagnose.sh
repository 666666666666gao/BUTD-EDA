#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage75_stage16_e8_val_geometry_semantic_dump/stage16_e8_val_geometry_semantic_frozen_legacy.pt"
OUT="${ROOT}/stage77_stage29_semantic_pair_diagnostic.json"

cd "${R}"
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 277; }
"${PYTHON}" "${P}/train_semantic_override_gate.py" diagnose \
  "${DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${P}/train_joint_option_ranker_semantic.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage74_semantic_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage74_semantic_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" 2>&1 | tee "${P}/stage77_semantic_pair_diagnose.log"
