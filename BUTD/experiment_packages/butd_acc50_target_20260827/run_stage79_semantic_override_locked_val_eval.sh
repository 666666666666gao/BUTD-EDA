#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage75_stage16_e8_val_geometry_semantic_dump/stage16_e8_val_geometry_semantic_frozen_legacy.pt"
MODEL="${ROOT}/stage78_semantic_override_gate/semantic_override_gate.txt"
LOCK="${ROOT}/stage78_semantic_override_gate/locked_semantic_override_gate.json"
OUT="${ROOT}/stage79_semantic_override_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ] && [ -f "${MODEL}" ] && [ -f "${LOCK}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 279; }
sha256sum "${MODEL}" "${LOCK}" > "${ROOT}/stage79_lock_sha256_before_val.txt"
"${PYTHON}" "${P}/train_semantic_override_gate.py" evaluate \
  "${DUMP}" "${MODEL}" "${LOCK}" "${OUT}" \
  2>&1 | tee "${P}/stage79_semantic_override_locked_val_eval.log"
