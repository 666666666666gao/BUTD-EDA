#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage83_raw_query_pca_val/stage16_e8_val_geometry_semantic_rawpca.pt"
MODEL="${ROOT}/stage82_rawpca_binary50_ranker/binary50_option_ranker.txt"
LOCK="${ROOT}/stage82_rawpca_binary50_ranker/locked_binary50_policy.json"
OUT="${ROOT}/stage84_rawpca_binary50_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ] && [ -f "${MODEL}" ] && [ -f "${LOCK}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 284; }
sha256sum "${MODEL}" "${LOCK}" > "${ROOT}/stage84_lock_sha256_before_val.txt"
"${PYTHON}" "${P}/train_joint_option_ranker_semantic_rawpca.py" evaluate \
  "${DUMP}" "${MODEL}" "${LOCK}" "${OUT}" \
  2>&1 | tee "${P}/stage84_rawpca_binary50_locked_val_eval.log"
