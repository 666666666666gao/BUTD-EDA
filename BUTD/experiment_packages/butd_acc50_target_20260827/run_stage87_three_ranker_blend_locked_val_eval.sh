#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage83_raw_query_pca_val/stage16_e8_val_geometry_semantic_rawpca.pt"
LOCK="${ROOT}/stage86_three_ranker_blend/locked_three_ranker_blend.json"
OUT="${ROOT}/stage87_three_ranker_blend_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ] && [ -f "${LOCK}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 287; }
sha256sum "${LOCK}" > "${ROOT}/stage87_lock_sha256_before_val.txt"
"${PYTHON}" "${P}/train_three_ranker_blend.py" evaluate \
  "${DUMP}" "${LOCK}" "${OUT}" \
  2>&1 | tee "${P}/stage87_three_ranker_blend_locked_val_eval.log"
