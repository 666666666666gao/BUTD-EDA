#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage81_raw_query_pca_train/stage16_e8_train_geometry_semantic_rawpca.pt"
OUT="${ROOT}/stage86_three_ranker_blend"

cd "${R}"
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 286; }
"${PYTHON}" "${P}/train_three_ranker_blend.py" train \
  "${DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${P}/train_joint_option_ranker_semantic.py" \
  "${P}/train_joint_option_ranker_semantic_rawpca.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage74_semantic_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage82_rawpca_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage74_semantic_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage82_rawpca_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --preserve-acc025-drop 0.0 \
  2>&1 | tee "${P}/stage86_three_ranker_blend_train.log"
[ -f "${OUT}/locked_three_ranker_blend.json" ]
sha256sum "${OUT}/locked_three_ranker_blend.json" > "${OUT}/sha256.txt"
echo STAGE86_THREE_RANKER_BLEND_LOCKED
