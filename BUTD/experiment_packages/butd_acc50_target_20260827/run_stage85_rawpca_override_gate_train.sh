#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage81_raw_query_pca_train/stage16_e8_train_geometry_semantic_rawpca.pt"
OUT="${ROOT}/stage85_rawpca_override_gate"

cd "${R}"
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 285; }
"${PYTHON}" "${P}/train_semantic_override_gate.py" train \
  "${DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${P}/train_joint_option_ranker_semantic_rawpca.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage82_rawpca_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage82_rawpca_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --num-threads 32 --preserve-acc025-drop 0.0 \
  --meta-train-start 85 --meta-train-end 89 \
  --meta-dev-start 90 --meta-dev-end 94 \
  --meta-test-start 95 --meta-test-end 99 \
  2>&1 | tee "${P}/stage85_rawpca_override_gate_train.log"
[ -f "${OUT}/semantic_override_gate.txt" ]
[ -f "${OUT}/locked_semantic_override_gate.json" ]
sha256sum "${OUT}/semantic_override_gate.txt" \
  "${OUT}/locked_semantic_override_gate.json" > "${OUT}/sha256.txt"
echo STAGE85_RAWPCA_OVERRIDE_GATE_LOCKED
