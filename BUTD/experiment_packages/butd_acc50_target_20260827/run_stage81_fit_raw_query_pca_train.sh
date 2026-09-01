#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
INPUT="${ROOT}/stage73_stage16_e8_train_geometry_semantic_dump/stage16_e8_train_geometry_semantic.pt"
OUT="${ROOT}/stage81_raw_query_pca_train"
PROJECTOR="${OUT}/raw_query_pca64.npz"
DUMP="${OUT}/stage16_e8_train_geometry_semantic_rawpca.pt"
RECEIPT="${OUT}/fit_transform_receipt.json"

cd "${R}"
[ -f "${INPUT}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 281; }
mkdir -p "${OUT}"
"${PYTHON}" "${P}/fit_apply_raw_query_pca.py" fit-transform \
  "${INPUT}" "${PROJECTOR}" "${DUMP}" "${RECEIPT}" \
  --fit-bucket-end 69 --sample-size 50000 --output-dim 64 --seed 0 \
  2>&1 | tee "${P}/stage81_fit_raw_query_pca_train.log"
[ -f "${PROJECTOR}" ] && [ -f "${DUMP}" ] && [ -f "${RECEIPT}" ]
sha256sum "${PROJECTOR}" "${DUMP}" "${RECEIPT}" > "${OUT}/sha256.txt"
echo STAGE81_RAW_QUERY_PCA_TRAIN_DUMP_PASS
