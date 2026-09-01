#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
INPUT="${ROOT}/stage75_stage16_e8_val_geometry_semantic_dump/stage16_e8_val_geometry_semantic_frozen_legacy.pt"
PROJECTOR="${ROOT}/stage81_raw_query_pca_train/raw_query_pca64.npz"
OUT="${ROOT}/stage83_raw_query_pca_val"
DUMP="${OUT}/stage16_e8_val_geometry_semantic_rawpca.pt"
RECEIPT="${OUT}/apply_receipt.json"

cd "${R}"
[ -f "${INPUT}" ] && [ -f "${PROJECTOR}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 283; }
mkdir -p "${OUT}"
"${PYTHON}" "${P}/fit_apply_raw_query_pca.py" apply \
  "${INPUT}" "${PROJECTOR}" "${DUMP}" "${RECEIPT}" \
  2>&1 | tee "${P}/stage83_apply_raw_query_pca_val.log"
[ -f "${DUMP}" ] && [ -f "${RECEIPT}" ]
sha256sum "${PROJECTOR}" "${DUMP}" "${RECEIPT}" > "${OUT}/sha256.txt"
echo STAGE83_RAW_QUERY_PCA_VAL_DUMP_PASS
