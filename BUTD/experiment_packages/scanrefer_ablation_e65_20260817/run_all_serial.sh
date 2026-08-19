#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
RUN_TAG="${ABLATION_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export ABLATION_RUN_TAG="${RUN_TAG}"
export ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_e65_reproduction/${RUN_TAG}}"
CONTROL_DIR="${ABLATION_LOG_ROOT}/_package_control"
STATUS_DIR="${CONTROL_DIR}/status"

mkdir -p "${STATUS_DIR}"
bash "${PACKAGE_ROOT}/validate.sh"
bash "${PACKAGE_ROOT}/register_external_baseline.sh" "${CONTROL_DIR}/external_butd_paper_baseline.json"

SCRIPTS=(
  launch/row02_full_model.sh
  launch/row04_sacr_rapf_no_qahnl.sh
  launch/row08_sacr_only.sh
  launch/row09_sacr_qahnl_structured.sh
  launch/row03_qahnl_only_base.sh
  launch/row05_full_no_quality.sh
  launch/row10_full_qahnl_base_source.sh
  launch/row06_full_no_gate_supervision.sh
  launch/row07_full_no_relation.sh
)

for rel in "${SCRIPTS[@]}"; do
  name="$(basename "${rel}" .sh)"
  if [ -f "${STATUS_DIR}/${name}.done" ]; then
    echo "SKIP completed ${name}"
    continue
  fi
  if [ -e "${STATUS_DIR}/${name}.started" ]; then
    echo "ERROR: ${name} has an incomplete prior attempt under ${ABLATION_LOG_ROOT}; inspect it before restarting from official initialization." >&2
    exit 50
  fi
  date -Is > "${STATUS_DIR}/${name}.started"
  echo "START ${name} $(date -Is)"
  bash "${PACKAGE_ROOT}/${rel}" 2>&1 | tee "${CONTROL_DIR}/${name}.launcher.log"
  date -Is > "${STATUS_DIR}/${name}.done"
  echo "DONE ${name} $(date -Is)"
done

echo "ALL_NINE_TRAINED_ABLATIONS_COMPLETED $(date -Is)" | tee "${CONTROL_DIR}/ALL_COMPLETED"
