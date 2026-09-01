#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
BASE_PACKAGE="${BASE_PACKAGE:-${PACKAGE_ROOT}/scanrefer_three_table_ablations_20260821}"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"

bash -n "${PACKAGE_ROOT}/run_machine_queue.sh"
bash -n "${PACKAGE_ROOT}/start_machine_queue.sh"
"${PYTHON}" -m json.tool "${PACKAGE_ROOT}/plan_manifest.json" >/dev/null
[ -f "${BASE_PACKAGE}/launch/run_row.sh" ]
bash "${BASE_PACKAGE}/validate.sh"

for row in S0 S2 S3 R0 R1 R2 R3; do
  DRY_RUN=1 REPO_ROOT="${REPO_ROOT}" bash "${BASE_PACKAGE}/launch/run_row.sh" "${row}" >/dev/null
done

echo "dual-machine ablation package validation PASS"
