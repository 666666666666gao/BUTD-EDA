#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/03_sacr_only_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/04_rapf_quality_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/05_full_sacr_rapf_qahnl_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/06_full_no_gate_supervision_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/07_full_no_quality_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/08_full_no_qahnl_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/09_sacr_no_relation_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/10_qahnl_base_source_scanrefer_2stage.sh" "$@"
bash "${SCRIPT_DIR}/11_sacr_rank_scanrefer_2stage.sh" "$@"
