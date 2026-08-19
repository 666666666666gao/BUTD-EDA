#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/01_baseline_scanrefer_1stage.sh" "$@"
bash "${SCRIPT_DIR}/02_quality_only_scanrefer_1stage.sh" "$@"
bash "${SCRIPT_DIR}/05_full_sacr_rapf_qahnl_scanrefer_1stage.sh" "$@"
