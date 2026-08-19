#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/01_baseline_sr3d.sh" "$@"
bash "${SCRIPT_DIR}/02_quality_only_sr3d.sh" "$@"
bash "${SCRIPT_DIR}/03_full_sacr_rapf_qahnl_sr3d.sh" "$@"
