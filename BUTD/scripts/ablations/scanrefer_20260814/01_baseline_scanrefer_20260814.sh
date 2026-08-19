#!/bin/bash
set -euo pipefail
export ABLATION_ID="01_baseline"
export ABLATION_FLAGS=""
exec bash "$(dirname "${BASH_SOURCE[0]}")/scanrefer_ablation_common_20260814.sh" "$@"
