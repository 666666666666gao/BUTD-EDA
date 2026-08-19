#!/bin/bash
set -euo pipefail
export ABLATION_ID="03_no_sacr_rapf_qahnl_base"
export ABLATION_FLAGS="--use_qahnl --qahnl_score_source base"
exec bash "$(dirname "${BASH_SOURCE[0]}")/scanrefer_ablation_common_20260814.sh" "$@"
