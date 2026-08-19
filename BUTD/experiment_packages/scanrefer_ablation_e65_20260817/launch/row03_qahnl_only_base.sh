#!/bin/bash
set -euo pipefail
export ABLATION_ID="03_no_sacr_rapf_qahnl_base"
export ABLATION_FLAGS="--use_qahnl --qahnl_score_source base"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh" "$@"

