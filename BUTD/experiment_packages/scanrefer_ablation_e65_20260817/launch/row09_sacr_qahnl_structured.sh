#!/bin/bash
set -euo pipefail
export ABLATION_ID="09_sacr_qahnl"
export ABLATION_FLAGS="--use_structured_slots --use_sacr --use_qahnl --qahnl_score_source structured --eval_use_structured_scores"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh" "$@"

