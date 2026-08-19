#!/bin/bash
set -euo pipefail
export ABLATION_ID="08_sacr_only"
export ABLATION_FLAGS="--use_structured_slots --use_sacr --eval_use_structured_scores"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh" "$@"

