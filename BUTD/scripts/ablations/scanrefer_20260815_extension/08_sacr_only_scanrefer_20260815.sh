#!/bin/bash
set -euo pipefail
export ABLATION_ID="08_sacr_only"
export ABLATION_FLAGS="--use_structured_slots --use_sacr --eval_use_structured_scores"
exec bash "/home/gb/new butd/butd_detr-main/scripts/ablations/scanrefer_20260814/scanrefer_ablation_common_20260814.sh" "$@"
