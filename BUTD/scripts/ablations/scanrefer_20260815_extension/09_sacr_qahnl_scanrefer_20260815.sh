#!/bin/bash
set -euo pipefail
export ABLATION_ID="09_sacr_qahnl"
export ABLATION_FLAGS="--use_structured_slots --use_sacr --use_qahnl --qahnl_score_source structured --eval_use_structured_scores"
exec bash "/home/gb/new butd/butd_detr-main/scripts/ablations/scanrefer_20260814/scanrefer_ablation_common_20260814.sh" "$@"
