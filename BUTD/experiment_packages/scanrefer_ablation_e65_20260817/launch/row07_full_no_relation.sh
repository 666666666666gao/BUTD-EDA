#!/bin/bash
set -euo pipefail
export ABLATION_ID="07_no_relation"
export ABLATION_FLAGS="--use_structured_slots --use_sacr --sacr_disable_relation --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused --eval_use_fused_scores --rapf_quality_weight 0.75 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh" "$@"

