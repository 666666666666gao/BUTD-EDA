#!/usr/bin/env python3
"""CPU contract test for one-pass RAPF quality-weight calibration."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.reliability_fusion import ReliabilityFusion  # noqa: E402


def main():
    torch.manual_seed(20260827)
    batch_size, num_queries = 2, 7
    base_scores = torch.randn(batch_size, num_queries)
    structured_scores = torch.randn(batch_size, num_queries)
    quality_scores = torch.randn(batch_size, num_queries)
    structured_valid = torch.ones(batch_size, dtype=torch.bool)
    global_only = torch.zeros(batch_size, dtype=torch.bool)
    weak_generic = torch.zeros(batch_size, dtype=torch.bool)
    parse_confidence = torch.tensor([0.9, 0.6])
    error_count = torch.tensor([0.0, 1.0])
    anchor_entropy = torch.tensor([0.3, 0.8])
    anchor_mass = torch.tensor([0.7, 0.4])
    valid_query = torch.ones(batch_size, num_queries, dtype=torch.bool)

    module = ReliabilityFusion(
        hidden_dim=16,
        initial_gate_bias=-2.5,
        use_quality=True,
        quality_weight=0.25,
        generic_gate_cap=0.1,
        residual_clip=0.25,
        quality_anchor_structured_residual=False,
    ).eval()

    kwargs = dict(
        base_scores=base_scores,
        structured_scores=structured_scores,
        quality_scores=quality_scores,
        structured_valid_mask=structured_valid,
        global_only_mask=global_only,
        weak_generic_target_mask=weak_generic,
        parse_confidence=parse_confidence,
        decomposition_error_flags_count=error_count,
        anchor_entropy=anchor_entropy,
        anchor_top1_mass=anchor_mass,
        valid_query_mask=valid_query,
    )
    reference = module(**kwargs)
    grid_inputs = (
        reference["rapf_base_norm"],
        reference["rapf_gate"],
        reference["rapf_delta"],
        reference["rapf_quality_norm"],
    )

    for quality_weight in (
        0.00, 0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00
    ):
        base_norm, gate, delta, quality_norm = grid_inputs
        reconstructed = (
            base_norm + gate * delta + quality_weight * quality_norm
        )
        reconstructed = (
            reconstructed
            - reconstructed.min(dim=1, keepdim=True).values
            + 1e-6
        )
        module.quality_weight = quality_weight
        direct = module(**kwargs)["fused_scores"]
        if not torch.allclose(
            reconstructed, direct, atol=1e-6, rtol=1e-6
        ):
            max_error = (reconstructed - direct).abs().max().item()
            raise AssertionError(
                f"quality weight {quality_weight:.2f} mismatch: {max_error}"
            )
        if not torch.equal(
            reconstructed.argmax(dim=1), direct.argmax(dim=1)
        ):
            raise AssertionError(
                f"quality weight {quality_weight:.2f} changes ranking"
            )

    print("RAPF_QUALITY_GRID_CONTRACT_PASS")


if __name__ == "__main__":
    main()
