import torch

from models.reliability_fusion import ReliabilityFusion


def test_failclosed_fusion_is_positive_and_order_equivalent_to_base():
    fusion = ReliabilityFusion(
        use_quality=False,
        quality_weight=0.0,
        residual_clip=0.0,
    )
    base = torch.tensor([[0.001, 0.8, 0.2, 0.0]])
    structured = torch.tensor([[3.0, -1.0, 2.0, 0.5]])
    output = fusion(base, structured)
    fused = output["fused_scores"]
    assert torch.all(fused > 0)
    assert fused.argmax(dim=1).item() == base.argmax(dim=1).item()
    assert torch.equal(
        fused.argsort(dim=1, descending=True),
        base.argsort(dim=1, descending=True),
    )
