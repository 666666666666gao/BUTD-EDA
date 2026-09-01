import torch

from models.detector_policy_sources import DetectorPolicyAdapterHead


def _features(batch=2, queries=6, dim=8):
    torch.manual_seed(7)
    fused = torch.randn(batch, queries)
    fused[:, -1] = fused.max(dim=1).values + 2.0
    return {
        "quality_scores": torch.randn(batch, queries),
        "class_scores": torch.rand(batch, queries),
        "conf_scores": torch.rand(batch, queries),
        "det_count": torch.ones(batch),
        "low_count": torch.ones(batch, dtype=torch.bool),
        "detector_support": torch.rand(batch, queries),
        "query_feats": torch.randn(batch, queries, dim),
        "base_scores": torch.randn(batch, queries),
        "structured_scores": torch.randn(batch, queries),
        "contrastive_scores": torch.randn(batch, queries),
        "fused_scores": fused,
        "pred_boxes": torch.randn(batch, queries, 6),
    }


def test_rescue_gate_is_exact_full_fallback_at_initialization():
    features = _features()
    head = DetectorPolicyAdapterHead(
        hidden_dim=8, query_dim=8, candidate_k=6, delta_scale=4.0
    )
    output = head(features)
    assert torch.equal(output["scores"], features["fused_scores"])
    assert not output["rescue_gate"].any()
    assert torch.equal(
        output["fallback_query"], features["fused_scores"].argmax(dim=1)
    )


def test_high_confidence_hit50_decision_can_override_full_fallback():
    features = _features(batch=1)
    head = DetectorPolicyAdapterHead(
        hidden_dim=8, query_dim=8, candidate_k=6, delta_scale=4.0
    )
    head.eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        features["query_feats"].zero_()
        features["query_feats"][0, 0, 0] = 10.0
        head.query_mlp[0].weight[0, 0] = 1.0
        head.rerank_head[0].weight[0, 0] = 1.0
        head.rerank_head[-1].weight[2, 0] = 1.0
    output = head(features)
    assert output["rescue_gate"].item()
    assert output["rescue_query"].item() != output["fallback_query"].item()
    assert output["scores"].argmax(dim=1).item() == output["rescue_query"].item()
