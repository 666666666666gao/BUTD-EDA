import torch

from models.detector_policy_sources import DetectorPolicyAdapterHead
from models.losses import _detector_policy_adapter_losses


def _features(batch=2, queries=10, dim=8):
    torch.manual_seed(7)
    fused = torch.randn(batch, queries)
    return {
        'quality_scores': torch.rand(batch, queries),
        'class_scores': torch.rand(batch, queries),
        'conf_scores': torch.rand(batch, queries),
        'det_count': torch.ones(batch, dtype=torch.long),
        'low_count': torch.ones(batch),
        'detector_support': torch.rand(batch, queries),
        'query_feats': torch.randn(batch, queries, dim),
        'pred_boxes': torch.randn(batch, queries, 6),
        'base_scores': torch.randn(batch, queries),
        'structured_scores': torch.randn(batch, queries),
        'fused_scores': fused,
        'contrastive_scores': torch.randn(batch, queries),
    }


def test_candidate_reranker_starts_at_exact_fused_parity_and_trains():
    head = DetectorPolicyAdapterHead(
        query_dim=8, hidden_dim=16, candidate_k=3, delta_scale=0.25
    )
    features = _features()
    out = head(features)
    assert torch.equal(out['scores'], features['fused_scores'])
    assert out['candidate_mask'].shape == features['fused_scores'].shape
    if 'hit25_logits' in out:
        assert out['hit25_logits'].shape == features['fused_scores'].shape
        assert out['hit50_logits'].shape == features['fused_scores'].shape
    assert torch.all(out['candidate_mask'].sum(dim=1) >= 3)
    # The hard score path deliberately has no surrogate gradient: training is
    # driven by the explicitly supervised rescue classifier.
    loss = -out['rescue_logits'][:, 0].mean()
    loss.backward()
    assert head.rerank_head[-1].weight.grad is not None
    assert torch.isfinite(head.rerank_head[-1].weight.grad).all()


def test_geometry_head_starts_at_point3_and_has_safe_support_fallback():
    head = DetectorPolicyAdapterHead(
        query_dim=8, hidden_dim=16, candidate_k=3, delta_scale=0.25
    )
    features = _features(batch=1, queries=5)
    pred = features['pred_boxes'].clone()
    pred[..., 3:] = pred[..., 3:].abs() + 0.5
    matched = pred.clone()
    matched[..., :3] += 0.2
    matched[..., 3:] *= 1.1
    features.update({
        'pred_boxes': pred,
        'matched_det_boxes': matched,
        'matched_det_support': torch.tensor([[0.2, 0.3, 0.5, 0.8, 0.9]]),
        'matched_det_confidence': torch.full((1, 5), 0.8),
        'matched_det_valid': torch.ones(1, 5, dtype=torch.bool),
    })
    out = head(features)
    assert torch.equal(out['geometry_action'], torch.full((1, 5), 3))
    assert out['geometry_alpha'][0, 0].item() == 0.0
    assert torch.allclose(out['geometry_alpha'][0, 1:], torch.full((4,), 0.3))
    expected = pred + out['geometry_alpha'].unsqueeze(-1) * (matched - pred)
    assert torch.allclose(out['calibrated_boxes'], expected)
    assert out['geometry_candidate_boxes'].shape == (1, 5, 6, 6)

    loss = -out['geometry_logits'][..., 0].mean()
    loss.backward()
    assert head.geometry_head[-1].weight.grad is not None
    assert torch.isfinite(head.geometry_head[-1].weight.grad).all()
    assert head.rerank_head[-1].weight.grad is None


def test_geometry_only_loss_backpropagates_to_action_logits():
    batch, queries, actions = 2, 3, 6
    logits = torch.zeros(batch, queries, actions, requires_grad=True)
    pred = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.4, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]).view(1, queries, 6).repeat(batch, 1, 1)
    matched = pred.clone()
    matched[..., 0] -= 0.4
    alphas = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    candidates = (
        pred.unsqueeze(2)
        + alphas.view(1, 1, actions, 1)
        * (matched - pred).unsqueeze(2)
    )
    end_points = {
        'detector_policy_adapter_scores': torch.randn(batch, queries),
        'detector_policy_adapter_candidate_mask': torch.ones(
            batch, queries, dtype=torch.bool
        ),
        'detector_policy_adapter_hit25_logits': torch.zeros(batch, queries),
        'detector_policy_adapter_hit50_logits': torch.zeros(batch, queries),
        'detector_policy_adapter_rescue_logits': torch.zeros(batch, queries),
        'detector_policy_adapter_rescue_query': torch.zeros(
            batch, dtype=torch.long
        ),
        'detector_policy_adapter_geometry_logits': logits,
        'detector_policy_adapter_geometry_candidate_boxes': candidates,
        'detector_policy_adapter_geometry_enabled': torch.ones(
            batch, queries, dtype=torch.bool
        ),
        'detector_policy_adapter_geometry_alpha': torch.full(
            (batch, queries), 0.3
        ),
        'last_center': pred[..., :3],
        'last_pred_size': pred[..., 3:],
        'center_label': torch.zeros(batch, 1, 3),
        'size_gts': torch.ones(batch, 1, 3),
        'box_label_mask': torch.ones(batch, 1),
        'fused_scores': torch.randn(batch, queries),
    }
    losses = _detector_policy_adapter_losses(
        end_points, weight=0.0, geometry_weight=1.0
    )
    loss = losses['loss_detector_policy_adapter']
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert torch.allclose(
        losses['dbg_detector_policy_geometry_valid_ratio'],
        torch.tensor(1.0 / queries),
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
