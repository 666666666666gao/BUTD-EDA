from types import SimpleNamespace

import torch
import torch.nn as nn

from main_utils import (
    _freeze_non_detector_policy_adapter_parameters,
    _set_non_detector_policy_adapter_modules_eval,
)
from models.detector_policy_sources import DetectorPolicyAdapterHead
from models.losses import _detector_policy_tier_pair_losses


def _features(batch=2, queries=6, dim=8):
    torch.manual_seed(147)
    pred_boxes = torch.randn(batch, queries, 6)
    pred_boxes[..., 3:] = pred_boxes[..., 3:].abs() + 0.5
    matched_boxes = pred_boxes.clone()
    matched_boxes[..., 0] += 0.1
    fused = torch.randn(batch, queries)
    fused[:, -1] = fused.max(dim=1).values + 2.0
    return {
        "quality_scores": torch.randn(batch, queries),
        "class_scores": torch.rand(batch, queries),
        "conf_scores": torch.rand(batch, queries),
        "det_count": torch.full((batch,), 3, dtype=torch.long),
        "low_count": torch.zeros(batch),
        "detector_support": torch.rand(batch, queries),
        "pred_boxes": pred_boxes,
        "matched_det_boxes": matched_boxes,
        "matched_det_support": torch.full((batch, queries), 0.7),
        "matched_det_confidence": torch.full((batch, queries), 0.8),
        "matched_det_valid": torch.ones(batch, queries, dtype=torch.bool),
        "query_feats": torch.randn(batch, queries, dim),
        "base_scores": torch.randn(batch, queries),
        "structured_scores": torch.randn(batch, queries),
        "fused_scores": fused,
        "contrastive_scores": torch.randn(batch, queries),
    }


def test_fresh_tier_pair_head_has_exact_checkpoint_parity():
    legacy = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        tier_pair_rescue_head=False,
    )
    upgraded = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        tier_pair_rescue_head=True, tier_pair_candidate_k=2,
    )
    incompatible = upgraded.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert any("tier_pair_" in key for key in incompatible.missing_keys)

    features = _features()
    legacy.eval()
    upgraded.eval()
    with torch.no_grad():
        expected = legacy(features)
        actual = upgraded(features)
    assert not actual["tier_pair_gate"].any()
    assert torch.equal(actual["scores"], expected["scores"])
    assert torch.equal(actual["calibrated_boxes"], expected["calibrated_boxes"])
    assert actual["tier_pair_candidate_query"].shape == (2, 2)


class _ForceSecondCandidate(nn.Module):
    def forward(self, pair_features):
        batch, candidates = pair_features.shape[:2]
        logits = pair_features.new_zeros(batch, candidates, 4)
        logits[:, 1, 2] = 4.0
        logits[:, 1, 3] = 4.0
        logits[:, 0, 3] = -4.0
        return logits


def test_tier_pair_gate_can_promote_the_second_current_query():
    head = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        tier_pair_rescue_head=True, tier_pair_candidate_k=2,
        tier_pair_override_threshold=0.0,
    )
    head.tier_pair_rescue_head = _ForceSecondCandidate()
    head.eval()
    with torch.no_grad():
        output = head(_features(batch=1))
    assert output["tier_pair_gate"].item()
    assert (
        output["tier_pair_query"].item()
        == output["tier_pair_candidate_query"][0, 1].item()
    )
    assert output["scores"].argmax(dim=1).item() == output["tier_pair_query"].item()


def test_tier_pair_loss_uses_train_only_gt_and_backpropagates():
    logits = torch.zeros(3, 2, 4, requires_grad=True)
    end_points = {
        "detector_policy_tier_pair_logits": logits,
        "detector_policy_tier_pair_candidate_query": torch.tensor([[0, 1], [0, 1], [0, 1]]),
        "detector_policy_tier_pair_incumbent_query": torch.tensor([0, 0, 0]),
        "last_center": torch.tensor([
            [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]],
        ]),
        "last_pred_size": torch.ones(3, 2, 3),
        "center_label": torch.zeros(3, 1, 3),
        "size_gts": torch.ones(3, 1, 3),
        "box_label_mask": torch.ones(3, 1),
    }
    losses = _detector_policy_tier_pair_losses(end_points, weight=1.0)
    loss = losses["loss_detector_policy_tier_pair"]
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert losses["dbg_detector_policy_tier_pair_positive_ratio"].item() > 0
    assert losses["dbg_detector_policy_tier_pair_no_rescue_ratio"].item() > 0
    assert losses["dbg_detector_policy_tier_pair_ambiguous_ratio"].item() > 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.detector_policy_adapter = DetectorPolicyAdapterHead(
            hidden_dim=4, query_dim=8, tier_pair_rescue_head=True,
        )


def test_tier_pair_train_only_freezes_every_accepted_parameter_and_mode():
    model = nn.DataParallel(_Wrapper())
    count = _freeze_non_detector_policy_adapter_parameters(
        SimpleNamespace(
            detector_policy_adapter_train_only=True,
            source_pool_selector_train_only=False,
            detector_policy_geometry_train_only=False,
            detector_policy_geometry_extension_train_only=False,
            detector_policy_rank2_rescue_train_only=False,
            detector_policy_alignment_rescue_train_only=False,
            detector_policy_tier_pair_train_only=True,
            detector_policy_boundary_refiner_train_only=False,
        ),
        model,
    )
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert count > 0
    assert trainable
    assert all("detector_policy_adapter.tier_pair_" in name for name in trainable)

    model.train()
    _set_non_detector_policy_adapter_modules_eval(
        model, tier_pair_only=True
    )
    assert not model.module.detector_policy_adapter.query_mlp.training
    assert model.module.detector_policy_adapter.tier_pair_rescue_head.training
    assert not model.module.backbone.training
