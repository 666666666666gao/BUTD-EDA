from types import SimpleNamespace

import torch
import torch.nn as nn

from main_utils import (
    _freeze_non_detector_policy_adapter_parameters,
    _set_non_detector_policy_adapter_modules_eval,
)
from models.detector_policy_sources import DetectorPolicyAdapterHead
from models.losses import _detector_policy_alignment_rescue_losses


def _features(batch=2, queries=6, detections=5, dim=8):
    torch.manual_seed(145)
    pred_boxes = torch.randn(batch, queries, 6)
    pred_boxes[..., 3:] = pred_boxes[..., 3:].abs() + 0.5
    matched_boxes = pred_boxes.clone()
    matched_boxes[..., 0] += 0.1
    fused = torch.randn(batch, queries)
    fused[:, -1] = fused.max(dim=1).values + 2.0
    det_boxes = torch.randn(batch, detections, 6)
    det_boxes[..., 3:] = det_boxes[..., 3:].abs() + 0.5
    det_class_ids = torch.ones(batch, detections, dtype=torch.long)
    det_class_ids[:, 0] = 0
    det_logits = torch.full((batch, detections, 3), -2.0)
    det_logits[..., 1] = 2.0
    det_logits[:, 0, 0] = 5.0
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
        "det_boxes": det_boxes,
        "det_bbox_label_mask": torch.ones(
            batch, detections, dtype=torch.bool
        ),
        "det_class_ids": det_class_ids,
        "det_logits": det_logits,
        "target_cid": torch.zeros(batch, dtype=torch.long),
        "detected_feats": torch.randn(batch, detections, dim),
        "alignment_text_context": torch.randn(batch, dim),
    }


def test_fresh_alignment_rescue_has_exact_legacy_parity():
    legacy = DetectorPolicyAdapterHead(
        hidden_dim=4,
        query_dim=8,
        candidate_k=3,
        alignment_rescue_head=False,
    )
    upgraded = DetectorPolicyAdapterHead(
        hidden_dim=4,
        query_dim=8,
        candidate_k=3,
        alignment_rescue_head=True,
        alignment_candidate_k=3,
    )
    incompatible = upgraded.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert any("alignment_" in key for key in incompatible.missing_keys)

    features = _features()
    legacy.eval()
    upgraded.eval()
    with torch.no_grad():
        expected = legacy(features)
        actual = upgraded(features)

    assert not actual["alignment_gate"].any()
    assert torch.equal(actual["scores"], expected["scores"])
    assert torch.equal(
        actual["calibrated_boxes"], expected["calibrated_boxes"]
    )
    assert actual["alignment_candidate_mask"].shape == (2, 5)


def test_alignment_rescue_can_replace_incumbent_with_raw_detector_box():
    head = DetectorPolicyAdapterHead(
        hidden_dim=4,
        query_dim=8,
        candidate_k=3,
        alignment_rescue_head=True,
        alignment_candidate_k=1,
        alignment_override_threshold=0.0,
    )
    features = _features(batch=1)
    head.eval()
    with torch.no_grad():
        head.alignment_rescue_head[-1].bias[2] = 1.0
        output = head(features)

    incumbent = output["alignment_incumbent_query"].item()
    assert output["alignment_gate"].item()
    assert output["alignment_candidate"].item() == 0
    assert torch.equal(
        output["calibrated_boxes"][0, incumbent],
        features["det_boxes"][0, 0],
    )
    assert torch.equal(output["scores"], features["fused_scores"])


def test_alignment_rescue_loss_is_train_only_supervised_and_backpropagates():
    batch, detections, queries = 2, 3, 2
    logits = torch.zeros(batch, detections, 3, requires_grad=True)
    candidate_boxes = torch.tensor([
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [4.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [4.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
    ])
    incumbent_boxes = torch.tensor([
        [
            [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
    ])
    end_points = {
        "detector_policy_alignment_logits": logits,
        "detector_policy_alignment_candidate_mask": torch.ones(
            batch, detections, dtype=torch.bool
        ),
        "detector_policy_alignment_candidate_boxes": candidate_boxes,
        "detector_policy_alignment_incumbent_query": torch.zeros(
            batch, dtype=torch.long
        ),
        "detector_policy_alignment_incumbent_boxes": incumbent_boxes,
        "center_label": torch.zeros(batch, 1, 3),
        "size_gts": torch.ones(batch, 1, 3),
        "box_label_mask": torch.ones(batch, 1),
    }
    losses = _detector_policy_alignment_rescue_losses(end_points, weight=1.0)
    loss = losses["loss_detector_policy_alignment_rescue"]
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert losses["dbg_detector_policy_alignment_positive_ratio"].item() > 0
    assert losses["dbg_detector_policy_alignment_no_rescue_ratio"].item() > 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.detector_policy_adapter = DetectorPolicyAdapterHead(
            hidden_dim=4,
            query_dim=8,
            alignment_rescue_head=True,
        )


def test_alignment_rescue_train_only_freezes_legacy_adapter_and_modes():
    model = nn.DataParallel(_Wrapper())
    count = _freeze_non_detector_policy_adapter_parameters(
        SimpleNamespace(
            detector_policy_adapter_train_only=True,
            source_pool_selector_train_only=False,
            detector_policy_geometry_train_only=False,
            detector_policy_geometry_extension_train_only=False,
            detector_policy_rank2_rescue_train_only=False,
            detector_policy_boundary_refiner_train_only=False,
            detector_policy_alignment_rescue_train_only=True,
        ),
        model,
    )
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert count > 0
    assert trainable
    assert all("detector_policy_adapter.alignment_" in name for name in trainable)

    model.train()
    _set_non_detector_policy_adapter_modules_eval(
        model, alignment_rescue_only=True
    )
    assert not model.module.detector_policy_adapter.query_mlp.training
    assert model.module.detector_policy_adapter.alignment_rescue_head.training
    assert model.module.detector_policy_adapter.alignment_det_mlp.training
    assert not model.module.backbone.training
