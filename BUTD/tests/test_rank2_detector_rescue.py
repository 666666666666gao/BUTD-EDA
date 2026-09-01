import torch

from models.detector_policy_sources import (
    DetectorPolicyAdapterHead,
    build_detector_policy_features,
)


def _adapter_features(batch=2, queries=6, query_dim=8):
    torch.manual_seed(17)
    pred_boxes = torch.randn(batch, queries, 6)
    pred_boxes[..., 3:] = pred_boxes[..., 3:].abs() + 0.5
    matched = pred_boxes.clone()
    matched[..., :3] += 0.1
    second = pred_boxes.clone()
    second[..., :3] -= 0.2
    base = torch.randn(batch, queries)
    fused = torch.randn(batch, queries)
    support = torch.full((batch, queries), 0.6)
    confidence = torch.full((batch, queries), 0.7)
    return {
        "quality_scores": torch.randn(batch, queries),
        "class_scores": torch.rand(batch, queries),
        "conf_scores": torch.rand(batch, queries),
        "det_count": torch.full((batch,), 3, dtype=torch.long),
        "low_count": torch.zeros(batch),
        "detector_support": support,
        "pred_boxes": pred_boxes,
        "matched_det_boxes": matched,
        "matched_det_support": support,
        "matched_det_confidence": confidence,
        "matched_det_valid": torch.ones(batch, queries, dtype=torch.bool),
        "second_matched_det_boxes": second,
        "second_matched_det_support": support * 0.8,
        "second_matched_det_confidence": confidence * 0.7,
        "second_matched_det_valid": torch.ones(
            batch, queries, dtype=torch.bool
        ),
        "query_feats": torch.randn(batch, queries, query_dim),
        "base_scores": base,
        "structured_scores": base + 0.1,
        "fused_scores": fused,
        "contrastive_scores": base - 0.1,
    }


def test_second_detector_match_is_distinct_and_deployable():
    pred = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
    det = torch.tensor([[[
        0.0, 0.0, 0.0, 2.0, 2.0, 2.0
    ], [
        0.4, 0.0, 0.0, 2.0, 2.0, 2.0
    ], [
        3.0, 0.0, 0.0, 2.0, 2.0, 2.0
    ]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    class_ids = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]])
    features = build_detector_policy_features(
        pred_boxes=pred,
        quality_scores=torch.ones(1, 1),
        det_boxes=det,
        det_bbox_label_mask=mask,
        det_class_ids=class_ids,
        det_logits=logits,
        target_cid=torch.zeros(1, dtype=torch.long),
    )
    assert features["matched_det_valid"].all()
    assert features["second_matched_det_valid"].all()
    assert torch.equal(features["matched_det_boxes"][0, 0], det[0, 0])
    assert torch.equal(
        features["second_matched_det_boxes"][0, 0], det[0, 1]
    )
    assert (
        features["matched_det_support"]
        > features["second_matched_det_support"]
    ).all()


def test_fresh_rank2_head_has_exact_stage18_parity():
    legacy = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        rank2_rescue_head=False,
    )
    upgraded = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        rank2_rescue_head=True, rank2_override_threshold=0.0,
    )
    incompatible = upgraded.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    features = _adapter_features()
    legacy.eval()
    upgraded.eval()
    with torch.no_grad():
        expected = legacy(features)
        actual = upgraded(features)
    assert not actual["rank2_gate"].any()
    assert torch.equal(actual["scores"], expected["scores"])
    assert torch.equal(
        actual["calibrated_boxes"], expected["calibrated_boxes"]
    )
    assert actual["rank2_candidate_boxes"].shape == (2, 6, 5, 6)


def test_rank2_head_can_promote_a_second_match_after_training():
    head = DetectorPolicyAdapterHead(
        hidden_dim=4, query_dim=8, candidate_k=3,
        rank2_rescue_head=True, rank2_override_threshold=0.0,
    )
    with torch.no_grad():
        head.rank2_rescue_head[-1].bias.fill_(1.0)
    output = head(_adapter_features())
    assert output["rank2_gate"].all()
    assert output["rank2_query"].shape == (2,)
    assert output["rank2_action"].shape == (2,)
