#!/usr/bin/env python3
"""CPU synthetic behavior checks for the three-table ablation switches."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.losses import _qahnl_losses
from models.reliability_fusion import ReliabilityFusion
from models.sacr_head import SACRHead


def sacr_inputs():
    torch.manual_seed(9)
    batch, queries, dim, relations = 2, 8, 16, 2
    slot_dict = {
        'global_slot': torch.randn(batch, dim),
        'target_slot': torch.randn(batch, dim),
        'attr_slot': torch.randn(batch, dim),
        'rel_slots': torch.randn(batch, relations, dim),
        'anchor_slots': torch.randn(batch, relations, dim),
        'slot_mask': torch.ones(batch, relations, dtype=torch.bool),
        'coverage_stats': {'has_target': torch.ones(batch, dtype=torch.bool)},
    }
    return (
        torch.randn(batch, queries, dim),
        torch.rand(batch, queries, 6),
        torch.randn(batch, queries),
        slot_dict,
    )


def test_sacr():
    query, boxes, base, slots = sacr_inputs()
    relation_only = SACRHead(
        d_model=16, hidden_dim=24, top_m_targets=4, top_k_anchors=4,
        disable_target_attr=True,
    )(query, boxes, base, slots)
    assert torch.count_nonzero(relation_only['target_attr_scores']) == 0
    assert torch.allclose(
        relation_only['structured_scores'],
        relation_only['relation_anchor_scores'],
    )

    no_geometry = SACRHead(d_model=16, hidden_dim=24, geo_dim=0)
    assert no_geometry.geo_encoder is None and not no_geometry.use_geometry
    no_geometry(query, boxes, base, slots)

    hard = SACRHead(
        d_model=16, hidden_dim=24, top_m_targets=4, top_k_anchors=4,
        anchor_aggregation='hard',
    )(query, boxes, base, slots)
    assert torch.allclose(hard['anchor_entropy'], torch.zeros(2), atol=1e-6)

    full = SACRHead(d_model=16, hidden_dim=24, global_residual_weight=0.1)
    no_global = SACRHead(d_model=16, hidden_dim=24, global_residual_weight=0.0)
    no_global.load_state_dict(full.state_dict())
    full_out = full(query, boxes, base, slots)
    zero_out = no_global(query, boxes, base, slots)
    assert not torch.allclose(
        full_out['target_attr_scores'], zero_out['target_attr_scores']
    )


def test_rapf():
    torch.manual_seed(10)
    base = torch.randn(2, 8)
    structured = torch.randn(2, 8) * 4.0
    common = dict(
        base_scores=base,
        structured_scores=structured,
        structured_valid_mask=torch.tensor([True, True]),
        global_only_mask=torch.tensor([False, True]),
        weak_generic_target_mask=torch.tensor([False, False]),
    )
    fixed_safe = ReliabilityFusion(fixed_alpha=0.1)(**common)
    assert torch.allclose(fixed_safe['rapf_gate'][0], torch.full((8,), 0.1))
    assert torch.count_nonzero(fixed_safe['rapf_gate'][1]) == 0

    fixed_unsafe = ReliabilityFusion(
        fixed_alpha=0.1, disable_safety=True
    )(**common)
    assert torch.allclose(fixed_unsafe['rapf_gate'][1], torch.full((8,), 0.1))

    unclipped = ReliabilityFusion(
        fixed_alpha=0.1, residual_clip=0.01,
        disable_residual_clipping=True,
        disable_agreement_features=True,
        disable_parser_anchor_features=True,
    )(**common)
    assert unclipped['rapf_delta'].abs().max().item() > 0.01
    assert unclipped['dbg_rapf_residual_clip_ratio'].item() == 0.0
    assert unclipped['dbg_rapf_agreement_features_disabled'] == 1.0
    assert unclipped['dbg_rapf_parser_anchor_features_disabled'] == 1.0

    captured = {}
    cue_free = ReliabilityFusion(
        hidden_dim=8,
        use_quality=True,
        quality_weight=0.75,
        generic_gate_cap=0.1,
        disable_parser_anchor_features=True,
    )
    def capture_features(module, args):
        captured['features'] = args[0].detach()

    handle = cue_free.gate_mlp[0].register_forward_pre_hook(capture_features)
    cue_out = cue_free(
        base_scores=base,
        structured_scores=structured,
        quality_scores=torch.randn(2, 8),
        structured_valid_mask=torch.tensor([True, True]),
        global_only_mask=torch.tensor([True, False]),
        weak_generic_target_mask=torch.tensor([False, True]),
        parse_confidence=torch.tensor([0.2, 0.9]),
        decomposition_error_flags_count=torch.tensor([3.0, 2.0]),
        anchor_entropy=torch.tensor([0.6, 0.7]),
        anchor_top1_mass=torch.tensor([0.8, 0.9]),
    )
    handle.remove()
    features = captured['features']
    # Parser confidence, global/generic feature bits, error count, anchor
    # entropy and mass are zeroed only in the learned gate feature vector.
    for index in (4, 10, 11, 12, 13, 14):
        assert torch.count_nonzero(features[..., index]) == 0
    # Agreement and quality cues remain present.
    assert torch.count_nonzero(features[..., 2]) > 0
    assert torch.count_nonzero(features[..., 5:10]) > 0
    # Original masks still enforce safety after gate prediction.
    assert torch.count_nonzero(cue_out['rapf_gate'][0]) == 0
    assert cue_out['rapf_gate'][1].max().item() <= 0.100001
    assert cue_out['dbg_rapf_parser_anchor_cues_disabled'] == 1.0


def qahnl_inputs():
    batch, queries = 2, 8
    centers = torch.full((batch, queries, 3), 4.0)
    centers[:, 0, :] = 0.0
    end_points = {
        'last_center': centers,
        'last_pred_size': torch.ones(batch, queries, 3),
        'center_label': torch.zeros(batch, 1, 3),
        'size_gts': torch.ones(batch, 1, 3),
        'box_label_mask': torch.ones(batch, 1),
        'dataset': ['scanrefer_spacy', 'scanrefer_spacy'],
        'base_grounding_scores': torch.randn(batch, queries),
        'fused_scores': torch.randn(batch, queries),
    }
    indices = [
        (torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long))
        for _ in range(batch)
    ]
    return end_points, indices


def test_qahnl():
    end_points, indices = qahnl_inputs()
    random_result = _qahnl_losses(end_points, indices, {
        'score_source': 'fused', 'negative_sampling': 'random',
        'num_hard_neg': 3, 'pos_iou_thresh': 0.25,
        'neg_iou_thresh': 0.10, 'topk_iou_pos': 3,
    })
    assert random_result['dbg_qahnl_negative_sampling_random'] == 1.0
    rescue_off = _qahnl_losses(end_points, indices, {
        'score_source': 'fused', 'negative_sampling': 'hardest',
        'num_hard_neg': 3, 'pos_iou_thresh': 0.25,
        'neg_iou_thresh': 0.10, 'topk_iou_pos': 3,
        'disable_top_iou_pos': True,
        'disable_hungarian_pos_rescue': True,
    })
    assert rescue_off['dbg_qahnl_top_iou_pos_rescue_disabled'] == 1.0
    assert rescue_off['dbg_qahnl_hungarian_pos_rescue_disabled'] == 1.0
    assert torch.isfinite(rescue_off['loss_qahnl'])


if __name__ == '__main__':
    test_sacr()
    test_rapf()
    test_qahnl()
    print('THREE_TABLE_MODULE_ABLATION_SMOKE_PASS')
