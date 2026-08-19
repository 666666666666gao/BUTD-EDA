"""Synthetic smoke test for SACR, RAPF, and quality heads."""

import torch
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.quality_head import QualityHead
from models.sacr_head import SACRHead
from models.reliability_fusion import ReliabilityFusion
from models.losses import _quality_losses, _qahnl_losses, _rapf_gate_losses


def main():
    torch.manual_seed(0)
    batch_size = 2
    num_queries = 8
    d_model = 32
    num_pairs = 3

    query_feats = torch.randn(batch_size, num_queries, d_model)
    pred_boxes = torch.rand(batch_size, num_queries, 6)
    base_scores = torch.randn(batch_size, num_queries)
    slot_dict = {
        'global_slot': torch.randn(batch_size, d_model),
        'target_slot': torch.randn(batch_size, d_model),
        'attr_slot': torch.randn(batch_size, d_model),
        'rel_slots': torch.randn(batch_size, num_pairs, d_model),
        'anchor_slots': torch.randn(batch_size, num_pairs, d_model),
        'slot_mask': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        'parse_confidence': torch.tensor([0.9, 0.4]),
        'coverage_stats': {
            'has_target': torch.tensor([1, 1], dtype=torch.bool),
            'num_attrs': torch.tensor([2, 1]),
            'num_pairs': torch.tensor([2, 1]),
        },
    }

    quality = QualityHead(d_model=d_model, hidden_dim=64)
    q_out = quality(query_feats, pred_boxes)
    assert q_out['quality_logits'].shape == (batch_size, num_queries)
    assert q_out['pred_iou'].shape == (batch_size, num_queries)

    sacr = SACRHead(
        d_model=d_model, hidden_dim=64,
        top_m_targets=4, top_k_anchors=4, geo_dim=8
    )
    sacr_out = sacr(
        query_feats, pred_boxes, base_scores, slot_dict,
        global_only_mask=torch.tensor([False, True]),
        weak_generic_target_mask=torch.tensor([False, True]),
    )
    assert sacr_out['structured_scores'].shape == (batch_size, num_queries)
    assert sacr_out['anchor_entropy'].shape == (batch_size,)
    assert sacr_out['anchor_top1_mass'].shape == (batch_size,)
    assert sacr_out['structured_valid_mask'].tolist() == [True, False]
    assert torch.allclose(sacr_out['structured_scores'][1], torch.zeros(num_queries))
    assert torch.isfinite(sacr_out['anchor_entropy']).all()
    assert torch.isfinite(sacr_out['anchor_top1_mass']).all()

    sacr_no_relation = SACRHead(
        d_model=d_model, hidden_dim=64,
        top_m_targets=4, top_k_anchors=4, geo_dim=8,
        disable_relation=True,
    )
    sacr_no_relation_out = sacr_no_relation(
        query_feats, pred_boxes, base_scores, slot_dict,
        global_only_mask=torch.tensor([False, False]),
        weak_generic_target_mask=torch.tensor([False, False]),
    )
    assert torch.allclose(
        sacr_no_relation_out['relation_anchor_scores'],
        torch.zeros_like(sacr_no_relation_out['relation_anchor_scores']),
    )
    assert sacr_no_relation_out['relation_active_ratio'].item() == 0.0
    assert torch.isfinite(sacr_no_relation_out['anchor_entropy']).all()

    rapf = ReliabilityFusion(
        hidden_dim=32, initial_gate_bias=-2.0,
        use_quality=True, quality_weight=0.25,
        residual_clip=0.05,
    )
    fused = rapf(
        base_scores=base_scores,
        structured_scores=sacr_out['structured_scores'],
        quality_scores=q_out['pred_iou'],
        structured_valid_mask=sacr_out['structured_valid_mask'],
        global_only_mask=sacr_out['global_only_mask'],
        weak_generic_target_mask=sacr_out['weak_generic_target_mask'],
        parse_confidence=slot_dict['parse_confidence'],
    )
    assert fused['fused_scores'].shape == (batch_size, num_queries)
    assert fused['rapf_gate'].shape == (batch_size, num_queries)
    assert fused['rapf_delta'].abs().max().item() <= 0.05001
    assert torch.allclose(fused['rapf_gate'][1], torch.zeros(num_queries))
    for key in (
        'dbg_rapf_gate_std',
        'dbg_rapf_gate_min',
        'dbg_rapf_gate_max',
        'dbg_rapf_residual_clip_ratio',
        'dbg_rapf_delta_mean',
        'dbg_rapf_entropy_mean',
        'dbg_rapf_margin_mean',
        'dbg_rapf_disagree_ratio',
        'dbg_rapf_js_mean',
    ):
        assert key in fused
        assert torch.isfinite(torch.as_tensor(fused[key]))

    no_quality_fused = ReliabilityFusion(
        hidden_dim=32, initial_gate_bias=-2.0,
        use_quality=False, generic_gate_cap=0.35,
    )(
        base_scores=base_scores,
        structured_scores=sacr_out['structured_scores'],
        quality_scores=None,
        structured_valid_mask=sacr_out['structured_valid_mask'],
        global_only_mask=sacr_out['global_only_mask'],
        weak_generic_target_mask=sacr_out['weak_generic_target_mask'],
        parse_confidence=slot_dict['parse_confidence'],
    )
    assert no_quality_fused['fused_scores'].shape == (batch_size, num_queries)
    assert no_quality_fused['dbg_rapf_quality_enabled'] == 0.0

    capped = ReliabilityFusion(
        hidden_dim=32, initial_gate_bias=3.0,
        use_quality=False, generic_gate_cap=0.35,
    )
    capped_out = capped(
        base_scores=base_scores,
        structured_scores=base_scores + 1.0,
        structured_valid_mask=torch.tensor([True, True]),
        global_only_mask=torch.tensor([False, False]),
        weak_generic_target_mask=torch.tensor([False, True]),
        parse_confidence=slot_dict['parse_confidence'],
    )
    assert capped_out['rapf_gate'][1].max().item() <= 0.35001

    gt_center = torch.zeros(batch_size, 1, 3)
    gt_size = torch.ones(batch_size, 1, 3)
    loss_end_points = {
        'quality_logits': torch.randn(batch_size, num_queries),
        'pred_iou': torch.rand(batch_size, num_queries),
        'last_center': torch.rand(batch_size, num_queries, 3) * 0.2,
        'last_pred_size': torch.ones(batch_size, num_queries, 3),
        'center_label': gt_center,
        'size_gts': gt_size,
        'box_label_mask': torch.ones(batch_size, 1),
        'dataset': ['sr3d_spacy', 'nr3d_spacy'],
    }
    q_losses = _quality_losses(loss_end_points)
    assert 'loss_quality' in q_losses
    assert 'dbg_quality_reg_raw' in q_losses
    assert 'dbg_quality_cls_raw' in q_losses
    for key in (
        'dbg_quality_iou50_positive_ratio',
        'dbg_quality_target_iou_mean',
        'dbg_quality_target_iou_std',
        'dbg_quality_pred_iou_std',
        'dbg_quality_pred_target_iou_corr',
        'dbg_quality_top1_base_iou',
        'dbg_quality_top1_quality_iou',
        'dbg_quality_top1_improves_ratio',
    ):
        assert key in q_losses
    assert torch.isfinite(q_losses['loss_quality'])

    rapf_loss_end_points = dict(loss_end_points)
    rapf_loss_end_points.update({
        'rapf_gate': fused['rapf_gate'],
        'base_grounding_scores': base_scores,
        'structured_scores': sacr_out['structured_scores'],
        'fused_scores': fused['fused_scores'],
        'structured_valid_mask': sacr_out['structured_valid_mask'],
        'global_only_mask': sacr_out['global_only_mask'],
    })
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        rapf_gate_losses = _rapf_gate_losses(rapf_loss_end_points, indices=[])
    assert 'loss_rapf_gate' in rapf_gate_losses
    assert torch.isfinite(rapf_gate_losses['loss_rapf_gate'])

    qahnl_end_points = dict(loss_end_points)
    qahnl_centers = torch.full((batch_size, num_queries, 3), 4.0)
    qahnl_centers[:, 0, :] = 0.0
    qahnl_end_points['last_center'] = qahnl_centers
    qahnl_end_points['last_pred_size'] = torch.ones(batch_size, num_queries, 3)
    qahnl_end_points.update({
        'base_grounding_scores': torch.randn(batch_size, num_queries),
        'structured_scores': torch.randn(batch_size, num_queries),
        'global_only_mask': torch.tensor([False, True]),
        'weak_generic_target_mask': torch.tensor([False, True]),
    })
    indices = [
        (torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long))
        for _ in range(batch_size)
    ]
    hn_losses = _qahnl_losses(qahnl_end_points, indices, {
        'score_source': 'base',
        'num_hard_neg': 3,
        'pos_iou_thresh': 0.25,
        'neg_iou_thresh': 0.10,
        'topk_iou_pos': 2,
        'margin_base': 0.2,
        'margin_iou_lambda': 0.0,
        'margin_min': 0.05,
        'margin_max': 0.5,
        'temperature': 2.0,
        'temperature_max': 1.5,
        'loss_weight': 0.2,
        'use_entity_hardneg': True,
    })
    assert 'loss_qahnl' in hn_losses
    assert torch.isfinite(hn_losses['loss_qahnl'])
    assert hn_losses['dbg_qahnl_temperature'] == 1.5
    assert hn_losses['dbg_qahnl_margin_base'] == 0.2
    assert hn_losses['dbg_qahnl_ambiguous_as_negative_ratio'].item() == 0.0
    assert hn_losses['dbg_qahnl_global_only_used_ratio'].item() > 0.0
    assert hn_losses['dbg_qahnl_global_only_skipped_structured_ratio'].item() == 0.0
    for key in (
        'dbg_qahnl_hard_negative_query_ratio',
        'dbg_qahnl_ambiguous_ignore_ratio',
        'dbg_qahnl_hardneg_iou',
        'dbg_qahnl_iou_gap',
        'dbg_qahnl_score_gap',
        'dbg_qahnl_margin_mean',
        'dbg_qahnl_loss_unweighted',
        'dbg_qahnl_loss_weighted',
        'dbg_warn_qahnl_no_positive_ratio',
        'dbg_warn_qahnl_no_hard_negative_ratio',
    ):
        assert key in hn_losses
        assert torch.isfinite(torch.as_tensor(hn_losses[key]))

    structured_hn = _qahnl_losses(qahnl_end_points, indices, {
        'score_source': 'structured',
        'num_hard_neg': 3,
        'pos_iou_thresh': 0.25,
        'neg_iou_thresh': 0.10,
        'topk_iou_pos': 2,
        'margin_base': 0.2,
        'margin_iou_lambda': 0.0,
        'margin_min': 0.05,
        'margin_max': 0.5,
        'temperature': 1.0,
        'temperature_max': 6.0,
        'loss_weight': 0.2,
    })
    assert structured_hn['dbg_qahnl_global_only_used_ratio'].item() == 0.0
    assert structured_hn['dbg_qahnl_global_only_skipped_structured_ratio'].item() > 0.0
    print("module_smoke: ok")


if __name__ == '__main__':
    main()
