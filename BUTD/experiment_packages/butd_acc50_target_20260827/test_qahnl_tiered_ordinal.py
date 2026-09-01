import math
import importlib.util
import os
from unittest import mock

import torch

if os.environ.get('STAGE148_LOSSES_PATH'):
    import models

    spec = importlib.util.spec_from_file_location(
        'models.losses_stage148_candidate',
        os.environ['STAGE148_LOSSES_PATH'],
    )
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)
else:
    from models import losses


def _run(scores, iou, candidate_mask=None):
    scores = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    iou = torch.tensor(iou, dtype=torch.float32)
    if candidate_mask is not None:
        candidate_mask = torch.tensor(candidate_mask, dtype=torch.bool)
    result = losses._qahnl_tiered_ordinal_losses(
        scores=scores,
        target_iou=iou,
        candidate_mask=candidate_mask,
        tier2_iou_thresh=0.50,
        tier1_iou_thresh=0.25,
        tier0_iou_thresh=0.10,
        margin21=0.20,
        margin10=0.10,
        temperature=1.0,
    )
    return scores, result


def test_ordered_scores_have_lower_loss():
    _, bad = _run(
        [[0.0, 2.0, 3.0, 100.0]],
        [[0.70, 0.30, 0.05, 0.20]],
    )
    _, good = _run(
        [[3.0, 2.0, 0.0, -100.0]],
        [[0.70, 0.30, 0.05, 0.20]],
    )
    assert good['loss_raw'].item() < bad['loss_raw'].item()
    assert good['active21_count'].item() == 1
    assert good['active10_count'].item() == 1
    assert good['active20_count'].item() == 0
    assert math.isclose(good['ambiguous_ratio'].item(), 0.25)


def test_ambiguous_and_masked_candidates_are_ignored():
    candidate_mask = [[True, True, True, False, True]]
    iou = [[0.70, 0.30, 0.05, 0.80, 0.20]]
    _, reference = _run(
        [[3.0, 2.0, 0.0, -999.0, -999.0]], iou, candidate_mask
    )
    _, changed = _run(
        [[3.0, 2.0, 0.0, 999.0, 999.0]], iou, candidate_mask
    )
    assert torch.equal(reference['loss_raw'], changed['loss_raw'])


def test_missing_middle_tier_uses_direct_tier2_over_tier0_constraint():
    _, bad = _run([[0.0, 2.0]], [[0.70, 0.05]])
    scores, good = _run([[2.0, 0.0]], [[0.70, 0.05]])
    assert good['active21_count'].item() == 0
    assert good['active10_count'].item() == 0
    assert good['active20_count'].item() == 1
    assert good['loss_raw'].item() < bad['loss_raw'].item()
    good['loss_raw'].backward()
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 1].item() > 0.0


def test_qahnl_integration_uses_adapter_hit50_logits_and_candidate_mask():
    logits = torch.tensor(
        [[0.0, 2.0, 3.0, 100.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    iou = torch.tensor([[0.70, 0.30, 0.05, 0.80]])
    end_points = {
        'detector_policy_adapter_hit50_logits': logits,
        'detector_policy_adapter_candidate_mask': torch.tensor(
            [[True, True, True, False]], dtype=torch.bool
        ),
        'box_label_mask': torch.ones(1, 1),
    }
    config = {
        'score_source': 'adapter_hit50',
        'loss_weight': 0.75,
        'tiered_quality': True,
        'tier2_iou_thresh': 0.50,
        'pos_iou_thresh': 0.25,
        'neg_iou_thresh': 0.10,
        'tiered_margin21': 0.20,
        'tiered_margin10': 0.10,
        'tiered_temperature': 1.0,
    }
    with mock.patch.object(
        losses, '_target_iou_matrix', return_value=iou
    ), mock.patch.object(
        losses,
        '_dataset_not_scannet_mask',
        return_value=torch.ones(1, dtype=torch.bool),
    ):
        result = losses._qahnl_losses(end_points, indices=None, config=config)
    assert torch.allclose(
        result['loss_qahnl'],
        result['loss_qahnl_raw'] * 0.75,
    )
    result['loss_qahnl'].backward()
    assert logits.grad is not None
    assert logits.grad[0, 3].item() == 0.0
    assert logits.grad[0, 0].item() < 0.0
    assert logits.grad[0, 2].item() > 0.0


if __name__ == '__main__':
    test_ordered_scores_have_lower_loss()
    test_ambiguous_and_masked_candidates_are_ignored()
    test_missing_middle_tier_uses_direct_tier2_over_tier0_constraint()
    test_qahnl_integration_uses_adapter_hit50_logits_and_candidate_mask()
    print('STAGE148_TIERED_QAHNL_TESTS_PASS')
