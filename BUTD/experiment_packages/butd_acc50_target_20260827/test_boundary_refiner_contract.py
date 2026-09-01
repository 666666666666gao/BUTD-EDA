#!/usr/bin/env python3
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main_utils
from models.detector_policy_sources import DetectorPolicyAdapterHead
from models.losses import (
    _detector_policy_boundary_refiner_losses,
    _iou3d_par,
    box_cxcyczwhd_to_xyzxyz,
)


torch.manual_seed(17)
B, Q, D, H = 2, 4, 8, 6
pred_boxes = torch.zeros(B, Q, 6)
pred_boxes[..., 3:] = 1.0
pred_boxes[:, :, 0] = torch.tensor([0.0, 1.5, 3.0, 4.5])
fused = torch.tensor([[4.0, 3.0, 2.0, 1.0]]).expand(B, -1).clone()
features = {
    'quality_scores': torch.zeros(B, Q),
    'class_scores': torch.zeros(B, Q),
    'conf_scores': torch.zeros(B, Q),
    'det_count': torch.full((B,), 4),
    'low_count': torch.zeros(B),
    'detector_support': torch.ones(B, Q),
    'pred_boxes': pred_boxes,
    'matched_det_boxes': pred_boxes.clone(),
    'matched_det_support': torch.ones(B, Q),
    'matched_det_confidence': torch.ones(B, Q),
    'matched_det_valid': torch.ones(B, Q, dtype=torch.bool),
    'query_feats': torch.randn(B, Q, D),
    'fused_scores': fused,
    'base_scores': fused,
    'structured_scores': fused,
    'contrastive_scores': fused,
}

head = DetectorPolicyAdapterHead(
    hidden_dim=H,
    query_dim=D,
    candidate_k=2,
    boundary_refiner=True,
    boundary_refiner_scale=0.25,
)
head.eval()
initial = head(features)
assert torch.equal(
    initial['pre_refined_calibrated_boxes'], initial['calibrated_boxes']
)
assert torch.count_nonzero(initial['boundary_refiner_raw']) == 0
assert torch.count_nonzero(initial['boundary_refiner_delta']) == 0


class Wrapper(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.detector_policy_adapter = adapter
        self.other = nn.Linear(D, D)


wrapper = Wrapper(head)
args = SimpleNamespace(
    detector_policy_adapter_train_only=True,
    source_pool_selector_train_only=False,
    detector_policy_geometry_train_only=False,
    detector_policy_geometry_extension_train_only=False,
    detector_policy_rank2_rescue_train_only=False,
    detector_policy_boundary_refiner_train_only=True,
)
count = main_utils._freeze_non_detector_policy_adapter_parameters(args, wrapper)
trainable = [name for name, p in wrapper.named_parameters() if p.requires_grad]
assert trainable
assert all('boundary_refiner_head.' in name for name in trainable), trainable
assert count == sum(p.numel() for p in head.boundary_refiner_head.parameters())
wrapper.train()
main_utils._set_non_detector_policy_adapter_modules_eval(
    wrapper, boundary_refiner_only=True
)
assert not wrapper.detector_policy_adapter.training
assert wrapper.detector_policy_adapter.boundary_refiner_head.training
assert not wrapper.other.training

gt_center = torch.tensor([[[0.35, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
gt_size = torch.ones(B, 1, 3)


def endpoints(output):
    return {
        'detector_policy_adapter_scores': output['scores'],
        'detector_policy_adapter_pre_refined_calibrated_boxes': (
            output['pre_refined_calibrated_boxes']
        ),
        'detector_policy_adapter_calibrated_boxes': output['calibrated_boxes'],
        'center_label': gt_center,
        'size_gts': gt_size,
        'box_label_mask': torch.ones(B, 1),
        'dataset': ['scanrefer', 'scanrefer'],
    }


def selected_ious(output):
    chosen = output['scores'].argmax(dim=1)
    boxes = output['calibrated_boxes'][torch.arange(B), chosen]
    gt = torch.cat([gt_center[:, 0], gt_size[:, 0]], dim=-1)
    return torch.diag(_iou3d_par(
        box_cxcyczwhd_to_xyzxyz(boxes),
        box_cxcyczwhd_to_xyzxyz(gt),
    )[0])


initial_iou = selected_ious(initial).detach()
initial_scores = initial['scores'].detach().clone()
initial_loss = _detector_policy_boundary_refiner_losses(
    endpoints(initial), weight=1.0, iou_min=0.25, iou_max=0.55,
    stability_weight=0.25,
)['loss_detector_policy_boundary_refiner']
assert initial_loss.item() > 0

optimizer = torch.optim.Adam(head.boundary_refiner_head.parameters(), lr=0.02)
for _ in range(80):
    optimizer.zero_grad(set_to_none=True)
    output = head(features)
    loss = _detector_policy_boundary_refiner_losses(
        endpoints(output), weight=1.0, iou_min=0.25, iou_max=0.55,
        stability_weight=0.25,
    )['loss_detector_policy_boundary_refiner']
    loss.backward()
    optimizer.step()

trained = head(features)
trained_iou = selected_ious(trained).detach()
assert torch.equal(initial_scores, trained['scores'])
assert trained_iou[0] > initial_iou[0] + 0.05, (initial_iou, trained_iou)
assert trained_iou[1] > 0.50, (initial_iou, trained_iou)
assert torch.isfinite(trained['calibrated_boxes']).all()
print(
    'BOUNDARY_REFINER_CONTRACT_PASS',
    'trainable=', count,
    'initial_iou=', initial_iou.tolist(),
    'trained_iou=', trained_iou.tolist(),
)
