import importlib.util
import os
from types import SimpleNamespace

import torch
from torch import nn


def _load_candidate(module_name, env_name, fallback_name):
    path = os.environ.get(env_name)
    if path:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return __import__(fallback_name, fromlist=['*'])


main_utils = _load_candidate(
    'main_utils_stage150_candidate',
    'STAGE150_MAIN_UTILS_PATH',
    'main_utils',
)
losses = _load_candidate(
    'models.losses_stage150_candidate',
    'STAGE150_LOSSES_PATH',
    'models.losses',
)


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_mlp = nn.Linear(3, 4)
        self.scalar_mlp = nn.Linear(3, 4)
        self.rerank_head = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.geometry_head = nn.Linear(4, 2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.detector_policy_adapter = _Adapter()


def test_rerank_head_only_freezes_feature_and_geometry_parameters():
    model = _Model()
    args = SimpleNamespace(
        detector_policy_adapter_train_only=True,
        source_pool_selector_train_only=False,
        detector_policy_geometry_train_only=False,
        detector_policy_geometry_extension_train_only=False,
        detector_policy_rank2_rescue_train_only=False,
        detector_policy_alignment_rescue_train_only=False,
        detector_policy_tier_pair_train_only=False,
        detector_policy_boundary_refiner_train_only=False,
        detector_policy_rerank_head_train_only=True,
    )
    count = main_utils._freeze_non_detector_policy_adapter_parameters(
        args, model
    )
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        'detector_policy_adapter.rerank_head.' in name
        for name in trainable
    )
    assert count == sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if 'detector_policy_adapter.rerank_head.' in name
    )


def test_tier2_relation_weight_emphasizes_tier2_ordering():
    scores = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
    iou = torch.tensor([[0.70, 0.30, 0.05]])
    base = losses._qahnl_tiered_ordinal_losses(
        scores, iou, tier2_relation_weight=1.0
    )
    emphasized = losses._qahnl_tiered_ordinal_losses(
        scores, iou, tier2_relation_weight=4.0
    )
    assert emphasized['loss_raw'].item() > base['loss_raw'].item()
    assert emphasized['weighted_relation_count'].item() == 5.0
    emphasized['loss_raw'].backward()
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 1].item() > 0.0


if __name__ == '__main__':
    test_rerank_head_only_freezes_feature_and_geometry_parameters()
    test_tier2_relation_weight_emphasizes_tier2_ordering()
    print('STAGE150_RERANK_ONLY_TIER_WEIGHT_TESTS_PASS')
