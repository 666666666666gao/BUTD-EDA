from types import SimpleNamespace

import torch

from main_utils import (
    _freeze_non_universal_module_parameters,
    _set_non_universal_modules_eval,
    _should_restore_checkpoint_train_state,
)


class TinyUniversalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_net = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)
        )
        self.structured_slot_builder = torch.nn.Linear(2, 2)
        self.sacr_head = torch.nn.Linear(2, 2)
        self.reliability_fusion = torch.nn.Linear(2, 2)
        self.quality_head = torch.nn.Linear(2, 2)


def test_only_shared_universal_modules_remain_trainable():
    model = TinyUniversalModel()
    count = _freeze_non_universal_module_parameters(
        SimpleNamespace(universal_modules_train_only=True), model
    )
    expected_prefixes = (
        'structured_slot_builder.',
        'sacr_head.',
        'reliability_fusion.',
        'quality_head.',
    )
    assert count > 0
    assert all(
        param.requires_grad == name.startswith(expected_prefixes)
        for name, param in model.named_parameters()
    )


def test_frozen_backbone_stays_eval_while_shared_modules_train():
    model = TinyUniversalModel()
    model.train()
    _set_non_universal_modules_eval(model)
    assert not model.backbone_net.training
    assert model.structured_slot_builder.training
    assert model.sacr_head.training
    assert model.reliability_fusion.training
    assert model.quality_head.training


def test_calibration_does_not_restore_full_optimizer_state():
    args = SimpleNamespace(
        eval=False,
        reduce_lr=False,
        universal_modules_train_only=True,
        source_pool_selector_train_only=False,
        detector_policy_adapter_train_only=False,
        freeze_rapf=False,
        freeze_quality_head=False,
    )
    assert not _should_restore_checkpoint_train_state(args, False)
